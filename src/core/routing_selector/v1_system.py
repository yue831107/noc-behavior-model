"""
V1 System: NI + Selector + Mesh for Host-to-NoC architecture.

Single entry/exit point topology where all traffic goes through
the Routing Selector.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..router import ChannelMode
from src.verification import GoldenManager, VerificationReport
from .config import RoutingSelectorConfig
from .selector import RoutingSelector

if TYPE_CHECKING:
    from ..mesh import Mesh
    from ..ni import SlaveNI
    from src.testbench.memory import Memory
    from src.testbench.host_axi_master import HostAXIMaster
    from src.config import TransferConfig
    from src.testbench.axi_master import AXIIdConfig


class V1System:
    """
    Complete V1 System integrating NI, Selector, and Mesh.

    This is the top-level container for the Single Entry Routing
    Selector architecture.

    Optionally supports HostAXIMaster for DMA-style transfers.
    """

    @property
    def current_cycle(self) -> int:
        """Current simulation cycle."""
        return self.current_time

    @property
    def mesh_dimensions(self) -> Tuple[int, int]:
        """Return (cols, rows) of the mesh."""
        return (self._mesh_cols, self._mesh_rows)

    def get_buffer_occupancy(self) -> Dict[Tuple[int, int], int]:
        """
        Get flits in transit for each router in the mesh.

        Counts all flits currently in the router:
        - Input buffer occupancy
        - Pending output signals (out_valid = True)
        - Flits in pipeline stages (for multi-stage routers)

        This provides accurate buffer utilization even in fast mode
        where flits move through in a single cycle.
        """
        from ..router import AXIModeRouter

        occupancy = {}
        for coord, router in self.mesh.routers.items():
            occ = 0

            if isinstance(router, AXIModeRouter):
                # AXI Mode: 5 Sub-Routers
                for channel_router in router._channel_routers.values():
                    for port in channel_router.ports.values():
                        if hasattr(port, 'input_buffer'):
                            occ += port.input_buffer.occupancy
                        if port.out_valid and port.out_flit is not None:
                            occ += 1
                    occ += channel_router.flits_in_pipeline
            else:
                # General Mode: 2 Sub-Routers
                for port in router.req_router.ports.values():
                    if hasattr(port, 'input_buffer'):
                        occ += port.input_buffer.occupancy
                    if port.out_valid and port.out_flit is not None:
                        occ += 1
                occ += router.req_router.flits_in_pipeline

                for port in router.resp_router.ports.values():
                    if hasattr(port, 'input_buffer'):
                        occ += port.input_buffer.occupancy
                    if port.out_valid and port.out_flit is not None:
                        occ += 1
                occ += router.resp_router.flits_in_pipeline

            occupancy[coord] = occ
        return occupancy

    def get_flit_stats(self) -> Dict[Tuple[int, int], int]:
        """Get flit forwarding stats for each router (all sub-routers)."""
        from ..router import AXIModeRouter

        stats = {}
        for coord, router in self.mesh.routers.items():
            count = 0
            if isinstance(router, AXIModeRouter):
                # AXI Mode: 5 Sub-Routers
                for channel_router in router._channel_routers.values():
                    count += channel_router.stats.flits_forwarded
            else:
                # General Mode: 2 Sub-Routers
                count = router.req_router.stats.flits_forwarded
                if hasattr(router, 'resp_router'):
                    count += router.resp_router.stats.flits_forwarded
            stats[coord] = count
        return stats

    def get_transfer_stats(self) -> Tuple[int, int, int]:
        """Get transfer completion statistics."""
        if self.host_axi_master:
            stats = self.host_axi_master.controller_stats
            completed = stats.completed_transactions + stats.read_completed
            bytes_transferred = stats.completed_bytes + stats.read_bytes_received

            config = self.host_axi_master.transfer_config
            from src.config import TransferMode
            if config.transfer_mode in (TransferMode.BROADCAST, TransferMode.BROADCAST_READ):
                size = config.src_size  # BROADCAST: 每個 node 收到完整數據
            else:
                size = config.src_size // max(1, len(config.get_target_node_list(16)))

            return (completed, bytes_transferred, size)
        return (0, 0, 0)

    def set_packet_arrival_callback(self, callback: Callable[[int, int, int], None]) -> None:
        """
        Set callback for packet arrival notification.

        Propagates callback to all MasterNIs in the mesh.

        Args:
            callback: Function that receives (packet_id, creation_time, arrival_time).
        """
        for ni in self.mesh.nis.values():
            ni.set_packet_arrival_callback(callback)

    def __init__(
        self,
        mesh_cols: int = 5,
        mesh_rows: int = 4,
        buffer_depth: int = 32,
        max_outstanding: int = 16,
        selector_config: Optional[RoutingSelectorConfig] = None,
        host_memory: Optional["Memory"] = None,
        channel_mode: ChannelMode = ChannelMode.GENERAL,
    ):
        """
        Initialize V1 System.

        Args:
            mesh_cols: Mesh columns.
            mesh_rows: Mesh rows.
            buffer_depth: Router buffer depth.
            max_outstanding: Max outstanding transactions (affects NI and safety buffer sizing).
            selector_config: Selector configuration.
            host_memory: Optional Host Memory for DMA transfers.
            channel_mode: Physical channel mode (GENERAL or AXI).
        """
        from ..mesh import create_mesh
        from ..ni import SlaveNI, NIConfig
        from src.address.address_map import SystemAddressMap, AddressMapConfig

        self._mesh_cols = mesh_cols
        self._mesh_rows = mesh_rows
        self._buffer_depth = buffer_depth
        self._max_outstanding = max_outstanding
        self.channel_mode = channel_mode

        # Create mesh
        self.mesh = create_mesh(
            cols=mesh_cols,
            rows=mesh_rows,
            edge_column=0,
            buffer_depth=buffer_depth,
            channel_mode=channel_mode,
        )

        # Create selector with safe buffer depths
        # CRITICAL: ingress_buffer_depth must be >= max_outstanding to prevent deadlock
        # When max_outstanding transactions are submitted, all their flits must fit
        # in the ingress buffer, otherwise transactions block and cause deadlock.
        # Ensure ingress buffer can hold all outstanding transaction flits
        # Each transaction may produce multiple flits, so we need extra margin
        safe_ingress_depth = max(buffer_depth, max_outstanding)

        if selector_config is None:
            selector_config = RoutingSelectorConfig(
                ingress_buffer_depth=safe_ingress_depth,
                egress_buffer_depth=safe_ingress_depth,
                channel_mode=channel_mode,  # Pass channel mode to selector
            )
        else:
            # Ensure channel_mode is set even if config was provided
            selector_config.channel_mode = channel_mode
        self.selector = RoutingSelector(selector_config)

        # Connect selector to edge routers
        self.selector.connect_edge_routers(self.mesh.edge_routers)

        # Create Slave NI for Host side (AXI Slave interface)
        # This receives AXI Master requests from Host CPU/DMA
        # and converts them to NoC Request Flits
        self.address_map = SystemAddressMap(
            AddressMapConfig(
                mesh_cols=mesh_cols,
                mesh_rows=mesh_rows,
                edge_column=0,
            )
        )
        # Note: This is a SlaveNI (receives from AXI Master)
        # Named "master_ni" for backward compatibility with V1System API
        # NI buffer depth must be large enough to hold flits for max_outstanding transactions
        # Each transaction can produce multiple flits (e.g., 256B / 32B = 8 flits)
        # Use safe_ingress_depth to ensure adequate buffering
        self.master_ni = SlaveNI(
            coord=(0, 0),  # Virtual coord for Host Slave NI
            address_map=self.address_map,
            config=NIConfig(
                req_buffer_depth=safe_ingress_depth,
                resp_buffer_depth=safe_ingress_depth,
                max_outstanding=max_outstanding,
                channel_mode=channel_mode,  # Pass channel mode to NI
            ),
            ni_id=0,
        )

        # Optional Host Memory and AXI Master for DMA transfers
        self.host_memory: Optional["Memory"] = host_memory
        self.host_axi_master: Optional["HostAXIMaster"] = None

        # Simulation time
        self.current_time = 0

        # Golden Manager for all memory verification (Write and Read)
        self.golden_manager = GoldenManager()

    def process_cycle(self) -> None:
        """
        Process one simulation cycle with coordinated PortWire timing.

        The Selector and Mesh must coordinate their phased processing:
        1. Host AXI Master sends requests (if enabled)
        2. Master NI generates request flits
        3. Selector propagates request outputs
        4. Mesh samples, processes, sets response outputs
        5. Selector clears accepted request outputs
        6. Selector samples response inputs, processes ingress/egress
        7. Master NI handles responses
        8. Host AXI Master receives responses (if enabled)
        """
        # 0. Host AXI Master: Send requests to master_ni (but not receive yet)
        if self.host_axi_master is not None:
            # Only generate and send, don't receive responses yet
            self.host_axi_master._generate_transactions(self.current_time)
            self.host_axi_master._send_axi_requests(self.current_time)
            self.host_axi_master.stats.total_cycles = self.current_time + 1

        # 1. Master NI: Process input FIFOs and generate request flits
        # First, process AXI input FIFOs to convert requests to flits
        self.master_ni.req_path.process_cycle(self.current_time)

        # Then transfer flits from output buffers to selector
        while self.master_ni.req_ni.has_pending_output():
            if self.selector.ingress_buffer.is_full():
                break
            flit = self.master_ni.get_req_flit(self.current_time)
            if flit is not None:
                if not self.selector.accept_request(flit):
                    break

        # 2. Selector Phase 1: Update ready and propagate request outputs
        # This must happen BEFORE Mesh samples so EdgeRouters see Selector's outputs
        self.selector.update_all_ready()
        self.selector.propagate_all_wires()

        # 3. Mesh: Process all routers and NIs
        # EdgeRouters will sample Selector's propagated outputs
        # Then process and set response outputs
        self.mesh.process_cycle(self.current_time)

        # 4. Clear accepted request outputs AFTER Mesh has sampled
        # This handles the handshake completion
        self.selector.clear_accepted_outputs()
        self.selector.handle_credit_release()

        # 5. Selector Phase 2: Propagate to get Mesh's response outputs
        self.selector.propagate_all_wires()

        # 6. Selector Phase 3-5: Sample responses, process ingress/egress
        self.selector.sample_all_inputs()
        self.selector.clear_all_input_signals()

        # 6b. Clear EdgeRouter response LOCAL outputs after sampling
        # This completes the handshake (EdgeRouter skips LOCAL in its clear_accepted_outputs)
        self.selector.clear_edge_resp_outputs()

        self.selector._process_ingress(self.current_time)
        self.selector._process_egress(self.current_time)

        # 7. Propagate new request outputs (set during _process_ingress)
        self.selector.propagate_all_wires()

        # 8. Master NI: Process responses
        while self.selector.has_pending_responses:
            flit = self.selector.get_response()
            if flit is not None:
                self.master_ni.receive_resp_flit(flit)

        self.master_ni.process_cycle(self.current_time)

        # 9. Host AXI Master: Receive responses and check completion
        if self.host_axi_master is not None:
            from src.testbench.host_axi_master import HostAXIMasterState
            self.host_axi_master._receive_axi_responses(self.current_time)
            # Check completion based on mode
            if self.host_axi_master._is_read_mode:
                # Read mode completion
                if (self.host_axi_master._controller.read_is_complete and
                    not self.host_axi_master._pending_ar_queue):
                    self.host_axi_master._state = HostAXIMasterState.COMPLETE
                    self.host_axi_master.stats.last_r_cycle = self.current_time
                    # Queue mode: advance to next transfer
                    if self.host_axi_master._queue_mode:
                        self.host_axi_master._advance_queue()
            else:
                # Write mode completion
                if (self.host_axi_master._controller.is_complete and
                    not self.host_axi_master._pending_aw_queue and
                    not self.host_axi_master._pending_w_beats):
                    self.host_axi_master._state = HostAXIMasterState.COMPLETE
                    self.host_axi_master.stats.last_b_cycle = self.current_time
                    # Queue mode: advance to next transfer
                    if self.host_axi_master._queue_mode:
                        self.host_axi_master._advance_queue()

        self.current_time += 1

    def submit_write(
        self,
        addr: int,
        data: bytes,
        axi_id: int = 0
    ) -> bool:
        """
        Submit AXI write transaction.

        Args:
            addr: 64-bit AXI address.
            data: Data to write.
            axi_id: AXI transaction ID.

        Returns:
            True if accepted.
        """
        from src.axi.interface import AXI_AW, AXI_W, AXISize

        aw = AXI_AW(
            awid=axi_id,
            awaddr=addr,
            awlen=0,  # Single beat
            awsize=AXISize.SIZE_8,
        )

        if not self.master_ni.process_aw(aw, self.current_time):
            return False

        w = AXI_W(
            wdata=data,
            wstrb=0xFF,
            wlast=True,
        )

        result = self.master_ni.process_w(w, axi_id, self.current_time)

        # Record golden pattern for verification
        if result:
            node_id = self.address_map.extract_node_id(addr)
            local_addr = self.address_map.extract_local_addr(addr)
            self.golden_manager.capture_write(
                node_id=node_id,
                addr=local_addr,
                data=data,
                cycle=self.current_time
            )

        return result

    def submit_read(
        self,
        addr: int,
        size: int = 8,
        axi_id: int = 0
    ) -> bool:
        """
        Submit AXI read transaction.

        Args:
            addr: 64-bit AXI address.
            size: Read size in bytes.
            axi_id: AXI transaction ID.

        Returns:
            True if accepted.
        """
        from src.axi.interface import AXI_AR, AXISize

        # Calculate burst length based on size
        beat_size = 8  # AXISize.SIZE_8 = 8 bytes per beat
        burst_length = (size + beat_size - 1) // beat_size  # Round up
        burst_length = max(1, min(burst_length, 256))  # AXI4 max is 256 beats
        arlen = burst_length - 1  # arlen = burst_length - 1

        ar = AXI_AR(
            arid=axi_id,
            araddr=addr,
            arlen=arlen,
            arsize=AXISize.SIZE_8,
        )

        return self.master_ni.process_ar(ar, self.current_time)

    def run(self, cycles: int) -> None:
        """
        Run simulation for given cycles.

        Args:
            cycles: Number of cycles to run.
        """
        for _ in range(cycles):
            self.process_cycle()

    def print_status(self) -> None:
        """Print system status."""
        print(f"=== V1 System Status (cycle {self.current_time}) ===")
        print(f"Master NI: outstanding={self.master_ni.req_ni.outstanding_count}")
        self.selector.print_status()
        print(f"Mesh: cycles={self.mesh.stats.total_cycles}")
        print()


    def verify_all_writes(self, verbose: bool = True) -> Tuple[int, int]:
        """
        Verify all submitted writes against golden patterns.

        Compares the actual memory contents at each destination node
        with the expected data stored in GoldenManager.

        Args:
            verbose: If True, print summary and failures.

        Returns:
            Tuple of (pass_count, fail_count).
        """
        # Collect actual data from all nodes mentioned in golden patterns
        read_results = {}
        for entry in self.golden_manager.entries:
            if isinstance(entry.node_id, int):
                node_coords = self.address_map.get_coord(entry.node_id)
                ni = self.mesh.nis.get(node_coords)
                if ni is not None:
                    actual_data, _ = ni.local_memory.read(entry.local_addr, len(entry.data))
                    read_results[(entry.node_id, entry.local_addr)] = actual_data

        # Verify using GoldenManager
        report = self.golden_manager.verify(read_results)

        if verbose:
            self.golden_manager.print_report(report)

        return report.passed, report.failed

    def clear_golden_patterns(self) -> None:
        """Clear all recorded golden patterns."""
        self.golden_manager.clear()

    @property
    def golden_pattern_count(self) -> int:
        """Number of recorded golden patterns."""
        return self.golden_manager.entry_count

    # === Host AXI Master DMA Transfer API ===

    def configure_transfer(
        self,
        transfer_config: "TransferConfig",
        axi_id_config: Optional["AXIIdConfig"] = None,
    ) -> None:
        """
        Configure DMA transfer using Host AXI Master.

        Args:
            transfer_config: Transfer configuration.
            axi_id_config: Optional AXI ID configuration.

        Raises:
            ValueError: If host_memory is not set.
        """
        from src.testbench.host_axi_master import HostAXIMaster
        from src.testbench.axi_master import AXIIdConfig

        if self.host_memory is None:
            raise ValueError(
                "host_memory must be set to use DMA transfers. "
                "Pass host_memory to V1System constructor."
            )

        self.host_axi_master = HostAXIMaster(
            host_memory=self.host_memory,
            transfer_config=transfer_config,
            axi_id_config=axi_id_config or AXIIdConfig(),
            mesh_cols=self._mesh_cols,
            mesh_rows=self._mesh_rows,
        )
        self.host_axi_master.connect_to_slave_ni(self.master_ni)

    def start_transfer(self) -> bool:
        """
        Start the configured DMA transfer.

        Returns:
            True if transfer started successfully.
        """
        if self.host_axi_master is None:
            return False

        self.host_axi_master.start()
        return True

    @property
    def transfer_complete(self) -> bool:
        """Check if DMA transfer is complete."""
        if self.host_axi_master is None:
            return True  # No transfer configured
        return self.host_axi_master.is_complete

    @property
    def transfer_progress(self) -> float:
        """Get DMA transfer progress (0.0 - 1.0)."""
        if self.host_axi_master is None:
            return 0.0  # No transfer configured yet
        return self.host_axi_master.progress

    def run_until_transfer_complete(self, max_cycles: int = 10000) -> int:
        """
        Run simulation until DMA transfer completes.

        Args:
            max_cycles: Maximum cycles to run.

        Returns:
            Number of cycles run.
        """
        cycles_run = 0
        while not self.transfer_complete and cycles_run < max_cycles:
            self.process_cycle()
            cycles_run += 1
        return cycles_run

    def get_transfer_summary(self) -> Optional[Dict]:
        """Get DMA transfer summary."""
        if self.host_axi_master is None:
            return None
        return self.host_axi_master.get_summary()

    # === Read Transfer and Verification API ===

    def configure_read_transfer(
        self,
        transfer_config: "TransferConfig",
        axi_id_config: Optional["AXIIdConfig"] = None,
        use_golden: bool = True,
    ) -> None:
        """
        Configure read-back transfer using Host AXI Master.

        Args:
            transfer_config: Transfer configuration (must be read mode).
            axi_id_config: Optional AXI ID configuration.
            use_golden: If True, use golden_manager data for verification.

        Raises:
            ValueError: If host_memory is not set or mode is not read.
        """
        from src.testbench.host_axi_master import HostAXIMaster
        from src.testbench.axi_master import AXIIdConfig

        if self.host_memory is None:
            raise ValueError(
                "host_memory must be set to use DMA transfers. "
                "Pass host_memory to V1System constructor."
            )

        if not transfer_config.is_read:
            raise ValueError(
                f"Transfer mode must be read (BROADCAST_READ or GATHER), "
                f"got {transfer_config.transfer_mode.value}"
            )

        self.host_axi_master = HostAXIMaster(
            host_memory=self.host_memory,
            transfer_config=transfer_config,
            axi_id_config=axi_id_config or AXIIdConfig(),
            mesh_cols=self._mesh_cols,
            mesh_rows=self._mesh_rows,
        )
        self.host_axi_master.connect_to_slave_ni(self.master_ni)

        # Configure for read with optional golden verification
        golden_store = None
        if use_golden:
            golden_store = self.golden_manager.get_golden_store()
        self.host_axi_master.configure_read(golden_store)

    def start_read_transfer(self) -> bool:
        """
        Start the configured read transfer.

        Returns:
            True if transfer started successfully.
        """
        if self.host_axi_master is None:
            return False
        if not self.host_axi_master._is_read_mode:
            return False

        self.host_axi_master.start()
        return True

    def get_read_data(self) -> Dict[Tuple[int, int], bytes]:
        """
        Get read data collected from nodes.

        Returns:
            Dict of (node_id, local_addr) -> data bytes.
        """
        if self.host_axi_master is None:
            return {}
        return self.host_axi_master.read_data

    def verify_read_results(self) -> VerificationReport:
        """
        Verify read results against golden data.

        Returns:
            VerificationReport with detailed results.
        """
        read_data = self.get_read_data()
        return self.golden_manager.verify(read_data)

    def print_verification_report(self, show_data_bytes: int = 64) -> None:
        """
        Print read-back verification report.

        Args:
            show_data_bytes: Max bytes to show in data preview.
        """
        report = self.verify_read_results()
        self.golden_manager.print_report(report, show_data_bytes)

    def capture_golden_from_write(
        self,
        node_id: int,
        local_addr: int,
        data: bytes,
        cycle: int = 0
    ) -> None:
        """
        Capture golden data during write operation.

        This is called automatically by DMA write transfers.
        Can also be called manually for non-DMA writes.

        Args:
            node_id: Target node ID.
            local_addr: Local memory address.
            data: Data being written.
            cycle: Simulation cycle when captured.
        """
        self.golden_manager.capture_write(node_id, local_addr, data, cycle)

    def reset_for_read(self) -> None:
        """
        Reset system state for a new read operation.

        Call this after a write transfer completes and before
        configuring a read transfer on the same system.
        """
        if self.host_axi_master is not None:
            self.host_axi_master.reset()
        # Note: golden_manager is NOT cleared - we need it for verification
