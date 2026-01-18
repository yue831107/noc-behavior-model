"""
NoC-to-NoC System for multi-node traffic simulation.

Each compute node has:
- LocalAXIMaster: Initiates transfers to other nodes
- SlaveNI: Converts local AXI requests to NoC flits (user signal routing)
- MasterNI: Receives NoC flits and writes to local memory
- LocalMemory: Storage for this node

This system enables bidirectional communication between nodes using
5 traffic patterns: neighbor, shuffle, bit_reverse, random, transpose.
"""

from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple, TYPE_CHECKING

from ..router import ChannelMode
from src.verification import GoldenManager

if TYPE_CHECKING:
    from ..flit import Flit
    from src.config import NoCTrafficConfig
    from src.testbench.node_controller import NodeController


class NoCSystem:
    """
    NoC-to-NoC System for multi-node traffic simulation.

    Each compute node has:
    - LocalAXIMaster: Initiates transfers to other nodes
    - SlaveNI: Converts local AXI requests to NoC flits (user signal routing)
    - MasterNI: Receives NoC flits and writes to local memory
    - LocalMemory: Storage for this node

    This system enables bidirectional communication between nodes using
    5 traffic patterns: neighbor, shuffle, bit_reverse, random, transpose.
    """

    @property
    def mesh_dimensions(self) -> Tuple[int, int]:
        """Return (cols, rows) of the mesh."""
        return (self.mesh_cols, self.mesh_rows)

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
        """Get flit forwarding stats for each router (all channels)."""
        from ..router import AXIModeRouter

        stats = {}
        for coord, router in self.mesh.routers.items():
            count = 0

            if isinstance(router, AXIModeRouter):
                # AXI Mode: Sum all 5 channels
                for channel_router in router._channel_routers.values():
                    count += channel_router.stats.flits_forwarded
            else:
                # General Mode: Req + Resp
                count = router.req_router.stats.flits_forwarded
                if hasattr(router, 'resp_router'):
                    count += router.resp_router.stats.flits_forwarded

            stats[coord] = count
        return stats

    def get_transfer_stats(self) -> Tuple[int, int, int]:
        """Get transfer completion statistics."""
        completed = 0
        for controller in self.node_controllers.values():
            completed += controller.stats.transfers_completed

        bytes_transferred = 0
        size = 0
        if self._traffic_config:
            size = self._traffic_config.transfer_size
            bytes_transferred = completed * size

        return (completed, bytes_transferred, size)

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
        memory_size: int = 0x100000000,
        channel_mode: ChannelMode = ChannelMode.GENERAL,
    ):
        """
        Initialize NoC-to-NoC System.

        Args:
            mesh_cols: Mesh columns.
            mesh_rows: Mesh rows.
            buffer_depth: Router buffer depth.
            memory_size: Local memory size per node.
            channel_mode: Physical channel mode (GENERAL or AXI).
        """
        from ..mesh import create_mesh
        from src.testbench.node_controller import NodeController
        from ..ni import NIConfig

        self.mesh_cols = mesh_cols
        self.mesh_rows = mesh_rows
        self.buffer_depth = buffer_depth
        self.memory_size = memory_size
        self.channel_mode = channel_mode

        # Calculate number of compute nodes (exclude edge column)
        self.num_nodes = (mesh_cols - 1) * mesh_rows

        # Create mesh with specified channel mode
        self.mesh = create_mesh(
            cols=mesh_cols,
            rows=mesh_rows,
            edge_column=0,
            buffer_depth=buffer_depth,
            channel_mode=channel_mode
        )

        # Create NodeControllers for each compute node
        self.node_controllers: Dict[int, "NodeController"] = {}
        ni_config = NIConfig(
            use_user_signal_routing=True,
            req_buffer_depth=buffer_depth,
            resp_buffer_depth=buffer_depth,
            channel_mode=channel_mode,
        )

        for node_id in range(self.num_nodes):
            self.node_controllers[node_id] = NodeController(
                node_id=node_id,
                mesh_cols=mesh_cols,
                mesh_rows=mesh_rows,
                memory_size=memory_size,
                ni_config=ni_config,
            )

        # Wire NodeControllers to Mesh
        self._wire_nodes_to_mesh()

        # Traffic configuration
        self._traffic_config: Optional["NoCTrafficConfig"] = None

        # Simulation state
        self.current_cycle = 0

        # Statistics
        self._transfers_started = 0
        self._transfers_completed = 0

        # Golden data manager for verification
        self.golden_manager = GoldenManager()

        # Pending flits that couldn't be injected (back-pressure handling)
        self._pending_req_flits: Dict[int, "Flit"] = {}  # node_id -> pending flit

    def _wire_nodes_to_mesh(self) -> None:
        """
        Wire NodeControllers to Mesh routers.

        Share memory between NodeController and mesh.nis so that:
        - When mesh.nis receives a request and writes, NodeController sees it
        - Source data from NodeController is visible for golden generation
        """
        for node_id, controller in self.node_controllers.items():
            coord = controller.coord
            mesh_ni = self.mesh.nis.get(coord)
            if mesh_ni is not None:
                # Share memory: mesh.nis uses NodeController's memory
                # This way writes by mesh.nis go to NodeController.local_memory
                mesh_ni.local_memory = controller.local_memory
                # Also update the AXI slave's memory reference
                mesh_ni.axi_slave.memory = controller.local_memory

    def _node_id_to_coord(self, node_id: int) -> Tuple[int, int]:
        """Convert node ID to (x, y) coordinate."""
        compute_cols = self.mesh_cols - 1
        x = (node_id % compute_cols) + 1
        y = node_id // compute_cols
        return (x, y)

    def configure_traffic(self, config: "NoCTrafficConfig") -> None:
        """
        Configure traffic pattern for all nodes.

        Args:
            config: Traffic configuration with pattern.
        """
        from src.traffic.pattern_generator import TrafficPatternGenerator

        self._traffic_config = config

        # Generate per-node configs from pattern
        generator = TrafficPatternGenerator(self.mesh_cols, self.mesh_rows)
        node_configs = generator.generate(config)

        # Store node configs in traffic config for golden generation
        self._traffic_config.node_configs = node_configs

        # Apply configs to NodeControllers
        for node_config in node_configs:
            node_id = node_config.src_node_id
            if node_id in self.node_controllers:
                self.node_controllers[node_id].configure_transfer(node_config)

    def initialize_node_memory(
        self,
        pattern: str = "sequential",
        seed: int = 42,
        value: int = 0,
    ) -> None:
        """
        Initialize source data in each node's local memory.

        Each node's data is made unique by incorporating node_id into the pattern,
        ensuring proper verification after transfers.

        Args:
            pattern: Data pattern - one of:
                - "sequential": 0x00, 0x01, 0x02, ... with node_id offset
                - "random": Random bytes (deterministic per node with seed)
                - "constant": Fixed value fill (with node_id in first byte)
                - "address": 4-byte address values (with node_id offset)
                - "walking_ones": 0x01, 0x02, 0x04, ... rotated by node_id
                - "walking_zeros": 0xFE, 0xFD, 0xFB, ... rotated by node_id
                - "checkerboard": 0xAA, 0x55, ... (inverted for odd nodes)
                - "node_id": Fill with node_id value (legacy, same as constant)
            seed: Random seed base (for random pattern).
            value: Constant value (for constant pattern).
        """
        import random
        import struct

        for node_id, controller in self.node_controllers.items():
            if self._traffic_config is None:
                continue

            size = self._traffic_config.transfer_size
            src_addr = self._traffic_config.local_src_addr

            if pattern == "sequential":
                # Sequential bytes with node_id offset for uniqueness
                # Node 0: 0x00, 0x01, 0x02, ...
                # Node 1: 0x10, 0x11, 0x12, ...
                data = bytes((node_id * 16 + i) & 0xFF for i in range(size))

            elif pattern == "random":
                # Random data with deterministic seed per node
                rng = random.Random(seed + node_id)
                data = bytes(rng.randint(0, 255) for _ in range(size))

            elif pattern == "constant" or pattern == "node_id":
                # Constant fill - use node_id as first byte for uniqueness
                fill_value = value if pattern == "constant" else node_id
                # First byte is node_id for identification, rest is fill_value
                data = bytes([node_id & 0xFF] + [fill_value & 0xFF] * (size - 1))

            elif pattern == "address":
                # 4-byte little-endian address values with node_id offset
                # Each node uses different base address for uniqueness
                base_addr = src_addr + (node_id << 16)
                data_list = []
                for offset in range(0, size, 4):
                    addr = base_addr + offset
                    addr_bytes = struct.pack("<I", addr & 0xFFFFFFFF)
                    data_list.extend(addr_bytes[:min(4, size - offset)])
                data = bytes(data_list)

            elif pattern == "walking_ones":
                # Walking ones: 0x01, 0x02, 0x04, 0x08, ...
                # First byte is node_id, rest is rotated pattern
                pattern_data = [1 << ((i + node_id) % 8) for i in range(size - 1)]
                data = bytes([node_id & 0xFF] + pattern_data)

            elif pattern == "walking_zeros":
                # Walking zeros: 0xFE, 0xFD, 0xFB, 0xF7, ...
                # First byte is node_id, rest is rotated pattern
                pattern_data = [~(1 << ((i + node_id) % 8)) & 0xFF for i in range(size - 1)]
                data = bytes([node_id & 0xFF] + pattern_data)

            elif pattern == "checkerboard":
                # Checkerboard: 0xAA, 0x55, 0xAA, 0x55, ...
                # First byte is node_id, rest is pattern (inverted for odd nodes)
                if node_id % 2 == 0:
                    pattern_data = [0xAA if i % 2 == 0 else 0x55 for i in range(size - 1)]
                else:
                    pattern_data = [0x55 if i % 2 == 0 else 0xAA for i in range(size - 1)]
                data = bytes([node_id & 0xFF] + pattern_data)

            else:
                # Default: zeros with node_id in first byte
                data = bytes([node_id & 0xFF] + [0] * (size - 1))

            controller.initialize_memory(src_addr, data)

    def load_node_memory_from_files(
        self,
        payload_dir: str,
    ) -> int:
        """
        Load node memory data from per-node binary files.

        Expects files named: node_00.bin, node_01.bin, ..., node_15.bin
        Payload file size must be >= config.transfer_size.

        Args:
            payload_dir: Directory containing node_XX.bin files.

        Returns:
            Number of nodes loaded successfully.
        """
        from pathlib import Path

        payload_path = Path(payload_dir)
        if not payload_path.exists():
            raise FileNotFoundError(f"Payload directory not found: {payload_dir}")

        if self._traffic_config is None:
            raise ValueError("Traffic not configured")

        loaded_count = 0
        expected_size = self._traffic_config.transfer_size

        for node_id, controller in self.node_controllers.items():
            bin_file = payload_path / f"node_{node_id:02d}.bin"
            if not bin_file.exists():
                raise FileNotFoundError(f"Payload file not found: {bin_file}")

            # Read binary data from file
            data = bin_file.read_bytes()

            # Check size
            if len(data) < expected_size:
                raise ValueError(
                    f"Payload file {bin_file} too small: "
                    f"{len(data)} bytes < {expected_size} required"
                )

            # Truncate to transfer_size
            data = data[:expected_size]

            # Load into node memory
            src_addr = self._traffic_config.local_src_addr
            controller.initialize_memory(src_addr, data)
            loaded_count += 1

        return loaded_count

    def start_all_transfers(self) -> None:
        """Start all configured transfers."""
        for controller in self.node_controllers.values():
            controller.start_transfer()
        self._transfers_started = len(self.node_controllers)

    def process_cycle(self) -> None:
        """
        Process one simulation cycle.

        This coordinates between NodeControllers and Mesh:
        1. Each node generates outgoing flits
        2. Mesh routes flits
        3. Each node receives incoming flits
        """
        # Phase 1: NodeControllers generate outgoing flits
        # and process their internal state
        for controller in self.node_controllers.values():
            controller.process_cycle(self.current_cycle)

        # Phase 2: Transfer flits from NodeControllers to Mesh
        self._inject_flits_to_mesh()

        # Phase 3: Mesh processes routing
        self.mesh.process_cycle(self.current_cycle)

        # Phase 4: Transfer flits from Mesh to NodeControllers
        self._deliver_flits_from_mesh()

        self.current_cycle += 1

    def _inject_flits_to_mesh(self) -> None:
        """
        Inject outgoing request flits from NodeControllers into Mesh.

        NodeController.SlaveNI generates request flits.
        These are injected into the local router's request LOCAL input.
        Handles back-pressure by buffering rejected flits for retry.
        """
        from ..router import Direction, AXIModeRouter

        for node_id, controller in self.node_controllers.items():
            coord = controller.coord
            router = self.mesh.routers.get(coord)
            if router is None:
                continue

            # First, try to inject any pending flit from previous cycle
            if node_id in self._pending_req_flits:
                pending = self._pending_req_flits[node_id]
                if self._inject_flit_to_router(router, pending):
                    # Successfully injected pending flit
                    del self._pending_req_flits[node_id]
                # If still failed, keep it pending and skip getting new flit
                else:
                    continue

            # Get outgoing request flit from NodeController's SlaveNI
            flit = controller.get_outgoing_flit()
            if flit is not None:
                # Inject into router's request LOCAL port
                success = self._inject_flit_to_router(router, flit)
                if not success:
                    # Router couldn't accept - buffer for retry next cycle
                    self._pending_req_flits[node_id] = flit

    def _inject_flit_to_router(self, router, flit) -> bool:
        """
        Inject flit into router, handling both General and AXI modes.

        Args:
            router: Router instance (Router or AXIModeRouter)
            flit: Flit to inject

        Returns:
            True if accepted, False if rejected.
        """
        from ..router import Direction, AXIModeRouter

        if isinstance(router, AXIModeRouter):
            # AXI Mode: route to channel-specific Sub-Router
            channel = flit.hdr.axi_ch
            sub_router = router.get_channel_router(channel)
            return sub_router.receive_flit(Direction.LOCAL, flit)
        else:
            # General Mode: use receive_request
            return router.receive_request(Direction.LOCAL, flit)

    def _deliver_flits_from_mesh(self) -> None:
        """
        Deliver response flits from Mesh back to NodeControllers.

        When a response flit arrives at a compute node's router,
        it needs to be delivered to the source NodeController's SlaveNI.

        Also inject response flits from mesh.nis (MasterNI) into router.
        """
        from ..router import Direction, AXIModeRouter

        for node_id, controller in self.node_controllers.items():
            coord = controller.coord
            router = self.mesh.routers.get(coord)
            if router is None:
                continue

            if isinstance(router, AXIModeRouter):
                # AXI Mode: check both B and R channels
                self._deliver_channel_flit(router.get_b_port(Direction.LOCAL), controller)
                self._deliver_channel_flit(router.get_r_port(Direction.LOCAL), controller)
            else:
                # General Mode: check response LOCAL port
                resp_local = router.get_resp_port(Direction.LOCAL)
                self._deliver_channel_flit(resp_local, controller)

    def _deliver_channel_flit(self, port, controller) -> bool:
        """
        Deliver flit from a router port to controller.

        Args:
            port: RouterPort to check for outgoing flit
            controller: NodeController to deliver to

        Returns:
            True if flit was delivered, False otherwise.
        """
        if port.out_valid and port.out_flit is not None:
            flit = port.out_flit
            # Deliver to NodeController's SlaveNI response path
            if controller.receive_response_flit(flit):
                # Clear router's output
                port.out_valid = False
                port.out_flit = None
                return True
        return False

    @property
    def all_transfers_complete(self) -> bool:
        """Check if all node transfers are complete."""
        return all(
            controller.is_transfer_complete
            for controller in self.node_controllers.values()
        )

    @property
    def transfer_progress(self) -> float:
        """Get overall transfer progress (0.0 - 1.0)."""
        if not self.node_controllers:
            return 0.0
        completed = sum(
            1 for controller in self.node_controllers.values()
            if controller.is_transfer_complete
        )
        return completed / len(self.node_controllers)

    def run_until_complete(self, max_cycles: int = 10000) -> int:
        """
        Run simulation until all transfers complete.

        Args:
            max_cycles: Maximum cycles to run.

        Returns:
            Number of cycles run.
        """
        cycles_run = 0
        while not self.all_transfers_complete and cycles_run < max_cycles:
            self.process_cycle()
            cycles_run += 1
        return cycles_run


    def _coord_to_node_id(self, coord: Tuple[int, int]) -> int:
        """Convert coordinate to node ID."""
        x, y = coord
        if x < 1:
            return -1
        compute_cols = self.mesh_cols - 1
        return y * compute_cols + (x - 1)

    def get_node_summary(self, node_id: int) -> Optional[Dict]:
        """Get summary for specific node."""
        controller = self.node_controllers.get(node_id)
        if controller is None:
            return None
        return controller.get_summary()

    def print_status(self) -> None:
        """Print system status."""
        print(f"=== NoC-to-NoC System Status (cycle {self.current_cycle}) ===")
        print(f"Nodes: {self.num_nodes}")
        print(f"Mesh: {self.mesh_cols}x{self.mesh_rows}")

        complete = sum(1 for c in self.node_controllers.values()
                       if c.is_transfer_complete)
        print(f"Transfers complete: {complete}/{len(self.node_controllers)}")
        print()

    def generate_golden(self) -> int:
        """
        Generate golden data based on current traffic config.

        Must be called AFTER:
          1. configure_traffic() - so node_configs exist
          2. initialize_node_memory() - so source data exists

        Golden is generated by reading each source node's memory
        and storing as expected data for the destination node.

        Returns:
            Number of golden entries generated.
        """
        if self._traffic_config is None:
            raise ValueError("Traffic not configured")

        node_configs = self._traffic_config.node_configs
        if node_configs is None:
            raise ValueError("No node configs available")

        def get_node_memory(node_id: int):
            return self.node_controllers[node_id].local_memory

        return self.golden_manager.generate_noc_golden(
            node_configs=node_configs,
            get_node_memory=get_node_memory,
            mesh_cols=self.mesh_cols,
        )

    def verify_transfers(self):
        """
        Verify all transfers against golden data.

        Reads actual data from each destination node's memory
        and compares against the golden data.

        Returns:
            VerificationReport with detailed results.
        """
        from src.verification import GoldenKey

        if self._traffic_config is None:
            raise ValueError("Traffic not configured")

        node_configs = self._traffic_config.node_configs
        if node_configs is None:
            raise ValueError("No node configs available")

        # Collect actual data from all destination nodes
        read_results: Dict[Tuple[int, int], bytes] = {}

        for nc in node_configs:
            dest_x, dest_y = nc.dest_coord
            if dest_x < 1 or dest_x >= self.mesh_cols:
                continue

            dst_node_id = self._coord_to_node_id(nc.dest_coord)
            if dst_node_id < 0:
                continue

            dst_controller = self.node_controllers.get(dst_node_id)
            if dst_controller is None:
                continue

            actual_data = dst_controller.read_local_memory(
                nc.local_dst_addr, nc.transfer_size
            )
            key = (dst_node_id, nc.local_dst_addr)
            read_results[key] = actual_data

        return self.golden_manager.verify(read_results)
