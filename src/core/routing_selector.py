"""
Routing Selector implementation for Version 1 architecture.

The Routing Selector is the single entry/exit point between the
master NI and the 2D mesh network. It connects to 4 Edge Routers
(Column 0) via their Local ports.

Features:
- Ingress Path Selection: Route packets into mesh based on hop count and credits
- Egress Arbitration: Collect responses from mesh based on buffer occupancy
- Req/Resp physical separation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Deque, Callable
from collections import deque
from enum import Enum, auto

from .flit import Flit, FlitType
from .buffer import FlitBuffer, CreditFlowControl
from .router import RouterPort, Direction, EdgeRouter, PortWire
from .golden_manager import GoldenManager, VerificationReport


@dataclass
class RoutingSelectorConfig:
    """Routing Selector configuration."""
    num_directions: int = 4         # Number of connected Edge Routers
    ingress_buffer_depth: int = 8   # Ingress buffer depth
    egress_buffer_depth: int = 8    # Egress buffer depth
    hop_weight: float = 1.0         # Hop count weight for path selection
    credit_weight: float = 1.0      # Credit weight for path selection


@dataclass
class SelectorStats:
    """Routing Selector statistics."""
    # Ingress (into mesh)
    req_flits_received: int = 0
    req_flits_injected: int = 0
    req_blocked_no_credit: int = 0

    # Egress (from mesh)
    resp_flits_collected: int = 0
    resp_flits_sent: int = 0

    # Path selection
    path_selections: Dict[int, int] = field(default_factory=dict)  # row -> count

    def __post_init__(self):
        for i in range(4):
            self.path_selections[i] = 0


class EdgeRouterPort:
    """
    Connection to a single Edge Router.

    Manages both request (to mesh) and response (from mesh) channels.
    Uses valid/ready handshake via PortWire for communication.
    """

    def __init__(
        self,
        row: int,
        buffer_depth: int = 8
    ):
        """
        Initialize edge router port.

        Args:
            row: Edge router row (0-3).
            buffer_depth: Buffer depth for both channels.
        """
        self.row = row
        self.coord = (0, row)  # Edge routers are in column 0
        self._buffer_depth = buffer_depth

        # NEW: Signal-based interface using RouterPort
        # Request port: Selector sends TO EdgeRouter
        self._req_port = RouterPort(
            direction=Direction.LOCAL,
            buffer_depth=buffer_depth,
            name=f"Sel_req_to_edge{row}"
        )

        # Response port: Selector receives FROM EdgeRouter
        self._resp_port = RouterPort(
            direction=Direction.LOCAL,
            buffer_depth=buffer_depth,
            name=f"Sel_resp_from_edge{row}"
        )

        # Wire connections (set during connect_edge_routers)
        self._req_wire: Optional[PortWire] = None
        self._resp_wire: Optional[PortWire] = None

        # Connected edge router
        self._edge_router: Optional[EdgeRouter] = None

    def connect_edge_router(self, edge_router: EdgeRouter) -> None:
        """Connect to an Edge Router."""
        self._edge_router = edge_router

    # =========================================================================
    # Request Path Signal Methods (Selector -> EdgeRouter)
    # =========================================================================

    def update_req_ready(self) -> None:
        """Update request port's ready signal (not used for output direction)."""
        # For request path, we're the sender - no ready to update
        # But we may receive ready from EdgeRouter via wire
        pass

    def can_send_request(self) -> bool:
        """Check if we can send request via signal interface."""
        return self._req_port.can_send()

    def set_req_output(self, flit: Flit) -> bool:
        """
        Set request output for sending to EdgeRouter via PortWire.

        Args:
            flit: Request flit to send.

        Returns:
            True if output was set successfully.
        """
        return self._req_port.set_output(flit)

    def clear_req_if_accepted(self) -> bool:
        """
        Clear request output if EdgeRouter accepted (handshake completed).

        Returns:
            True if output was accepted and cleared.
        """
        return self._req_port.clear_output_if_accepted()

    # =========================================================================
    # Response Path Signal Methods (EdgeRouter -> Selector)
    # =========================================================================

    def update_resp_ready(self) -> None:
        """Update response port's ready signal based on buffer availability."""
        self._resp_port.update_ready()

    def sample_resp_input(self) -> bool:
        """
        Sample response input from EdgeRouter via PortWire.

        Called after wire propagation.

        Returns:
            True if a flit was received.
        """
        return self._resp_port.sample_input()

    def clear_resp_input_signals(self) -> None:
        """Clear response input signals after sampling."""
        self._resp_port.clear_input_signals()

    def get_response(self) -> Optional[Flit]:
        """
        Get response flit from the response port buffer.

        Returns:
            Response flit if available.
        """
        return self._resp_port.pop_for_routing()

    def release_credit(self, count: int = 1) -> None:
        """Release credits (edge router freed buffer space)."""
        self._req_port._output_credit.release(count)

    @property
    def available_credits(self) -> int:
        """Available request credits."""
        return self._req_port._output_credit.credits

    @property
    def resp_occupancy(self) -> int:
        """Response buffer occupancy."""
        return self._resp_port._buffer.occupancy

    def __repr__(self) -> str:
        return (
            f"EdgeRouterPort(row={self.row}, "
            f"req_credit={self.available_credits}, "
            f"resp_occ={self.resp_occupancy})"
        )


class RoutingSelector:
    """
    Routing Selector for V1 Architecture.

    Single entry/exit point between master NI and 2D mesh.
    Connects to 4 Edge Routers in Column 0 via Local ports.
    """

    def __init__(self, config: Optional[RoutingSelectorConfig] = None):
        """
        Initialize Routing Selector.

        Args:
            config: Selector configuration.
        """
        self.config = config or RoutingSelectorConfig()

        # Edge router ports (one per row: 0, 1, 2, 3)
        self.edge_ports: Dict[int, EdgeRouterPort] = {}
        for row in range(self.config.num_directions):
            self.edge_ports[row] = EdgeRouterPort(
                row=row,
                buffer_depth=self.config.egress_buffer_depth
            )

        # Ingress buffer (from master NI)
        self.ingress_buffer = FlitBuffer(
            self.config.ingress_buffer_depth,
            "Sel_ingress"
        )

        # Egress buffer (to master NI)
        self.egress_buffer = FlitBuffer(
            self.config.egress_buffer_depth,
            "Sel_egress"
        )

        # Round-robin state for egress arbitration
        self._egress_rr_index = 0

        # Statistics
        self.stats = SelectorStats()

        # Mesh reference for hop calculation
        self._mesh_rows = 4

        # Packet-to-path mapping: track which row a packet is using
        # This ensures all flits of a packet go through the same edge router
        self._packet_path: Dict[int, int] = {}  # packet_id -> row

    def connect_edge_routers(self, edge_routers: List[EdgeRouter]) -> None:
        """
        Connect to Edge Routers via PortWire.

        For each edge router:
        - Request Wire: EdgeRouterPort._req_port <-> EdgeRouter.req.LOCAL
        - Response Wire: EdgeRouter.resp.LOCAL <-> EdgeRouterPort._resp_port

        Args:
            edge_routers: List of EdgeRouter instances.
        """
        for router in edge_routers:
            row = router.coord[1]
            if row not in self.edge_ports:
                continue

            port = self.edge_ports[row]
            port.connect_edge_router(router)

            # Request Wire: Selector._req_port <-> EdgeRouter.req.LOCAL
            # Selector sends requests TO EdgeRouter's request network
            req_local = router.req_router.ports[Direction.LOCAL]
            port._req_wire = PortWire(port._req_port, req_local)

            # Initialize credits based on EdgeRouter's buffer depth
            port._req_port._output_credit = CreditFlowControl(
                initial_credits=req_local._buffer_depth
            )

            # Response Wire: EdgeRouter.resp.LOCAL <-> Selector._resp_port
            # Selector receives responses FROM EdgeRouter's response network
            resp_local = router.resp_router.ports[Direction.LOCAL]
            port._resp_wire = PortWire(resp_local, port._resp_port)

            # Initialize response port credits (for credit release back to EdgeRouter)
            # EdgeRouter's resp_local output_credit should match Selector's resp buffer
            resp_local._output_credit = CreditFlowControl(
                initial_credits=port._buffer_depth
            )

            # Note: Legacy _virtual_req_port/_virtual_resp_port aliases have been
            # removed. Response collection now uses signal-based interface via
            # sample_all_inputs() and wire propagation.

    # =========================================================================
    # Phased Cycle Processing Methods (PortWire interface)
    # =========================================================================

    def update_all_ready(self) -> None:
        """
        Update ready signals for all EdgeRouterPorts.

        Phase 1 of cycle processing.
        """
        for port in self.edge_ports.values():
            port.update_resp_ready()

    def propagate_all_wires(self) -> None:
        """
        Propagate signals through all PortWires.

        Phase 2 of cycle processing.
        """
        for port in self.edge_ports.values():
            if port._req_wire is not None:
                port._req_wire.propagate_signals()
            if port._resp_wire is not None:
                port._resp_wire.propagate_signals()

    def sample_all_inputs(self) -> None:
        """
        Sample response inputs from all EdgeRouterPorts.

        Phase 3 of cycle processing.
        """
        for port in self.edge_ports.values():
            port.sample_resp_input()

    def clear_all_input_signals(self) -> None:
        """
        Clear input signals for all EdgeRouterPorts.

        Phase 3b of cycle processing.
        """
        for port in self.edge_ports.values():
            port.clear_resp_input_signals()

    def clear_accepted_outputs(self) -> None:
        """
        Clear request outputs that were accepted by EdgeRouters.

        Phase 4 of cycle processing.
        """
        for port in self.edge_ports.values():
            port.clear_req_if_accepted()

    def handle_credit_release(self) -> None:
        """
        Handle credit release for all PortWires.

        Phase 5 of cycle processing.
        """
        for port in self.edge_ports.values():
            if port._req_wire is not None:
                port._req_wire.propagate_credit_release()
            if port._resp_wire is not None:
                port._resp_wire.propagate_credit_release()

    def clear_edge_resp_outputs(self) -> None:
        """
        Clear EdgeRouter response LOCAL outputs after Selector has sampled.

        This completes the handshake for response flits. EdgeRouter's LOCAL
        port output is not cleared by mesh.process_cycle() to avoid timing
        issues, so Selector must clear it explicitly after sampling.
        """
        for port in self.edge_ports.values():
            if port._edge_router is not None:
                resp_local = port._edge_router.resp_router.ports[Direction.LOCAL]
                # Clear if handshake completed (out_valid=True and in_ready=True)
                resp_local.clear_output_if_accepted()

    # =========================================================================
    # Request/Response Interface
    # =========================================================================

    def accept_request(self, flit: Flit) -> bool:
        """
        Accept request flit from master NI.

        Args:
            flit: Request flit.

        Returns:
            True if accepted into ingress buffer.
        """
        if self.ingress_buffer.is_full():
            return False
        self.ingress_buffer.push(flit)
        self.stats.req_flits_received += 1
        return True

    def get_response(self) -> Optional[Flit]:
        """
        Get response flit for master NI.

        Returns:
            Response flit if available.
        """
        flit = self.egress_buffer.pop()
        if flit is not None:
            self.stats.resp_flits_sent += 1
        return flit

    def process_cycle(self, current_time: int = 0) -> None:
        """
        Process one simulation cycle with phased PortWire processing.

        Phases:
        1. Update ready signals
        2. Propagate wire signals (for sampling)
        3. Sample response inputs
        4. Clear input signals
        5. Process ingress/egress logic
        6. Propagate wire signals (for outputs)
        7. Clear accepted outputs
        8. Handle credit release
        """
        # Phase 1: Update ready signals
        self.update_all_ready()

        # Phase 2: Propagate signals (for sampling response inputs)
        self.propagate_all_wires()

        # Phase 3-4: Sample inputs and clear input signals
        self.sample_all_inputs()
        self.clear_all_input_signals()

        # Phase 5: Process ingress and egress
        self._process_ingress(current_time)
        self._process_egress(current_time)

        # Phase 6: Propagate signals (for outputs)
        self.propagate_all_wires()

        # Phase 7: Clear accepted outputs
        self.clear_accepted_outputs()

        # Phase 8: Handle credit release
        self.handle_credit_release()

    def _process_ingress(self, current_time: int) -> None:
        """
        Process ingress path: route requests into mesh via PortWire.

        Sets request outputs on EdgeRouterPorts. The actual transfer
        to EdgeRouters happens during signal propagation phase.
        """
        while not self.ingress_buffer.is_empty():
            flit = self.ingress_buffer.peek()
            if flit is None:
                break

            # Select best path
            best_row = self._select_ingress_path(flit)
            if best_row is None:
                self.stats.req_blocked_no_credit += 1
                break

            # Check if we can send via signal interface
            edge_port = self.edge_ports[best_row]
            if not edge_port.can_send_request():
                self.stats.req_blocked_no_credit += 1
                break

            # Pop and set output for wire propagation
            flit = self.ingress_buffer.pop()

            # Update flit source to edge router coord (for response routing)
            # This is critical for XY routing to work correctly
            flit.src = edge_port.coord

            if edge_port.set_req_output(flit):
                self.stats.req_flits_injected += 1
                self.stats.path_selections[best_row] += 1
                # Clear packet path tracking after TAIL is sent
                self._clear_packet_path(flit)
            else:
                # Should not happen - we checked can_send_request_signal
                # Put back to buffer
                self.ingress_buffer.push(flit)
                break

    def _select_ingress_path(self, flit: Flit) -> Optional[int]:
        """
        Select best ingress path using hop count and credits.

        IMPORTANT: For multi-flit packets, all flits must go through
        the same edge router to maintain packet integrity (VCT requirement).

        Formula for new packets: cost = hop_weight * hops - credit_weight * credits
        Lower cost is better.

        Args:
            flit: Flit to route.

        Returns:
            Best row (0-3), or None if no path available.
        """
        packet_id = flit.packet_id

        # If this packet already has an assigned path, use it
        if packet_id in self._packet_path:
            assigned_row = self._packet_path[packet_id]
            port = self.edge_ports[assigned_row]
            if port.can_send_request():
                # NOTE: Don't delete path here - it will be deleted after successful send
                return assigned_row
            else:
                # Path blocked - wait
                return None

        # New packet - select best path
        dest = flit.dest
        best_row = None
        min_cost = float('inf')

        for row, port in self.edge_ports.items():
            if not port.can_send_request():
                continue

            # Calculate hop count from this edge router to destination
            hops = self._calculate_hops((0, row), dest)

            # Get available credits
            credits = port.available_credits

            # Calculate cost
            cost = (self.config.hop_weight * hops -
                    self.config.credit_weight * credits)

            if cost < min_cost:
                min_cost = cost
                best_row = row

        # If found a path and this is a multi-flit packet, remember the path
        if best_row is not None and not flit.is_single_flit():
            if flit.is_head():
                self._packet_path[packet_id] = best_row

        return best_row

    def _clear_packet_path(self, flit: Flit) -> None:
        """Clear packet path after successful TAIL send."""
        if flit.is_tail() and flit.packet_id in self._packet_path:
            del self._packet_path[flit.packet_id]

    def _calculate_hops(
        self,
        src: Tuple[int, int],
        dest: Tuple[int, int]
    ) -> int:
        """
        Calculate Manhattan distance (hop count).

        Args:
            src: Source coordinate.
            dest: Destination coordinate.

        Returns:
            Number of hops.
        """
        return abs(dest[0] - src[0]) + abs(dest[1] - src[1])

    def _process_egress(self, current_time: int) -> None:
        """
        Process egress path: collect responses from mesh.

        Uses buffer occupancy-based arbitration.
        """
        if self.egress_buffer.is_full():
            return

        # Select edge router to read from
        row = self._select_egress_source()
        if row is None:
            return

        port = self.edge_ports[row]
        flit = port.get_response()
        if flit is not None:
            self.egress_buffer.push(flit)
            self.stats.resp_flits_collected += 1

    def _select_egress_source(self) -> Optional[int]:
        """
        Select edge router to read response from.

        Strategy: Read from the edge router with highest buffer occupancy
        to prevent upstream blocking.

        Returns:
            Row to read from, or None if all empty.
        """
        max_occupancy = 0
        best_row = None

        for row, port in self.edge_ports.items():
            occ = port.resp_occupancy
            if occ > max_occupancy:
                max_occupancy = occ
                best_row = row

        # If all equal occupancy, use round-robin
        if best_row is None:
            # Check if any has data
            for i in range(self.config.num_directions):
                row = (self._egress_rr_index + i) % self.config.num_directions
                if self.edge_ports[row].resp_occupancy > 0:
                    best_row = row
                    self._egress_rr_index = (row + 1) % self.config.num_directions
                    break

        return best_row

    def has_pending_requests(self) -> bool:
        """Check if requests are pending."""
        return not self.ingress_buffer.is_empty()

    @property
    def has_pending_responses(self) -> bool:
        """Check if responses are available."""
        return not self.egress_buffer.is_empty()

    def print_status(self) -> None:
        """Print selector status for debugging."""
        print(f"Routing Selector Status:")
        print(f"  Ingress buffer: {self.ingress_buffer.occupancy}/{self.config.ingress_buffer_depth}")
        print(f"  Egress buffer: {self.egress_buffer.occupancy}/{self.config.egress_buffer_depth}")
        print(f"  Edge ports:")
        for row, port in self.edge_ports.items():
            print(f"    Row {row}: credit={port.available_credits}, resp_occ={port.resp_occupancy}")
        print(f"  Stats:")
        print(f"    Req received: {self.stats.req_flits_received}")
        print(f"    Req injected: {self.stats.req_flits_injected}")
        print(f"    Resp collected: {self.stats.resp_flits_collected}")
        print(f"    Resp sent: {self.stats.resp_flits_sent}")
        print(f"    Path selections: {self.stats.path_selections}")

    def __repr__(self) -> str:
        return (
            f"RoutingSelector("
            f"ingress={self.ingress_buffer.occupancy}, "
            f"egress={self.egress_buffer.occupancy})"
        )


# =============================================================================
# V1 System: NI + Selector + Mesh
# =============================================================================

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
        occupancy = {}
        for coord, router in self.mesh.routers.items():
            occ = 0
            # Req router
            for port in router.req_router.ports.values():
                if hasattr(port, 'input_buffer'):
                    occ += port.input_buffer.occupancy
                # Count pending output (flit waiting to be accepted)
                if port.out_valid and port.out_flit is not None:
                    occ += 1
            # Flits in pipeline stages
            occ += router.req_router.flits_in_pipeline

            # Resp router
            for port in router.resp_router.ports.values():
                if hasattr(port, 'input_buffer'):
                    occ += port.input_buffer.occupancy
                if port.out_valid and port.out_flit is not None:
                    occ += 1
            occ += router.resp_router.flits_in_pipeline

            occupancy[coord] = occ
        return occupancy

    def get_flit_stats(self) -> Dict[Tuple[int, int], int]:
        """Get flit forwarding stats for each router (req + resp)."""
        stats = {}
        for coord, router in self.mesh.routers.items():
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
            from ..config import TransferMode
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
        selector_config: Optional[RoutingSelectorConfig] = None,
        host_memory: Optional["Memory"] = None,
    ):
        """
        Initialize V1 System.

        Args:
            mesh_cols: Mesh columns.
            mesh_rows: Mesh rows.
            buffer_depth: Router buffer depth.
            selector_config: Selector configuration.
            host_memory: Optional Host Memory for DMA transfers.
        """
        from .mesh import create_mesh
        from .ni import SlaveNI, NIConfig
        from ..address.address_map import SystemAddressMap, AddressMapConfig

        self._mesh_cols = mesh_cols
        self._mesh_rows = mesh_rows
        self._buffer_depth = buffer_depth

        # Create mesh
        self.mesh = create_mesh(
            cols=mesh_cols,
            rows=mesh_rows,
            edge_column=0,
            buffer_depth=buffer_depth
        )

        # Create selector with matching buffer depth
        if selector_config is None:
            selector_config = RoutingSelectorConfig(
                ingress_buffer_depth=buffer_depth,
                egress_buffer_depth=buffer_depth,
            )
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
        # NI buffer depth should match router buffer depth to support same max packet size
        self.master_ni = SlaveNI(
            coord=(0, 0),  # Virtual coord for Host Slave NI
            address_map=self.address_map,
            config=NIConfig(
                req_buffer_depth=buffer_depth,
                resp_buffer_depth=buffer_depth,
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

        # 1. Master NI: Generate request flits (transfer to selector)
        while self.master_ni.req_ni.has_pending_output():
            if self.selector.ingress_buffer.is_full():
                break
            flit = self.master_ni.get_req_flit()
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
            from .host_axi_master import HostAXIMasterState
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
        from ..axi.interface import AXI_AW, AXI_W, AXISize

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
        from ..axi.interface import AXI_AR, AXISize

        ar = AXI_AR(
            arid=axi_id,
            araddr=addr,
            arlen=0,  # Single beat
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
        from .host_axi_master import HostAXIMaster
        from .axi_master import AXIIdConfig

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
        from .host_axi_master import HostAXIMaster
        from .axi_master import AXIIdConfig

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


# =============================================================================
# NoC-to-NoC System
# =============================================================================

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
        occupancy = {}
        for coord, router in self.mesh.routers.items():
            occ = 0
            # Req router
            for port in router.req_router.ports.values():
                if hasattr(port, 'input_buffer'):
                    occ += port.input_buffer.occupancy
                # Count pending output (flit waiting to be accepted)
                if port.out_valid and port.out_flit is not None:
                    occ += 1
            # Flits in pipeline stages
            occ += router.req_router.flits_in_pipeline

            # Resp router
            for port in router.resp_router.ports.values():
                if hasattr(port, 'input_buffer'):
                    occ += port.input_buffer.occupancy
                if port.out_valid and port.out_flit is not None:
                    occ += 1
            occ += router.resp_router.flits_in_pipeline

            occupancy[coord] = occ
        return occupancy

    def get_flit_stats(self) -> Dict[Tuple[int, int], int]:
        """Get flit forwarding stats for each router (req + resp)."""
        stats = {}
        for coord, router in self.mesh.routers.items():
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
        buffer_depth: int = 4,
        memory_size: int = 0x100000000,
    ):
        """
        Initialize NoC-to-NoC System.

        Args:
            mesh_cols: Mesh columns.
            mesh_rows: Mesh rows.
            buffer_depth: Router buffer depth.
            memory_size: Local memory size per node.
        """
        from .mesh import create_mesh
        from .node_controller import NodeController
        from .ni import NIConfig

        self.mesh_cols = mesh_cols
        self.mesh_rows = mesh_rows
        self.buffer_depth = buffer_depth
        self.memory_size = memory_size

        # Calculate number of compute nodes (exclude edge column)
        self.num_nodes = (mesh_cols - 1) * mesh_rows

        # Create mesh
        self.mesh = create_mesh(
            cols=mesh_cols,
            rows=mesh_rows,
            edge_column=0,
            buffer_depth=buffer_depth
        )

        # Create NodeControllers for each compute node
        self.node_controllers: Dict[int, "NodeController"] = {}
        ni_config = NIConfig(
            use_user_signal_routing=True,
            req_buffer_depth=buffer_depth,
            resp_buffer_depth=buffer_depth,
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
        from .golden_manager import GoldenManager
        self.golden_manager = GoldenManager()

        # Pending flits that couldn't be injected (back-pressure handling)
        from .flit import Flit
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
        from ..traffic.pattern_generator import TrafficPatternGenerator

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
        from .router import Direction

        for node_id, controller in self.node_controllers.items():
            coord = controller.coord
            router = self.mesh.routers.get(coord)
            if router is None:
                continue

            # First, try to inject any pending flit from previous cycle
            if node_id in self._pending_req_flits:
                pending = self._pending_req_flits[node_id]
                if router.receive_request(Direction.LOCAL, pending):
                    # Successfully injected pending flit
                    del self._pending_req_flits[node_id]
                # If still failed, keep it pending and skip getting new flit
                else:
                    continue

            # Get outgoing request flit from NodeController's SlaveNI
            flit = controller.get_outgoing_flit()
            if flit is not None:
                # Inject into router's request LOCAL port
                success = router.receive_request(Direction.LOCAL, flit)
                if not success:
                    # Router couldn't accept - buffer for retry next cycle
                    self._pending_req_flits[node_id] = flit

    def _deliver_flits_from_mesh(self) -> None:
        """
        Deliver response flits from Mesh back to NodeControllers.

        When a response flit arrives at a compute node's router,
        it needs to be delivered to the source NodeController's SlaveNI.

        Also inject response flits from mesh.nis (MasterNI) into router.
        """
        from .router import Direction

        for node_id, controller in self.node_controllers.items():
            coord = controller.coord
            router = self.mesh.routers.get(coord)
            if router is None:
                continue

            # Check if router's response LOCAL output has a flit for us
            # This happens when a response is routed back to this node
            resp_local = router.get_resp_port(Direction.LOCAL)
            if resp_local.out_valid and resp_local.out_flit is not None:
                flit = resp_local.out_flit
                # Deliver to NodeController's SlaveNI response path
                if controller.receive_response_flit(flit):
                    # Clear router's output
                    resp_local.out_valid = False
                    resp_local.out_flit = None

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
        from .golden_manager import GoldenKey

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

