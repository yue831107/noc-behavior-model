"""
Routing Selector core logic.

The central component that manages request injection into mesh
and response collection from mesh.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from ..buffer import FlitBuffer
from ..flit import Flit, AxiChannel, encode_node_id
from ..router import (
    Direction, PortWire, CreditFlowControl, ChannelMode
)
from .config import RoutingSelectorConfig, SelectorStats
from .edge_port import EdgeRouterPort, AXIModeEdgeRouterPort

if TYPE_CHECKING:
    from ..router import EdgeRouter, AXIModeEdgeRouter


class RoutingSelector:
    """
    Routing Selector for V1 Architecture.

    Single entry/exit point between master NI and 2D mesh.
    Connects to 4 Edge Routers in Column 0 via Local ports.

    Supports both General Mode (2 sub-routers) and AXI Mode (5 sub-routers).
    """

    def __init__(self, config: Optional[RoutingSelectorConfig] = None):
        """
        Initialize Routing Selector.

        Args:
            config: Selector configuration.
        """
        self.config = config or RoutingSelectorConfig()
        self._is_axi_mode = (self.config.channel_mode == ChannelMode.AXI)

        # Edge router ports (one per row: 0, 1, 2, 3)
        # Type depends on channel mode
        self.edge_ports: Dict[int, Union[EdgeRouterPort, AXIModeEdgeRouterPort]] = {}
        if self._is_axi_mode:
            for row in range(self.config.num_directions):
                self.edge_ports[row] = AXIModeEdgeRouterPort(
                    row=row,
                    buffer_depth=self.config.egress_buffer_depth
                )
        else:
            for row in range(self.config.num_directions):
                self.edge_ports[row] = EdgeRouterPort(
                    row=row,
                    buffer_depth=self.config.egress_buffer_depth
                )

        # Ingress buffer (from master NI) - shared for all channels
        self.ingress_buffer = FlitBuffer(
            self.config.ingress_buffer_depth,
            "Sel_ingress"
        )

        # Egress buffer (to master NI) - shared for all channels
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
        # Key: (src_id, dst_id, rob_idx), Value: row
        self._packet_path: Dict[Tuple[int, int, int], int] = {}

        # Active W packet per row: track which row has W flits in-flight
        # In General mode, only one W packet can use a row at a time to prevent
        # wormhole deadlock from out-of-order flit arrival at the router
        # Key: row, Value: packet_key (src_id, dst_id, rob_idx)
        self._active_w_packet: Dict[int, Tuple[int, int, int]] = {}

    def connect_edge_routers(self, edge_routers: List["EdgeRouter"]) -> None:
        """
        Connect to Edge Routers via PortWire.

        Supports both General Mode and AXI Mode:
        - General Mode: 2 wires (req, resp)
        - AXI Mode: 5 wires (AW, W, AR, B, R)

        Args:
            edge_routers: List of EdgeRouter instances.
        """
        for router in edge_routers:
            row = router.coord[1]
            if row not in self.edge_ports:
                continue

            port = self.edge_ports[row]
            port.connect_edge_router(router)

            if self._is_axi_mode:
                # AXI Mode: Connect 5 independent channel wires
                self._connect_axi_mode_wires(port, router)
            else:
                # General Mode: Connect 2 wires (req, resp)
                self._connect_general_mode_wires(port, router)

    def _connect_general_mode_wires(
        self,
        port: EdgeRouterPort,
        router: "EdgeRouter"
    ) -> None:
        """Connect wires for General Mode (2 sub-routers)."""
        # Request Wire: Selector._req_port <-> EdgeRouter.req.LOCAL
        req_local = router.req_router.ports[Direction.LOCAL]
        port._req_wire = PortWire(port._req_port, req_local)

        # Initialize credits based on EdgeRouter's buffer depth
        port._req_port._output_credit = CreditFlowControl(
            initial_credits=req_local._buffer_depth
        )

        # Response Wire: EdgeRouter.resp.LOCAL <-> Selector._resp_port
        resp_local = router.resp_router.ports[Direction.LOCAL]
        port._resp_wire = PortWire(resp_local, port._resp_port)

        # Initialize response port credits
        resp_local._output_credit = CreditFlowControl(
            initial_credits=port._buffer_depth
        )

    def _connect_axi_mode_wires(
        self,
        port: AXIModeEdgeRouterPort,
        router: "AXIModeEdgeRouter"
    ) -> None:
        """Connect wires for AXI Mode (5 sub-routers)."""
        # Request channels: Selector -> EdgeRouter (AW, W, AR)
        aw_local = router.aw_router.ports[Direction.LOCAL]
        port._aw_wire = PortWire(port._aw_port, aw_local)
        port._aw_port._output_credit = CreditFlowControl(
            initial_credits=aw_local._buffer_depth
        )

        w_local = router.w_router.ports[Direction.LOCAL]
        port._w_wire = PortWire(port._w_port, w_local)
        port._w_port._output_credit = CreditFlowControl(
            initial_credits=w_local._buffer_depth
        )

        ar_local = router.ar_router.ports[Direction.LOCAL]
        port._ar_wire = PortWire(port._ar_port, ar_local)
        port._ar_port._output_credit = CreditFlowControl(
            initial_credits=ar_local._buffer_depth
        )

        # Response channels: EdgeRouter -> Selector (B, R)
        b_local = router.b_router.ports[Direction.LOCAL]
        port._b_wire = PortWire(b_local, port._b_port)
        b_local._output_credit = CreditFlowControl(
            initial_credits=port._buffer_depth
        )

        r_local = router.r_router.ports[Direction.LOCAL]
        port._r_wire = PortWire(r_local, port._r_port)
        r_local._output_credit = CreditFlowControl(
            initial_credits=port._buffer_depth
        )

    # =========================================================================
    # Phased Cycle Processing Methods (PortWire interface)
    # =========================================================================

    def update_all_ready(self) -> None:
        """
        Update ready signals for all EdgeRouterPorts.

        Phase 1 of cycle processing.
        """
        if self._is_axi_mode:
            for port in self.edge_ports.values():
                port.update_b_ready()
                port.update_r_ready()
        else:
            for port in self.edge_ports.values():
                port.update_resp_ready()

    def propagate_all_wires(self) -> None:
        """
        Propagate signals through all PortWires.

        Phase 2 of cycle processing.
        """
        if self._is_axi_mode:
            for port in self.edge_ports.values():
                # Request wires
                if port._aw_wire is not None:
                    port._aw_wire.propagate_signals()
                if port._w_wire is not None:
                    port._w_wire.propagate_signals()
                if port._ar_wire is not None:
                    port._ar_wire.propagate_signals()
                # Response wires
                if port._b_wire is not None:
                    port._b_wire.propagate_signals()
                if port._r_wire is not None:
                    port._r_wire.propagate_signals()
        else:
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
        if self._is_axi_mode:
            for port in self.edge_ports.values():
                port.sample_b_input()
                port.sample_r_input()
        else:
            for port in self.edge_ports.values():
                port.sample_resp_input()

    def clear_all_input_signals(self) -> None:
        """
        Clear input signals for all EdgeRouterPorts.

        Phase 3b of cycle processing.
        """
        if self._is_axi_mode:
            for port in self.edge_ports.values():
                port.clear_b_input_signals()
                port.clear_r_input_signals()
        else:
            for port in self.edge_ports.values():
                port.clear_resp_input_signals()

    def clear_accepted_outputs(self) -> None:
        """
        Clear request outputs that were accepted by EdgeRouters.

        Phase 4 of cycle processing.
        """
        if self._is_axi_mode:
            for port in self.edge_ports.values():
                port.clear_aw_if_accepted()
                port.clear_w_if_accepted()
                port.clear_ar_if_accepted()
        else:
            for port in self.edge_ports.values():
                port.clear_req_if_accepted()

    def handle_credit_release(self) -> None:
        """
        Handle credit release for all PortWires.

        Phase 5 of cycle processing.
        """
        if self._is_axi_mode:
            for port in self.edge_ports.values():
                # Request wires
                if port._aw_wire is not None:
                    port._aw_wire.propagate_credit_release()
                if port._w_wire is not None:
                    port._w_wire.propagate_credit_release()
                if port._ar_wire is not None:
                    port._ar_wire.propagate_credit_release()
                # Response wires
                if port._b_wire is not None:
                    port._b_wire.propagate_credit_release()
                if port._r_wire is not None:
                    port._r_wire.propagate_credit_release()
        else:
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
        if self._is_axi_mode:
            for port in self.edge_ports.values():
                if port._edge_router is not None:
                    # Clear B and R channel outputs
                    b_local = port._edge_router.b_router.ports[Direction.LOCAL]
                    b_local.clear_output_if_accepted()
                    r_local = port._edge_router.r_router.ports[Direction.LOCAL]
                    r_local.clear_output_if_accepted()
        else:
            for port in self.edge_ports.values():
                if port._edge_router is not None:
                    resp_local = port._edge_router.resp_router.ports[Direction.LOCAL]
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

        In AXI Mode, routes to channel-specific ports (AW, W, AR).
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

            edge_port = self.edge_ports[best_row]

            # Check if we can send via signal interface (mode-specific)
            if self._is_axi_mode:
                if not self._can_send_axi_request(edge_port, flit):
                    self.stats.req_blocked_no_credit += 1
                    break
            else:
                if not edge_port.can_send_request():
                    self.stats.req_blocked_no_credit += 1
                    break

            # Pop and set output for wire propagation
            flit = self.ingress_buffer.pop()

            # Save original packet_key BEFORE modifying src_id
            # This is critical for correct path cleanup
            original_packet_key = (flit.hdr.src_id, flit.hdr.dst_id, flit.hdr.rob_idx)

            # Update flit source to edge router coord (for response routing)
            flit.hdr.src_id = encode_node_id(edge_port.coord)

            # Set output (mode-specific)
            if self._is_axi_mode:
                success = self._set_axi_req_output(edge_port, flit)
            else:
                success = edge_port.set_req_output(flit)

            if success:
                self.stats.req_flits_injected += 1
                self.stats.path_selections[best_row] += 1

                # Track active W packet per row (General mode only)
                # SET when first W flit is sent, CLEAR happens in _clear_packet_path_by_key
                if not self._is_axi_mode and flit.hdr.axi_ch == AxiChannel.W:
                    if flit.is_head():
                        self._active_w_packet[best_row] = original_packet_key

                self._clear_packet_path_by_key(flit, original_packet_key, best_row)
            else:
                # Put back to buffer
                self.ingress_buffer.push(flit)
                break

    def _can_send_axi_request(
        self,
        port: AXIModeEdgeRouterPort,
        flit: Flit
    ) -> bool:
        """Check if AXI Mode port can send flit on its channel."""
        channel = flit.hdr.axi_ch
        if channel == AxiChannel.AW:
            return port.can_send_aw()
        elif channel == AxiChannel.W:
            return port.can_send_w()
        elif channel == AxiChannel.AR:
            return port.can_send_ar()
        return False

    def _set_axi_req_output(
        self,
        port: AXIModeEdgeRouterPort,
        flit: Flit
    ) -> bool:
        """Set AXI Mode request output on correct channel."""
        channel = flit.hdr.axi_ch
        if channel == AxiChannel.AW:
            return port.set_aw_output(flit)
        elif channel == AxiChannel.W:
            return port.set_w_output(flit)
        elif channel == AxiChannel.AR:
            return port.set_ar_output(flit)
        return False

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
        # Use (src_id, dst_id, rob_idx) as packet key
        packet_key = (flit.hdr.src_id, flit.hdr.dst_id, flit.hdr.rob_idx)

        # If this packet already has an assigned path, use it
        if packet_key in self._packet_path:
            assigned_row = self._packet_path[packet_key]
            port = self.edge_ports[assigned_row]

            # In General mode, check if row is blocked by another W packet
            # This applies to W flits that need to wait for their turn
            is_w_flit = flit.hdr.axi_ch == AxiChannel.W
            if not self._is_axi_mode and is_w_flit and assigned_row in self._active_w_packet:
                active_key = self._active_w_packet[assigned_row]
                if active_key != packet_key:
                    # Row is busy with another W packet - must wait
                    return None

            # Check if can send (mode-specific)
            if self._is_axi_mode:
                can_send = self._can_send_axi_request(port, flit)
            else:
                can_send = port.can_send_request()
            if can_send:
                return assigned_row
            else:
                return None

        # New packet - select best path
        dest = flit.dest
        best_row = None
        min_cost = float('inf')

        for row, port in self.edge_ports.items():
            # Check if can send (mode-specific)
            if self._is_axi_mode:
                can_send = self._can_send_axi_request(port, flit)
                credits = self._get_axi_credits(port, flit)
            else:
                can_send = port.can_send_request()
                credits = port.available_credits

            if not can_send:
                continue

            # In General mode, check if this row has an active W packet
            # from a different transaction. If so, skip to prevent wormhole deadlock
            if not self._is_axi_mode and row in self._active_w_packet:
                active_key = self._active_w_packet[row]
                if active_key != packet_key:
                    # Row is busy with another W packet
                    continue

            # Calculate hop count from this edge router to destination
            hops = self._calculate_hops((0, row), dest)

            # Calculate cost
            cost = (self.config.hop_weight * hops -
                    self.config.credit_weight * credits)

            if cost < min_cost:
                min_cost = cost
                best_row = row

        # If found a path, remember it for subsequent flits
        # For AW (single-flit): store path so W uses same edge router
        # For multi-flit HEAD: store path for body/tail flits
        if best_row is not None:
            is_aw_flit = flit.hdr.axi_ch == AxiChannel.AW
            is_multi_flit_head = flit.is_head() and not flit.is_single_flit()
            if is_aw_flit or is_multi_flit_head:
                self._packet_path[packet_key] = best_row

        return best_row

    def _get_axi_credits(
        self,
        port: AXIModeEdgeRouterPort,
        flit: Flit
    ) -> int:
        """Get AXI Mode credits for flit's channel."""
        channel = flit.hdr.axi_ch
        if channel == AxiChannel.AW:
            return port.aw_credits
        elif channel == AxiChannel.W:
            return port.w_credits
        elif channel == AxiChannel.AR:
            return port.ar_credits
        return 0

    def _clear_packet_path_by_key(
        self,
        flit: Flit,
        packet_key: Tuple[int, int, int],
        row: int
    ) -> None:
        """Clear packet path using the original packet key.

        This method must be called with the packet_key computed BEFORE
        flit.hdr.src_id is modified to edge router coord.

        Args:
            flit: The flit being sent (to check if W TAIL).
            packet_key: Original (src_id, dst_id, rob_idx) before src_id modification.
            row: Edge router row that was used for this flit.
        """
        # Only clear on W TAIL (not AW, which needs to keep path for W)
        is_w_tail = flit.hdr.axi_ch == AxiChannel.W and flit.is_tail()
        if is_w_tail:
            if packet_key in self._packet_path:
                del self._packet_path[packet_key]

            # Clear active W packet tracking (General mode only)
            if not self._is_axi_mode and row in self._active_w_packet:
                if self._active_w_packet[row] == packet_key:
                    del self._active_w_packet[row]

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
        In AXI Mode, collects from both B and R channels.
        """
        if self.egress_buffer.is_full():
            return

        if self._is_axi_mode:
            self._process_egress_axi_mode()
        else:
            self._process_egress_general_mode()

    def _process_egress_general_mode(self) -> None:
        """Process egress for General Mode (single resp channel)."""
        row = self._select_egress_source()
        if row is None:
            return

        port = self.edge_ports[row]
        flit = port.get_response()
        if flit is not None:
            self.egress_buffer.push(flit)
            self.stats.resp_flits_collected += 1

    def _process_egress_axi_mode(self) -> None:
        """Process egress for AXI Mode (B and R channels)."""
        # Try to collect from both B and R channels
        # Use round-robin between channels to avoid starvation
        channels = ['B', 'R']
        start_idx = self._egress_rr_index % 2

        for i in range(2):
            if self.egress_buffer.is_full():
                break

            ch_idx = (start_idx + i) % 2
            channel = channels[ch_idx]

            # Select row with highest occupancy for this channel
            row = self._select_egress_source_axi(channel)
            if row is None:
                continue

            port = self.edge_ports[row]
            if channel == 'B':
                flit = port.get_b_response()
            else:
                flit = port.get_r_response()

            if flit is not None:
                self.egress_buffer.push(flit)
                self.stats.resp_flits_collected += 1
                self._egress_rr_index = ch_idx + 1

    def _select_egress_source(self) -> Optional[int]:
        """
        Select edge router to read response from (General Mode).

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

    def _select_egress_source_axi(self, channel: str) -> Optional[int]:
        """
        Select edge router to read from for AXI Mode.

        Args:
            channel: 'B' or 'R'

        Returns:
            Row to read from, or None if all empty.
        """
        max_occupancy = 0
        best_row = None

        for row, port in self.edge_ports.items():
            if channel == 'B':
                occ = port.b_occupancy
            else:
                occ = port.r_occupancy

            if occ > max_occupancy:
                max_occupancy = occ
                best_row = row

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
            if self._is_axi_mode:
                axi_port = port  # type: AXIModeEdgeRouterPort
                print(f"    Row {row}: AW={axi_port.aw_credits}, W={axi_port.w_credits}, "
                      f"AR={axi_port.ar_credits}, B_occ={axi_port.b_occupancy}, R_occ={axi_port.r_occupancy}")
            else:
                gen_port = port  # type: EdgeRouterPort
                print(f"    Row {row}: credit={gen_port.available_credits}, resp_occ={gen_port.resp_occupancy}")
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
