"""
2D Mesh Topology implementation.

Creates and connects routers in a 2D mesh pattern with:
- Edge routers in column 0 (no NI, connect to Selector)
- Compute nodes in columns 1+ (Router + NI)
- Req/Resp physical separation throughout
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

from .router import (
    Router, EdgeRouter, Direction, RouterConfig, RouterPort,
    PortWire, create_router, ChannelMode,
    AXIModeRouter, AXIModeEdgeRouter
)
from .ni import SlaveNI, MasterNI, NIConfig
from .flit import AxiChannel
from ..address.address_map import SystemAddressMap, AddressMapConfig


@dataclass
class MeshConfig:
    """Mesh topology configuration."""
    cols: int = 5                   # Total columns (including edge)
    rows: int = 4                   # Total rows
    edge_column: int = 0            # Edge router column (no NI)

    # Physical channel mode
    channel_mode: ChannelMode = ChannelMode.GENERAL

    # Router configuration
    router_config: RouterConfig = field(default_factory=RouterConfig)

    # NI configuration
    ni_config: NIConfig = field(default_factory=NIConfig)

    def __post_init__(self):
        if self.edge_column >= self.cols:
            raise ValueError("edge_column must be less than cols")
        if self.cols < 2:
            raise ValueError("cols must be at least 2")
        if self.rows < 1:
            raise ValueError("rows must be at least 1")
        # Propagate channel_mode to router_config
        self.router_config.channel_mode = self.channel_mode

    @property
    def compute_cols(self) -> range:
        """Range of compute node columns."""
        return range(self.edge_column + 1, self.cols)

    @property
    def num_compute_nodes(self) -> int:
        """Number of compute nodes (with NI)."""
        return (self.cols - self.edge_column - 1) * self.rows


@dataclass
class MeshStats:
    """Mesh-level statistics."""
    total_cycles: int = 0
    total_req_flits: int = 0
    total_resp_flits: int = 0

    # Aggregated from routers
    total_buffer_full_events: int = 0

    def aggregate_from_routers(self, routers: Dict[Tuple[int, int], Router]) -> None:
        """Aggregate stats from all routers."""
        self.total_buffer_full_events = sum(
            r.req_router.stats.buffer_full_events +
            r.resp_router.stats.buffer_full_events
            for r in routers.values()
        )


class Mesh:
    """
    2D Mesh Network topology.

    Creates a mesh of routers with:
    - Column 0: Edge routers (connect to Selector via Local port)
    - Columns 1+: Compute nodes (Router + NI)

    All routers have Req/Resp physical separation.
    """

    def __init__(self, config: Optional[MeshConfig] = None):
        """
        Initialize mesh topology.

        Args:
            config: Mesh configuration.
        """
        self.config = config or MeshConfig()

        # Address map for NI address translation
        self.address_map = SystemAddressMap(
            AddressMapConfig(
                mesh_cols=self.config.cols,
                mesh_rows=self.config.rows,
                edge_column=self.config.edge_column,
            )
        )

        # Router storage: (x, y) -> Router
        self.routers: Dict[Tuple[int, int], Router] = {}

        # NI storage: (x, y) -> MasterNI (only for compute nodes)
        # Note: Compute nodes have MasterNI (AXI Master to Memory)
        self.nis: Dict[Tuple[int, int], MasterNI] = {}

        # Edge routers quick access
        self.edge_routers: List[EdgeRouter] = []

        # Wire connections (for valid/ready signal propagation)
        # General Mode: Req/Resp separation
        self._req_wires: List[PortWire] = []
        self._resp_wires: List[PortWire] = []

        # AXI Mode: 5-channel separation
        self._aw_wires: List[PortWire] = []
        self._w_wires: List[PortWire] = []
        self._ar_wires: List[PortWire] = []
        self._b_wires: List[PortWire] = []
        self._r_wires: List[PortWire] = []

        # Statistics
        self.stats = MeshStats()

        # Build the mesh
        self._build_mesh()
        self._connect_routers()
        self._connect_nis()

    def _build_mesh(self) -> None:
        """Create all routers and NIs."""
        is_axi_mode = self.config.channel_mode == ChannelMode.AXI

        for y in range(self.config.rows):
            for x in range(self.config.cols):
                coord = (x, y)

                # Create router based on channel mode
                if is_axi_mode:
                    # AXI Mode: 5 Sub-Routers per node
                    if x == self.config.edge_column:
                        router = AXIModeEdgeRouter(coord, self.config.router_config)
                        self.edge_routers.append(router)
                    else:
                        router = AXIModeRouter(coord, self.config.router_config)
                else:
                    # General Mode: 2 Sub-Routers per node
                    if x == self.config.edge_column:
                        router = EdgeRouter(coord, self.config.router_config)
                        self.edge_routers.append(router)
                    else:
                        router = Router(coord, self.config.router_config)

                # Create Master NI for compute nodes (both modes)
                if x != self.config.edge_column:
                    node_id = self.address_map.get_node_id(coord)
                    ni = MasterNI(
                        coord=coord,
                        config=self.config.ni_config,
                        ni_id=node_id,
                        node_id=node_id,
                    )
                    self.nis[coord] = ni

                self.routers[coord] = router

    def _connect_routers(self) -> None:
        """Connect routers in mesh pattern."""
        for y in range(self.config.rows):
            for x in range(self.config.cols):
                router = self.routers[(x, y)]

                # Connect EAST
                if x < self.config.cols - 1:
                    east_router = self.routers[(x + 1, y)]
                    self._connect_bidirectional(
                        router, Direction.EAST,
                        east_router, Direction.WEST
                    )

                # Connect NORTH
                if y < self.config.rows - 1:
                    north_router = self.routers[(x, y + 1)]
                    self._connect_bidirectional(
                        router, Direction.NORTH,
                        north_router, Direction.SOUTH
                    )

        # Special: Edge routers N/S interconnection for response routing
        # This is already handled by the NORTH connections above

    def _connect_bidirectional(
        self,
        router1,
        dir1: Direction,
        router2,
        dir2: Direction
    ) -> None:
        """
        Connect two routers bidirectionally using PortWire.

        General Mode: Req/Resp networks (2 wires per direction pair)
        AXI Mode: 5 channel networks (5 wires per direction pair)
        """
        if self.config.channel_mode == ChannelMode.AXI:
            # AXI Mode: Connect all 5 channels
            # AW channel
            aw_wire = PortWire(
                router1.get_aw_port(dir1),
                router2.get_aw_port(dir2)
            )
            self._aw_wires.append(aw_wire)

            # W channel
            w_wire = PortWire(
                router1.get_w_port(dir1),
                router2.get_w_port(dir2)
            )
            self._w_wires.append(w_wire)

            # AR channel
            ar_wire = PortWire(
                router1.get_ar_port(dir1),
                router2.get_ar_port(dir2)
            )
            self._ar_wires.append(ar_wire)

            # B channel
            b_wire = PortWire(
                router1.get_b_port(dir1),
                router2.get_b_port(dir2)
            )
            self._b_wires.append(b_wire)

            # R channel
            r_wire = PortWire(
                router1.get_r_port(dir1),
                router2.get_r_port(dir2)
            )
            self._r_wires.append(r_wire)
        else:
            # General Mode: Req/Resp networks
            # Request network
            req_wire = PortWire(
                router1.get_req_port(dir1),
                router2.get_req_port(dir2)
            )
            self._req_wires.append(req_wire)

            # Response network
            resp_wire = PortWire(
                router1.get_resp_port(dir1),
                router2.get_resp_port(dir2)
            )
            self._resp_wires.append(resp_wire)

    def _connect_nis(self) -> None:
        """Connect NIs to their local routers."""
        for coord, ni in self.nis.items():
            router = self.routers[coord]

            if self.config.channel_mode == ChannelMode.AXI:
                # AXI Mode: Create virtual ports for all 5 channels
                # Request channels: AW, W, AR
                for channel in [AxiChannel.AW, AxiChannel.W, AxiChannel.AR]:
                    ch_local = router.get_channel_port(channel, Direction.LOCAL)
                    ni_ch_port = RouterPort(
                        direction=Direction.LOCAL,
                        buffer_depth=self.config.ni_config.req_buffer_depth,
                        name=f"NI({coord})_{channel.name}_in"
                    )
                    ch_local.neighbor = ni_ch_port
                    ni_ch_port.neighbor = ch_local

                # Response channels: B, R
                for channel in [AxiChannel.B, AxiChannel.R]:
                    ch_local = router.get_channel_port(channel, Direction.LOCAL)
                    ni_ch_port = RouterPort(
                        direction=Direction.LOCAL,
                        buffer_depth=self.config.ni_config.resp_buffer_depth,
                        name=f"NI({coord})_{channel.name}_out"
                    )
                    ni_ch_port.neighbor = ch_local

                # Store single virtual port for backward compatibility
                # (Phase 3 will add per-channel port storage)
                ni._router_req_port = router.get_aw_port(Direction.LOCAL).neighbor
                ni._router_resp_port = RouterPort(
                    direction=Direction.LOCAL,
                    buffer_depth=self.config.ni_config.resp_buffer_depth,
                    name=f"NI({coord})_resp_out"
                )
            else:
                # General Mode: Create virtual ports for Req/Resp
                # Router Req LOCAL -> NI (requests going to local NI)
                # NI -> Router Resp LOCAL (responses from local NI)

                # For request path: router's LOCAL output needs a target
                req_local = router.get_req_port(Direction.LOCAL)
                ni_req_port = RouterPort(
                    direction=Direction.LOCAL,
                    buffer_depth=self.config.ni_config.req_buffer_depth,
                    name=f"NI({coord})_req_in"
                )
                # Bidirectional connection for proper credit flow
                req_local.neighbor = ni_req_port
                ni_req_port.neighbor = req_local

                # For response path: NI's output needs to reach router's LOCAL
                resp_local = router.get_resp_port(Direction.LOCAL)
                ni_resp_port = RouterPort(
                    direction=Direction.LOCAL,
                    buffer_depth=self.config.ni_config.resp_buffer_depth,
                    name=f"NI({coord})_resp_out"
                )
                ni_resp_port.neighbor = resp_local

                # Store virtual ports for transfer logic
                ni._router_req_port = ni_req_port
                ni._router_resp_port = ni_resp_port

    def get_router(self, coord: Tuple[int, int]) -> Optional[Router]:
        """Get router at coordinate."""
        return self.routers.get(coord)

    def get_ni(self, coord: Tuple[int, int]) -> Optional[MasterNI]:
        """Get NI at coordinate (only for compute nodes)."""
        return self.nis.get(coord)

    def get_edge_router(self, row: int):
        """
        Get edge router at given row.

        Returns EdgeRouter (General Mode) or AXIModeEdgeRouter (AXI Mode).
        """
        coord = (self.config.edge_column, row)
        router = self.routers.get(coord)
        if isinstance(router, (EdgeRouter, AXIModeEdgeRouter)):
            return router
        return None

    def process_cycle(self, current_time: int = 0) -> None:
        """
        Process one simulation cycle for all components.

        Phased processing with wire signal propagation:
        1. Sample inputs (from signals propagated at end of last cycle)
        2. Clear input signals
        3. Update ready signals for all routers
        4. Route and forward flits
        5. Propagate wire signals (for next cycle to sample)
        6. Clear accepted outputs
        7. Handle credit release
        8. Process NIs and transfer flits

        IMPORTANT: Sampling happens FIRST using signals set at the END of
        the previous cycle. Propagation happens at the END to set up signals
        for the NEXT cycle to sample. This matches the cycle-accurate sequence
        in conftest.py's run_multi_router_cycle().
        """
        # Phase 1: Sample inputs (from signals set at end of last cycle)
        for router in self.routers.values():
            router.sample_all_inputs()

        # Phase 2: Clear input signals after sampling
        for router in self.routers.values():
            router.clear_all_input_signals()

        # Phase 3: Update ready signals (based on new buffer state)
        for router in self.routers.values():
            router.update_all_ready()

        # Phase 4: Route and forward flits
        for router in self.routers.values():
            router.route_and_forward(current_time)

        # Phase 5: Propagate wire signals (for next cycle to sample)
        self._propagate_all_wires()

        # Phase 6: Clear accepted outputs
        for router in self.routers.values():
            router.clear_accepted_outputs()

        # Phase 7: Handle credit release for consumed flits
        self._handle_credit_release()

        # Phase 8: Process NIs and transfer flits
        for ni in self.nis.values():
            ni.process_cycle(current_time)

        # Transfer flits from NIs to routers and vice versa
        self._transfer_ni_flits(current_time)

        self.stats.total_cycles += 1

    def _propagate_all_wires(self) -> None:
        """Propagate signals on all wires."""
        if self.config.channel_mode == ChannelMode.AXI:
            # AXI Mode: 5 channels
            for wire in self._aw_wires:
                wire.propagate_signals()
            for wire in self._w_wires:
                wire.propagate_signals()
            for wire in self._ar_wires:
                wire.propagate_signals()
            for wire in self._b_wires:
                wire.propagate_signals()
            for wire in self._r_wires:
                wire.propagate_signals()
        else:
            # General Mode: Req/Resp
            for wire in self._req_wires:
                wire.propagate_signals()
            for wire in self._resp_wires:
                wire.propagate_signals()

    def _handle_credit_release(self) -> None:
        """Handle credit release for all wires."""
        if self.config.channel_mode == ChannelMode.AXI:
            # AXI Mode: 5 channels
            for wire in self._aw_wires:
                wire.propagate_credit_release()
            for wire in self._w_wires:
                wire.propagate_credit_release()
            for wire in self._ar_wires:
                wire.propagate_credit_release()
            for wire in self._b_wires:
                wire.propagate_credit_release()
            for wire in self._r_wires:
                wire.propagate_credit_release()
        else:
            # General Mode: Req/Resp
            for wire in self._req_wires:
                wire.propagate_credit_release()
            for wire in self._resp_wires:
                wire.propagate_credit_release()

    def _transfer_ni_flits(self, current_time: int) -> None:
        """Transfer flits between NIs and their local routers using signal interface."""
        for coord, ni in self.nis.items():
            router = self.routers[coord]

            if self.config.channel_mode == ChannelMode.AXI:
                # AXI Mode: Check all 3 request channels (AW, W, AR)
                for channel in [AxiChannel.AW, AxiChannel.W, AxiChannel.AR]:
                    req_local = router.get_channel_port(channel, Direction.LOCAL)
                    if req_local.out_valid and req_local.out_flit is not None:
                        flit = req_local.out_flit
                        if ni.receive_req_flit(flit):
                            req_local.out_valid = False
                            req_local.out_flit = None
                            self.stats.total_req_flits += 1

                # AXI Mode: Responses go to B or R channel
                # For now, use B channel for write responses, R for read
                # (NI Phase 3 will add per-channel output selection)
                if ni.has_pending_response():
                    flit = ni.peek_resp_flit()
                    if flit is not None:
                        # Determine channel from flit's axi_ch
                        resp_channel = flit.hdr.axi_ch
                        resp_local = router.get_channel_port(resp_channel, Direction.LOCAL)
                        if not resp_local.in_valid:
                            flit = ni.get_resp_flit()
                            if flit is not None:
                                resp_local.in_valid = True
                                resp_local.in_flit = flit
                                self.stats.total_resp_flits += 1
            else:
                # General Mode: Single req/resp paths
                # Transfer from Router req_LOCAL output to NI
                req_local = router.get_req_port(Direction.LOCAL)
                if req_local.out_valid and req_local.out_flit is not None:
                    flit = req_local.out_flit
                    if ni.receive_req_flit(flit):
                        req_local.out_valid = False
                        req_local.out_flit = None
                        self.stats.total_req_flits += 1

                # Transfer from NI response output to Router resp_LOCAL input
                resp_local = router.get_resp_port(Direction.LOCAL)
                if ni.has_pending_response() and not resp_local.in_valid:
                    flit = ni.get_resp_flit()
                    if flit is not None:
                        resp_local.in_valid = True
                        resp_local.in_flit = flit
                        self.stats.total_resp_flits += 1


    def sample_stats(self) -> None:
        """Sample statistics from all components."""
        for router in self.routers.values():
            router.sample_stats()
        self.stats.aggregate_from_routers(self.routers)

    def get_buffer_occupancy_map(self, is_request: bool = True) -> Dict[Tuple[int, int], int]:
        """
        Get buffer occupancy for visualization.

        Args:
            is_request: True for request network.

        Returns:
            Dict of (x, y) -> total occupancy.
        """
        result = {}
        for coord, router in self.routers.items():
            if is_request:
                result[coord] = router.total_req_occupancy
            else:
                result[coord] = router.total_resp_occupancy
        return result

    def print_topology(self) -> None:
        """Print mesh topology for debugging."""
        print(f"Mesh Topology: {self.config.cols}x{self.config.rows}")
        print(f"Edge column: {self.config.edge_column}")
        print(f"Compute nodes: {self.config.num_compute_nodes}")
        print()

        # Print mesh layout (top to bottom = high Y to low Y)
        for y in range(self.config.rows - 1, -1, -1):
            row_str = ""
            for x in range(self.config.cols):
                coord = (x, y)
                if x == self.config.edge_column:
                    row_str += f"[E{y}]"
                else:
                    node_id = self.address_map.get_node_id(coord)
                    row_str += f"[N{node_id:02d}]"
                if x < self.config.cols - 1:
                    row_str += "──"
            print(row_str)
            if y > 0:
                # Print vertical connections
                vert_str = ""
                for x in range(self.config.cols):
                    vert_str += "  │  "
                    if x < self.config.cols - 1:
                        vert_str += "  "
                print(vert_str)
        print()

    def __repr__(self) -> str:
        return (
            f"Mesh({self.config.cols}x{self.config.rows}, "
            f"routers={len(self.routers)}, "
            f"nis={len(self.nis)})"
        )


# =============================================================================
# Factory functions
# =============================================================================

def create_mesh(
    cols: int = 5,
    rows: int = 4,
    edge_column: int = 0,
    buffer_depth: int = 4,
    channel_mode: ChannelMode = ChannelMode.GENERAL
) -> Mesh:
    """
    Create a mesh with default configuration.

    Args:
        cols: Number of columns.
        rows: Number of rows.
        edge_column: Edge router column.
        buffer_depth: Router buffer depth.
        channel_mode: Physical channel mode (GENERAL or AXI).

    Returns:
        Configured Mesh instance.
    """
    router_config = RouterConfig(
        buffer_depth=buffer_depth,
        channel_mode=channel_mode
    )
    ni_config = NIConfig(req_buffer_depth=buffer_depth, resp_buffer_depth=buffer_depth)

    config = MeshConfig(
        cols=cols,
        rows=rows,
        edge_column=edge_column,
        channel_mode=channel_mode,
        router_config=router_config,
        ni_config=ni_config,
    )

    return Mesh(config)
