"""
Shared pytest fixtures for NoC Behavior Model tests.

This module provides common fixtures for creating routers, NIs, flits,
and network topologies used across unit and integration tests.
"""

import pytest
from typing import Tuple, List, Optional

from src.core.flit import (
    Flit, FlitFactory, FlitHeader, AxiChannel,
    AxiAwPayload, AxiWPayload, AxiArPayload, AxiBPayload, AxiRPayload,
    encode_node_id, decode_node_id
)
from src.core.buffer import FlitBuffer
from src.core.router import (
    Direction, RouterConfig, RouterPort, PortWire,
    XYRouter, ReqRouter, RespRouter, Router, EdgeRouter,
    WormholeArbiter
)
from src.core.ni import MasterNI, NIConfig
from src.core.mesh import Mesh, MeshConfig, create_mesh


# ==============================================================================
# Configuration Fixtures
# ==============================================================================

@pytest.fixture
def router_config() -> RouterConfig:
    """Default router configuration for testing."""
    return RouterConfig(
        buffer_depth=4,
        output_buffer_depth=0,
        flit_width=64,
        routing_algorithm="XY",
        arbitration="wormhole",
    )


@pytest.fixture
def router_config_with_output_buffer() -> RouterConfig:
    """Router configuration with output buffer enabled."""
    return RouterConfig(
        buffer_depth=4,
        output_buffer_depth=2,
        flit_width=64,
        routing_algorithm="XY",
        arbitration="wormhole",
    )


@pytest.fixture
def ni_config() -> NIConfig:
    """Default NI configuration for testing."""
    return NIConfig(
        req_buffer_depth=8,
        resp_buffer_depth=8,
    )


# ==============================================================================
# Flit Fixtures
# ==============================================================================

@pytest.fixture(autouse=True)
def reset_flit_counters():
    """Reset RoB index counter before each test.

    Note: This is process-safe for pytest-xdist parallel execution.
    Each worker process has its own Python interpreter and class variables,
    so FlitFactory._rob_idx_counter is effectively process-local.
    """
    FlitFactory.reset()
    yield


@pytest.fixture
def single_flit_factory():
    """Factory for creating single-flit packets (FlooNoC style)."""
    def _create(
        src: Tuple[int, int],
        dest: Tuple[int, int],
        is_request: bool = True,
        payload: bytes = b"test_data",
        axi_ch: AxiChannel = None,
    ) -> Flit:
        """
        Create a single-flit packet.

        For requests: creates AR flit (read address)
        For responses: creates B flit (write response)
        """
        if axi_ch is None:
            axi_ch = AxiChannel.AR if is_request else AxiChannel.B

        if axi_ch == AxiChannel.AR:
            return FlitFactory.create_ar(
                src=src,
                dest=dest,
                addr=0x1000,
                axi_id=0,
                length=0,
            )
        elif axi_ch == AxiChannel.B:
            return FlitFactory.create_b(
                src=src,
                dest=dest,
                axi_id=0,
                resp=0,
            )
        elif axi_ch == AxiChannel.AW:
            return FlitFactory.create_aw(
                src=src,
                dest=dest,
                addr=0x1000,
                axi_id=0,
                length=0,
                last=True,  # Single flit AW (no W follows)
            )
        elif axi_ch == AxiChannel.R:
            return FlitFactory.create_r(
                src=src,
                dest=dest,
                data=payload if len(payload) <= 32 else payload[:32],
                axi_id=0,
                last=True,
            )
        else:  # W
            return FlitFactory.create_w(
                src=src,
                dest=dest,
                data=payload if len(payload) <= 32 else payload[:32],
                last=True,
            )
    return _create


@pytest.fixture
def multi_flit_packet_factory():
    """Factory for creating multi-flit packets (FlooNoC style)."""
    def _create(
        src: Tuple[int, int],
        dest: Tuple[int, int],
        num_flits: int = 3,
        is_request: bool = True,
    ) -> List[Flit]:
        """
        Create a packet with specified number of flits.

        For requests: creates AW + W flit(s)
        For responses: creates R flit(s)

        FlooNoC format uses 'last' bit instead of HEAD/BODY/TAIL types.
        """
        flits = []
        rob_idx = FlitFactory._next_rob_idx()

        if is_request:
            # Request: AW flit + W flit(s)
            if num_flits == 1:
                # Single W flit (for simple data transfer)
                flits.append(FlitFactory.create_w(
                    src=src,
                    dest=dest,
                    data=b"SINGLE" + bytes(26),
                    last=True,
                    rob_idx=rob_idx,
                    seq_num=0,
                ))
            else:
                # AW flit (not last, W follows)
                flits.append(FlitFactory.create_aw(
                    src=src,
                    dest=dest,
                    addr=0x1000,
                    axi_id=0,
                    length=num_flits - 2,  # Number of W beats - 1
                    rob_idx=rob_idx,
                    rob_req=True,
                    last=False,  # W flit(s) follow
                ))

                # W flits
                for i in range(num_flits - 1):
                    is_last = (i == num_flits - 2)
                    data = b"TAIL" if is_last else f"BODY{i}".encode()
                    data = data + bytes(32 - len(data))
                    flits.append(FlitFactory.create_w(
                        src=src,
                        dest=dest,
                        data=data,
                        last=is_last,
                        rob_idx=rob_idx,
                        seq_num=i,
                    ))
        else:
            # Response: R flit(s)
            for i in range(num_flits):
                is_last = (i == num_flits - 1)
                if num_flits == 1:
                    data = b"SINGLE" + bytes(26)
                elif is_last:
                    data = b"TAIL" + bytes(28)
                elif i == 0:
                    data = b"HEAD" + bytes(28)
                else:
                    data = f"BODY{i}".encode()
                    data = data + bytes(32 - len(data))

                flits.append(FlitFactory.create_r(
                    src=src,
                    dest=dest,
                    data=data,
                    axi_id=0,
                    last=is_last,
                    rob_idx=rob_idx,
                    seq_num=i,
                ))

        return flits
    return _create


# ==============================================================================
# Router Fixtures
# ==============================================================================

@pytest.fixture
def xy_router(router_config) -> XYRouter:
    """Create a single XYRouter at (2,2)."""
    return XYRouter(coord=(2, 2), config=router_config, name="TestRouter")


@pytest.fixture
def req_router(router_config) -> ReqRouter:
    """Create a Request Router at (2,2)."""
    return ReqRouter(coord=(2, 2), config=router_config)


@pytest.fixture
def resp_router(router_config) -> RespRouter:
    """Create a Response Router at (2,2)."""
    return RespRouter(coord=(2, 2), config=router_config)


@pytest.fixture
def combined_router(router_config) -> Router:
    """Create a combined Router (Req + Resp) at (2,2)."""
    return Router(coord=(2, 2), config=router_config)


@pytest.fixture
def edge_router(router_config) -> EdgeRouter:
    """Create an EdgeRouter at (0,1)."""
    return EdgeRouter(coord=(0, 1), config=router_config)


@pytest.fixture
def wormhole_arbiter() -> WormholeArbiter:
    """Create a fresh WormholeArbiter."""
    return WormholeArbiter()


# ==============================================================================
# Connected Router Fixtures
# ==============================================================================

@pytest.fixture
def two_routers_horizontal(router_config) -> Tuple[XYRouter, XYRouter, PortWire]:
    """
    Create two horizontally connected routers.

    R1(1,1) --EAST-- R2(2,1)

    Returns:
        Tuple of (R1, R2, PortWire connecting them)
    """
    r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")
    r2 = XYRouter(coord=(2, 1), config=router_config, name="R2")

    # Connect R1.EAST <-> R2.WEST using PortWire
    wire = PortWire(r1.ports[Direction.EAST], r2.ports[Direction.WEST])

    return r1, r2, wire


@pytest.fixture
def two_routers_vertical(router_config) -> Tuple[XYRouter, XYRouter, PortWire]:
    """
    Create two vertically connected routers.

    R2(1,2)
       |
     NORTH
       |
    R1(1,1)

    Returns:
        Tuple of (R1, R2, PortWire connecting them)
    """
    r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")
    r2 = XYRouter(coord=(1, 2), config=router_config, name="R2")

    # Connect R1.NORTH <-> R2.SOUTH
    wire = PortWire(r1.ports[Direction.NORTH], r2.ports[Direction.SOUTH])

    return r1, r2, wire


@pytest.fixture
def router_chain_horizontal(router_config) -> Tuple[List[XYRouter], List[PortWire]]:
    """
    Create 3 horizontally connected routers.

    R1(1,1) --E-- R2(2,1) --E-- R3(3,1)

    Returns:
        Tuple of ([R1, R2, R3], [wire1, wire2])
    """
    r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")
    r2 = XYRouter(coord=(2, 1), config=router_config, name="R2")
    r3 = XYRouter(coord=(3, 1), config=router_config, name="R3")

    wire1 = PortWire(r1.ports[Direction.EAST], r2.ports[Direction.WEST])
    wire2 = PortWire(r2.ports[Direction.EAST], r3.ports[Direction.WEST])

    return [r1, r2, r3], [wire1, wire2]


@pytest.fixture
def router_chain_vertical(router_config) -> Tuple[List[XYRouter], List[PortWire]]:
    """
    Create 3 vertically connected routers.

    R3(1,3)
       |
    R2(1,2)
       |
    R1(1,1)

    Returns:
        Tuple of ([R1, R2, R3], [wire1, wire2])
    """
    r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")
    r2 = XYRouter(coord=(1, 2), config=router_config, name="R2")
    r3 = XYRouter(coord=(1, 3), config=router_config, name="R3")

    wire1 = PortWire(r1.ports[Direction.NORTH], r2.ports[Direction.SOUTH])
    wire2 = PortWire(r2.ports[Direction.NORTH], r3.ports[Direction.SOUTH])

    return [r1, r2, r3], [wire1, wire2]


# ==============================================================================
# NI Fixtures
# ==============================================================================

@pytest.fixture
def master_ni(ni_config) -> MasterNI:
    """Create a MasterNI at (1,1)."""
    return MasterNI(
        coord=(1, 1),
        config=ni_config,
        ni_id=0,
        node_id=0,
    )


@pytest.fixture
def router_with_ni(router_config, ni_config) -> Tuple[Router, MasterNI]:
    """
    Create a Router connected to a MasterNI at (2,2).

    Sets up virtual port connections as done in Mesh._connect_nis().

    Returns:
        Tuple of (Router, MasterNI)
    """
    router = Router(coord=(2, 2), config=router_config)
    ni = MasterNI(
        coord=(2, 2),
        config=ni_config,
        ni_id=0,
        node_id=0,
    )

    # Setup the virtual port connections as done in Mesh._connect_nis()
    # Request path: Router LOCAL -> NI
    req_local = router.get_req_port(Direction.LOCAL)
    ni_req_port = RouterPort(
        direction=Direction.LOCAL,
        buffer_depth=ni_config.req_buffer_depth,
        name="NI_req_in"
    )
    req_local.neighbor = ni_req_port
    ni_req_port.neighbor = req_local

    # Response path: NI -> Router LOCAL
    resp_local = router.get_resp_port(Direction.LOCAL)
    ni_resp_port = RouterPort(
        direction=Direction.LOCAL,
        buffer_depth=ni_config.resp_buffer_depth,
        name="NI_resp_out"
    )
    ni_resp_port.neighbor = resp_local

    # Store virtual ports in NI for transfer logic
    ni._router_req_port = ni_req_port
    ni._router_resp_port = ni_resp_port

    return router, ni


# ==============================================================================
# Mesh Fixtures
# ==============================================================================

@pytest.fixture
def small_mesh() -> Mesh:
    """Create a small 3x2 mesh for testing."""
    return create_mesh(cols=3, rows=2, edge_column=0, buffer_depth=4)


@pytest.fixture
def standard_mesh() -> Mesh:
    """Create a standard 5x4 mesh."""
    return create_mesh(cols=5, rows=4, edge_column=0, buffer_depth=4)


# ==============================================================================
# Helper Functions (not fixtures - can be imported directly)
# ==============================================================================

def run_router_cycle(
    router: XYRouter,
    wires: Optional[List[PortWire]] = None,
    time: int = 0
) -> List[Tuple[Flit, Direction]]:
    """
    Run a complete processing cycle for a router with wire propagation.

    Cycle-accurate sequence (samples signals from previous cycle first):
    1. Sample inputs (from signals propagated at end of last cycle)
    2. Clear input signals
    3. Update ready signals (based on new buffer state)
    4. Route and forward
    5. Propagate wire signals (propagates ready AND valid/flit for next cycle)
    6. Clear accepted outputs
    7. Handle credit release

    Args:
        router: Router to process
        wires: Optional list of PortWires connected to this router
        time: Current simulation time

    Returns:
        List of forwarded (flit, direction) pairs
    """
    # Phase 1: Sample inputs (from signals set at end of last cycle)
    router.sample_all_inputs()

    # Phase 2: Clear input signals
    router.clear_all_input_signals()

    # Phase 3: Update ready (based on new buffer state)
    router.update_all_ready()

    # Phase 4: Route and forward
    forwarded = router.route_and_forward(time)

    # Phase 5: Propagate signals (ready AND valid/flit for next cycle)
    if wires:
        for w in wires:
            w.propagate_signals()

    # Phase 6: Clear accepted outputs
    router.clear_accepted_outputs()

    # Phase 7: Handle credit release
    if wires:
        for w in wires:
            w.propagate_credit_release()

    return forwarded


def run_multi_router_cycle(
    routers: List[XYRouter],
    wires: List[PortWire],
    time: int = 0
) -> List[Tuple[Flit, Direction]]:
    """
    Run a complete cycle for multiple connected routers.

    Cycle-accurate sequence (samples signals from previous cycle first):
    1. Sample inputs (from signals propagated at end of last cycle)
    2. Clear input signals
    3. Update ready signals
    4. Route and forward
    5. Propagate wire signals (for next cycle to sample)
    6. Clear accepted outputs
    7. Handle credit release

    Args:
        routers: List of routers to process
        wires: List of PortWires connecting the routers
        time: Current simulation time

    Returns:
        List of all forwarded (flit, direction) pairs from all routers
    """
    # Phase 1: Sample inputs (from signals set at end of last cycle)
    for r in routers:
        r.sample_all_inputs()

    # Phase 2: Clear input signals
    for r in routers:
        r.clear_all_input_signals()

    # Phase 3: Update ready (based on new buffer state)
    for r in routers:
        r.update_all_ready()

    # Phase 4: Route and forward
    all_forwarded = []
    for r in routers:
        fwd = r.route_and_forward(time)
        all_forwarded.extend(fwd)

    # Phase 5: Propagate signals (for next cycle to sample)
    for w in wires:
        w.propagate_signals()

    # Phase 6: Clear accepted outputs
    for r in routers:
        r.clear_accepted_outputs()

    # Phase 7: Handle credit release
    for w in wires:
        w.propagate_credit_release()

    return all_forwarded


def transfer_ni_flits(router: Router, ni: MasterNI) -> Tuple[int, int]:
    """
    Transfer flits between Router and NI (as done in Mesh._transfer_ni_flits).

    Args:
        router: Router connected to NI
        ni: MasterNI to transfer flits with

    Returns:
        Tuple of (req_flits_transferred, resp_flits_transferred)
    """
    req_count = 0
    resp_count = 0

    # Transfer from virtual port (router LOCAL output) to NI
    if hasattr(ni, '_router_req_port'):
        while not ni._router_req_port.input_buffer.is_empty():
            flit = ni._router_req_port.pop()
            if flit is None:
                break
            if not ni.receive_req_flit(flit):
                break
            req_count += 1

    # Transfer from NI response output to router LOCAL input
    while ni.has_pending_response():
        flit = ni.get_resp_flit()
        if flit is None:
            break
        if not router.receive_response(Direction.LOCAL, flit):
            break
        resp_count += 1

    return req_count, resp_count
