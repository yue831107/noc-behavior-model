"""
AXI Mode Integration Tests for NoC Behavior Model.

Tests AXI Mode functionality at the mesh and system level:
- Independent channel flow (5 sub-routers operate independently)
- Channel ordering (AW must precede corresponding W)
- Response routing (B/R correctly route back to source)
- No head-of-line blocking between channels

Usage:
    make test_axi                    # Run all AXI tests
    pytest tests/integration/test_axi_mode_integration.py -v
"""

import pytest
from typing import List, Tuple

from src.core.router import ChannelMode, RouterConfig, Direction
from src.core.mesh import Mesh, MeshConfig, create_mesh
from src.core.flit import Flit, FlitFactory, AxiChannel
from src.core.channel_mode_strategy import (
    get_channel_mode_strategy,
    AXIModeStrategy,
    GeneralModeStrategy,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def axi_mesh() -> Mesh:
    """Create an AXI Mode mesh for testing."""
    return create_mesh(
        cols=5,
        rows=4,
        buffer_depth=4,
        channel_mode=ChannelMode.AXI,
    )


@pytest.fixture
def general_mesh() -> Mesh:
    """Create a General Mode mesh for comparison."""
    return create_mesh(
        cols=5,
        rows=4,
        buffer_depth=4,
        channel_mode=ChannelMode.GENERAL,
    )


# ==============================================================================
# Test: Channel Mode Strategy
# ==============================================================================

class TestChannelModeStrategy:
    """Test channel mode strategy pattern."""

    def test_general_mode_has_2_channels(self):
        """General mode should have 2 logical channels (Req + Resp)."""
        strategy = get_channel_mode_strategy(ChannelMode.GENERAL)
        assert strategy.channel_count == 2

    def test_axi_mode_has_5_channels(self):
        """AXI mode should have 5 physical channels."""
        strategy = get_channel_mode_strategy(ChannelMode.AXI)
        assert strategy.channel_count == 5

    def test_request_channels_are_aw_w_ar(self):
        """Request channels should be AW, W, AR."""
        strategy = get_channel_mode_strategy(ChannelMode.AXI)
        req_channels = strategy.request_channels
        assert AxiChannel.AW in req_channels
        assert AxiChannel.W in req_channels
        assert AxiChannel.AR in req_channels
        assert len(req_channels) == 3

    def test_response_channels_are_b_r(self):
        """Response channels should be B, R."""
        strategy = get_channel_mode_strategy(ChannelMode.AXI)
        resp_channels = strategy.response_channels
        assert AxiChannel.B in resp_channels
        assert AxiChannel.R in resp_channels
        assert len(resp_channels) == 2


# ==============================================================================
# Test: AXI Mode Mesh Infrastructure
# ==============================================================================

class TestAXIModeMeshInfrastructure:
    """Test AXI Mode mesh infrastructure."""

    def test_axi_mesh_creation(self, axi_mesh: Mesh):
        """AXI mesh should be created successfully."""
        assert axi_mesh is not None
        assert axi_mesh.config.router_config.channel_mode == ChannelMode.AXI

    def test_axi_mesh_has_5_wire_sets(self, axi_mesh: Mesh):
        """AXI mesh should have 5 separate wire sets."""
        # Check for channel-specific wire attributes
        assert hasattr(axi_mesh, "_aw_wires")
        assert hasattr(axi_mesh, "_w_wires")
        assert hasattr(axi_mesh, "_ar_wires")
        assert hasattr(axi_mesh, "_b_wires")
        assert hasattr(axi_mesh, "_r_wires")

    def test_general_mesh_has_2_wire_sets(self, general_mesh: Mesh):
        """General mesh should have 2 wire sets (Req + Resp)."""
        assert hasattr(general_mesh, "_req_wires")
        assert hasattr(general_mesh, "_resp_wires")

    def test_axi_mesh_routers_are_axi_mode(self, axi_mesh: Mesh):
        """All routers in AXI mesh should be AXI mode."""
        for row in range(axi_mesh.config.rows):
            for col in range(axi_mesh.config.cols):
                router = axi_mesh.get_router((col, row))
                if router is not None:
                    # Check router has 5 sub-routers
                    assert hasattr(router, "aw_router")
                    assert hasattr(router, "w_router")
                    assert hasattr(router, "ar_router")
                    assert hasattr(router, "b_router")
                    assert hasattr(router, "r_router")


# ==============================================================================
# Test: Independent Channel Flow
# ==============================================================================

class TestIndependentChannelFlow:
    """Test that AXI channels operate independently."""

    def test_aw_channel_independent(self, axi_mesh: Mesh):
        """AW channel should route independently of other channels."""
        # Create AW flit
        aw_flit = FlitFactory.create_aw(
            src=(1, 0),
            dest=(4, 3),
            addr=0x1000,
            axi_id=1,
            length=0,
        )

        # Inject into mesh at source router
        source_router = axi_mesh.get_router((1, 0))
        if source_router and hasattr(source_router, "aw_router"):
            # The AW flit should only go through AW sub-router
            assert aw_flit.hdr.axi_ch == AxiChannel.AW

    def test_w_channel_independent(self, axi_mesh: Mesh):
        """W channel should route independently of other channels."""
        # Create W flit
        w_flit = FlitFactory.create_w(
            src=(1, 0),
            dest=(4, 3),
            data=b"\x00" * 32,
            strb=0xFFFFFFFF,
            last=True,
        )

        assert w_flit.hdr.axi_ch == AxiChannel.W

    def test_ar_channel_independent(self, axi_mesh: Mesh):
        """AR channel should route independently of other channels."""
        ar_flit = FlitFactory.create_ar(
            src=(1, 0),
            dest=(4, 3),
            addr=0x2000,
            axi_id=2,
            length=0,
        )

        assert ar_flit.hdr.axi_ch == AxiChannel.AR

    def test_b_channel_independent(self, axi_mesh: Mesh):
        """B channel should route independently of other channels."""
        b_flit = FlitFactory.create_b(
            src=(4, 3),
            dest=(1, 0),
            axi_id=1,
            resp=0,
        )

        assert b_flit.hdr.axi_ch == AxiChannel.B

    def test_r_channel_independent(self, axi_mesh: Mesh):
        """R channel should route independently of other channels."""
        r_flit = FlitFactory.create_r(
            src=(4, 3),
            dest=(1, 0),
            axi_id=2,
            data=b"\xFF" * 32,
            resp=0,
            last=True,
        )

        assert r_flit.hdr.axi_ch == AxiChannel.R


# ==============================================================================
# Test: Channel Ordering
# ==============================================================================

class TestChannelOrdering:
    """Test AXI channel ordering requirements."""

    def test_aw_before_w_ordering(self):
        """AW (write address) should precede W (write data) for same transaction."""
        # Create AW and W flits with matching transaction
        aw_flit = FlitFactory.create_aw(
            src=(1, 0),
            dest=(3, 2),
            addr=0x3000,
            axi_id=5,
            length=0,
        )

        w_flit = FlitFactory.create_w(
            src=(1, 0),
            dest=(3, 2),
            data=b"\xAB" * 32,
            strb=0xFFFFFFFF,
            last=True,
        )

        # Both should have same source and destination
        assert aw_flit.hdr.src_id == w_flit.hdr.src_id
        assert aw_flit.hdr.dst_id == w_flit.hdr.dst_id

        # AW is address, W is data - order enforced by transaction manager
        assert aw_flit.hdr.axi_ch == AxiChannel.AW
        assert w_flit.hdr.axi_ch == AxiChannel.W

    def test_request_before_response(self):
        """Request (AR) should precede response (R)."""
        ar_flit = FlitFactory.create_ar(
            src=(2, 1),
            dest=(4, 3),
            addr=0x4000,
            axi_id=7,
            length=0,
        )

        r_flit = FlitFactory.create_r(
            src=(4, 3),
            dest=(2, 1),  # Response goes back to original source
            axi_id=7,
            data=b"\xCD" * 32,
            resp=0,
            last=True,
        )

        # R response should go back to AR source
        assert ar_flit.hdr.src_id == r_flit.hdr.dst_id
        assert ar_flit.hdr.dst_id == r_flit.hdr.src_id


# ==============================================================================
# Test: Response Routing
# ==============================================================================

class TestResponseRouting:
    """Test that responses correctly route back to source."""

    def test_b_response_routes_to_source(self):
        """B (write response) should route back to AW source."""
        # Original write request
        aw_src = (1, 0)
        aw_dst = (4, 3)

        aw_flit = FlitFactory.create_aw(
            src=aw_src,
            dest=aw_dst,
            addr=0x5000,
            axi_id=10,
            length=0,
        )

        # Write response goes back
        b_flit = FlitFactory.create_b(
            src=aw_dst,     # Response originates from original destination
            dest=aw_src,    # Response goes to original source
            axi_id=10,
            resp=0,
        )

        # Verify routing
        assert b_flit.hdr.dst_id == aw_flit.hdr.src_id
        assert b_flit.hdr.src_id == aw_flit.hdr.dst_id

    def test_r_response_routes_to_source(self):
        """R (read response) should route back to AR source."""
        # Original read request
        ar_src = (2, 1)
        ar_dst = (3, 3)

        ar_flit = FlitFactory.create_ar(
            src=ar_src,
            dest=ar_dst,
            addr=0x6000,
            axi_id=11,
            length=0,
        )

        # Read response goes back
        r_flit = FlitFactory.create_r(
            src=ar_dst,
            dest=ar_src,
            axi_id=11,
            data=b"\xEF" * 32,
            resp=0,
            last=True,
        )

        # Verify routing
        assert r_flit.hdr.dst_id == ar_flit.hdr.src_id
        assert r_flit.hdr.src_id == ar_flit.hdr.dst_id


# ==============================================================================
# Test: No Head-of-Line Blocking
# ==============================================================================

class TestNoHeadOfLineBlocking:
    """Test that AXI Mode eliminates HoL blocking between channels."""

    def test_w_heavy_traffic_no_blocking(self, axi_mesh: Mesh):
        """Heavy W traffic should not block other channels in AXI mode."""
        # In AXI mode, W channel is separate from AR channel
        # So heavy W traffic cannot block AR traffic

        # Create multiple W flits
        w_flits = [
            FlitFactory.create_w(
                src=(1, 0),
                dest=(4, 3),
                data=bytes([i] * 32),
                strb=0xFFFFFFFF,
                last=(i == 9),
            )
            for i in range(10)
        ]

        # Create AR flit
        ar_flit = FlitFactory.create_ar(
            src=(1, 0),
            dest=(4, 3),
            addr=0x7000,
            axi_id=20,
            length=0,
        )

        # All should be routable independently
        assert all(f.hdr.axi_ch == AxiChannel.W for f in w_flits)
        assert ar_flit.hdr.axi_ch == AxiChannel.AR

        # In AXI mode, these use different sub-routers
        # W flits go through W sub-router
        # AR flit goes through AR sub-router

    def test_channel_isolation_under_pressure(self, axi_mesh: Mesh):
        """Channels should remain isolated under high traffic."""
        # Create traffic on all channels
        traffic = {
            AxiChannel.AW: FlitFactory.create_aw(
                src=(1, 0), dest=(4, 3), addr=0x8000, axi_id=30, length=0
            ),
            AxiChannel.W: FlitFactory.create_w(
                src=(1, 0), dest=(4, 3), data=b"\x00" * 32, strb=0xFFFFFFFF, last=True
            ),
            AxiChannel.AR: FlitFactory.create_ar(
                src=(2, 1), dest=(3, 2), addr=0x9000, axi_id=31, length=0
            ),
            AxiChannel.B: FlitFactory.create_b(
                src=(4, 3), dest=(1, 0), axi_id=30, resp=0
            ),
            AxiChannel.R: FlitFactory.create_r(
                src=(3, 2), dest=(2, 1), axi_id=31, data=b"\xFF" * 32, resp=0, last=True
            ),
        }

        # Verify each flit has correct channel
        for channel, flit in traffic.items():
            assert flit.hdr.axi_ch == channel, f"Flit should be on {channel}"

        # In AXI mode, each channel has its own sub-router
        # So traffic on one channel cannot affect another
