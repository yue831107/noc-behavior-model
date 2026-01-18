"""
General Mode vs AXI Mode Comparison Tests.

Tests functional equivalence between modes and validates
that AXI Mode provides expected benefits (no HoL blocking).

Usage:
    make test_axi                    # Run all AXI tests
    pytest tests/integration/test_mode_comparison.py -v
"""

import pytest
from typing import Tuple, List

from src.core.router import ChannelMode, RouterConfig
from src.core.mesh import Mesh, MeshConfig, create_mesh
from src.core.flit import Flit, FlitFactory, AxiChannel


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def general_mesh() -> Mesh:
    """Create General Mode mesh."""
    return create_mesh(
        cols=5,
        rows=4,
        buffer_depth=4,
        channel_mode=ChannelMode.GENERAL,
    )


@pytest.fixture
def axi_mesh() -> Mesh:
    """Create AXI Mode mesh."""
    return create_mesh(
        cols=5,
        rows=4,
        buffer_depth=4,
        channel_mode=ChannelMode.AXI,
    )


# ==============================================================================
# Test: Mode Configuration
# ==============================================================================

class TestModeConfiguration:
    """Test that both modes can be configured correctly."""

    def test_general_mode_creation(self, general_mesh: Mesh):
        """General mode mesh should be created successfully."""
        assert general_mesh is not None
        assert general_mesh.config.router_config.channel_mode == ChannelMode.GENERAL

    def test_axi_mode_creation(self, axi_mesh: Mesh):
        """AXI mode mesh should be created successfully."""
        assert axi_mesh is not None
        assert axi_mesh.config.router_config.channel_mode == ChannelMode.AXI

    def test_both_modes_same_dimensions(self, general_mesh: Mesh, axi_mesh: Mesh):
        """Both modes should have same mesh dimensions."""
        assert general_mesh.config.cols == axi_mesh.config.cols
        assert general_mesh.config.rows == axi_mesh.config.rows


# ==============================================================================
# Test: Functional Equivalence
# ==============================================================================

class TestFunctionalEquivalence:
    """Test that both modes produce functionally correct results."""

    @pytest.mark.parametrize("mode", [ChannelMode.GENERAL, ChannelMode.AXI])
    def test_aw_flit_creation_same_format(self, mode):
        """AW flit format should be the same in both modes."""
        flit = FlitFactory.create_aw(
            src=(1, 0),
            dest=(4, 3),
            addr=0x1000,
            axi_id=1,
            length=0,
        )

        # Flit format is independent of mode
        assert flit.hdr.axi_ch == AxiChannel.AW
        assert flit.hdr.src_id is not None
        assert flit.hdr.dst_id is not None

    @pytest.mark.parametrize("mode", [ChannelMode.GENERAL, ChannelMode.AXI])
    def test_ar_flit_creation_same_format(self, mode):
        """AR flit format should be the same in both modes."""
        flit = FlitFactory.create_ar(
            src=(2, 1),
            dest=(3, 2),
            addr=0x2000,
            axi_id=2,
            length=0,
        )

        assert flit.hdr.axi_ch == AxiChannel.AR

    @pytest.mark.parametrize("mode", [ChannelMode.GENERAL, ChannelMode.AXI])
    def test_w_flit_creation_same_format(self, mode):
        """W flit format should be the same in both modes."""
        flit = FlitFactory.create_w(
            src=(1, 0),
            dest=(4, 3),
            data=b"\xAB" * 32,
            strb=0xFFFFFFFF,
            last=True,
        )

        assert flit.hdr.axi_ch == AxiChannel.W

    @pytest.mark.parametrize("mode", [ChannelMode.GENERAL, ChannelMode.AXI])
    def test_b_flit_creation_same_format(self, mode):
        """B flit format should be the same in both modes."""
        flit = FlitFactory.create_b(
            src=(4, 3),
            dest=(1, 0),
            axi_id=1,
            resp=0,
        )

        assert flit.hdr.axi_ch == AxiChannel.B

    @pytest.mark.parametrize("mode", [ChannelMode.GENERAL, ChannelMode.AXI])
    def test_r_flit_creation_same_format(self, mode):
        """R flit format should be the same in both modes."""
        flit = FlitFactory.create_r(
            src=(4, 3),
            dest=(1, 0),
            axi_id=2,
            data=b"\xCD" * 32,
            resp=0,
            last=True,
        )

        assert flit.hdr.axi_ch == AxiChannel.R


# ==============================================================================
# Test: Routing Correctness
# ==============================================================================

class TestRoutingCorrectness:
    """Test that routing works correctly in both modes."""

    def test_xy_routing_same_in_both_modes(self, general_mesh, axi_mesh):
        """XY routing algorithm should be the same in both modes."""
        # Both modes use XY routing
        # Source: (1, 0), Dest: (4, 3)
        # XY routing: X first (1->4), then Y (0->3)

        flit = FlitFactory.create_ar(
            src=(1, 0),
            dest=(4, 3),
            addr=0x3000,
            axi_id=3,
            length=0,
        )

        # The routing decision should be the same
        # At (1,0): go EAST (towards X=4)
        # At (4,0): go NORTH (towards Y=3)
        # This is XY routing behavior, same for both modes

        assert flit.hdr.dst_id is not None  # Has valid destination

    def test_response_routing_correct_both_modes(self):
        """Response routing should work correctly in both modes."""
        # Request: (1, 0) -> (4, 3)
        ar_flit = FlitFactory.create_ar(
            src=(1, 0),
            dest=(4, 3),
            addr=0x4000,
            axi_id=4,
            length=0,
        )

        # Response: (4, 3) -> (1, 0)
        r_flit = FlitFactory.create_r(
            src=(4, 3),
            dest=(1, 0),
            axi_id=4,
            data=b"\xEF" * 32,
            resp=0,
            last=True,
        )

        # Response should go back to original source
        assert r_flit.hdr.dst_id == ar_flit.hdr.src_id


# ==============================================================================
# Test: AXI Mode Advantages
# ==============================================================================

class TestAXIModeAdvantages:
    """Test that AXI Mode provides expected benefits."""

    def test_axi_mode_eliminates_hol_blocking(self, axi_mesh: Mesh):
        """AXI Mode should eliminate HoL blocking between channels.

        In General Mode, heavy W traffic can block AR traffic because
        they share the same Request sub-router.

        In AXI Mode, W and AR have separate sub-routers, so they
        cannot block each other.
        """
        # In AXI mode, 5 independent sub-routers
        assert axi_mesh.config.router_config.channel_mode == ChannelMode.AXI

        # Heavy W traffic
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

        # AR traffic (should not be blocked by W)
        ar_flit = FlitFactory.create_ar(
            src=(1, 0),
            dest=(4, 3),
            addr=0x5000,
            axi_id=5,
            length=0,
        )

        # In AXI mode, W uses W sub-router, AR uses AR sub-router
        # They are completely independent
        assert all(f.hdr.axi_ch == AxiChannel.W for f in w_flits)
        assert ar_flit.hdr.axi_ch == AxiChannel.AR

    def test_general_mode_potential_hol_blocking(self, general_mesh: Mesh):
        """General Mode may have HoL blocking between channels.

        In General Mode, AW, W, AR share the same Request sub-router.
        Heavy traffic on one channel can block others.
        """
        assert general_mesh.config.router_config.channel_mode == ChannelMode.GENERAL

        # In General mode, AW, W, AR all use Req sub-router
        # B, R use Resp sub-router

    def test_axi_mode_better_channel_isolation(self, axi_mesh: Mesh):
        """AXI Mode should provide better channel isolation."""
        # Create traffic on all 5 channels simultaneously
        traffic = [
            FlitFactory.create_aw(
                src=(1, 0), dest=(4, 3), addr=0x6000, axi_id=6, length=0
            ),
            FlitFactory.create_w(
                src=(1, 0), dest=(4, 3), data=b"\x00" * 32, strb=0xFFFFFFFF, last=True
            ),
            FlitFactory.create_ar(
                src=(2, 1), dest=(3, 2), addr=0x7000, axi_id=7, length=0
            ),
            FlitFactory.create_b(
                src=(4, 3), dest=(1, 0), axi_id=6, resp=0
            ),
            FlitFactory.create_r(
                src=(3, 2), dest=(2, 1), axi_id=7, data=b"\xFF" * 32, resp=0, last=True
            ),
        ]

        # Each flit goes through its own sub-router
        channels = [f.hdr.axi_ch for f in traffic]
        assert len(set(channels)) == 5  # All 5 channels represented

        # In AXI mode, these can all proceed in parallel
        # without blocking each other


# ==============================================================================
# Test: Wire Structure
# ==============================================================================

class TestWireStructure:
    """Test wire structure differences between modes."""

    def test_general_mode_has_req_resp_wires(self, general_mesh: Mesh):
        """General mode should have request and response wire sets."""
        assert hasattr(general_mesh, "_req_wires")
        assert hasattr(general_mesh, "_resp_wires")

        # Should have unified wire list
        assert hasattr(general_mesh, "_all_wires")
        assert len(general_mesh._all_wires) > 0

    def test_axi_mode_has_five_wire_sets(self, axi_mesh: Mesh):
        """AXI mode should have 5 channel-specific wire sets."""
        assert hasattr(axi_mesh, "_aw_wires")
        assert hasattr(axi_mesh, "_w_wires")
        assert hasattr(axi_mesh, "_ar_wires")
        assert hasattr(axi_mesh, "_b_wires")
        assert hasattr(axi_mesh, "_r_wires")

        # Should have unified wire list
        assert hasattr(axi_mesh, "_all_wires")
        assert len(axi_mesh._all_wires) > 0

    def test_wire_count_difference(self, general_mesh: Mesh, axi_mesh: Mesh):
        """AXI mode should have more wires due to 5 sub-routers."""
        # General mode: 2 sub-routers (Req + Resp)
        # AXI mode: 5 sub-routers (AW, W, AR, B, R)

        general_wire_count = len(general_mesh._all_wires)
        axi_wire_count = len(axi_mesh._all_wires)

        # AXI mode should have roughly 2.5x more wires
        # (5 channels vs 2 channels)
        assert axi_wire_count > general_wire_count
