"""
Unit tests for AXI Mode implementation.

Tests the channel-specific Sub-Router and wire infrastructure.
"""

import pytest
from typing import Tuple

from src.core.router import (
    ChannelMode, RouterConfig, Direction,
    AXIModeRouter, AXIModeEdgeRouter,
    AWRouter, WRouter, ARRouter, BRouter, RRouter,
    create_router
)
from src.core.mesh import Mesh, MeshConfig, create_mesh
from src.core.flit import AxiChannel, FlitFactory


class TestChannelMode:
    """Test ChannelMode enum."""

    def test_general_mode_exists(self):
        """General mode should exist."""
        assert ChannelMode.GENERAL is not None

    def test_axi_mode_exists(self):
        """AXI mode should exist."""
        assert ChannelMode.AXI is not None

    def test_modes_are_distinct(self):
        """General and AXI modes should be distinct."""
        assert ChannelMode.GENERAL != ChannelMode.AXI


class TestRouterConfigChannelMode:
    """Test RouterConfig channel_mode field."""

    def test_default_is_general(self):
        """Default channel mode should be GENERAL."""
        config = RouterConfig()
        assert config.channel_mode == ChannelMode.GENERAL

    def test_can_set_axi_mode(self):
        """Should be able to set AXI mode."""
        config = RouterConfig(channel_mode=ChannelMode.AXI)
        assert config.channel_mode == ChannelMode.AXI


class TestChannelSpecificRouters:
    """Test channel-specific Sub-Router classes."""

    @pytest.fixture
    def coord(self) -> Tuple[int, int]:
        return (1, 1)

    @pytest.fixture
    def config(self) -> RouterConfig:
        return RouterConfig(channel_mode=ChannelMode.AXI)

    def test_aw_router_creation(self, coord, config):
        """AWRouter should be creatable."""
        router = AWRouter(coord, config)
        assert router.coord == coord
        assert "AW" in router.name

    def test_w_router_creation(self, coord, config):
        """WRouter should be creatable."""
        router = WRouter(coord, config)
        assert router.coord == coord
        assert "W" in router.name

    def test_ar_router_creation(self, coord, config):
        """ARRouter should be creatable."""
        router = ARRouter(coord, config)
        assert router.coord == coord
        assert "AR" in router.name

    def test_b_router_creation(self, coord, config):
        """BRouter should be creatable."""
        router = BRouter(coord, config)
        assert router.coord == coord
        assert "B" in router.name

    def test_r_router_creation(self, coord, config):
        """RRouter should be creatable."""
        router = RRouter(coord, config)
        assert router.coord == coord
        assert "R" in router.name


class TestAXIModeRouter:
    """Test AXIModeRouter composite class."""

    @pytest.fixture
    def router(self) -> AXIModeRouter:
        return AXIModeRouter((2, 2))

    def test_has_all_five_subrouters(self, router):
        """AXIModeRouter should have all 5 Sub-Routers."""
        assert router.aw_router is not None
        assert router.w_router is not None
        assert router.ar_router is not None
        assert router.b_router is not None
        assert router.r_router is not None

    def test_get_channel_router(self, router):
        """Should be able to get Sub-Router by channel."""
        assert router.get_channel_router(AxiChannel.AW) is router.aw_router
        assert router.get_channel_router(AxiChannel.W) is router.w_router
        assert router.get_channel_router(AxiChannel.AR) is router.ar_router
        assert router.get_channel_router(AxiChannel.B) is router.b_router
        assert router.get_channel_router(AxiChannel.R) is router.r_router

    def test_get_channel_port(self, router):
        """Should be able to get port by channel and direction."""
        for channel in AxiChannel:
            for direction in [Direction.NORTH, Direction.EAST, Direction.SOUTH,
                              Direction.WEST, Direction.LOCAL]:
                port = router.get_channel_port(channel, direction)
                assert port is not None

    def test_channel_specific_accessors(self, router):
        """Should have convenience accessors for each channel."""
        assert router.get_aw_port(Direction.EAST) is not None
        assert router.get_w_port(Direction.EAST) is not None
        assert router.get_ar_port(Direction.EAST) is not None
        assert router.get_b_port(Direction.EAST) is not None
        assert router.get_r_port(Direction.EAST) is not None


class TestAXIModeEdgeRouter:
    """Test AXIModeEdgeRouter for Column 0."""

    @pytest.fixture
    def edge_router(self) -> AXIModeEdgeRouter:
        return AXIModeEdgeRouter((0, 1))

    def test_inherits_from_axi_mode_router(self, edge_router):
        """AXIModeEdgeRouter should inherit from AXIModeRouter."""
        assert isinstance(edge_router, AXIModeRouter)

    def test_selector_connected_initially_false(self, edge_router):
        """Initially not connected to selector."""
        assert edge_router.selector_connected is False


class TestCreateRouterFactory:
    """Test create_router factory function."""

    def test_general_mode_creates_router(self):
        """General mode should create standard Router."""
        from src.core.router import Router
        config = RouterConfig(channel_mode=ChannelMode.GENERAL)
        router = create_router((1, 1), is_edge=False, config=config)
        assert isinstance(router, Router)
        assert not isinstance(router, AXIModeRouter)

    def test_axi_mode_creates_axi_router(self):
        """AXI mode should create AXIModeRouter."""
        config = RouterConfig(channel_mode=ChannelMode.AXI)
        router = create_router((1, 1), is_edge=False, config=config)
        assert isinstance(router, AXIModeRouter)

    def test_axi_mode_edge_creates_axi_edge_router(self):
        """AXI mode edge should create AXIModeEdgeRouter."""
        config = RouterConfig(channel_mode=ChannelMode.AXI)
        router = create_router((0, 0), is_edge=True, config=config)
        assert isinstance(router, AXIModeEdgeRouter)


class TestAXIModeMesh:
    """Test Mesh creation in AXI Mode."""

    def test_create_mesh_general_mode(self):
        """General mode mesh should use standard routers."""
        from src.core.router import Router, EdgeRouter
        mesh = create_mesh(cols=3, rows=2, channel_mode=ChannelMode.GENERAL)
        assert mesh.config.channel_mode == ChannelMode.GENERAL

        # Check router types
        edge = mesh.get_edge_router(0)
        compute = mesh.routers[(1, 0)]
        assert isinstance(edge, EdgeRouter)
        assert isinstance(compute, Router)

    def test_create_mesh_axi_mode(self):
        """AXI mode mesh should use AXI mode routers."""
        mesh = create_mesh(cols=3, rows=2, channel_mode=ChannelMode.AXI)
        assert mesh.config.channel_mode == ChannelMode.AXI

        # Check router types
        edge = mesh.get_edge_router(0)
        compute = mesh.routers[(1, 0)]
        assert isinstance(edge, AXIModeEdgeRouter)
        assert isinstance(compute, AXIModeRouter)

    def test_axi_mode_wire_counts(self):
        """AXI mode should have 5x wires instead of 2x."""
        mesh = create_mesh(cols=3, rows=2, channel_mode=ChannelMode.AXI)

        # For 3x2 mesh: 7 connections (4 horizontal + 3 vertical)
        expected = 7
        assert len(mesh._aw_wires) == expected
        assert len(mesh._w_wires) == expected
        assert len(mesh._ar_wires) == expected
        assert len(mesh._b_wires) == expected
        assert len(mesh._r_wires) == expected

        # General mode wires should be empty
        assert len(mesh._req_wires) == 0
        assert len(mesh._resp_wires) == 0

    def test_general_mode_wire_counts(self):
        """General mode should have 2 wire lists."""
        mesh = create_mesh(cols=3, rows=2, channel_mode=ChannelMode.GENERAL)

        # For 3x2 mesh: 7 connections
        expected = 7
        assert len(mesh._req_wires) == expected
        assert len(mesh._resp_wires) == expected

        # AXI mode wires should be empty
        assert len(mesh._aw_wires) == 0


class TestMeshConfigChannelMode:
    """Test MeshConfig channel_mode field."""

    def test_default_is_general(self):
        """Default channel mode should be GENERAL."""
        config = MeshConfig()
        assert config.channel_mode == ChannelMode.GENERAL

    def test_propagates_to_router_config(self):
        """Channel mode should propagate to router_config."""
        config = MeshConfig(channel_mode=ChannelMode.AXI)
        assert config.router_config.channel_mode == ChannelMode.AXI
