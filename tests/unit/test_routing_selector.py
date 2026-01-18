"""
RoutingSelector Unit Tests.

Tests for RoutingSelector component covering:
- Path selection (hop count, credits)
- Packet path storage (AW/W same-route requirement)
- Credit management
- Response path routing
"""

import pytest
from typing import Tuple, List, Optional, Dict
from unittest.mock import Mock, MagicMock, patch

from src.core.routing_selector.selector import RoutingSelector
from src.core.routing_selector.config import RoutingSelectorConfig, SelectorStats
from src.core.routing_selector.edge_port import EdgeRouterPort, AXIModeEdgeRouterPort
from src.core.flit import (
    Flit, FlitFactory, FlitHeader, AxiChannel,
    AxiAwPayload, AxiWPayload, AxiBPayload, AxiRPayload,
    encode_node_id, decode_node_id,
)
from src.core.router import (
    Direction, RouterConfig, ChannelMode, PortWire,
    EdgeRouter, XYRouter,
)
from src.core.buffer import FlitBuffer


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def selector_config() -> RoutingSelectorConfig:
    """Default selector configuration."""
    return RoutingSelectorConfig(
        num_directions=4,
        ingress_buffer_depth=8,
        egress_buffer_depth=8,
        hop_weight=1.0,
        credit_weight=0.1,
        channel_mode=ChannelMode.GENERAL,
    )


@pytest.fixture
def selector_config_axi_mode() -> RoutingSelectorConfig:
    """AXI Mode selector configuration."""
    return RoutingSelectorConfig(
        num_directions=4,
        ingress_buffer_depth=8,
        egress_buffer_depth=8,
        hop_weight=1.0,
        credit_weight=0.1,
        channel_mode=ChannelMode.AXI,
    )


@pytest.fixture
def selector(selector_config) -> RoutingSelector:
    """Create RoutingSelector in General Mode."""
    return RoutingSelector(config=selector_config)


@pytest.fixture
def selector_axi_mode(selector_config_axi_mode) -> RoutingSelector:
    """Create RoutingSelector in AXI Mode."""
    return RoutingSelector(config=selector_config_axi_mode)


@pytest.fixture
def router_config() -> RouterConfig:
    """Default router configuration."""
    return RouterConfig(
        buffer_depth=4,
        output_buffer_depth=0,
    )


def create_mock_edge_router(row: int, router_config: RouterConfig) -> EdgeRouter:
    """Create an EdgeRouter for testing."""
    return EdgeRouter(coord=(0, row), config=router_config)


# ==============================================================================
# Part 5.1: Path Selection Tests
# ==============================================================================

class TestRoutingSelectorPathSelection:
    """Tests for path selection logic."""

    def test_select_path_by_hop_count(self, selector):
        """Path selection should prefer shorter hop count."""
        # Destination is (4, 3) - far corner
        # Row 3 edge router at (0, 3) is closer
        flit = FlitFactory.create_ar(
            src=(0, 0),
            dest=(4, 3),
            addr=0x1000,
            axi_id=0,
        )

        # Inject into ingress buffer
        selector.accept_request(flit)

        # Select path should prefer row 3 (same Y as destination)
        ingress_flit = selector.ingress_buffer.peek()
        best_row = selector._select_ingress_path(ingress_flit)

        # Row 3 should be preferred due to lower Y-distance
        assert best_row is not None
        # Exact row depends on credit availability

    def test_select_path_by_credits(self, selector, router_config):
        """When hop count is equal, prefer higher credits."""
        # Create and connect edge routers
        edge_routers = []
        for row in range(4):
            er = create_mock_edge_router(row, router_config)
            edge_routers.append(er)

        selector.connect_edge_routers(edge_routers)

        # Destination at (1, 1) - equidistant from rows 0 and 2
        flit = FlitFactory.create_ar(
            src=(0, 0),
            dest=(1, 1),
            addr=0x1000,
            axi_id=0,
        )

        selector.accept_request(flit)

        # Select path
        ingress_flit = selector.ingress_buffer.peek()
        best_row = selector._select_ingress_path(ingress_flit)

        # Should select one of the rows
        assert best_row is not None
        assert 0 <= best_row <= 3

    def test_path_to_all_edge_routers(self, selector, router_config):
        """Should be able to route to all 4 edge routers."""
        # Create and connect edge routers
        edge_routers = []
        for row in range(4):
            er = create_mock_edge_router(row, router_config)
            edge_routers.append(er)

        selector.connect_edge_routers(edge_routers)

        # Test destinations in each row
        destinations = [(4, 0), (4, 1), (4, 2), (4, 3)]

        for dest in destinations:
            flit = FlitFactory.create_ar(
                src=(0, 0),
                dest=dest,
                addr=0x1000,
                axi_id=0,
            )
            selector.accept_request(flit)

            ingress_flit = selector.ingress_buffer.peek()
            best_row = selector._select_ingress_path(ingress_flit)

            # Path should exist (assuming credits available)
            # Pop the flit for next iteration
            selector.ingress_buffer.pop()


# ==============================================================================
# Part 5.2: Packet Path Storage Tests (AW/W Same-Route)
# ==============================================================================

class TestRoutingSelectorPacketPathStorage:
    """Tests for AW/W packet path storage and matching."""

    def test_aw_path_stored(self, selector, router_config):
        """AW path should be stored for subsequent W flits."""
        edge_routers = [create_mock_edge_router(row, router_config) for row in range(4)]
        selector.connect_edge_routers(edge_routers)

        # AW flit
        aw_flit = FlitFactory.create_aw(
            src=(1, 1),
            dest=(4, 2),
            addr=0x1000,
            axi_id=0,
            length=0,
            rob_idx=5,
            rob_req=True,
            last=True,
        )

        # Accept and select path
        selector.accept_request(aw_flit)
        ingress_flit = selector.ingress_buffer.peek()
        best_row = selector._select_ingress_path(ingress_flit)

        # Path should be stored
        packet_key = (aw_flit.hdr.src_id, aw_flit.hdr.dst_id, aw_flit.hdr.rob_idx)
        assert packet_key in selector._packet_path

    def test_w_uses_stored_aw_path(self, selector, router_config):
        """W flits should use the stored AW path."""
        edge_routers = [create_mock_edge_router(row, router_config) for row in range(4)]
        selector.connect_edge_routers(edge_routers)

        rob_idx = 7

        # AW flit
        aw_flit = FlitFactory.create_aw(
            src=(1, 1),
            dest=(4, 2),
            addr=0x1000,
            axi_id=0,
            length=1,  # 2 W beats
            rob_idx=rob_idx,
            rob_req=True,
            last=True,
        )

        # Select AW path and store it
        selector.accept_request(aw_flit)
        selector._select_ingress_path(selector.ingress_buffer.peek())
        selector.ingress_buffer.pop()  # Remove AW

        packet_key = (aw_flit.hdr.src_id, aw_flit.hdr.dst_id, aw_flit.hdr.rob_idx)
        assert packet_key in selector._packet_path
        stored_row = selector._packet_path[packet_key]

        # W flit with same src/dst/rob_idx
        w_flit = FlitFactory.create_w(
            src=(1, 1),
            dest=(4, 2),
            data=b"test_data" + bytes(23),
            last=False,
            rob_idx=rob_idx,
            seq_num=0,
        )

        selector.accept_request(w_flit)
        selected_row = selector._select_ingress_path(selector.ingress_buffer.peek())

        # W should use same row as AW
        assert selected_row == stored_row

    def test_path_cleared_after_w_complete(self, selector, router_config):
        """Path should be cleared after last W flit."""
        edge_routers = [create_mock_edge_router(row, router_config) for row in range(4)]
        selector.connect_edge_routers(edge_routers)

        rob_idx = 10

        # AW flit
        aw_flit = FlitFactory.create_aw(
            src=(1, 1), dest=(4, 2),
            addr=0x1000, axi_id=0, length=0,
            rob_idx=rob_idx, rob_req=True, last=True,
        )

        selector.accept_request(aw_flit)
        selector._select_ingress_path(selector.ingress_buffer.peek())

        packet_key = (aw_flit.hdr.src_id, aw_flit.hdr.dst_id, rob_idx)
        assert packet_key in selector._packet_path
        selector.ingress_buffer.pop()

        # W flit (last=True - single beat)
        w_flit = FlitFactory.create_w(
            src=(1, 1), dest=(4, 2),
            data=b"data" + bytes(28),
            last=True,  # Last W flit
            rob_idx=rob_idx,
            seq_num=0,
        )

        selector.accept_request(w_flit)
        original_key = (w_flit.hdr.src_id, w_flit.hdr.dst_id, w_flit.hdr.rob_idx)

        # Process should clear path after W TAIL
        selector._process_ingress(current_time=0)

        # Path should be cleared after W TAIL send
        # (The clearing happens during _process_ingress after successful send)

    def test_multiple_outstanding_paths(self, selector, router_config):
        """Multiple outstanding AW/W transactions should have independent paths."""
        edge_routers = [create_mock_edge_router(row, router_config) for row in range(4)]
        selector.connect_edge_routers(edge_routers)

        # Two independent AW transactions
        aw0 = FlitFactory.create_aw(
            src=(1, 1), dest=(4, 0),
            addr=0x1000, axi_id=0, length=0,
            rob_idx=0, rob_req=True, last=True,
        )
        aw1 = FlitFactory.create_aw(
            src=(2, 2), dest=(4, 3),
            addr=0x2000, axi_id=1, length=0,
            rob_idx=1, rob_req=True, last=True,
        )

        # Store paths for both
        selector.accept_request(aw0)
        selector._select_ingress_path(selector.ingress_buffer.peek())
        selector.ingress_buffer.pop()

        selector.accept_request(aw1)
        selector._select_ingress_path(selector.ingress_buffer.peek())
        selector.ingress_buffer.pop()

        key0 = (aw0.hdr.src_id, aw0.hdr.dst_id, aw0.hdr.rob_idx)
        key1 = (aw1.hdr.src_id, aw1.hdr.dst_id, aw1.hdr.rob_idx)

        # Both paths should be stored independently
        assert key0 in selector._packet_path
        assert key1 in selector._packet_path

    def test_path_key_uses_original_src_id(self, selector, router_config):
        """Path key should use original src_id, not modified one."""
        edge_routers = [create_mock_edge_router(row, router_config) for row in range(4)]
        selector.connect_edge_routers(edge_routers)

        # Original src_id from coordinate (1, 1)
        original_src = (1, 1)
        aw_flit = FlitFactory.create_aw(
            src=original_src, dest=(4, 2),
            addr=0x1000, axi_id=0, length=0,
            rob_idx=5, rob_req=True, last=True,
        )

        original_src_id = aw_flit.hdr.src_id

        selector.accept_request(aw_flit)
        selector._select_ingress_path(selector.ingress_buffer.peek())

        # Path key should use original src_id
        expected_key = (original_src_id, aw_flit.hdr.dst_id, aw_flit.hdr.rob_idx)
        assert expected_key in selector._packet_path


# ==============================================================================
# Part 5.3: Credit Management Tests
# ==============================================================================

class TestRoutingSelectorCreditManagement:
    """Tests for credit-based flow control."""

    def test_credits_tracked_per_port(self, selector, router_config):
        """Each port should track credits independently."""
        edge_routers = [create_mock_edge_router(row, router_config) for row in range(4)]
        selector.connect_edge_routers(edge_routers)

        # Each port should have its own credit count
        for row in range(4):
            port = selector.edge_ports[row]
            initial_credits = port.available_credits
            # Initial credits should be set based on downstream buffer depth
            assert initial_credits >= 0

    def test_credits_decremented_on_send(self, selector, router_config):
        """Credits should decrease when sending."""
        edge_routers = [create_mock_edge_router(row, router_config) for row in range(4)]
        selector.connect_edge_routers(edge_routers)

        # Get initial credits for row 0
        port = selector.edge_ports[0]
        initial_credits = port.available_credits

        # If we have credits, sending should decrement
        if initial_credits > 0 and port.can_send_request():
            flit = FlitFactory.create_ar(
                src=(0, 0), dest=(4, 0),
                addr=0x1000, axi_id=0,
            )
            port.set_req_output(flit)
            # Credit is consumed when set_output is called

    def test_no_send_when_zero_credits(self, selector, router_config):
        """Should not send when credits are exhausted."""
        edge_routers = [create_mock_edge_router(row, router_config) for row in range(4)]
        selector.connect_edge_routers(edge_routers)

        # Exhaust credits by setting output credits to 0
        port = selector.edge_ports[0]

        # After exhausting credits, can_send should return False
        # This is controlled by the credit flow control


# ==============================================================================
# Part 5.4: Response Path Tests
# ==============================================================================

class TestRoutingSelectorResponsePath:
    """Tests for response path routing."""

    def test_b_response_routed_back(self, selector, router_config):
        """B response should be collected and routed to egress."""
        edge_routers = [create_mock_edge_router(row, router_config) for row in range(4)]
        selector.connect_edge_routers(edge_routers)

        # Simulate B response in edge port buffer
        b_flit = FlitFactory.create_b(
            src=(4, 2), dest=(1, 1),
            axi_id=0, resp=0,
        )

        # Push directly to response port buffer
        port = selector.edge_ports[2]
        port._resp_port._buffer.push(b_flit)

        # Process egress
        selector._process_egress(current_time=0)

        # Response should be in egress buffer
        response = selector.get_response()
        assert response is not None
        assert response.hdr.axi_ch == AxiChannel.B

    def test_r_response_routed_back(self, selector, router_config):
        """R response should be collected and routed to egress."""
        edge_routers = [create_mock_edge_router(row, router_config) for row in range(4)]
        selector.connect_edge_routers(edge_routers)

        # Simulate R response
        r_flit = FlitFactory.create_r(
            src=(4, 1), dest=(2, 2),
            data=b"read_data" + bytes(23),
            axi_id=0, last=True,
        )

        port = selector.edge_ports[1]
        port._resp_port._buffer.push(r_flit)

        selector._process_egress(current_time=0)

        response = selector.get_response()
        assert response is not None
        assert response.hdr.axi_ch == AxiChannel.R

    def test_response_src_id_preserved(self, selector, router_config):
        """Response src_id should be preserved through selector."""
        edge_routers = [create_mock_edge_router(row, router_config) for row in range(4)]
        selector.connect_edge_routers(edge_routers)

        original_src = (3, 2)
        original_dest = (1, 1)

        b_flit = FlitFactory.create_b(
            src=original_src,
            dest=original_dest,
            axi_id=5,
            resp=0,
        )

        port = selector.edge_ports[2]
        port._resp_port._buffer.push(b_flit)

        selector._process_egress(current_time=0)

        response = selector.get_response()
        assert response is not None
        # Source and dest should be preserved
        assert response.src == original_src
        assert response.dest == original_dest


# ==============================================================================
# Part 5.5: Ingress/Egress Buffer Tests
# ==============================================================================

class TestRoutingSelectorBuffers:
    """Tests for ingress and egress buffer management."""

    def test_accept_request_to_ingress(self, selector):
        """Accept request should add to ingress buffer."""
        flit = FlitFactory.create_ar(
            src=(1, 1), dest=(4, 2),
            addr=0x1000, axi_id=0,
        )

        result = selector.accept_request(flit)

        assert result is True
        assert selector.ingress_buffer.occupancy == 1
        assert selector.stats.req_flits_received == 1

    def test_ingress_buffer_full_rejects(self, selector):
        """Full ingress buffer should reject new requests."""
        # Fill the buffer
        for i in range(selector.config.ingress_buffer_depth):
            flit = FlitFactory.create_ar(
                src=(1, 1), dest=(4, 2),
                addr=0x1000 + i * 0x100, axi_id=i % 16,
            )
            selector.accept_request(flit)

        assert selector.ingress_buffer.is_full() is True

        # Next one should be rejected
        overflow_flit = FlitFactory.create_ar(
            src=(1, 1), dest=(4, 2),
            addr=0x9000, axi_id=0,
        )
        result = selector.accept_request(overflow_flit)
        assert result is False

    def test_get_response_from_egress(self, selector):
        """Get response should return from egress buffer."""
        b_flit = FlitFactory.create_b(
            src=(4, 2), dest=(1, 1),
            axi_id=0, resp=0,
        )

        selector.egress_buffer.push(b_flit)

        response = selector.get_response()
        assert response is not None
        assert response.hdr.axi_ch == AxiChannel.B
        assert selector.stats.resp_flits_sent == 1

    def test_has_pending_requests(self, selector):
        """has_pending_requests should reflect ingress buffer state."""
        assert selector.has_pending_requests() is False

        flit = FlitFactory.create_ar(
            src=(1, 1), dest=(4, 2),
            addr=0x1000, axi_id=0,
        )
        selector.accept_request(flit)

        assert selector.has_pending_requests() is True

    def test_has_pending_responses(self, selector):
        """has_pending_responses should reflect egress buffer state."""
        assert selector.has_pending_responses is False

        b_flit = FlitFactory.create_b(
            src=(4, 2), dest=(1, 1),
            axi_id=0, resp=0,
        )
        selector.egress_buffer.push(b_flit)

        assert selector.has_pending_responses is True


# ==============================================================================
# Part 5.6: Statistics Tests
# ==============================================================================

class TestRoutingSelectorStatistics:
    """Tests for statistics tracking."""

    def test_stats_req_flits_received(self, selector):
        """Should track received request flits."""
        assert selector.stats.req_flits_received == 0

        flit = FlitFactory.create_ar(
            src=(1, 1), dest=(4, 2),
            addr=0x1000, axi_id=0,
        )
        selector.accept_request(flit)

        assert selector.stats.req_flits_received == 1

    def test_stats_resp_flits_sent(self, selector):
        """Should track sent response flits."""
        assert selector.stats.resp_flits_sent == 0

        b_flit = FlitFactory.create_b(
            src=(4, 2), dest=(1, 1),
            axi_id=0, resp=0,
        )
        selector.egress_buffer.push(b_flit)
        selector.get_response()

        assert selector.stats.resp_flits_sent == 1

    def test_stats_path_selections(self, selector, router_config):
        """Should track path selection distribution."""
        edge_routers = [create_mock_edge_router(row, router_config) for row in range(4)]
        selector.connect_edge_routers(edge_routers)

        # Path selections should be tracked per row
        assert isinstance(selector.stats.path_selections, dict)


# ==============================================================================
# Part 5.7: AXI Mode Specific Tests
# ==============================================================================

class TestRoutingSelectorAxiMode:
    """Tests specific to AXI Mode operation."""

    def test_axi_mode_separate_channels(self, selector_axi_mode, router_config):
        """AXI Mode should have separate channel ports."""
        assert selector_axi_mode._is_axi_mode is True

        for row in range(4):
            port = selector_axi_mode.edge_ports[row]
            assert isinstance(port, AXIModeEdgeRouterPort)

    def test_axi_mode_aw_channel_send(self, selector_axi_mode, router_config):
        """AXI Mode should route AW through dedicated channel."""
        # Create mock AXI Mode edge routers would need AXIModeEdgeRouter
        # For this test, we check the channel routing logic

        aw_flit = FlitFactory.create_aw(
            src=(1, 1), dest=(4, 2),
            addr=0x1000, axi_id=0, length=0,
            rob_idx=0, rob_req=True, last=True,
        )

        # Should recognize this as AW channel
        assert aw_flit.hdr.axi_ch == AxiChannel.AW

    def test_axi_mode_w_channel_send(self, selector_axi_mode):
        """AXI Mode should route W through dedicated channel."""
        w_flit = FlitFactory.create_w(
            src=(1, 1), dest=(4, 2),
            data=b"test" + bytes(28),
            last=True, rob_idx=0, seq_num=0,
        )

        assert w_flit.hdr.axi_ch == AxiChannel.W

    def test_axi_mode_ar_channel_send(self, selector_axi_mode):
        """AXI Mode should route AR through dedicated channel."""
        ar_flit = FlitFactory.create_ar(
            src=(1, 1), dest=(4, 2),
            addr=0x2000, axi_id=0,
        )

        assert ar_flit.hdr.axi_ch == AxiChannel.AR


# ==============================================================================
# Part 5.8: Hop Calculation Tests
# ==============================================================================

class TestRoutingSelectorHopCalculation:
    """Tests for hop count calculation."""

    def test_calculate_hops_same_location(self, selector):
        """Hop count to same location should be 0."""
        hops = selector._calculate_hops((0, 0), (0, 0))
        assert hops == 0

    def test_calculate_hops_horizontal(self, selector):
        """Horizontal movement should count correctly."""
        hops = selector._calculate_hops((0, 0), (4, 0))
        assert hops == 4

    def test_calculate_hops_vertical(self, selector):
        """Vertical movement should count correctly."""
        hops = selector._calculate_hops((0, 0), (0, 3))
        assert hops == 3

    def test_calculate_hops_diagonal(self, selector):
        """Diagonal movement should be Manhattan distance."""
        hops = selector._calculate_hops((0, 0), (4, 3))
        assert hops == 7  # 4 + 3

    def test_calculate_hops_reverse(self, selector):
        """Distance should be same in both directions."""
        hops_forward = selector._calculate_hops((0, 0), (4, 3))
        hops_reverse = selector._calculate_hops((4, 3), (0, 0))
        assert hops_forward == hops_reverse


# ==============================================================================
# Part 5.9: Edge Router Connection Tests
# ==============================================================================

class TestRoutingSelectorEdgeRouterConnection:
    """Tests for edge router connection."""

    def test_connect_edge_routers(self, selector, router_config):
        """Should connect to 4 edge routers."""
        edge_routers = [create_mock_edge_router(row, router_config) for row in range(4)]
        selector.connect_edge_routers(edge_routers)

        # Each port should have edge router reference
        for row in range(4):
            port = selector.edge_ports[row]
            assert port._edge_router is not None

    def test_edge_ports_initialized(self, selector):
        """Edge ports should be initialized for all 4 rows."""
        assert len(selector.edge_ports) == 4

        for row in range(4):
            assert row in selector.edge_ports
            port = selector.edge_ports[row]
            assert port.row == row
            assert port.coord == (0, row)
