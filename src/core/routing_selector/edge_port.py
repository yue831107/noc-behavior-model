"""
Edge Router Port classes for Routing Selector.

Provides connection interfaces between RoutingSelector and Edge Routers.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..flit import Flit
from ..router import RouterPort, Direction, PortWire

if TYPE_CHECKING:
    from ..router import EdgeRouter, AXIModeEdgeRouter


class EdgeRouterPort:
    """
    Connection to a single Edge Router.

    Manages both request (to mesh) and response (from mesh) channels.
    Uses valid/ready handshake via PortWire for communication.
    """

    def __init__(
        self,
        row: int,
        buffer_depth: int = 32
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

        # Signal-based interface using RouterPort
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

    def connect_edge_router(self, edge_router: "EdgeRouter") -> None:
        """Connect to an Edge Router."""
        self._edge_router = edge_router

    # =========================================================================
    # Request Path Signal Methods (Selector -> EdgeRouter)
    # =========================================================================

    def update_req_ready(self) -> None:
        """Update request port's ready signal (not used for output direction)."""
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


class AXIModeEdgeRouterPort:
    """
    Connection to a single AXI Mode Edge Router.

    Unlike EdgeRouterPort which has 2 channels (req/resp),
    this has 5 independent channels (AW, W, AR, B, R).

    Request channels (Selector -> EdgeRouter): AW, W, AR
    Response channels (EdgeRouter -> Selector): B, R
    """

    def __init__(
        self,
        row: int,
        buffer_depth: int = 32
    ):
        """
        Initialize AXI Mode edge router port.

        Args:
            row: Edge router row (0-3).
            buffer_depth: Buffer depth for each channel.
        """
        self.row = row
        self.coord = (0, row)
        self._buffer_depth = buffer_depth

        # Request channel ports (Selector -> EdgeRouter)
        self._aw_port = RouterPort(
            direction=Direction.LOCAL,
            buffer_depth=buffer_depth,
            name=f"Sel_aw_to_edge{row}"
        )
        self._w_port = RouterPort(
            direction=Direction.LOCAL,
            buffer_depth=buffer_depth,
            name=f"Sel_w_to_edge{row}"
        )
        self._ar_port = RouterPort(
            direction=Direction.LOCAL,
            buffer_depth=buffer_depth,
            name=f"Sel_ar_to_edge{row}"
        )

        # Response channel ports (EdgeRouter -> Selector)
        self._b_port = RouterPort(
            direction=Direction.LOCAL,
            buffer_depth=buffer_depth,
            name=f"Sel_b_from_edge{row}"
        )
        self._r_port = RouterPort(
            direction=Direction.LOCAL,
            buffer_depth=buffer_depth,
            name=f"Sel_r_from_edge{row}"
        )

        # Wire connections (set during connect_edge_routers)
        self._aw_wire: Optional[PortWire] = None
        self._w_wire: Optional[PortWire] = None
        self._ar_wire: Optional[PortWire] = None
        self._b_wire: Optional[PortWire] = None
        self._r_wire: Optional[PortWire] = None

        # Connected edge router
        self._edge_router: Optional["AXIModeEdgeRouter"] = None

    def connect_edge_router(self, edge_router: "AXIModeEdgeRouter") -> None:
        """Connect to an AXI Mode Edge Router."""
        self._edge_router = edge_router

    # =========================================================================
    # Request Path Methods (Selector -> EdgeRouter)
    # =========================================================================

    def can_send_aw(self) -> bool:
        """Check if AW channel can send."""
        return self._aw_port.can_send()

    def can_send_w(self) -> bool:
        """Check if W channel can send."""
        return self._w_port.can_send()

    def can_send_ar(self) -> bool:
        """Check if AR channel can send."""
        return self._ar_port.can_send()

    def set_aw_output(self, flit: Flit) -> bool:
        """Set AW output."""
        return self._aw_port.set_output(flit)

    def set_w_output(self, flit: Flit) -> bool:
        """Set W output."""
        return self._w_port.set_output(flit)

    def set_ar_output(self, flit: Flit) -> bool:
        """Set AR output."""
        return self._ar_port.set_output(flit)

    def clear_aw_if_accepted(self) -> bool:
        """Clear AW output if accepted."""
        return self._aw_port.clear_output_if_accepted()

    def clear_w_if_accepted(self) -> bool:
        """Clear W output if accepted."""
        return self._w_port.clear_output_if_accepted()

    def clear_ar_if_accepted(self) -> bool:
        """Clear AR output if accepted."""
        return self._ar_port.clear_output_if_accepted()

    # =========================================================================
    # Response Path Methods (EdgeRouter -> Selector)
    # =========================================================================

    def update_b_ready(self) -> None:
        """Update B channel ready signal."""
        self._b_port.update_ready()

    def update_r_ready(self) -> None:
        """Update R channel ready signal."""
        self._r_port.update_ready()

    def sample_b_input(self) -> bool:
        """Sample B input from wire."""
        return self._b_port.sample_input()

    def sample_r_input(self) -> bool:
        """Sample R input from wire."""
        return self._r_port.sample_input()

    def clear_b_input_signals(self) -> None:
        """Clear B input signals."""
        self._b_port.clear_input_signals()

    def clear_r_input_signals(self) -> None:
        """Clear R input signals."""
        self._r_port.clear_input_signals()

    def get_b_response(self) -> Optional[Flit]:
        """Get B response flit."""
        return self._b_port.pop_for_routing()

    def get_r_response(self) -> Optional[Flit]:
        """Get R response flit."""
        return self._r_port.pop_for_routing()

    # =========================================================================
    # Credit Management
    # =========================================================================

    @property
    def aw_credits(self) -> int:
        """Available AW credits."""
        return self._aw_port._output_credit.credits if self._aw_port._output_credit else 0

    @property
    def w_credits(self) -> int:
        """Available W credits."""
        return self._w_port._output_credit.credits if self._w_port._output_credit else 0

    @property
    def ar_credits(self) -> int:
        """Available AR credits."""
        return self._ar_port._output_credit.credits if self._ar_port._output_credit else 0

    @property
    def b_occupancy(self) -> int:
        """B buffer occupancy."""
        return self._b_port._buffer.occupancy

    @property
    def r_occupancy(self) -> int:
        """R buffer occupancy."""
        return self._r_port._buffer.occupancy

    def __repr__(self) -> str:
        return (
            f"AXIModeEdgeRouterPort(row={self.row}, "
            f"aw_cr={self.aw_credits}, w_cr={self.w_credits}, ar_cr={self.ar_credits})"
        )
