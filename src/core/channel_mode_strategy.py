"""
Channel Mode Strategy pattern for abstracting General vs AXI mode operations.

This module implements the Strategy design pattern to handle the differences
between General Mode and AXI Mode in the NoC router architecture.

Channel Mode Architecture:
--------------------------

General Mode (2 Sub-Routers):
    - Request Sub-Router: Handles AW, W, AR channels (shared resources)
    - Response Sub-Router: Handles B, R channels (shared resources)
    - Simpler design but may suffer from Head-of-Line (HoL) blocking
    - All request channels compete for the same buffer/routing resources

AXI Mode (5 Sub-Routers):
    - Each AXI channel (AW, W, AR, B, R) has dedicated sub-router
    - Eliminates HoL blocking between different transaction types
    - Higher area cost but better performance for mixed traffic
    - Each channel has independent buffer and routing resources

Usage:
------
    strategy = get_channel_mode_strategy(ChannelMode.AXI)
    for ch in strategy.request_channels:
        buffer = get_buffer_for_channel(ch)

Benefits:
---------
- Eliminates 15+ `if channel_mode == AXI:` conditionals across codebase
- Single point of configuration for channel mode behavior
- Easy to extend with new modes if needed
- Improves testability by isolating mode-specific logic
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, TYPE_CHECKING

from .flit import AxiChannel

if TYPE_CHECKING:
    from .router import PortWire


class ChannelModeStrategy(ABC):
    """
    Abstract base class defining the interface for channel mode strategies.

    This strategy pattern allows Router and NI components to operate
    polymorphically regardless of whether they're in General or AXI mode.

    Responsibilities:
    -----------------
    1. Channel Classification: Define which channels are request vs response
    2. Buffer Management: Determine if per-channel or shared buffers are used
    3. Sub-Router Count: Report how many physical sub-routers exist

    Implementers:
    -------------
    - GeneralModeStrategy: 2 sub-routers (Req + Resp)
    - AXIModeStrategy: 5 sub-routers (AW, W, AR, B, R)

    Thread Safety:
    --------------
    Strategy instances are stateless and can be safely shared across
    multiple Router/NI instances (singleton pattern used).
    """

    @property
    @abstractmethod
    def request_channels(self) -> List[AxiChannel]:
        """Get request direction channel list."""
        pass

    @property
    @abstractmethod
    def response_channels(self) -> List[AxiChannel]:
        """Get response direction channel list."""
        pass

    @property
    @abstractmethod
    def all_channels(self) -> List[AxiChannel]:
        """Get all channels in order."""
        pass

    @property
    @abstractmethod
    def channel_count(self) -> int:
        """Number of physical channels (sub-routers)."""
        pass

    @property
    @abstractmethod
    def uses_per_channel_buffers(self) -> bool:
        """Whether this mode uses per-channel buffers (True for AXI, False for General)."""
        pass

    @abstractmethod
    def get_buffer_channels_for_request(self) -> List[AxiChannel]:
        """Get channels that need separate buffers for request direction."""
        pass

    @abstractmethod
    def get_buffer_channels_for_response(self) -> List[AxiChannel]:
        """Get channels that need separate buffers for response direction."""
        pass


class GeneralModeStrategy(ChannelModeStrategy):
    """
    General Mode: 2 logical sub-routers (Request + Response).

    Architecture:
    -------------
    ┌─────────────────────────────────────┐
    │           General Mode              │
    │  ┌─────────────┐  ┌─────────────┐   │
    │  │   Request   │  │  Response   │   │
    │  │ Sub-Router  │  │ Sub-Router  │   │
    │  │ (AW,W,AR)   │  │   (B,R)     │   │
    │  └─────────────┘  └─────────────┘   │
    └─────────────────────────────────────┘

    Characteristics:
    ----------------
    - Request channels (AW, W, AR) share one sub-router's resources
    - Response channels (B, R) share another sub-router's resources
    - Uses shared buffers (one buffer per direction)
    - Simpler implementation, lower area cost
    - May experience HoL blocking when different request types compete

    Use Case:
    ---------
    Best for systems with homogeneous traffic patterns or when area
    is constrained. Not recommended for latency-sensitive mixed workloads.
    """

    @property
    def request_channels(self) -> List[AxiChannel]:
        return [AxiChannel.AW, AxiChannel.W, AxiChannel.AR]

    @property
    def response_channels(self) -> List[AxiChannel]:
        return [AxiChannel.B, AxiChannel.R]

    @property
    def all_channels(self) -> List[AxiChannel]:
        return [AxiChannel.AW, AxiChannel.W, AxiChannel.AR, AxiChannel.B, AxiChannel.R]

    @property
    def channel_count(self) -> int:
        return 2  # Req + Resp

    @property
    def uses_per_channel_buffers(self) -> bool:
        return False

    def get_buffer_channels_for_request(self) -> List[AxiChannel]:
        return []  # General mode uses single shared buffer

    def get_buffer_channels_for_response(self) -> List[AxiChannel]:
        return []  # General mode uses single shared buffer


class AXIModeStrategy(ChannelModeStrategy):
    """
    AXI Mode: 5 independent sub-routers (AW, W, AR, B, R).

    Architecture:
    -------------
    ┌───────────────────────────────────────────────────────┐
    │                     AXI Mode                          │
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐             │
    │  │ AW  │ │  W  │ │ AR  │ │  B  │ │  R  │             │
    │  │Sub- │ │Sub- │ │Sub- │ │Sub- │ │Sub- │             │
    │  │Rtr  │ │Rtr  │ │Rtr  │ │Rtr  │ │Rtr  │             │
    │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘             │
    └───────────────────────────────────────────────────────┘

    Characteristics:
    ----------------
    - Each AXI channel has dedicated sub-router and buffers
    - Eliminates Head-of-Line (HoL) blocking between channels
    - Write address (AW) won't block read address (AR)
    - Higher area cost (5x routing resources)
    - Better performance for mixed read/write workloads

    Use Case:
    ---------
    Recommended for latency-sensitive applications with mixed traffic.
    Essential for systems where write and read performance must be
    independent (e.g., streaming + random access patterns).

    Performance Benefit:
    --------------------
    In General Mode, a large write burst blocks subsequent reads.
    In AXI Mode, reads can proceed on AR/R channels while writes
    use AW/W channels, improving overall system throughput.
    """

    @property
    def request_channels(self) -> List[AxiChannel]:
        return [AxiChannel.AW, AxiChannel.W, AxiChannel.AR]

    @property
    def response_channels(self) -> List[AxiChannel]:
        return [AxiChannel.B, AxiChannel.R]

    @property
    def all_channels(self) -> List[AxiChannel]:
        return [AxiChannel.AW, AxiChannel.W, AxiChannel.AR, AxiChannel.B, AxiChannel.R]

    @property
    def channel_count(self) -> int:
        return 5  # AW, W, AR, B, R

    @property
    def uses_per_channel_buffers(self) -> bool:
        return True

    def get_buffer_channels_for_request(self) -> List[AxiChannel]:
        return [AxiChannel.AW, AxiChannel.W, AxiChannel.AR]

    def get_buffer_channels_for_response(self) -> List[AxiChannel]:
        return [AxiChannel.B, AxiChannel.R]


# Singleton instances for reuse
GENERAL_MODE_STRATEGY = GeneralModeStrategy()
AXI_MODE_STRATEGY = AXIModeStrategy()


def get_channel_mode_strategy(channel_mode) -> ChannelModeStrategy:
    """
    Get strategy instance for the given channel mode.

    Args:
        channel_mode: ChannelMode enum value.

    Returns:
        Appropriate strategy instance.
    """
    from .router import ChannelMode

    if channel_mode == ChannelMode.AXI:
        return AXI_MODE_STRATEGY
    return GENERAL_MODE_STRATEGY
