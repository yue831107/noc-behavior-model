"""
Routing Selector configuration classes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

from ..router import ChannelMode


@dataclass
class RoutingSelectorConfig:
    """Routing Selector configuration."""
    num_directions: int = 4         # Number of connected Edge Routers
    ingress_buffer_depth: int = 32  # Ingress buffer depth
    egress_buffer_depth: int = 32   # Egress buffer depth
    hop_weight: float = 1.0         # Hop count weight for path selection
    credit_weight: float = 1.0      # Credit weight for path selection
    channel_mode: ChannelMode = ChannelMode.GENERAL  # Physical channel mode


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
