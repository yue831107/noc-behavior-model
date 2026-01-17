"""
BookSim2-style metrics module (simplified for line-rate injection).

This module provides performance metrics compatible with BookSim2 terminology
while maintaining the full-speed (line-rate) injection behavior of our
hardware-accurate model.

Key differences from BookSim2:
- No configurable injection_rate (we inject at line-rate)
- throughput is measured, not derived from injection_rate
- Saturation detection (L > 3×L₀) is preserved
"""

from .booksim_metrics import (
    BookSimMetrics,
    SaturationStatus,
    calculate_saturation_status,
)

__all__ = [
    "BookSimMetrics",
    "SaturationStatus",
    "calculate_saturation_status",
]
