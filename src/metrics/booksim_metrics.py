"""
BookSim2-style performance metrics.

Simplified for line-rate (full-speed) injection model.
No injection_rate configuration - system runs at maximum rate
as fast as backpressure allows.

Key metrics:
- throughput: bytes/cycle (measured, per-flit tracking)
- avg_latency: cycles (per-flit latency)
- zero_load_latency (L₀): theoretical minimum latency
- saturation_factor: L_avg / L₀
- saturation_status: NORMAL / WARNING / SATURATED

Note: "packet" in variable names refers to flits for BookSim2 compatibility.
Each flit is ~32 bytes (W channel data width).
"""

from dataclasses import dataclass
from enum import Enum


class SaturationStatus(Enum):
    """
    Network saturation status based on normalized latency (L/L₀).

    BookSim2 defines saturation point as L = 3×L₀.
    """

    NORMAL = "normal"  # L/L₀ < 2.0
    WARNING = "warning"  # 2.0 ≤ L/L₀ < 3.0
    SATURATED = "saturated"  # L/L₀ ≥ 3.0


@dataclass
class BookSimMetrics:
    """
    BookSim2-style performance metrics.

    Simplified for line-rate (full-speed) injection model.
    No injection_rate configuration - system runs at maximum rate.

    Attributes:
        throughput: Measured throughput in bytes/cycle
        total_packets: Total completed packets (flits)
        total_cycles: Total simulation cycles
        bytes_per_flit: Bytes per flit (default 32)
        avg_latency: Average packet latency in cycles
        min_latency: Minimum observed latency
        max_latency: Maximum observed latency
        zero_load_latency: Theoretical minimum latency (L₀)
        saturation_factor: Normalized latency (L_avg / L₀)
        saturation_status: Current saturation status
    """

    # Throughput (measured)
    throughput: float  # bytes/cycle
    total_packets: int  # flits
    total_cycles: int

    # Latency
    avg_latency: float  # cycles
    min_latency: float
    max_latency: float

    # Zero-Load Reference
    zero_load_latency: float  # L₀: theoretical minimum

    # Saturation Detection
    saturation_factor: float  # L_avg / L₀
    saturation_status: SaturationStatus

    # Flit size configuration (must be last - has default value)
    bytes_per_flit: int = 32  # Default flit payload size

    @property
    def is_saturated(self) -> bool:
        """Check if network is saturated (L > 3×L₀)."""
        return self.saturation_status == SaturationStatus.SATURATED

    @property
    def normalized_latency(self) -> float:
        """Get normalized latency (L_avg / L₀)."""
        return self.saturation_factor

    @property
    def total_bytes(self) -> int:
        """Total bytes transferred."""
        return self.total_packets * self.bytes_per_flit

    def __str__(self) -> str:
        """Pretty print metrics."""
        lines = [
            "=" * 70,
            "Performance Metrics (BookSim2-Style)",
            "=" * 70,
            f"Throughput:          {self.throughput:.2f} bytes/cycle",
            f"Total Bytes:         {self.total_bytes}",
            f"Total Flits:         {self.total_packets}",
            f"Total Cycles:        {self.total_cycles}",
            "-" * 70,
            f"Average Latency:     {self.avg_latency:.1f} cycles",
            f"Min Latency:         {self.min_latency:.1f} cycles",
            f"Max Latency:         {self.max_latency:.1f} cycles",
            "-" * 70,
            f"Zero-Load Latency:   {self.zero_load_latency:.1f} cycles (L0)",
            f"Normalized Latency:  {self.saturation_factor:.2f} (L/L0)",
            f"Saturation Status:   {self.saturation_status.value.upper()}",
        ]
        if self.is_saturated:
            lines.append("  WARNING: Network is SATURATED (L > 3*L0)")
        lines.append("=" * 70)
        return "\n".join(lines)


def calculate_saturation_status(factor: float) -> SaturationStatus:
    """
    Determine saturation status based on normalized latency factor.

    Args:
        factor: Normalized latency (L_avg / L₀)

    Returns:
        SaturationStatus enum value

    Thresholds (from BookSim2):
        - NORMAL: factor < 2.0
        - WARNING: 2.0 ≤ factor < 3.0
        - SATURATED: factor ≥ 3.0
    """
    if factor >= 3.0:
        return SaturationStatus.SATURATED
    elif factor >= 2.0:
        return SaturationStatus.WARNING
    else:
        return SaturationStatus.NORMAL
