"""
Theory-based Performance Validator (BookSim2-Style).

Validates performance metrics using BookSim2 concepts:
- Zero-load latency (L₀): theoretical minimum latency
- Saturation detection: L > 3×L₀
- Buffer utilization bounds

This validator uses a monitor-based approach (no core modification required).
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from src.metrics.booksim_metrics import SaturationStatus, calculate_saturation_status


@dataclass
class MeshConfig:
    """Mesh configuration for theoretical calculations."""

    cols: int = 5
    rows: int = 4
    edge_column: int = 0

    @property
    def compute_nodes(self) -> int:
        """Number of compute nodes (excluding edge column)."""
        return (self.cols - 1) * self.rows

    @property
    def total_routers(self) -> int:
        """Total number of routers including edge routers."""
        return self.cols * self.rows


@dataclass
class RouterConfig:
    """Router configuration for theoretical calculations."""

    flit_data_bytes: int = 32  # 256 bits = 32 bytes (FlooNoC W/R payload)
    pipeline_depth: int = 1  # Fast mode: 1-cycle
    buffer_depth: int = 4
    switching: str = "wormhole"


class TheoryValidator:
    """
    Validates performance metrics against theoretical bounds (BookSim2-Style).

    Monitor-based validator: reads metrics from system without
    modifying core implementation.

    Key validations:
    - Latency lower bound: L ≥ L₀ (zero-load latency)
    - Saturation detection: L > 3×L₀
    - Buffer utilization: 0 ≤ U ≤ 1
    """

    def __init__(
        self,
        mesh_config: Optional[MeshConfig] = None,
        router_config: Optional[RouterConfig] = None,
    ):
        """
        Initialize validator with system configuration.

        Args:
            mesh_config: Mesh topology configuration
            router_config: Router hardware configuration
        """
        self.mesh_config = mesh_config or MeshConfig()
        self.router_config = router_config or RouterConfig()

    # =========================================================================
    # Theoretical Calculations
    # =========================================================================

    def calculate_zero_load_latency(
        self,
        src: Tuple[int, int],
        dest: Tuple[int, int],
        include_overhead: bool = True,
    ) -> float:
        """
        Calculate zero-load latency (L₀) - BookSim2 terminology.

        Zero-load latency is the theoretical minimum latency when there is
        no contention in the network. This is equivalent to the old
        calculate_min_latency() but uses BookSim2 naming.

        Formula: L₀ = hops × pipeline_depth + overhead

        Overhead (when include_overhead=True):
        - NI packetization: ~1 cycle
        - Routing Selector: ~1 cycle

        Args:
            src: Source coordinate (x, y)
            dest: Destination coordinate (x, y)
            include_overhead: Include NI/Selector overhead (default True)

        Returns:
            Zero-load latency in cycles
        """
        # Manhattan distance (XY routing)
        hops = abs(dest[0] - src[0]) + abs(dest[1] - src[1])

        # Pipeline latency per hop
        router_latency = hops * self.router_config.pipeline_depth

        if include_overhead:
            # NI + Selector overhead (~2 cycles)
            return float(router_latency + 2)

        return float(router_latency)

    def calculate_average_zero_load_latency(
        self,
        src: Tuple[int, int],
        destinations: list,
        include_overhead: bool = True,
    ) -> float:
        """
        Calculate average zero-load latency for multiple destinations.

        Useful for broadcast/scatter transfers.

        Args:
            src: Source coordinate (x, y)
            destinations: List of destination coordinates
            include_overhead: Include NI/Selector overhead

        Returns:
            Average zero-load latency in cycles
        """
        if not destinations:
            return 0.0

        total = sum(
            self.calculate_zero_load_latency(src, dest, include_overhead)
            for dest in destinations
        )
        return total / len(destinations)

    # =========================================================================
    # Validation Methods (BookSim2-Style)
    # =========================================================================

    def validate_latency(
        self,
        avg_latency: float,
        zero_load_latency: float,
        tolerance: float = 0.05,
    ) -> Tuple[bool, str, Dict]:
        """
        Validate latency using BookSim2 style.

        Checks:
        1. L ≥ L₀ (cannot be faster than zero-load)
        2. Calculates saturation_factor = L / L₀
        3. Determines saturation status

        Args:
            avg_latency: Measured average latency in cycles
            zero_load_latency: Theoretical zero-load latency (L₀)
            tolerance: Allowed tolerance below L₀ (default 5%)

        Returns:
            Tuple of (is_valid, message, details)
            details contains saturation_factor and saturation_status
        """
        # Handle edge case
        if zero_load_latency <= 0:
            return False, "Invalid zero-load latency (must be > 0)", {}

        # Lower bound check
        lower_bound = zero_load_latency * (1 - tolerance)
        if avg_latency < lower_bound:
            return (
                False,
                f"Below zero-load latency (L={avg_latency:.1f} < L0={zero_load_latency:.1f})",
                {},
            )

        # Calculate saturation factor
        saturation_factor = avg_latency / zero_load_latency
        status = calculate_saturation_status(saturation_factor)

        details = {
            "saturation_factor": saturation_factor,
            "saturation_status": status.value,
            "is_saturated": status == SaturationStatus.SATURATED,
        }

        return True, f"OK (L/L0={saturation_factor:.2f}, {status.value})", details

    def validate_buffer_utilization(
        self, actual_utilization: float
    ) -> Tuple[bool, str]:
        """
        Validate buffer utilization is in valid range [0, 1].

        Monitor-based: reads buffer utilization from system.

        Args:
            actual_utilization: Measured buffer utilization (0.0 to 1.0)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if actual_utilization < 0.0:
            return False, f"Invalid negative buffer utilization: {actual_utilization}"

        if actual_utilization > 1.0:
            error_msg = (
                f"Buffer utilization {actual_utilization:.2%} exceeds 100% "
                f"(indicates measurement error or overflow)"
            )
            return False, error_msg

        return True, "OK"

    def validate_all(self, metrics: Dict) -> Dict[str, Tuple[bool, str]]:
        """
        Validate all metrics in a single call (BookSim2-Style).

        Monitor-based: reads all metrics from system via dictionary.

        Args:
            metrics: Dictionary containing performance metrics
                Required for latency: avg_latency, zero_load_latency
                Optional: buffer_utilization

        Returns:
            Dictionary of validation results for each metric
        """
        results = {}

        # Validate latency (BookSim2 style)
        if all(k in metrics for k in ["avg_latency", "zero_load_latency"]):
            is_valid, msg, details = self.validate_latency(
                metrics["avg_latency"],
                metrics["zero_load_latency"],
            )
            results["latency"] = (is_valid, msg)
            # Store saturation info
            if details:
                results["saturation"] = (
                    not details.get("is_saturated", False),
                    f"factor={details.get('saturation_factor', 0):.2f}",
                )

        # Validate buffer utilization
        if "buffer_utilization" in metrics:
            results["buffer_utilization"] = self.validate_buffer_utilization(
                metrics["buffer_utilization"]
            )

        return results


def print_validation_results(results: Dict[str, Tuple[bool, str]]) -> None:
    """
    Pretty-print validation results.

    Args:
        results: Dictionary of validation results from validate_all()
    """
    print("=" * 70)
    print("Theory Validation Results (BookSim2-Style)")
    print("=" * 70)

    all_passed = True
    for metric, (is_valid, message) in results.items():
        status = "PASS" if is_valid else "FAIL"
        symbol = "+" if is_valid else "!"
        print(f"  [{symbol}] {metric:25s} {status:6s} {message}")
        all_passed = all_passed and is_valid

    print("=" * 70)
    print(f"  Overall: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 70)
