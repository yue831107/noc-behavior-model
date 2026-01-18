"""
Performance Regression Tests for NoC Behavior Model.

This module tests for performance regressions by comparing current
simulation results against stored baselines. If no baseline exists,
it creates one automatically.

Usage:
    make perf_baseline   # Run regression tests via Makefile
    pytest tests/performance/test_performance_regression.py -v
"""

import pytest
from pathlib import Path
from typing import Dict, Any

from .baseline_manager import BaselineManager, ComparisonResult
from .utils import extract_metrics_from_simulation


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def baseline_manager() -> BaselineManager:
    """Create baseline manager pointing to test baselines directory."""
    baselines_dir = Path(__file__).parent / "baselines"
    return BaselineManager(baselines_dir)


@pytest.fixture
def host_to_noc_system():
    """Create a minimal V1System for regression testing."""
    from src.core.routing_selector import V1System

    system = V1System(
        mesh_cols=5,
        mesh_rows=4,
        buffer_depth=4,
    )
    return system


@pytest.fixture
def noc_to_noc_system():
    """Create a minimal NoCSystem for regression testing."""
    from src.core.routing_selector import NoCSystem

    system = NoCSystem(
        mesh_cols=5,
        mesh_rows=4,
        buffer_depth=4,
    )
    return system


# ==============================================================================
# Helper Functions
# ==============================================================================

def run_simple_benchmark(system, num_transfers: int = 10) -> Dict[str, float]:
    """
    Run a simple benchmark and return metrics.

    This is a lightweight benchmark for regression detection,
    not a full performance characterization.
    """
    from src.core.flit import FlitFactory, AxiChannel

    # Generate simple write traffic
    payload = bytes([i % 256 for i in range(64)])

    # Determine system type and run appropriate benchmark
    system_type = type(system).__name__

    if system_type == "V1System":
        # V1System: Host-to-NoC benchmark
        for i in range(num_transfers):
            target_node = i % 16  # 4x4 = 16 nodes
            row = target_node // 4
            col = (target_node % 4) + 1  # cols 1-4

            # Create AW flit
            aw_flit = FlitFactory.create_aw(
                src=(0, 0),
                dest=(col, row),
                addr=0x1000 + i * 64,
                axi_id=i % 8,
                length=0,
            )

            # Inject into routing selector
            if hasattr(system, "_selector"):
                system._selector.inject_request(aw_flit)

    elif system_type == "NoCSystem":
        # NoCSystem: NoC-to-NoC benchmark
        for i in range(num_transfers):
            src_node = i % 16
            dst_node = (src_node + 1) % 16  # neighbor pattern

            src_row = src_node // 4
            src_col = (src_node % 4) + 1
            dst_row = dst_node // 4
            dst_col = (dst_node % 4) + 1

            # Create AR flit (read request)
            ar_flit = FlitFactory.create_ar(
                src=(src_col, src_row),
                dest=(dst_col, dst_row),
                addr=0x2000 + i * 64,
                axi_id=i % 8,
                length=0,
            )

            # Would inject into source node's SlaveNI
            # For now, just count the cycle overhead

    # Run simulation cycles
    max_cycles = 1000

    # Use appropriate run method based on system type
    if hasattr(system, "run"):
        system.run(max_cycles)
    else:
        # Fallback for systems with step() method
        for cycle in range(max_cycles):
            if hasattr(system, "step"):
                system.step()
            if hasattr(system, "is_complete") and system.is_complete():
                break

    # Extract metrics
    metrics = extract_metrics_from_simulation(system)

    return {
        "throughput_gbps": metrics.get("throughput", 0) * 8 / 1e9,  # Convert to Gbps
        "avg_latency_cycles": metrics.get("cycle", 0) / max(num_transfers, 1),
        "completed_transactions": metrics.get("completed_transactions", 0),
        "total_cycles": metrics.get("cycle", 0),
    }


# ==============================================================================
# Regression Tests
# ==============================================================================

class TestPerformanceRegression:
    """
    Performance regression test suite.

    Tests ensure that performance does not degrade beyond acceptable thresholds:
    - Throughput: Must be >= 95% of baseline
    - Latency: Must be <= 110% of baseline

    If no baseline exists, one is created automatically.
    """

    @pytest.mark.parametrize("config_name,num_transfers", [
        ("host_to_noc/simple_write", 10),
        ("host_to_noc/medium_write", 50),
    ])
    def test_host_to_noc_no_regression(
        self,
        config_name: str,
        num_transfers: int,
        baseline_manager: BaselineManager,
        host_to_noc_system,
    ):
        """Ensure Host-to-NoC performance doesn't regress."""
        # Run benchmark
        current = run_simple_benchmark(host_to_noc_system, num_transfers)

        # Load baseline
        baseline = baseline_manager.load_baseline(config_name)

        if baseline is None:
            # No baseline exists, create one
            baseline_manager.save_baseline(
                config_name,
                current,
                description=f"Initial baseline: {num_transfers} transfers"
            )
            pytest.skip(f"No baseline for {config_name}, created new one")

        # Compare
        result = baseline_manager.compare(current, baseline.metrics)

        # Assert no regression
        assert not result.regression_detected, (
            f"Performance regression detected for {config_name}:\n"
            f"  Throughput: {result.throughput_delta:+.1f}% "
            f"(threshold: {baseline_manager.throughput_threshold * 100:.0f}%)\n"
            f"  Latency: {result.latency_delta:+.1f}% "
            f"(threshold: {baseline_manager.latency_threshold * 100:.0f}%)\n"
            f"  Current: {current}\n"
            f"  Baseline: {baseline.metrics}"
        )

    @pytest.mark.parametrize("config_name,num_transfers", [
        ("noc_to_noc/neighbor_simple", 10),
    ])
    def test_noc_to_noc_no_regression(
        self,
        config_name: str,
        num_transfers: int,
        baseline_manager: BaselineManager,
        noc_to_noc_system,
    ):
        """Ensure NoC-to-NoC performance doesn't regress."""
        # Run benchmark
        current = run_simple_benchmark(noc_to_noc_system, num_transfers)

        # Load baseline
        baseline = baseline_manager.load_baseline(config_name)

        if baseline is None:
            # No baseline exists, create one
            baseline_manager.save_baseline(
                config_name,
                current,
                description=f"Initial baseline: {num_transfers} transfers"
            )
            pytest.skip(f"No baseline for {config_name}, created new one")

        # Compare
        result = baseline_manager.compare(current, baseline.metrics)

        # Assert no regression
        assert not result.regression_detected, (
            f"Performance regression detected for {config_name}:\n"
            f"  Throughput: {result.throughput_delta:+.1f}%\n"
            f"  Latency: {result.latency_delta:+.1f}%\n"
            f"  Details: {result.details}"
        )


class TestBaselineManager:
    """Unit tests for BaselineManager functionality."""

    def test_save_and_load_baseline(self, baseline_manager: BaselineManager, tmp_path):
        """Test saving and loading baselines."""
        # Use tmp_path for isolated testing
        manager = BaselineManager(tmp_path / "baselines")

        metrics = {
            "throughput_gbps": 10.5,
            "avg_latency_cycles": 45.2,
        }

        # Save
        manager.save_baseline("test/example", metrics, description="Test baseline")

        # Load
        loaded = manager.load_baseline("test/example")

        assert loaded is not None
        assert loaded.name == "test/example"
        assert loaded.metrics["throughput_gbps"] == 10.5
        assert loaded.metrics["avg_latency_cycles"] == 45.2

    def test_compare_no_regression(self, baseline_manager: BaselineManager):
        """Test comparison when no regression."""
        baseline = {"throughput_gbps": 10.0, "avg_latency_cycles": 50.0}
        current = {"throughput_gbps": 10.5, "avg_latency_cycles": 48.0}

        result = baseline_manager.compare(current, baseline)

        assert not result.regression_detected
        assert result.throughput_delta > 0  # Improved
        assert result.latency_delta < 0     # Improved (lower is better)

    def test_compare_throughput_regression(self, baseline_manager: BaselineManager):
        """Test comparison when throughput regresses."""
        baseline = {"throughput_gbps": 10.0, "avg_latency_cycles": 50.0}
        current = {"throughput_gbps": 9.0, "avg_latency_cycles": 50.0}  # 10% worse

        result = baseline_manager.compare(current, baseline)

        assert result.regression_detected
        assert result.throughput_delta < 0

    def test_compare_latency_regression(self, baseline_manager: BaselineManager):
        """Test comparison when latency regresses."""
        baseline = {"throughput_gbps": 10.0, "avg_latency_cycles": 50.0}
        current = {"throughput_gbps": 10.0, "avg_latency_cycles": 60.0}  # 20% worse

        result = baseline_manager.compare(current, baseline)

        assert result.regression_detected
        assert result.latency_delta > 0  # Higher latency is worse

    def test_update_if_better(self, tmp_path):
        """Test automatic baseline update when performance improves."""
        manager = BaselineManager(tmp_path / "baselines")

        # Save initial baseline
        initial = {"throughput_gbps": 10.0, "avg_latency_cycles": 50.0}
        manager.save_baseline("test/update", initial)

        # Better performance
        better = {"throughput_gbps": 11.0, "avg_latency_cycles": 45.0}
        updated = manager.update_if_better("test/update", better)

        assert updated is True

        # Verify update
        loaded = manager.load_baseline("test/update")
        assert loaded.metrics["throughput_gbps"] == 11.0

    def test_list_baselines(self, tmp_path):
        """Test listing available baselines."""
        manager = BaselineManager(tmp_path / "baselines")

        # Create some baselines
        manager.save_baseline("host_to_noc/test1", {"throughput_gbps": 10.0})
        manager.save_baseline("host_to_noc/test2", {"throughput_gbps": 11.0})
        manager.save_baseline("noc_to_noc/test1", {"throughput_gbps": 9.0})

        # List all
        all_baselines = manager.list_baselines()
        assert len(all_baselines) == 3

        # List by category
        host_baselines = manager.list_baselines("host_to_noc")
        assert len(host_baselines) == 2
        assert "host_to_noc/test1" in host_baselines
