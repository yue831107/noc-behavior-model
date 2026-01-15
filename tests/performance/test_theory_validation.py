"""
Theory-based validation tests (BookSim2-Style).

Tests that verify performance metrics using BookSim2 concepts:
- Zero-load latency (L0): theoretical minimum latency
- Saturation detection: L > 3×L0
- Buffer utilization bounds

Uses Monitor-based approach - no core modification required.
"""

import pytest
from src.verification.theory_validator import (
    TheoryValidator,
    MeshConfig,
    RouterConfig,
    print_validation_results
)
from src.metrics import SaturationStatus, calculate_saturation_status


class TestTheoryValidation:
    """Theory-based validation test cases (BookSim2-Style)."""

    def test_zero_load_latency_calculation(self):
        """
        Test zero-load latency (L0) calculation.

        L0 = hops × pipeline_depth + overhead
        Overhead = 2 cycles (NI + Selector)
        """
        validator = TheoryValidator()

        # 1 hop distance: 1 + 2 = 3 cycles
        assert validator.calculate_zero_load_latency((1, 1), (2, 1)) == 3.0

        # 3 hops distance (2 east + 1 north): 3 + 2 = 5 cycles
        assert validator.calculate_zero_load_latency((1, 1), (3, 2)) == 5.0

        # Without overhead: 1 hop = 1 cycle
        l0 = validator.calculate_zero_load_latency((1, 1), (2, 1), include_overhead=False)
        assert l0 == 1.0

    def test_average_zero_load_latency(self):
        """
        Test average zero-load latency for multiple destinations.
        """
        validator = TheoryValidator()

        # Single destination: same as calculate_zero_load_latency
        avg_l0 = validator.calculate_average_zero_load_latency(
            (1, 1), [(2, 1)]
        )
        assert avg_l0 == 3.0  # 1 hop + 2 overhead

        # Multiple destinations
        # (1,1) -> (2,1): 1 hop, L0=3
        # (1,1) -> (3,1): 2 hops, L0=4
        # Average = (3 + 4) / 2 = 3.5
        avg_l0 = validator.calculate_average_zero_load_latency(
            (1, 1), [(2, 1), (3, 1)]
        )
        assert avg_l0 == 3.5

    def test_validate_latency_booksim_style(self):
        """
        Test BookSim2-style latency validation.

        Checks:
        1. L >= L0 (cannot be faster than zero-load)
        2. Saturation detection (L > 3×L0)
        """
        validator = TheoryValidator()

        # PASS: normal operation (L = 4.0, L0 = 3.0)
        is_valid, msg, details = validator.validate_latency(4.0, 3.0)
        assert is_valid
        assert details['saturation_status'] == 'normal'
        assert details['saturation_factor'] == pytest.approx(4.0 / 3.0, rel=0.01)

        # PASS: warning level (L = 6.5, L0 = 3.0, factor = 2.17)
        is_valid, msg, details = validator.validate_latency(6.5, 3.0)
        assert is_valid
        assert details['saturation_status'] == 'warning'

        # PASS: saturated but still valid (L = 10.0, L0 = 3.0, factor = 3.33)
        is_valid, msg, details = validator.validate_latency(10.0, 3.0)
        assert is_valid
        assert details['saturation_status'] == 'saturated'
        assert details['is_saturated'] is True

        # FAIL: below zero-load latency (impossible)
        is_valid, msg, details = validator.validate_latency(2.0, 3.0)
        assert not is_valid
        assert "below" in msg.lower()

    def test_saturation_status_calculation(self):
        """
        Test saturation status calculation.

        Thresholds (from BookSim2):
        - NORMAL: factor < 2.0
        - WARNING: 2.0 <= factor < 3.0
        - SATURATED: factor >= 3.0
        """
        assert calculate_saturation_status(1.0) == SaturationStatus.NORMAL
        assert calculate_saturation_status(1.9) == SaturationStatus.NORMAL
        assert calculate_saturation_status(2.0) == SaturationStatus.WARNING
        assert calculate_saturation_status(2.9) == SaturationStatus.WARNING
        assert calculate_saturation_status(3.0) == SaturationStatus.SATURATED
        assert calculate_saturation_status(5.0) == SaturationStatus.SATURATED

    def test_validate_buffer_utilization(self):
        """
        Test buffer utilization validation: 0 <= U <= 1
        """
        validator = TheoryValidator()

        # PASS: 0.35 in [0, 1]
        is_valid, _ = validator.validate_buffer_utilization(0.35)
        assert is_valid

        # PASS: boundary values
        is_valid, _ = validator.validate_buffer_utilization(0.0)
        assert is_valid
        is_valid, _ = validator.validate_buffer_utilization(1.0)
        assert is_valid

        # FAIL: 1.5 > 1 (overflow)
        is_valid, msg = validator.validate_buffer_utilization(1.5)
        assert not is_valid and "exceeds" in msg.lower()

        # FAIL: -0.1 < 0 (negative)
        is_valid, msg = validator.validate_buffer_utilization(-0.1)
        assert not is_valid and "negative" in msg.lower()

    def test_validate_all_metrics(self):
        """Test batch validation with BookSim2-style metrics."""
        validator = TheoryValidator()

        # All valid metrics
        results = validator.validate_all({
            'avg_latency': 4.0,
            'zero_load_latency': 3.0,
            'buffer_utilization': 0.35,
        })
        assert 'latency' in results
        assert 'buffer_utilization' in results
        assert all(is_valid for is_valid, _ in results.values())

        # Latency below zero-load (invalid)
        results = validator.validate_all({
            'avg_latency': 2.0,  # Below L0
            'zero_load_latency': 3.0,
            'buffer_utilization': 0.35,
        })
        assert not results['latency'][0]  # Should fail

        # Buffer overflow (invalid)
        results = validator.validate_all({
            'avg_latency': 4.0,
            'zero_load_latency': 3.0,
            'buffer_utilization': 1.5,  # > 1
        })
        assert not results['buffer_utilization'][0]  # Should fail


def test_theory_validator_integration():
    """
    Integration test: verify complete validation flow.
    """
    validator = TheoryValidator(
        mesh_config=MeshConfig(cols=5, rows=4),
        router_config=RouterConfig(flit_data_bytes=32, pipeline_depth=1)
    )

    # Calculate L0 for a specific path
    zero_load_latency = validator.calculate_zero_load_latency((1, 1), (2, 2))
    # 2-hop path: 2 + 2 = 4 cycles

    # Mock simulation metrics with normal operation
    results = validator.validate_all({
        'avg_latency': 5.0,  # Slightly above L0
        'zero_load_latency': zero_load_latency,
        'buffer_utilization': 0.42,
    })

    print_validation_results(results)
    assert all(is_valid for is_valid, _ in results.values())

    # Test with saturated network
    results = validator.validate_all({
        'avg_latency': 15.0,  # L/L0 = 15/4 = 3.75 -> SATURATED
        'zero_load_latency': zero_load_latency,
        'buffer_utilization': 0.85,
    })

    # Latency validation should pass (saturation is not a failure)
    assert results['latency'][0] is True
    # Buffer utilization should pass
    assert results['buffer_utilization'][0] is True
    # Saturation result is (not is_saturated, message) - so it's False when saturated
    assert 'saturation' in results
    assert results['saturation'][0] is False  # is_saturated=True -> stored as (False, ...)
