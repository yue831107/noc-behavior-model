"""
Consistency-based validation tests.

Tests that verify performance metrics are internally consistent.
Uses Monitor-based approach - no core modification required.
"""

import pytest
from src.verification.consistency_validator import (
    ConsistencyValidator,
    print_consistency_results
)


class TestConsistencyValidation:
    """Consistency-based validation test cases."""

    def test_flit_conservation(self):
        """
        驗證公式: Total_Sent = Total_Received
        """
        validator = ConsistencyValidator()
        
        # PASS: Perfect conservation
        is_valid, msg = validator.validate_flit_conservation(1000, 1000)
        assert is_valid and "conservation holds" in msg.lower()
        
        # FAIL: Loss detected
        is_valid, msg = validator.validate_flit_conservation(1000, 995)
        assert not is_valid and "loss" in msg.lower()
        
        # FAIL: Duplication detected
        is_valid, msg = validator.validate_flit_conservation(1000, 1010)
        assert not is_valid and "duplication" in msg.lower()
    
    def test_bandwidth_conservation(self):
        """
        驗證公式: Injection_Rate ≈ Ejection_Rate (steady state)
        """
        validator = ConsistencyValidator(tolerance=0.05)
        
        # PASS: Perfect match
        is_valid, msg = validator.validate_bandwidth_conservation(30.0, 30.0)
        assert is_valid and "OK" in msg
        
        # PASS: Within tolerance (2.67%)
        is_valid, _ = validator.validate_bandwidth_conservation(30.0, 30.8)
        assert is_valid
        
        # FAIL: Violation (16.7% > 5%)
        is_valid, msg = validator.validate_bandwidth_conservation(30.0, 35.0)
        assert not is_valid and "violation" in msg.lower()
    
    def test_router_logic(self):
        """
        驗證公式: Received = Forwarded + Local_Consumed
        """
        validator = ConsistencyValidator()
        
        # PASS: 100 = 70 + 30
        is_valid, msg = validator.validate_router_logic(100, 70, 30)
        assert is_valid and msg == "OK"
        
        # FAIL: 100 ≠ 70 + 35
        is_valid, msg = validator.validate_router_logic(100, 70, 35)
        assert not is_valid and "logic error" in msg.lower()
    
    def test_validate_all_metrics(self):
        """整合驗證：所有一致性檢查"""
        validator = ConsistencyValidator(tolerance=0.10)

        # All valid
        results = validator.validate_all({
            'total_sent': 1000, 'total_received': 1000,
            'injection_rate': 30.0, 'ejection_rate': 30.0,
        })
        assert all(is_valid for is_valid, _ in results.values())

        # All invalid
        results = validator.validate_all({
            'total_sent': 1000, 'total_received': 980,  # ❌
            'injection_rate': 30.0, 'ejection_rate': 40.0,  # ❌
        })
        assert all(not is_valid for is_valid, _ in results.values())


def test_consistency_validator_integration():
    """整合測試：驗證完整模擬流程"""
    validator = ConsistencyValidator(tolerance=0.10)

    results = validator.validate_all({
        'total_sent': 512, 'total_received': 512,
        'injection_rate': 28.0, 'ejection_rate': 28.0,
    })

    print_consistency_results(results)
    assert all(is_valid for is_valid, _ in results.values())
