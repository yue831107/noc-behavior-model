"""
Consistency-based Performance Validator.

Validates that performance metrics are internally consistent using
Monitor-based approach (no core modification required).

For line-rate injection model:
- λ (arrival rate) in Little's Law is the measured throughput, not a configured
  injection_rate as in BookSim2. This is because our model injects at line-rate
  (as fast as backpressure allows).
"""

from typing import Dict, Tuple, Optional


class ConsistencyValidator:
    """
    Validates internal consistency between performance metrics.
    
    Monitor-based validator: reads metrics from system without
    modifying core implementation.
    
    Checks mathematical relationships between different metrics:
    - Little's Law: L = λ × W
    - Flit Conservation: Sent = Received
    - Bandwidth Conservation: Input = Output
    """
    
    def __init__(self, tolerance: float = 0.10):
        """
        Initialize validator with tolerance.
        
        Args:
            tolerance: Allowed deviation for consistency checks (default 10%)
        """
        self.tolerance = tolerance
    
    def validate_littles_law(
        self,
        throughput: float,
        avg_latency: float,
        avg_occupancy: float,
        flit_width_bytes: int = 32
    ) -> Tuple[bool, str]:
        """
        Validate Little's Law: L = λ × W

        Little's Law (Queuing Theory fundamental):
        - L: Average number of items in system (occupancy)
        - λ: Arrival rate (flits/cycle)
        - W: Average time in system (latency in cycles)

        For NoC (line-rate injection model):
        - L = avg_occupancy (flits in network)
        - λ = throughput (bytes/cycle) / flit_width (flits/cycle)
        - W = avg_latency (cycles)

        Note: In BookSim2, λ is a configured injection_rate.
        In our model, λ is the measured throughput since we inject
        at line-rate (as fast as backpressure allows).

        Args:
            throughput: Measured throughput in bytes/cycle
            avg_latency: Average latency in cycles
            avg_occupancy: Average number of flits in network
            flit_width_bytes: Flit width in bytes (default 32 for FlooNoC)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Convert throughput to flits/cycle
        lambda_rate = throughput / flit_width_bytes  # flits/cycle
        
        # Calculate expected occupancy using Little's Law
        expected_occupancy = lambda_rate * avg_latency
        
        # Check if actual matches expected (within tolerance)
        if expected_occupancy == 0:
            # Edge case: zero throughput
            if avg_occupancy == 0:
                return True, "OK (zero load)"
            else:
                return False, f"Zero throughput but occupancy = {avg_occupancy}"
        
        deviation = abs(avg_occupancy - expected_occupancy) / expected_occupancy
        
        if deviation > self.tolerance:
            error_msg = (
                f"Little's Law violation: "
                f"L={avg_occupancy:.2f}, λ={lambda_rate:.2f}, W={avg_latency:.2f} "
                f"(expected L=λ×W={expected_occupancy:.2f}, "
                f"deviation={deviation:.1%} > {self.tolerance:.0%})"
            )
            return False, error_msg
        
        return True, f"OK (deviation={deviation:.1%})"
    
    def validate_flit_conservation(
        self,
        total_sent: int,
        total_received: int
    ) -> Tuple[bool, str]:
        """
        Validate Flit Conservation Law.
        
        In a lossless system:
        Total Flits Sent = Total Flits Received
        
        This detects:
        - Packet loss (received < sent)
        - Duplicate counting (received > sent)
        - Measurement errors
        
        Args:
            total_sent: Total number of flits sent by all sources
            total_received: Total number of flits received by all destinations
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if total_sent == total_received:
            return True, "OK (conservation holds)"
        
        difference = total_received - total_sent
        
        if difference < 0:
            error_msg = (
                f"Flit loss detected: "
                f"received {total_received} < sent {total_sent} "
                f"(lost {-difference} flits)"
            )
            return False, error_msg
        else:
            error_msg = (
                f"Flit duplication detected: "
                f"received {total_received} > sent {total_sent} "
                f"(extra {difference} flits)"
            )
            return False, error_msg
    
    def validate_bandwidth_conservation(
        self,
        total_injection_rate: float,
        total_ejection_rate: float
    ) -> Tuple[bool, str]:
        """
        Validate Bandwidth Conservation (Steady State).
        
        In steady state:
        Total Input Bandwidth = Total Output Bandwidth
        Σ(injection_rate) = Σ(ejection_rate)
        
        Args:
            total_injection_rate: Sum of all injection rates (bytes/cycle)
            total_ejection_rate: Sum of all ejection rates (bytes/cycle)
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if total_injection_rate == 0:
            # Edge case: zero load
            if total_ejection_rate == 0:
                return True, "OK (zero load)"
            else:
                return False, f"Zero injection but ejection = {total_ejection_rate}"
        
        deviation = abs(total_injection_rate - total_ejection_rate) / total_injection_rate
        
        if deviation > self.tolerance:
            error_msg = (
                f"Bandwidth conservation violation: "
                f"injection={total_injection_rate:.2f}, "
                f"ejection={total_ejection_rate:.2f} "
                f"(deviation={deviation:.1%} > {self.tolerance:.0%})"
            )
            return False, error_msg
        
        return True, f"OK (deviation={deviation:.1%})"
    
    def validate_router_logic(
        self,
        router_received: int,
        router_forwarded: int,
        local_consumed: int
    ) -> Tuple[bool, str]:
        """
        Validate Router Internal Logic.
        
        For each router:
        Flits Received = Flits Forwarded + Flits Consumed Locally
        
        Args:
            router_received: Flits received by this router
            router_forwarded: Flits forwarded to other routers
            local_consumed: Flits consumed by local NI
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        expected_total = router_forwarded + local_consumed
        
        if router_received == expected_total:
            return True, "OK"
        
        difference = router_received - expected_total
        error_msg = (
            f"Router logic error: "
            f"received {router_received} != "
            f"forwarded {router_forwarded} + consumed {local_consumed} "
            f"(difference={difference})"
        )
        return False, error_msg
    
    def validate_all(self, metrics: Dict) -> Dict[str, Tuple[bool, str]]:
        """
        Validate all consistency metrics in a single call.
        
        Monitor-based: reads all metrics from system via dictionary.
        
        Args:
            metrics: Dictionary containing performance metrics
                Required for Little's Law: throughput, avg_latency, avg_occupancy
                Required for Flit Conservation: total_sent, total_received
                Optional: injection_rate, ejection_rate
        
        Returns:
            Dictionary of validation results for each check
        """
        results = {}
        
        # Validate Little's Law
        if all(k in metrics for k in ['throughput', 'avg_latency', 'avg_occupancy']):
            results['littles_law'] = self.validate_littles_law(
                metrics['throughput'],
                metrics['avg_latency'],
                metrics['avg_occupancy'],
                metrics.get('flit_width_bytes', 8)
            )
        
        # Validate Flit Conservation
        if all(k in metrics for k in ['total_sent', 'total_received']):
            results['flit_conservation'] = self.validate_flit_conservation(
                metrics['total_sent'],
                metrics['total_received']
            )
        
        # Validate Bandwidth Conservation (optional)
        if all(k in metrics for k in ['injection_rate', 'ejection_rate']):
            results['bandwidth_conservation'] = self.validate_bandwidth_conservation(
                metrics['injection_rate'],
                metrics['ejection_rate']
            )
        
        return results


def print_consistency_results(results: Dict[str, Tuple[bool, str]]) -> None:
    """
    Pretty-print consistency validation results.
    
    Args:
        results: Dictionary of validation results from validate_all()
    """
    print("=" * 70)
    print("Consistency Validation Results")
    print("=" * 70)
    
    all_passed = True
    for check, (is_valid, message) in results.items():
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"{check:25s} {status:8s} {message}")
        all_passed = all_passed and is_valid
    
    print("=" * 70)
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 70)
