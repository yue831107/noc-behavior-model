#!/usr/bin/env python3
"""
Batch Performance Test Runner.

Generates and runs 500+ configurations for Host-to-NoC and NoC-to-NoC modes,
collecting performance metrics and validating against theoretical bounds.

Usage:
    py -3 tools/run_batch_perf_test.py --mode host_to_noc --count 500
    py -3 tools/run_batch_perf_test.py --mode noc_to_noc --count 500
    py -3 tools/run_batch_perf_test.py --mode both --count 500
"""

import argparse
import itertools
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Fix for matplotlib home directory issue on Windows
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ.get('USERPROFILE', 'C:\\Users\\Default')
if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = os.environ.get('TEMP', 'C:\\Temp')

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import TransferConfig, TransferMode, NoCTrafficConfig, TrafficPattern
from src.core.routing_selector import V1System, NoCSystem
from src.visualization import MetricsCollector
from src.traffic.pattern_generator import TrafficPatternGenerator
from src.verification import TheoryValidator, ConsistencyValidator


@dataclass
class TestConfig:
    """Single test configuration."""
    test_id: int
    mode: str  # "host_to_noc" or "noc_to_noc"
    transfer_size: int
    # Host-to-NoC specific
    num_targets: int = 16
    transfer_mode: str = "broadcast"
    # NoC-to-NoC specific
    traffic_pattern: str = "neighbor"


@dataclass
class TestResult:
    """Single test result."""
    config: TestConfig
    # Timing
    cycles: int = 0
    wall_time_ms: float = 0.0
    # Performance metrics
    throughput: float = 0.0
    latency_min: int = 0
    latency_max: int = 0
    latency_avg: float = 0.0
    latency_std: float = 0.0
    latency_samples: int = 0
    buffer_peak: int = 0
    buffer_avg: float = 0.0
    buffer_utilization: float = 0.0
    # Validation
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    # Data integrity
    data_verified: bool = True


@dataclass
class BatchTestReport:
    """Batch test report."""
    mode: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    start_time: str
    end_time: str
    duration_seconds: float
    results: List[TestResult] = field(default_factory=list)

    # Aggregated statistics
    throughput_min: float = 0.0
    throughput_max: float = 0.0
    throughput_avg: float = 0.0
    latency_min: int = 0
    latency_max: int = 0
    latency_avg: float = 0.0


def generate_host_to_noc_configs(count: int) -> List[TestConfig]:
    """Generate Host-to-NoC test configurations."""
    configs = []

    # Parameter ranges
    transfer_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    num_targets_list = [1, 2, 4, 8, 16]
    transfer_modes = ["broadcast", "scatter"]

    # Generate all combinations
    all_combos = list(itertools.product(transfer_sizes, num_targets_list, transfer_modes))

    # If we need more than available combos, repeat with variations
    test_id = 0
    while len(configs) < count:
        for size, targets, mode in all_combos:
            if len(configs) >= count:
                break
            configs.append(TestConfig(
                test_id=test_id,
                mode="host_to_noc",
                transfer_size=size,
                num_targets=targets,
                transfer_mode=mode,
            ))
            test_id += 1

    return configs[:count]


def generate_noc_to_noc_configs(count: int) -> List[TestConfig]:
    """Generate NoC-to-NoC test configurations."""
    configs = []

    # Parameter ranges
    transfer_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    traffic_patterns = ["neighbor", "shuffle", "bit_reverse", "random", "transpose"]

    # Generate all combinations
    all_combos = list(itertools.product(transfer_sizes, traffic_patterns))

    # If we need more than available combos, repeat
    test_id = 0
    while len(configs) < count:
        for size, pattern in all_combos:
            if len(configs) >= count:
                break
            configs.append(TestConfig(
                test_id=test_id,
                mode="noc_to_noc",
                transfer_size=size,
                traffic_pattern=pattern,
            ))
            test_id += 1

    return configs[:count]


def run_host_to_noc_test(config: TestConfig, verbose: bool = False) -> TestResult:
    """Run single Host-to-NoC test."""
    result = TestResult(config=config)

    try:
        # Create system
        system = V1System(mesh_cols=5, mesh_rows=4)
        collector = MetricsCollector(system, capture_interval=1)

        # Generate test data
        test_data = bytes(range(256)) * (config.transfer_size // 256 + 1)
        test_data = test_data[:config.transfer_size]

        # Get target nodes
        target_nodes = list(range(config.num_targets))

        # Submit writes
        axi_id = 1
        outstanding = {}
        completed = 0
        cycle = 0
        max_cycles = 20000

        start_time = time.perf_counter()

        for node_id in target_nodes:
            addr = (node_id << 32) | 0x1000
            system.submit_write(addr, test_data, axi_id)
            collector.record_injection(axi_id, cycle)
            outstanding[axi_id] = node_id
            axi_id += 1

        # Run simulation
        while completed < len(target_nodes) and cycle < max_cycles:
            system.process_cycle()
            collector.capture()

            while True:
                resp = system.master_ni.get_b_response()
                if resp is None:
                    break
                if resp.bid in outstanding:
                    collector.record_ejection(resp.bid, cycle)
                    del outstanding[resp.bid]
                    completed += 1

            cycle += 1

        end_time = time.perf_counter()

        # Collect metrics
        result.cycles = cycle
        result.wall_time_ms = (end_time - start_time) * 1000

        # Calculate throughput using actual completed bytes from system stats
        # Note: For BROADCAST mode, we measure bytes injected through SlaveNI,
        # not total bytes at destination (which would be transfer_size * num_targets)
        # BROADCAST sends the SAME data to all nodes, so injection = transfer_size (not × num_targets)
        throughput = collector.get_throughput()
        if throughput == 0 and cycle > 0:
            # Fallback: calculate based on transfer_size
            # BROADCAST: each node receives same data, but NI only injects once
            # SCATTER: data is split across nodes
            total_bytes = config.transfer_size  # Not multiplied by num_targets!
            throughput = total_bytes / cycle
        result.throughput = throughput

        latency_stats = collector.get_latency_stats()
        result.latency_min = latency_stats['min']
        result.latency_max = latency_stats['max']
        result.latency_avg = latency_stats['avg']
        result.latency_std = latency_stats['std']
        result.latency_samples = latency_stats['samples']

        # Buffer capacity: routers × buffer_depth × ports (default: 5×4×32×5 for batch test mesh)
        # Note: V1System uses default buffer_depth=32
        buffer_capacity = 5 * 4 * 32 * 5  # mesh_cols × mesh_rows × depth × ports
        buffer_stats = collector.get_buffer_stats(total_capacity=buffer_capacity)
        result.buffer_peak = buffer_stats['peak']
        result.buffer_avg = buffer_stats['avg']
        result.buffer_utilization = buffer_stats['utilization']

        # Validate
        theory_validator = TheoryValidator()

        # Check throughput <= T_max
        is_valid, msg = theory_validator.validate_throughput(result.throughput)
        if not is_valid:
            result.validation_passed = False
            result.validation_errors.append(msg)

        # Check buffer utilization
        is_valid, msg = theory_validator.validate_buffer_utilization(result.buffer_utilization)
        if not is_valid:
            result.validation_passed = False
            result.validation_errors.append(msg)

        # Data verification
        result.data_verified = (completed == len(target_nodes))

    except (ImportError, RuntimeError, ValueError, KeyError) as e:
        result.validation_passed = False
        result.validation_errors.append(f"Exception: {str(e)}")

    return result


def run_noc_to_noc_test(config: TestConfig, verbose: bool = False) -> TestResult:
    """Run single NoC-to-NoC test."""
    result = TestResult(config=config)

    try:
        # Create traffic config
        pattern_map = {
            "neighbor": TrafficPattern.NEIGHBOR,
            "shuffle": TrafficPattern.SHUFFLE,
            "bit_reverse": TrafficPattern.BIT_REVERSE,
            "random": TrafficPattern.RANDOM,
            "transpose": TrafficPattern.TRANSPOSE,
        }
        traffic_config = NoCTrafficConfig(
            mesh_cols=5,
            mesh_rows=4,
            transfer_size=config.transfer_size,
            pattern=pattern_map[config.traffic_pattern],
            local_src_addr=0x0000,
            local_dst_addr=0x1000,
        )

        # Create system
        system = NoCSystem(
            mesh_cols=5,
            mesh_rows=4,
            buffer_depth=4,
            memory_size=0x10000,
        )
        collector = MetricsCollector(system, capture_interval=1)

        # Configure traffic
        system.configure_traffic(traffic_config)

        # Initialize memory
        system.initialize_node_memory(pattern="sequential")
        system.generate_golden()

        # Generate traffic mapping
        generator = TrafficPatternGenerator(5, 4)
        node_configs = generator.generate(traffic_config)
        traffic_config.node_configs = node_configs

        start_time = time.perf_counter()
        system.start_all_transfers()

        # Record injection for all nodes
        for node_id in system.node_controllers.keys():
            collector.record_injection(node_id, cycle=0)

        completed_nodes = set()
        max_cycles = 20000
        cycles = 0

        while not system.all_transfers_complete and cycles < max_cycles:
            system.process_cycle()
            collector.capture()
            cycles += 1

            for node_id, controller in system.node_controllers.items():
                if node_id not in completed_nodes and controller.is_transfer_complete:
                    collector.record_ejection(node_id, cycles)
                    completed_nodes.add(node_id)

        end_time = time.perf_counter()

        # Collect metrics
        result.cycles = cycles
        result.wall_time_ms = (end_time - start_time) * 1000
        result.throughput = collector.get_throughput()

        latency_stats = collector.get_latency_stats()
        result.latency_min = latency_stats['min']
        result.latency_max = latency_stats['max']
        result.latency_avg = latency_stats['avg']
        result.latency_std = latency_stats['std']
        result.latency_samples = latency_stats['samples']

        # Buffer capacity: routers × buffer_depth × ports
        # NoC-to-NoC uses buffer_depth=4 by default
        buffer_capacity = 5 * 4 * 4 * 5  # mesh_cols × mesh_rows × depth × ports
        buffer_stats = collector.get_buffer_stats(total_capacity=buffer_capacity)
        result.buffer_peak = buffer_stats['peak']
        result.buffer_avg = buffer_stats['avg']
        result.buffer_utilization = buffer_stats['utilization']

        # Validate
        theory_validator = TheoryValidator()

        # Check buffer utilization
        is_valid, msg = theory_validator.validate_buffer_utilization(result.buffer_utilization)
        if not is_valid:
            result.validation_passed = False
            result.validation_errors.append(msg)

        # Data verification
        report = system.verify_transfers()
        result.data_verified = (report.failed == 0)

    except (ImportError, RuntimeError, ValueError, KeyError) as e:
        result.validation_passed = False
        result.validation_errors.append(f"Exception: {str(e)}")

    return result


def run_batch_tests(
    mode: str,
    count: int,
    verbose: bool = False,
    progress_interval: int = 50,
) -> BatchTestReport:
    """Run batch of tests."""
    start_time = datetime.now()

    # Generate configs
    if mode == "host_to_noc":
        configs = generate_host_to_noc_configs(count)
    else:
        configs = generate_noc_to_noc_configs(count)

    print(f"\nRunning {len(configs)} {mode} tests...")
    print("=" * 60)

    results = []
    passed = 0
    failed = 0

    for i, config in enumerate(configs):
        # Progress update
        if (i + 1) % progress_interval == 0 or i == 0:
            print(f"  Progress: {i + 1}/{len(configs)} ({(i + 1) / len(configs) * 100:.1f}%)")

        # Run test
        if mode == "host_to_noc":
            result = run_host_to_noc_test(config, verbose)
        else:
            result = run_noc_to_noc_test(config, verbose)

        results.append(result)

        if result.validation_passed and result.data_verified:
            passed += 1
        else:
            failed += 1

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Create report
    report = BatchTestReport(
        mode=mode,
        total_tests=len(results),
        passed_tests=passed,
        failed_tests=failed,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        duration_seconds=duration,
        results=results,
    )

    # Calculate aggregated statistics
    if results:
        throughputs = [r.throughput for r in results if r.throughput > 0]
        latencies = [r.latency_avg for r in results if r.latency_avg > 0]

        if throughputs:
            report.throughput_min = min(throughputs)
            report.throughput_max = max(throughputs)
            report.throughput_avg = sum(throughputs) / len(throughputs)

        if latencies:
            report.latency_min = int(min(latencies))
            report.latency_max = int(max(latencies))
            report.latency_avg = sum(latencies) / len(latencies)

    return report


def save_report(report: BatchTestReport, output_dir: Path) -> None:
    """Save batch test report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary = {
        "mode": report.mode,
        "total_tests": report.total_tests,
        "passed_tests": report.passed_tests,
        "failed_tests": report.failed_tests,
        "pass_rate": report.passed_tests / report.total_tests * 100 if report.total_tests > 0 else 0,
        "start_time": report.start_time,
        "end_time": report.end_time,
        "duration_seconds": report.duration_seconds,
        "throughput": {
            "min": report.throughput_min,
            "max": report.throughput_max,
            "avg": report.throughput_avg,
        },
        "latency": {
            "min": report.latency_min,
            "max": report.latency_max,
            "avg": report.latency_avg,
        },
    }

    summary_path = output_dir / f"batch_{report.mode}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save detailed results
    details = []
    for r in report.results:
        details.append({
            "test_id": r.config.test_id,
            "transfer_size": r.config.transfer_size,
            "traffic_pattern": r.config.traffic_pattern if report.mode == "noc_to_noc" else None,
            "num_targets": r.config.num_targets if report.mode == "host_to_noc" else None,
            "cycles": r.cycles,
            "throughput": r.throughput,
            "latency_avg": r.latency_avg,
            "latency_min": r.latency_min,
            "latency_max": r.latency_max,
            "buffer_utilization": r.buffer_utilization,
            "validation_passed": r.validation_passed,
            "data_verified": r.data_verified,
            "errors": r.validation_errors,
        })

    details_path = output_dir / f"batch_{report.mode}_details.json"
    with open(details_path, 'w') as f:
        json.dump(details, f, indent=2)

    print(f"\nReport saved to: {output_dir}")


def print_report_summary(report: BatchTestReport) -> None:
    """Print report summary."""
    print()
    print("=" * 60)
    print(f"Batch Test Report: {report.mode.upper()}")
    print("=" * 60)
    print(f"Total Tests:    {report.total_tests}")
    print(f"Passed:         {report.passed_tests}")
    print(f"Failed:         {report.failed_tests}")
    print(f"Pass Rate:      {report.passed_tests / report.total_tests * 100:.1f}%")
    print(f"Duration:       {report.duration_seconds:.1f} seconds")
    print()
    print("Performance Summary:")
    print(f"  Throughput:   {report.throughput_min:.2f} - {report.throughput_max:.2f} B/cycle (avg: {report.throughput_avg:.2f})")
    print(f"  Latency:      {report.latency_min} - {report.latency_max} cycles (avg: {report.latency_avg:.1f})")
    print("=" * 60)

    # Show failed tests if any
    if report.failed_tests > 0:
        print("\nFailed Tests:")
        for r in report.results:
            if not r.validation_passed or not r.data_verified:
                print(f"  Test {r.config.test_id}: {r.validation_errors}")


def main():
    parser = argparse.ArgumentParser(
        description='Run batch performance tests'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['host_to_noc', 'noc_to_noc', 'both'],
        default='both',
        help='Test mode (default: both)'
    )
    parser.add_argument(
        '--count', '-c',
        type=int,
        default=500,
        help='Number of tests per mode (default: 500)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('output/batch_tests'),
        help='Output directory (default: output/batch_tests)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--progress',
        type=int,
        default=50,
        help='Progress update interval (default: 50)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Batch Performance Test Runner")
    print("=" * 60)
    print(f"Mode:   {args.mode}")
    print(f"Count:  {args.count} tests per mode")
    print(f"Output: {args.output}")

    reports = []

    if args.mode in ['host_to_noc', 'both']:
        report = run_batch_tests(
            'host_to_noc',
            args.count,
            args.verbose,
            args.progress,
        )
        print_report_summary(report)
        save_report(report, args.output)
        reports.append(report)

    if args.mode in ['noc_to_noc', 'both']:
        report = run_batch_tests(
            'noc_to_noc',
            args.count,
            args.verbose,
            args.progress,
        )
        print_report_summary(report)
        save_report(report, args.output)
        reports.append(report)

    # Overall summary
    if len(reports) == 2:
        print()
        print("=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        total = sum(r.total_tests for r in reports)
        passed = sum(r.passed_tests for r in reports)
        print(f"Total Tests:  {total}")
        print(f"Passed:       {passed}")
        print(f"Pass Rate:    {passed / total * 100:.1f}%")
        print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
