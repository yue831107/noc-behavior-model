#!/usr/bin/env python3
"""
NoC-to-NoC Traffic Test Runner.

Usage:
    py -3 run.py neighbor                      # Run neighbor with dynamic data
    py -3 run.py neighbor -m random            # With random memory data
    py -3 run.py neighbor -P payload/          # Load from payload files
    py -3 run.py --all                         # Run all traffic patterns
    py -3 run.py --all -P payload/             # All patterns with payload files

Memory Patterns (dynamic generation):
    sequential, random, constant, node_id, address, walking_ones, walking_zeros, checkerboard

Traffic Patterns:
    neighbor, shuffle, bit_reverse, random, transpose

Generate payload files first:
    make gen_noc_payload                       # Generate payload/node_XX.bin files
    make gen_noc_payload NOC_PATTERN=random    # With specific pattern
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core import NoCSystem
from src.config import load_noc_traffic_config, NoCTrafficConfig, TrafficPattern
from src.traffic import TrafficPatternGenerator
from src.visualization import MetricsCollector


def print_header(title: str) -> None:
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title: str) -> None:
    """Print section divider."""
    print(f"\n--- {title} ---")


MEMORY_PATTERNS = [
    'sequential', 'random', 'constant', 'node_id',
    'address', 'walking_ones', 'walking_zeros', 'checkerboard'
]


def run_traffic_pattern(
    pattern: str,
    memory_pattern: str = "sequential",
    payload_dir: str = None,
    verbose: bool = True
) -> dict:
    """
    Run NoC-to-NoC traffic pattern test.

    Args:
        pattern: Traffic pattern name.
        memory_pattern: Memory initialization pattern (ignored if payload_dir set).
        payload_dir: Directory with node_XX.bin files (overrides memory_pattern).
        verbose: Enable verbose output.

    Returns:
        Result dictionary.
    """
    start_time = time.perf_counter()

    if verbose:
        print_header(f"NoC-to-NoC Traffic Test: {pattern.upper()}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load config
    config_path = Path(__file__).parent / "config" / f"{pattern}.yaml"
    if config_path.exists():
        config = load_noc_traffic_config(config_path)
        if verbose:
            print(f"Config file: {config_path}")
    else:
        if verbose:
            print(f"Config not found: {config_path}, using defaults")
        config = NoCTrafficConfig(
            pattern=TrafficPattern(pattern),
            mesh_cols=5,
            mesh_rows=4,
            transfer_size=256,
        )

    # Create system
    system = NoCSystem(
        mesh_cols=config.mesh_cols,
        mesh_rows=config.mesh_rows,
        buffer_depth=4,
        memory_size=0x10000,
    )

    if verbose:
        print_section("Configuration")
        print(f"  Pattern:       {config.pattern.value}")
        print(f"  Mesh Size:     {config.mesh_cols}x{config.mesh_rows}")
        print(f"  Compute Nodes: {system.num_nodes}")
        print(f"  Transfer Size: {config.transfer_size} bytes/node")
        print(f"  Total Data:    {config.transfer_size * system.num_nodes} bytes")
        print(f"  Src Address:   0x{config.local_src_addr:04X}")
        print(f"  Dst Address:   0x{config.local_dst_addr:04X}")
        if payload_dir:
            print(f"  Memory Source: {payload_dir} (files)")
        else:
            print(f"  Memory Source: {memory_pattern} (dynamic)")

    # Configure traffic
    system.configure_traffic(config)

    # Initialize node memory
    if payload_dir:
        # Load from binary files
        loaded = system.load_node_memory_from_files(payload_dir)
        if verbose:
            print(f"  Loaded:        {loaded} nodes from payload files")
    else:
        # Generate dynamically
        system.initialize_node_memory(pattern=memory_pattern)

    # Generate golden data (BEFORE simulation)
    golden_count = system.generate_golden()

    # Generate and display traffic mapping
    generator = TrafficPatternGenerator(config.mesh_cols, config.mesh_rows)
    node_configs = generator.generate(config)

    # Store node configs in traffic config for verification
    config.node_configs = node_configs

    if verbose:
        print(f"  Golden Entries:    {golden_count}")

    if verbose:
        print_section("Traffic Mapping")
        print(f"  {'Src':>4}  {'Src Coord':>10}  ->  {'Dst Coord':>10}  {'Dst':>4}")
        print("  " + "-" * 40)
        for nc in node_configs[:10]:  # Show first 10
            src_coord = generator._node_id_to_coord(nc.src_node_id)
            dst_id = generator._coord_to_node_id(nc.dest_coord)
            dst_id_str = f"{dst_id:4d}" if dst_id >= 0 else " N/A"
            print(f"  {nc.src_node_id:4d}  {str(src_coord):>10}  ->  {str(nc.dest_coord):>10}  {dst_id_str}")
        if len(node_configs) > 10:
            print(f"  ... ({len(node_configs) - 10} more nodes)")

    # Start transfers
    if verbose:
        print_section("Simulation")
        print("  Starting all node transfers...")

    # Create metrics collector for detailed performance data
    collector = MetricsCollector(system, capture_interval=1)

    sim_start = time.perf_counter()
    system.start_all_transfers()

    # Run simulation with metrics collection
    max_cycles = 10000
    cycles = 0
    while not system.all_transfers_complete and cycles < max_cycles:
        system.process_cycle()
        collector.capture()
        cycles += 1

    sim_end = time.perf_counter()
    sim_time_ms = (sim_end - sim_start) * 1000

    if verbose:
        print(f"  Simulation completed in {cycles} cycles")
        print(f"  Wall-clock time: {sim_time_ms:.2f} ms")

    # === Collect Performance Metrics ===
    total_bytes = config.transfer_size * system.num_nodes
    throughput_bytes_per_cycle = total_bytes / cycles if cycles > 0 else 0

    # Router flit statistics
    flit_stats = system.get_flit_stats()
    total_flits = sum(flit_stats.values())
    active_routers = sum(1 for v in flit_stats.values() if v > 0)
    avg_flits_per_router = total_flits / len(flit_stats) if flit_stats else 0
    max_flits_router = max(flit_stats.values()) if flit_stats else 0

    # Buffer utilization from collector
    buffer_occupancies = [s.flits_in_flight for s in collector.snapshots]
    peak_buffer_util = max(buffer_occupancies) if buffer_occupancies else 0
    avg_buffer_util = sum(buffer_occupancies) / len(buffer_occupancies) if buffer_occupancies else 0
    # Average during active periods (when flits are in flight)
    active_occupancies = [x for x in buffer_occupancies if x > 0]
    active_avg_buffer = sum(active_occupancies) / len(active_occupancies) if active_occupancies else 0
    active_pct = len(active_occupancies) / len(buffer_occupancies) * 100 if buffer_occupancies else 0

    # Latency statistics from collector
    latencies = collector.get_all_latencies()
    if latencies:
        import statistics
        min_latency = min(latencies)
        max_latency = max(latencies)
        avg_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
    else:
        min_latency = max_latency = avg_latency = std_latency = 0
        # Fallback: estimate from cycles
        avg_latency = cycles / system.num_nodes if system.num_nodes > 0 else 0

    if verbose:
        print_section("Performance Metrics")
        print(f"  Total Cycles:      {cycles}")
        print(f"  Total Data:        {total_bytes} bytes")
        print(f"  Throughput:        {throughput_bytes_per_cycle:.2f} bytes/cycle")
        print(f"  Simulation Speed:  {cycles / sim_time_ms * 1000:.0f} cycles/sec")

        print_section("Router Statistics")
        print(f"  Total Flits:       {total_flits}")
        print(f"  Active Routers:    {active_routers}/{len(flit_stats)}")
        print(f"  Avg Flits/Router:  {avg_flits_per_router:.1f}")
        print(f"  Max Flits (router):{max_flits_router}")

        print_section("Latency Distribution")
        if latencies:
            print(f"  Samples:           {len(latencies)}")
            print(f"  Min Latency:       {min_latency} cycles")
            print(f"  Max Latency:       {max_latency} cycles")
            print(f"  Avg Latency:       {avg_latency:.2f} cycles")
            print(f"  Std Dev:           {std_latency:.2f} cycles")
        else:
            print(f"  Avg Latency:       {avg_latency:.2f} cycles (estimated)")

        print_section("Buffer Utilization")
        print(f"  Peak Occupancy:    {peak_buffer_util} flits")
        print(f"  Avg Occupancy:     {avg_buffer_util:.3f} flits")
        print(f"  Active Avg:        {active_avg_buffer:.2f} flits ({active_pct:.1f}% of cycles active)")

    # Golden Comparison / Verification
    if verbose:
        print_section("Golden Data Verification")

    # Verify transfers against golden data
    report = system.verify_transfers()
    pass_count = report.passed
    fail_count = report.failed

    if verbose:
        print(f"  Total Checks:      {report.total_checks}")
        print(f"  PASSED:            {pass_count}")
        print(f"  FAILED:            {fail_count}")
        if report.missing_golden > 0:
            print(f"  Missing Golden:    {report.missing_golden}")
        if report.missing_actual > 0:
            print(f"  Missing Actual:    {report.missing_actual}")
        print(f"  Pass Rate:         {report.pass_rate * 100:.1f}%")

        # Show failure details if any
        if not report.all_passed:
            print("\n  Failure Details:")
            shown = 0
            for result in report.results:
                if not result.passed and shown < 5:
                    if result.first_mismatch_offset >= 0:
                        print(f"    Node {result.node_id} @ 0x{result.local_addr:04X}: "
                              f"mismatch at offset {result.first_mismatch_offset}")
                    else:
                        print(f"    Node {result.node_id} @ 0x{result.local_addr:04X}: "
                              f"size mismatch (expected {len(result.expected)}, "
                              f"got {len(result.actual)})")
                    shown += 1
            if fail_count > 5:
                print(f"    ... ({fail_count - 5} more failures)")

    # Summary
    # Note: Until mesh routing is fully wired, actual != expected is expected
    # The verification infrastructure is working correctly
    success = report.all_passed
    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000

    if verbose:
        print_section("Summary")
        status = "PASS" if success else "FAIL"
        status_color = "\033[92m" if success else "\033[91m"  # Green/Red
        reset_color = "\033[0m"
        print(f"  Status:            {status_color}{status}{reset_color}")
        print(f"  Total Time:        {total_time_ms:.2f} ms")
        print("=" * 70)

    # Save metrics to latest.json for visualization
    metrics_dir = Path("output/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        'pattern': pattern,
        'cycles': cycles,
        'total_bytes': total_bytes,
        'throughput': throughput_bytes_per_cycle,
        'pass_count': pass_count,
        'fail_count': fail_count,
        'sim_time_ms': sim_time_ms,
        'total_time_ms': total_time_ms,
        'success': success,
        'timestamp': datetime.now().isoformat(),
        'mesh_cols': config.mesh_cols,
        'mesh_rows': config.mesh_rows,
        'num_nodes': system.num_nodes,
        'transfer_size': config.transfer_size,
        # Router statistics
        'total_flits': total_flits,
        'active_routers': active_routers,
        'avg_flits_per_router': avg_flits_per_router,
        'max_flits_router': max_flits_router,
        # Latency statistics
        'latency_samples': len(latencies),
        'min_latency': min_latency,
        'max_latency': max_latency,
        'avg_latency': avg_latency,
        'std_latency': std_latency,
        # Buffer utilization
        'peak_buffer_util': peak_buffer_util,
        'avg_buffer_util': avg_buffer_util,
    }
    
    # Save as latest.json
    import json
    latest_path = metrics_dir / "latest.json"
    with open(latest_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    if verbose:
        print(f"\n  Metrics saved: {latest_path}")
    
    return result


def run_all_patterns(
    memory_pattern: str = "sequential",
    payload_dir: str = None,
    verbose: bool = True
) -> dict:
    """Run all traffic patterns."""
    patterns = ['neighbor', 'shuffle', 'bit_reverse', 'random', 'transpose']
    results = {}

    if verbose:
        print_header("NoC-to-NoC Traffic Test Suite")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Running {len(patterns)} traffic patterns...")
        if payload_dir:
            print(f"Memory Source: {payload_dir} (files)")
        else:
            print(f"Memory Source: {memory_pattern} (dynamic)")

    for pattern in patterns:
        result = run_traffic_pattern(pattern, memory_pattern, payload_dir, verbose)
        results[pattern] = result

    # Print summary table
    if verbose:
        print_header("Summary: All Traffic Patterns")
        print(f"{'Pattern':12s} | {'Status':6s} | {'Cycles':>8s} | {'Throughput':>12s} | {'Latency':>10s}")
        print("-" * 70)
        for pattern, result in results.items():
            status = "PASS" if result['success'] else "FAIL"
            throughput = f"{result['throughput']:.2f} B/cyc"
            latency = f"{result['avg_latency']:.1f} cyc"
            print(f"{pattern:12s} | {status:6s} | {result['cycles']:8d} | {throughput:>12s} | {latency:>10s}")
        print("-" * 70)

        total_pass = sum(1 for r in results.values() if r['success'])
        total_fail = len(results) - total_pass
        print(f"Total: {total_pass} PASSED, {total_fail} FAILED")
        print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="NoC-to-NoC Traffic Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Traffic Patterns:
  neighbor    - Ring topology: node i sends to node (i+1) %% N
  shuffle     - Perfect shuffle: left rotate bits of node ID
  bit_reverse - Bit reversal: reverse bits of node ID
  random      - Random: each node sends to random destination
  transpose   - Transpose: swap (x,y) coordinates

Memory Patterns (for -m option):
  sequential     - Sequential bytes: (node_id * 16 + i) & 0xFF
  random         - Deterministic random data per node
  constant       - Constant value with node_id prefix
  node_id        - Fill with node_id value
  address        - Address-based pattern with node offset
  walking_ones   - Walking ones pattern (1 << (i %% 8))
  walking_zeros  - Walking zeros pattern
  checkerboard   - Alternating 0xAA/0x55 pattern

Payload Files (for -P option):
  Use 'make gen_noc_payload' to generate per-node binary files.
  Files: payload/node_00.bin, node_01.bin, ..., node_15.bin

Examples:
  py -3 run.py neighbor              Dynamic sequential data
  py -3 run.py neighbor -m random    Dynamic random data
  py -3 run.py neighbor -P payload/  Load from payload files
  py -3 run.py --all -P payload/     All patterns with payload files
"""
    )
    parser.add_argument('pattern', nargs='?', default='neighbor',
                        choices=['neighbor', 'shuffle', 'bit_reverse',
                                 'random', 'transpose'],
                        help='Traffic pattern to run (default: neighbor)')
    parser.add_argument('-m', '--memory-pattern', default='sequential',
                        choices=MEMORY_PATTERNS,
                        help='Memory pattern for dynamic generation (default: sequential)')
    parser.add_argument('-P', '--payload-dir',
                        help='Load memory from payload files (overrides -m)')
    parser.add_argument('--all', action='store_true',
                        help='Run all traffic patterns')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet mode (minimal output)')

    args = parser.parse_args()
    verbose = not args.quiet

    if args.all:
        results = run_all_patterns(args.memory_pattern, args.payload_dir, verbose)
        success = all(r['success'] for r in results.values())
    else:
        result = run_traffic_pattern(
            args.pattern, args.memory_pattern, args.payload_dir, verbose
        )
        success = result['success']

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
