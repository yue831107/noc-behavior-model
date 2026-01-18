#!/usr/bin/env python3
"""
NoC-to-NoC Traffic Test Runner.

Usage:
    py -3 run.py neighbor                      # Run neighbor with dynamic data
    py -3 run.py neighbor -m random            # With random memory data
    py -3 run.py neighbor -P payload/          # Load from payload files
    py -3 run.py --all                         # Run all traffic patterns
    py -3 run.py --all -P payload/             # All patterns with payload files
    py -3 run.py --all --mode axi              # Run with AXI Mode (5 Sub-Routers)

Memory Patterns (dynamic generation):
    sequential, random, constant, node_id, address, walking_ones, walking_zeros, checkerboard

Traffic Patterns:
    neighbor, shuffle, bit_reverse, random, transpose

Channel Modes:
    general   - 2 Sub-Routers (Req + Resp), default
    axi       - 5 Sub-Routers (AW + W + AR + B + R), no HoL blocking

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

from typing import List, Tuple

from src.core import NoCSystem
from src.core.router import ChannelMode
from src.config import load_noc_traffic_config, NoCTrafficConfig, TrafficPattern
from src.traffic import TrafficPatternGenerator
from src.visualization import MetricsCollector
from src.metrics import BookSimMetrics, SaturationStatus


def calculate_noc_zero_load_latency(
    node_configs: List,
    mesh_cols: int = 5,
    mesh_rows: int = 4
) -> float:
    """
    Calculate average zero-load latency (L0) for NoC-to-NoC traffic.

    For NoC-to-NoC, each node sends to its destination based on the traffic
    pattern. This calculates the average Manhattan distance + overhead.

    Args:
        node_configs: List of NodeTrafficConfig with src_node_id and dest_coord
        mesh_cols: Total mesh columns (default 5, with col 0 as edge)
        mesh_rows: Mesh rows (default 4)

    Returns:
        Average zero-load latency in cycles
    """
    from src.verification import TheoryValidator
    from src.verification.theory_validator import MeshConfig

    mesh_config = MeshConfig(cols=mesh_cols, rows=mesh_rows, edge_column=0)
    validator = TheoryValidator(mesh_config=mesh_config)

    # Compute columns (excluding edge column 0)
    compute_cols = mesh_cols - 1

    total_latency = 0.0
    count = 0

    for nc in node_configs:
        # Convert src_node_id to mesh coordinate
        # node_id = row * compute_cols + (col - 1) for col > 0
        src_col = (nc.src_node_id % compute_cols) + 1
        src_row = nc.src_node_id // compute_cols
        src_coord = (src_col, src_row)

        # dest_coord is already in mesh coordinates (col, row)
        dest_coord = nc.dest_coord

        # Calculate zero-load latency for this src-dest pair
        l0 = validator.calculate_zero_load_latency(src_coord, dest_coord)
        total_latency += l0
        count += 1

    return total_latency / count if count > 0 else 0.0


def print_booksim_metrics(
    metrics: BookSimMetrics,
    title: str = "Performance Metrics (BookSim2-Style)"
) -> None:
    """
    Print BookSim2-style performance metrics summary.

    Args:
        metrics: BookSimMetrics dataclass
        title: Header title
    """
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)
    print(f"Throughput:          {metrics.throughput:.4f} packets/cycle")
    print(f"Total Packets:       {metrics.total_packets}")
    print(f"Total Cycles:        {metrics.total_cycles}")
    print("-" * 70)
    print(f"Average Latency:     {metrics.avg_latency:.1f} cycles")
    print(f"Min Latency:         {metrics.min_latency:.1f} cycles")
    print(f"Max Latency:         {metrics.max_latency:.1f} cycles")
    print("-" * 70)
    print(f"Zero-Load Latency:   {metrics.zero_load_latency:.1f} cycles (L0)")
    print(f"Normalized Latency:  {metrics.saturation_factor:.2f} (L/L0)")
    print(f"Saturation Status:   {metrics.saturation_status.value.upper()}")
    if metrics.is_saturated:
        print("  WARNING: Network is SATURATED (L > 3*L0)")
    print("=" * 70)


def validate_performance_metrics(
    metrics: dict,
    zero_load_latency: float = None,
    verbose: bool = True
) -> bool:
    """
    Validate performance metrics using BookSim2-style checks.

    Validations:
    - Latency: L >= L0 (zero-load), saturation detection (L > 3*L0)
    - Buffer utilization: [0, 1] range
    - Flit Conservation: Sent = Received

    Args:
        metrics: Dict with performance metrics
        zero_load_latency: Theoretical minimum latency (L0). If None, skip latency validation.
        verbose: Print validation results

    Returns:
        True if all validations pass.
    """
    all_passed = True
    results = []

    # === Theory Validation (BookSim2-Style) ===
    try:
        from src.verification import TheoryValidator
        theory_validator = TheoryValidator()

        # 1. Latency validation with saturation detection
        if 'avg_latency' in metrics and zero_load_latency is not None:
            is_valid, msg, details = theory_validator.validate_latency(
                avg_latency=metrics['avg_latency'],
                zero_load_latency=zero_load_latency
            )
            results.append(('Latency', is_valid, msg))
            if not is_valid:
                all_passed = False

        # 2. Buffer utilization validation
        if 'buffer_utilization' in metrics and metrics['buffer_utilization'] is not None:
            is_valid, msg = theory_validator.validate_buffer_utilization(
                metrics['buffer_utilization']
            )
            results.append(('Buffer Util', is_valid, msg))
            if not is_valid:
                all_passed = False

    except ImportError:
        if verbose:
            print_section("Performance Validation")
            print("  [SKIP] TheoryValidator not available")

    # === Consistency Validation ===
    try:
        from src.verification import ConsistencyValidator
        consistency_validator = ConsistencyValidator(tolerance=0.50)

        # Flit Conservation: Sent = Received
        if all(k in metrics for k in ['total_flits_sent', 'total_flits_received']):
            is_valid, msg = consistency_validator.validate_flit_conservation(
                total_sent=metrics['total_flits_sent'],
                total_received=metrics['total_flits_received']
            )
            results.append(('Flit Conserv', is_valid, msg))
            if not is_valid:
                all_passed = False

    except ImportError:
        pass  # Consistency validator not available

    # Print results
    if verbose and results:
        print_section("Performance Validation (BookSim2-Style)")
        for name, is_valid, msg in results:
            status = "PASS" if is_valid else "FAIL"
            print(f"  {name + ':':<17} {status}")
            if not is_valid:
                print(f"                     {msg}")

        overall = "PASS" if all_passed else "FAIL"
        print(f"\n  Validation:        {overall}")

    return all_passed


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
    verbose: bool = True,
    channel_mode: ChannelMode = ChannelMode.GENERAL
) -> dict:
    """
    Run NoC-to-NoC traffic pattern test.

    Args:
        pattern: Traffic pattern name.
        memory_pattern: Memory initialization pattern (ignored if payload_dir set).
        payload_dir: Directory with node_XX.bin files (overrides memory_pattern).
        verbose: Enable verbose output.
        channel_mode: Physical channel mode (GENERAL or AXI).

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

    # Create system with specified channel mode
    system = NoCSystem(
        mesh_cols=config.mesh_cols,
        mesh_rows=config.mesh_rows,
        buffer_depth=8,
        memory_size=0x10000,
        channel_mode=channel_mode,
    )

    mode_name = "AXI" if channel_mode == ChannelMode.AXI else "General"

    if verbose:
        print_section("Configuration")
        print(f"  Pattern:       {config.pattern.value}")
        print(f"  Channel Mode:  {mode_name}")
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

    # === Connect MasterNI callbacks for BookSim2-style per-flit latency tracking ===
    # This matches Host-to-NoC's approach: track each flit's latency individually
    def on_flit_latency(latency: int) -> None:
        """Record per-flit latency when flit arrives at destination."""
        collector.add_latency_sample(latency)

    for coord, master_ni in system.mesh.nis.items():
        master_ni.set_flit_latency_callback(on_flit_latency)

    sim_start = time.perf_counter()
    system.start_all_transfers()

    # Run simulation with metrics collection
    # Note: Per-flit latency is now tracked via MasterNI callbacks (like Host-to-NoC)
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

    # === Performance Metrics (BookSim2-Style) ===
    # Calculate zero-load latency based on traffic pattern
    zero_load_latency = calculate_noc_zero_load_latency(
        node_configs, mesh_cols=config.mesh_cols, mesh_rows=config.mesh_rows
    )
    collector.set_zero_load_latency(zero_load_latency)

    # Get BookSim2-style metrics
    booksim_metrics = collector.get_booksim_metrics()

    if verbose:
        print_booksim_metrics(booksim_metrics)
        print(f"  Simulation Speed:  {cycles / sim_time_ms * 1000:.0f} cycles/sec")

        # Additional router statistics
        flit_stats = system.get_flit_stats()
        total_flits = sum(flit_stats.values())
        active_routers = sum(1 for v in flit_stats.values() if v > 0)
        print_section("Router Statistics")
        print(f"  Total Flits:       {total_flits}")
        print(f"  Active Routers:    {active_routers}/{len(flit_stats)}")

    # === Performance Validation ===
    buffer_stats = collector.get_buffer_stats(total_capacity=20 * 4 * 5)
    perf_metrics = {
        'throughput': booksim_metrics.throughput * 32,
        'buffer_utilization': buffer_stats['utilization'],
        'avg_latency': booksim_metrics.avg_latency,
        'avg_occupancy': buffer_stats['avg'],
    }
    validation_passed = validate_performance_metrics(
        perf_metrics,
        zero_load_latency=zero_load_latency,
        verbose=verbose
    )

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
    success = report.all_passed and validation_passed
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

    # Get router stats for result
    flit_stats = system.get_flit_stats()
    total_flits = sum(flit_stats.values())
    active_routers = sum(1 for v in flit_stats.values() if v > 0)

    result = {
        'pattern': pattern,
        'channel_mode': mode_name,
        'cycles': cycles,
        'total_bytes': config.transfer_size * system.num_nodes,
        'throughput': booksim_metrics.throughput * 32,  # bytes/cycle
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
        # Latency statistics (BookSim2-style)
        'latency_samples': booksim_metrics.total_packets,
        'min_latency': booksim_metrics.min_latency,
        'max_latency': booksim_metrics.max_latency,
        'avg_latency': booksim_metrics.avg_latency,
        'zero_load_latency': booksim_metrics.zero_load_latency,
        'saturation_factor': booksim_metrics.saturation_factor,
        'saturation_status': booksim_metrics.saturation_status.value,
        # Buffer utilization
        'peak_buffer_util': buffer_stats['peak'],
        'avg_buffer_util': buffer_stats['avg'],
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
    verbose: bool = True,
    channel_mode: ChannelMode = ChannelMode.GENERAL
) -> dict:
    """Run all traffic patterns."""
    patterns = ['neighbor', 'shuffle', 'bit_reverse', 'random', 'transpose']
    results = {}

    mode_name = "AXI" if channel_mode == ChannelMode.AXI else "General"

    if verbose:
        print_header(f"NoC-to-NoC Traffic Test Suite ({mode_name} Mode)")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Running {len(patterns)} traffic patterns...")
        print(f"Channel Mode: {mode_name}")
        if payload_dir:
            print(f"Memory Source: {payload_dir} (files)")
        else:
            print(f"Memory Source: {memory_pattern} (dynamic)")

    for pattern in patterns:
        result = run_traffic_pattern(
            pattern, memory_pattern, payload_dir, verbose, channel_mode
        )
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
  py -3 run.py neighbor --mode axi   Run with AXI Mode (5 Sub-Routers)
  py -3 run.py --all --mode axi      Compare all patterns in AXI Mode
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
    parser.add_argument('--mode', default='general',
                        choices=['general', 'axi'],
                        help='Physical channel mode (default: general)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet mode (minimal output)')

    args = parser.parse_args()
    verbose = not args.quiet

    # Parse channel mode
    channel_mode = ChannelMode.AXI if args.mode == 'axi' else ChannelMode.GENERAL

    if args.all:
        results = run_all_patterns(
            args.memory_pattern, args.payload_dir, verbose, channel_mode
        )
        success = all(r['success'] for r in results.values())
    else:
        result = run_traffic_pattern(
            args.pattern, args.memory_pattern, args.payload_dir, verbose,
            channel_mode
        )
        success = result['success']

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
