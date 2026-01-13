#!/usr/bin/env python3
"""
Parameter Sweep for NoC Performance Tuning.

Performs systematic parameter sweeps on a fixed 5x4 mesh to find optimal
hardware configurations that meet specified performance targets.

Usage:
    py -3 tools/param_sweep.py                    # Run default sweep
    py -3 tools/param_sweep.py --mode host_to_noc # Host-to-NoC only
    py -3 tools/param_sweep.py --mode noc_to_noc  # NoC-to-NoC only
    py -3 tools/param_sweep.py -o output/sweep    # Custom output directory
"""

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Fix for matplotlib home directory issue on Windows
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ.get('USERPROFILE', 'C:\\Users\\Default')
if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = os.environ.get('TEMP', 'C:\\Temp')

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import NoCTrafficConfig, TrafficPattern
from src.core.routing_selector import V1System, NoCSystem
from src.visualization import MetricsCollector
from src.traffic.pattern_generator import TrafficPatternGenerator


# =============================================================================
# Performance Targets
# =============================================================================

@dataclass
class PerformanceTarget:
    """Performance target specification."""
    name: str
    # Throughput targets
    min_throughput: float  # B/cycle
    # Latency targets
    max_avg_latency: float  # cycles
    max_latency: Optional[int] = None  # cycles (absolute max)
    # Buffer targets
    max_buffer_utilization: float = 0.80  # ratio


# Default performance targets for 5x4 mesh
# Note: Host-to-NoC is limited by SlaveNI bottleneck (8-16 B/cycle)
HOST_TO_NOC_TARGET = PerformanceTarget(
    name="Host-to-NoC",
    min_throughput=6.0,       # 75% of T_max (8 B/cycle unidirectional)
    max_avg_latency=150.0,    # cycles
    max_latency=1000,         # absolute max
    max_buffer_utilization=0.70,
)

NOC_TO_NOC_TARGET = PerformanceTarget(
    name="NoC-to-NoC",
    min_throughput=80.0,      # B/cycle (higher due to parallel transfers)
    max_avg_latency=80.0,     # cycles (higher for transpose pattern)
    max_latency=200,          # absolute max
    max_buffer_utilization=0.70,
)


# =============================================================================
# Parameter Space
# =============================================================================

@dataclass
class HardwareParams:
    """Hardware parameters to sweep."""
    buffer_depth: int = 4
    max_outstanding: int = 16
    # Fixed parameters
    mesh_cols: int = 5
    mesh_rows: int = 4


# Default sweep ranges
BUFFER_DEPTH_RANGE = [2, 4, 8, 16]
MAX_OUTSTANDING_RANGE = [4, 8, 16, 32]


# =============================================================================
# Results
# =============================================================================

@dataclass
class SweepResult:
    """Single sweep result."""
    params: HardwareParams
    # Metrics
    throughput: float = 0.0
    latency_min: int = 0
    latency_max: int = 0
    latency_avg: float = 0.0
    latency_std: float = 0.0
    buffer_peak: int = 0
    buffer_avg: float = 0.0
    buffer_utilization: float = 0.0
    # Simulation info
    cycles: int = 0
    wall_time_ms: float = 0.0
    # Target evaluation
    meets_throughput: bool = False
    meets_latency: bool = False
    meets_buffer: bool = False
    meets_all: bool = False
    # Score (for ranking)
    score: float = 0.0


@dataclass
class SweepReport:
    """Complete sweep report."""
    mode: str
    target: PerformanceTarget
    total_configs: int
    configs_meeting_target: int
    best_config: Optional[SweepResult] = None
    all_results: List[SweepResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0


# =============================================================================
# Sweep Functions
# =============================================================================

def calculate_score(result: SweepResult, target: PerformanceTarget) -> float:
    """
    Calculate a composite score for ranking configurations.

    Higher is better. Penalizes failing to meet targets.
    """
    score = 0.0

    # Throughput contribution (40% weight)
    if result.throughput >= target.min_throughput:
        # Bonus for exceeding target
        excess = (result.throughput - target.min_throughput) / target.min_throughput
        score += 40 * (1 + min(excess, 0.5))  # Cap bonus at 50%
    else:
        # Penalty for missing target
        deficit = (target.min_throughput - result.throughput) / target.min_throughput
        score += 40 * (1 - deficit)

    # Latency contribution (40% weight, lower is better)
    if result.latency_avg <= target.max_avg_latency:
        # Bonus for beating target
        margin = (target.max_avg_latency - result.latency_avg) / target.max_avg_latency
        score += 40 * (1 + min(margin, 0.5))
    else:
        # Penalty for missing target
        excess = (result.latency_avg - target.max_avg_latency) / target.max_avg_latency
        score += 40 * max(0, 1 - excess)

    # Buffer utilization contribution (20% weight, lower is better)
    if result.buffer_utilization <= target.max_buffer_utilization:
        margin = (target.max_buffer_utilization - result.buffer_utilization) / target.max_buffer_utilization
        score += 20 * (1 + min(margin, 0.5))
    else:
        excess = (result.buffer_utilization - target.max_buffer_utilization) / target.max_buffer_utilization
        score += 20 * max(0, 1 - excess)

    return score


def run_host_to_noc_sweep(
    params: HardwareParams,
    config_path: Optional[Path] = None,
    transfer_size: int = 256,  # Smaller for faster sweep
    num_targets: int = 16,
) -> SweepResult:
    """
    Run single Host-to-NoC configuration.

    Args:
        params: Hardware parameters (buffer_depth, max_outstanding)
        config_path: Optional path to transfer config YAML file
        transfer_size: Transfer size per target (used if no config_path)
        num_targets: Number of target nodes (used if no config_path)
    """
    from src.config import load_transfer_configs, TransferMode
    from src.core import HostMemory, Memory
    from src.testbench import HostAXIMaster

    result = SweepResult(params=params)

    try:
        start_time = time.perf_counter()
        total_bytes = 0
        cycle = 0

        if config_path and config_path.exists():
            # === Use external config file ===
            configs = load_transfer_configs(config_path)

            # Create host memory
            host_memory = HostMemory(size=2 * 1024 * 1024)  # 2MB

            # Create system with specified parameters and host memory
            system = V1System(
                mesh_cols=params.mesh_cols,
                mesh_rows=params.mesh_rows,
                buffer_depth=params.buffer_depth,
                max_outstanding=params.max_outstanding,
                host_memory=host_memory,
            )
            collector = MetricsCollector(system, capture_interval=1)

            # For gather (read) mode, pre-write data to node memories
            # For write mode, write data to host memory
            for cfg in configs:
                test_data = bytes(range(256)) * ((cfg.src_size // 256) + 1)
                test_data = test_data[:cfg.src_size]

                if cfg.is_read:
                    # Pre-write golden data to node local memories for gather
                    target_nodes = cfg.get_target_node_list(total_nodes=16)
                    read_addr = cfg.read_src_addr if hasattr(cfg, 'read_src_addr') and cfg.read_src_addr > 0 else cfg.dst_addr
                    portion_size = cfg.src_size // len(target_nodes) if target_nodes else 0
                    for i, node_id in enumerate(target_nodes):
                        for coord, ni in system.mesh.nis.items():
                            if ni.node_id == node_id:
                                start_offset = i * portion_size
                                end_offset = start_offset + portion_size
                                node_data = test_data[start_offset:end_offset]
                                ni.local_memory.write(read_addr, node_data)
                                break
                else:
                    # Write mode: fill host memory
                    host_memory.write(cfg.src_addr, test_data)

                total_bytes += cfg.src_size

            # Run transfers sequentially
            max_cycles = 200000
            for cfg in configs:
                system.configure_transfer(cfg)
                system.start_transfer()

                # Run until this transfer completes
                while not system.transfer_complete and cycle < max_cycles:
                    system.process_cycle()
                    collector.capture()
                    cycle += 1

        else:
            # === Use internal generated test (original behavior) ===
            # Create system with specified parameters
            system = V1System(
                mesh_cols=params.mesh_cols,
                mesh_rows=params.mesh_rows,
                buffer_depth=params.buffer_depth,
                max_outstanding=params.max_outstanding,
            )
            collector = MetricsCollector(system, capture_interval=1)

            # Generate test data
            test_data = bytes(range(256)) * (transfer_size // 256 + 1)
            test_data = test_data[:transfer_size]

            # Submit writes to all targets
            target_nodes = list(range(num_targets))
            axi_id = 1
            outstanding = {}
            completed = 0
            max_cycles = 50000

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

            total_bytes = transfer_size * len(target_nodes)

        end_time = time.perf_counter()

        # Collect metrics
        result.cycles = cycle
        result.wall_time_ms = (end_time - start_time) * 1000

        # Calculate throughput
        result.throughput = total_bytes / cycle if cycle > 0 else 0.0

        latency_stats = collector.get_latency_stats()
        result.latency_min = latency_stats['min']
        result.latency_max = latency_stats['max']
        result.latency_avg = latency_stats['avg']
        result.latency_std = latency_stats['std']

        # Buffer stats
        total_capacity = params.mesh_cols * params.mesh_rows * params.buffer_depth * 5
        buffer_stats = collector.get_buffer_stats(total_capacity=total_capacity)
        result.buffer_peak = buffer_stats['peak']
        result.buffer_avg = buffer_stats['avg']
        result.buffer_utilization = buffer_stats['utilization']

    except (ImportError, RuntimeError, ValueError, KeyError) as e:
        print(f"  Error: {e}")

    return result


def run_noc_to_noc_sweep(
    params: HardwareParams,
    transfer_size: int = 1024,
    pattern: str = "transpose",
) -> SweepResult:
    """Run single NoC-to-NoC configuration."""
    result = SweepResult(params=params)

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
            mesh_cols=params.mesh_cols,
            mesh_rows=params.mesh_rows,
            transfer_size=transfer_size,
            pattern=pattern_map[pattern],
            local_src_addr=0x0000,
            local_dst_addr=0x1000,
        )

        # Create system
        system = NoCSystem(
            mesh_cols=params.mesh_cols,
            mesh_rows=params.mesh_rows,
            buffer_depth=params.buffer_depth,
            memory_size=0x10000,
        )
        collector = MetricsCollector(system, capture_interval=1)

        # Configure traffic
        system.configure_traffic(traffic_config)
        system.initialize_node_memory(pattern="sequential")
        system.generate_golden()

        # Generate traffic mapping
        generator = TrafficPatternGenerator(params.mesh_cols, params.mesh_rows)
        node_configs = generator.generate(traffic_config)
        traffic_config.node_configs = node_configs

        start_time = time.perf_counter()
        system.start_all_transfers()

        # Record injection for all nodes
        for node_id in system.node_controllers.keys():
            collector.record_injection(node_id, cycle=0)

        completed_nodes = set()
        max_cycles = 10000
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

        # Buffer stats
        total_capacity = params.mesh_cols * params.mesh_rows * params.buffer_depth * 5
        buffer_stats = collector.get_buffer_stats(total_capacity=total_capacity)
        result.buffer_peak = buffer_stats['peak']
        result.buffer_avg = buffer_stats['avg']
        result.buffer_utilization = buffer_stats['utilization']

    except (ImportError, RuntimeError, ValueError, KeyError) as e:
        print(f"  Error: {e}")

    return result


def evaluate_result(result: SweepResult, target: PerformanceTarget) -> SweepResult:
    """Evaluate result against performance target."""
    result.meets_throughput = result.throughput >= target.min_throughput
    result.meets_latency = result.latency_avg <= target.max_avg_latency
    if target.max_latency:
        result.meets_latency = result.meets_latency and result.latency_max <= target.max_latency
    result.meets_buffer = result.buffer_utilization <= target.max_buffer_utilization
    result.meets_all = result.meets_throughput and result.meets_latency and result.meets_buffer
    result.score = calculate_score(result, target)
    return result


def run_parameter_sweep(
    mode: str,
    buffer_depths: List[int] = BUFFER_DEPTH_RANGE,
    max_outstandings: List[int] = MAX_OUTSTANDING_RANGE,
    config_path: Optional[Path] = None,
    verbose: bool = True,
) -> SweepReport:
    """
    Run parameter sweep for specified mode.

    Args:
        mode: 'host_to_noc' or 'noc_to_noc'
        buffer_depths: List of buffer depth values to sweep
        max_outstandings: List of max outstanding values to sweep
        config_path: Optional path to transfer config YAML (for host_to_noc)
        verbose: Print progress
    """
    target = HOST_TO_NOC_TARGET if mode == "host_to_noc" else NOC_TO_NOC_TARGET

    # Generate all parameter combinations
    all_configs = list(itertools.product(buffer_depths, max_outstandings))

    report = SweepReport(
        mode=mode,
        target=target,
        total_configs=len(all_configs),
        configs_meeting_target=0,
        start_time=datetime.now().isoformat(),
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Parameter Sweep: {mode.upper()}")
        print(f"{'='*60}")
        if config_path:
            print(f"Config: {config_path}")
        print(f"Performance Target:")
        print(f"  Throughput:  >= {target.min_throughput:.1f} B/cycle")
        print(f"  Latency:     <= {target.max_avg_latency:.1f} cycles (avg)")
        if target.max_latency:
            print(f"               <= {target.max_latency} cycles (max)")
        print(f"  Buffer Util: <= {target.max_buffer_utilization:.0%}")
        print(f"\nSweeping {len(all_configs)} configurations...")
        print(f"  buffer_depth:    {buffer_depths}")
        print(f"  max_outstanding: {max_outstandings}")
        print()

    results = []

    for i, (buf_depth, max_out) in enumerate(all_configs):
        params = HardwareParams(
            buffer_depth=buf_depth,
            max_outstanding=max_out,
        )

        if verbose:
            print(f"  [{i+1}/{len(all_configs)}] buffer_depth={buf_depth}, max_outstanding={max_out}", end="")

        # Run test
        if mode == "host_to_noc":
            result = run_host_to_noc_sweep(params, config_path=config_path)
        else:
            result = run_noc_to_noc_sweep(params)

        # Evaluate against target
        result = evaluate_result(result, target)
        results.append(result)

        if result.meets_all:
            report.configs_meeting_target += 1

        if verbose:
            status = "PASS" if result.meets_all else "FAIL"
            print(f" -> T={result.throughput:.1f}, L={result.latency_avg:.1f}, B={result.buffer_utilization:.1%} [{status}]")

    # Find best configuration
    if results:
        results.sort(key=lambda r: r.score, reverse=True)
        report.best_config = results[0]
        report.all_results = results

    report.end_time = datetime.now().isoformat()
    report.duration_seconds = (
        datetime.fromisoformat(report.end_time) -
        datetime.fromisoformat(report.start_time)
    ).total_seconds()

    return report


def print_report(report: SweepReport) -> None:
    """Print sweep report."""
    print(f"\n{'='*60}")
    print(f"Sweep Report: {report.mode.upper()}")
    print(f"{'='*60}")
    print(f"Total Configurations:     {report.total_configs}")
    print(f"Configurations Meeting Target: {report.configs_meeting_target}")
    print(f"Duration: {report.duration_seconds:.1f} seconds")

    if report.best_config:
        best = report.best_config
        print(f"\nBest Configuration (Score: {best.score:.1f}):")
        print(f"  buffer_depth:    {best.params.buffer_depth}")
        print(f"  max_outstanding: {best.params.max_outstanding}")
        print(f"\n  Performance:")
        print(f"    Throughput:    {best.throughput:.2f} B/cycle", end="")
        print(f" [{'OK' if best.meets_throughput else 'NG'}]")
        print(f"    Latency (avg): {best.latency_avg:.1f} cycles", end="")
        print(f" [{'OK' if best.meets_latency else 'NG'}]")
        print(f"    Latency (max): {best.latency_max} cycles")
        print(f"    Buffer Util:   {best.buffer_utilization:.1%}", end="")
        print(f" [{'OK' if best.meets_buffer else 'NG'}]")
        print(f"\n  Target: {'MET' if best.meets_all else 'NOT MET'}")

    # Show top 5 configurations
    print(f"\nTop 5 Configurations:")
    print(f"  {'Rank':<5} {'Buf':<5} {'Out':<5} {'T(B/c)':<8} {'L(avg)':<8} {'B(%)':<6} {'Score':<7} {'Status'}")
    print(f"  {'-'*55}")
    for i, r in enumerate(report.all_results[:5]):
        status = "MET" if r.meets_all else "---"
        print(f"  {i+1:<5} {r.params.buffer_depth:<5} {r.params.max_outstanding:<5} "
              f"{r.throughput:<8.2f} {r.latency_avg:<8.1f} {r.buffer_utilization*100:<6.1f} "
              f"{r.score:<7.1f} {status}")


def save_report(report: SweepReport, output_dir: Path) -> None:
    """Save sweep report to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = {
        "mode": report.mode,
        "target": {
            "name": report.target.name,
            "min_throughput": report.target.min_throughput,
            "max_avg_latency": report.target.max_avg_latency,
            "max_latency": report.target.max_latency,
            "max_buffer_utilization": report.target.max_buffer_utilization,
        },
        "total_configs": report.total_configs,
        "configs_meeting_target": report.configs_meeting_target,
        "start_time": report.start_time,
        "end_time": report.end_time,
        "duration_seconds": report.duration_seconds,
        "best_config": None,
        "all_results": [],
    }

    if report.best_config:
        data["best_config"] = {
            "buffer_depth": report.best_config.params.buffer_depth,
            "max_outstanding": report.best_config.params.max_outstanding,
            "throughput": report.best_config.throughput,
            "latency_avg": report.best_config.latency_avg,
            "latency_max": report.best_config.latency_max,
            "buffer_utilization": report.best_config.buffer_utilization,
            "score": report.best_config.score,
            "meets_all": report.best_config.meets_all,
        }

    for r in report.all_results:
        data["all_results"].append({
            "buffer_depth": r.params.buffer_depth,
            "max_outstanding": r.params.max_outstanding,
            "throughput": r.throughput,
            "latency_avg": r.latency_avg,
            "latency_max": r.latency_max,
            "latency_min": r.latency_min,
            "buffer_utilization": r.buffer_utilization,
            "cycles": r.cycles,
            "score": r.score,
            "meets_throughput": r.meets_throughput,
            "meets_latency": r.meets_latency,
            "meets_buffer": r.meets_buffer,
            "meets_all": r.meets_all,
        })

    output_path = output_dir / f"sweep_{report.mode}.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nReport saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Parameter sweep for NoC performance tuning'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['host_to_noc', 'noc_to_noc', 'both'],
        default='both',
        help='Sweep mode (default: both)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('output/param_sweep'),
        help='Output directory (default: output/param_sweep)'
    )
    parser.add_argument(
        '--buffer-depths',
        type=int,
        nargs='+',
        default=BUFFER_DEPTH_RANGE,
        help=f'Buffer depth values to sweep (default: {BUFFER_DEPTH_RANGE})'
    )
    parser.add_argument(
        '--max-outstandings',
        type=int,
        nargs='+',
        default=MAX_OUTSTANDING_RANGE,
        help=f'Max outstanding values to sweep (default: {MAX_OUTSTANDING_RANGE})'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode (less output)'
    )
    parser.add_argument(
        '-c', '--config',
        type=Path,
        default=None,
        help='Transfer config YAML file for host_to_noc mode (default: use internal test)'
    )

    args = parser.parse_args()

    print("="*60)
    print("NoC Parameter Sweep")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Mesh: 5x4 (fixed)")
    if args.config:
        print(f"Config: {args.config}")
    print(f"Output: {args.output}")

    reports = []

    if args.mode in ['host_to_noc', 'both']:
        report = run_parameter_sweep(
            'host_to_noc',
            buffer_depths=args.buffer_depths,
            max_outstandings=args.max_outstandings,
            config_path=args.config,
            verbose=not args.quiet,
        )
        print_report(report)
        save_report(report, args.output)
        reports.append(report)

    if args.mode in ['noc_to_noc', 'both']:
        report = run_parameter_sweep(
            'noc_to_noc',
            buffer_depths=args.buffer_depths,
            max_outstandings=args.max_outstandings,
            verbose=not args.quiet,
        )
        print_report(report)
        save_report(report, args.output)
        reports.append(report)

    # Summary
    if len(reports) == 2:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        for r in reports:
            status = "MET" if r.best_config and r.best_config.meets_all else "NOT MET"
            print(f"  {r.mode}: {r.configs_meeting_target}/{r.total_configs} configs meet target [{status}]")

    return 0


if __name__ == '__main__':
    sys.exit(main())
