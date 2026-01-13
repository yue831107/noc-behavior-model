#!/usr/bin/env python3
"""
Regression Test Runner.

Searches hardware parameter space to find optimal configurations
that meet specified latency and throughput targets.

Usage:
    python tools/run_regression.py --config tools/regression_config.yaml
    python tools/run_regression.py --config tools/regression_config.yaml --early-stop
"""

import argparse
import os
import sys
from pathlib import Path

# Fix for matplotlib home directory issue on Windows
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ.get('USERPROFILE', 'C:\\Users\\Default')
if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = os.environ.get('TEMP', 'C:\\Temp')

import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tests.performance.regression import (
    PerformanceTarget,
    ParameterSpace,
    ParameterOptimizer,
    GridSearch,
    RegressionReport,
)


def load_config(config_path: Path) -> dict:
    """Load regression config YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_target(config: dict) -> PerformanceTarget:
    """Create PerformanceTarget from config."""
    target_cfg = config.get('target', {})

    return PerformanceTarget(
        max_latency=target_cfg.get('max_latency', 20),
        min_throughput=target_cfg.get('min_throughput', 16),
        max_buffer_utilization=target_cfg.get('max_buffer_utilization', 0.9),
        latency_weight=target_cfg.get('latency_weight', 0.5),
        throughput_weight=target_cfg.get('throughput_weight', 0.5),
    )


def create_parameter_space(config: dict) -> ParameterSpace:
    """Create ParameterSpace from config."""
    space_cfg = config.get('parameter_space', {})

    return ParameterSpace(
        mesh_rows=space_cfg.get('mesh_rows', [2, 4, 6, 8]),
        mesh_cols=space_cfg.get('mesh_cols', [3, 5, 7, 9]),
        buffer_depth=space_cfg.get('buffer_depth', [4, 8, 16, 32]),
        max_outstanding=space_cfg.get('max_outstanding', [8, 16, 32]),
        flit_width_bytes=space_cfg.get('flit_width_bytes', 8),
    )


def main():
    parser = argparse.ArgumentParser(
        description='Run regression test to find optimal hardware parameters'
    )
    parser.add_argument(
        '--config', '-c',
        type=Path,
        required=True,
        help='Regression config YAML file'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('output/regression'),
        help='Output directory (default: output/regression)'
    )
    parser.add_argument(
        '--early-stop',
        action='store_true',
        help='Stop when first satisfying solution is found'
    )
    parser.add_argument(
        '--system-type',
        choices=['host_to_noc', 'noc_to_noc'],
        default='host_to_noc',
        help='System type (default: host_to_noc)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode'
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate visualization plots (requires matplotlib)'
    )

    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"Error: Config not found: {args.config}")
        return 1

    config = load_config(args.config)

    # Override system type if specified in config
    system_type = config.get('system_type', args.system_type)

    # Create components
    target = create_target(config)
    space = create_parameter_space(config)
    strategy = GridSearch()

    # Create workload from config
    workload_cfg = config.get('workload', {})
    workload = {
        'transfer_size': workload_cfg.get('transfer_size', 1024),
        'num_transfers': workload_cfg.get('num_transfers', 50),
    }

    verbose = not args.quiet

    # Print header
    if verbose:
        print("=" * 60)
        print("Regression Test - Hardware Parameter Optimization")
        print("=" * 60)
        print(f"Config:       {args.config}")
        print(f"System type:  {system_type}")
        print(f"Output:       {args.output}")
        print()
        print("Target:")
        print(f"  Max latency:    {target.max_latency} cycles")
        print(f"  Min throughput: {target.min_throughput} B/cycle")
        print(f"  Weights:        latency={target.latency_weight}, "
              f"throughput={target.throughput_weight}")
        print()
        print("Parameter Space:")
        print(f"  Mesh rows:      {space.mesh_rows}")
        print(f"  Mesh cols:      {space.mesh_cols}")
        print(f"  Buffer depth:   {space.buffer_depth}")
        print(f"  Max outstanding:{space.max_outstanding}")
        print(f"  Total:          {space.total_combinations()} combinations")
        print()
        print("Workload:")
        print(f"  Transfer size:  {workload['transfer_size']} bytes")
        print(f"  Num transfers:  {workload['num_transfers']}")
        print("=" * 60)
        print()

    # Create optimizer
    optimizer = ParameterOptimizer(
        target=target,
        parameter_space=space,
        strategy=strategy,
        system_type=system_type,
        verbose=verbose,
    )

    # Run optimization
    result = optimizer.optimize(
        workload=workload,
        early_stop=args.early_stop,
    )

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Generate report
    report = RegressionReport(result)

    if verbose:
        print()
        print(report.generate_summary())
        print()
        print(report.generate_top_n_table())

    # Save full report
    report.save_full_report(args.output)

    # Generate plots if requested
    if args.plots:
        if verbose:
            print("\nGenerating plots...")

        try:
            report.plot_score_distribution(
                save_path=args.output / "score_distribution.png"
            )
            report.plot_latency_vs_throughput(
                save_path=args.output / "latency_vs_throughput.png"
            )
        except (ImportError, OSError, ValueError) as e:
            print(f"Warning: Could not generate plots: {e}")

    # Print final summary
    if verbose:
        print()
        print("=" * 60)
        if result.target_satisfied:
            print("Result: TARGET SATISFIED")
            print(f"Best configuration found with score {result.best_score:.4f}")
        else:
            print("Result: TARGET NOT SATISFIED")
            print(f"Best effort configuration has score {result.best_score:.4f}")
        print(f"\nOutput saved to: {args.output.absolute()}")
        print("=" * 60)

    return 0 if result.target_satisfied else 1


if __name__ == '__main__':
    sys.exit(main())
