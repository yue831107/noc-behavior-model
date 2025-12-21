#!/usr/bin/env python3
"""
NoC Report Generator.
Generates visualization charts from simulation metrics.

Usage:
    py -3 -m src.visualization.report_generator all --from-metrics output/metrics/latest.json
    py -3 -m src.visualization.report_generator throughput --from-metrics output/metrics/latest.json
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path if run as script
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def demo_utilization(collector, save_dir: Path, show: bool = False):
    """Generate router utilization analysis."""
    from src.visualization import plot_router_comparison, plot_utilization_heatmap
    import matplotlib.pyplot as plt
    
    print("Generating router utilization analysis...")
    
    # 1. Bar Chart (Detailed Distribution)
    save_path_bar = save_dir / "router_utilization_bar.png"
    fig_bar = plot_router_comparison(collector, metric="flits", save_path=str(save_path_bar))
    print(f"  Saved: {save_path_bar}")

    # 2. Heatmap (Spatial Distribution)
    save_path_heatmap = save_dir / "router_utilization_heatmap.png"
    fig_heatmap = plot_utilization_heatmap(collector, save_path=str(save_path_heatmap))
    print(f"  Saved: {save_path_heatmap}")
    
    if show:
        plt.show()
    else:
        plt.close(fig_bar)
        plt.close(fig_heatmap)


def demo_throughput(collector, save_dir: Path, show: bool = False):
    """Generate throughput curve."""
    from src.visualization import plot_throughput_curve, ThroughputConfig
    import matplotlib.pyplot as plt
    
    print("Generating throughput curve...")
    config = ThroughputConfig(title="NoC Throughput Over Time")
    
    save_path = save_dir / "throughput_curve.png"
    fig = plot_throughput_curve(collector, config, save_path=str(save_path))
    print(f"  Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def demo_curves(collector, save_dir: Path, show: bool = False):
    """Generate additional curve charts."""
    from src.visualization.charts import (
        plot_flit_count_curve,
        plot_buffer_utilization_curve,
        plot_transaction_progress,
    )
    import matplotlib.pyplot as plt
    
    print("Generating additional curves...")
    
    # Flit count
    save_path = save_dir / "flit_count_curve.png"
    fig = plot_flit_count_curve(collector, save_path=str(save_path))
    print(f"  Saved: {save_path}")
    if not show:
        plt.close(fig)
    
    # Buffer utilization
    save_path = save_dir / "buffer_utilization_curve.png"
    fig = plot_buffer_utilization_curve(collector, save_path=str(save_path))
    print(f"  Saved: {save_path}")
    if not show:
        plt.close(fig)
    
    # Transaction progress
    save_path = save_dir / "transaction_progress.png"
    fig = plot_transaction_progress(collector, save_path=str(save_path))
    print(f"  Saved: {save_path}")
    if not show:
        plt.close(fig)
    
    if show:
        plt.show()


def demo_dashboard(collector, save_dir: Path, show: bool = False):
    """Generate combined dashboard."""
    from src.visualization.charts import plot_combined_dashboard
    import matplotlib.pyplot as plt
    
    print("Generating combined dashboard...")
    
    save_path = save_dir / "dashboard.png"
    fig = plot_combined_dashboard(collector, save_path=str(save_path))
    print(f"  Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def demo_save_metrics(collector, save_dir: Path):
    """Save metrics to files."""
    from src.visualization import MetricsStore
    
    print("Saving metrics...")
    
    # Create metrics store
    metrics_dir = save_dir.parent / "metrics"
    store = MetricsStore(base_dir=metrics_dir)
    
    # Save in all formats
    json_path = store.save_json(collector, "report")
    print(f"  JSON: {json_path}")
    
    csv_path = store.save_csv(collector, "report")
    print(f"  CSV:  {csv_path}")
    
    npz_path = store.save_npz(collector, "report")
    print(f"  NPZ:  {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="NoC Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'command',
        choices=['heatmap', 'latency', 'throughput', 'curves', 'dashboard', 'all', 'save'],
        default='all',
        nargs='?',
        help='Visualization to generate (default: all)',
    )
    parser.add_argument(
        '--save-dir', '-o',
        default='output/charts',
        help='Output directory (default: output/charts)',
    )
    parser.add_argument(
        '--show', '-s',
        action='store_true',
        help='Show plots interactively',
    )
    parser.add_argument(
        '--from-metrics', '-f',
        required=True,
        help='Load data from metrics JSON file (e.g., output/metrics/latest.json)',
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(" NoC Report Generator")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Source:    {args.from_metrics}")
    print(f"Output:    {save_dir}")
    print()
    
    # Load metrics
    import json
    metrics_path = Path(args.from_metrics)
    if not metrics_path.exists():
        print(f"Error: Metrics file not found: {metrics_path}")
        return 1
        
    with open(metrics_path) as f:
        saved_metrics = json.load(f)

    # Create collector structure from saved metrics
    from src.core import NoCSystem
    from src.visualization import MetricsCollector
    
    # Use mesh size from metrics if available, else default 5x4
    mesh_cols = saved_metrics.get('mesh_cols', 5)
    mesh_rows = saved_metrics.get('mesh_rows', 4)
    
    system = NoCSystem(
        mesh_cols=mesh_cols,
        mesh_rows=mesh_rows,
    )
    collector = MetricsCollector(system)
    
    if 'snapshots' in saved_metrics and saved_metrics['snapshots']:
        print(f"Loading {len(saved_metrics['snapshots'])} snapshots from metrics file...")
        loaded_collector = MetricsCollector.from_dict(saved_metrics)
        collector.snapshots = loaded_collector.snapshots
    else:
        print("Error: No snapshots found in metrics file. Cannot generate reports.")
        return 1
    
    # Generate requested visualizations
    if args.command in ('utilization', 'all'):
        demo_utilization(collector, save_dir, args.show)
    
    if args.command in ('throughput', 'all'):
        demo_throughput(collector, save_dir, args.show)
    
    if args.command in ('curves', 'all'):
        demo_curves(collector, save_dir, args.show)
    
    if args.command in ('dashboard', 'all'):
        demo_dashboard(collector, save_dir, args.show)
    
    if args.command in ('save', 'all'):
        demo_save_metrics(collector, save_dir)
    
    print()
    print("=" * 60)
    print(" Done!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
