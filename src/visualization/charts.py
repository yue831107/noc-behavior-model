"""
Additional Charts for NoC Visualization.

Provides more curve charts for detailed analysis:
- Flit count over time
- Buffer utilization over time
- Transaction progress
- Per-router comparison
- Port utilization heatmap
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .metrics_collector import MetricsCollector


@dataclass
class ChartConfig:
    """Common configuration for charts."""
    figsize: Tuple[int, int] = (12, 5)
    line_width: float = 1.5
    show_grid: bool = True
    grid_alpha: float = 0.3


def plot_flit_count_curve(
    collector: "MetricsCollector",
    config: Optional[ChartConfig] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot flit count over time.
    
    Args:
        collector: MetricsCollector with captured data.
        config: Chart configuration.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    if config is None:
        config = ChartConfig()
    
    cycles, flit_counts = collector.get_flit_count_over_time()
    
    if not cycles:
        fig, ax = plt.subplots(figsize=config.figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title("Flit Count Over Time")
        return fig
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    ax.plot(cycles, flit_counts, color='steelblue', linewidth=config.line_width)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Total Flits Forwarded")
    ax.set_title("Flit Count Over Time", fontsize=14, fontweight='bold')
    
    if config.show_grid:
        ax.grid(True, alpha=config.grid_alpha)
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_buffer_utilization_curve(
    collector: "MetricsCollector",
    config: Optional[ChartConfig] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot buffer utilization (total flits in buffers) over time.
    
    Args:
        collector: MetricsCollector with captured data.
        config: Chart configuration.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    if config is None:
        config = ChartConfig()
    
    cycles, occupancies = collector.get_total_buffer_occupancy_over_time()
    
    if not cycles:
        fig, ax = plt.subplots(figsize=config.figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title("Buffer Utilization Over Time")
        return fig
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    ax.fill_between(cycles, occupancies, alpha=0.3, color='steelblue')
    ax.plot(cycles, occupancies, color='steelblue', linewidth=config.line_width)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Total Buffer Occupancy (flits)")
    ax.set_title("Buffer Utilization Over Time", fontsize=14, fontweight='bold')
    
    if config.show_grid:
        ax.grid(True, alpha=config.grid_alpha)
    
    # Add average line
    if occupancies:
        avg = np.mean(occupancies)
        ax.axhline(y=avg, color='red', linestyle='--', linewidth=1,
                   label=f'Average: {avg:.1f}')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_transaction_progress(
    collector: "MetricsCollector",
    total_expected: int = None,
    config: Optional[ChartConfig] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot transaction completion progress over time.
    
    Args:
        collector: MetricsCollector with captured data.
        total_expected: Total expected transactions for percentage.
        config: Chart configuration.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    if config is None:
        config = ChartConfig()
    
    cycles, progress = collector.get_transaction_progress_over_time(total_expected)
    
    if not cycles:
        fig, ax = plt.subplots(figsize=config.figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title("Transaction Progress")
        return fig
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    ax.plot(cycles, progress, color='green', linewidth=config.line_width)
    ax.fill_between(cycles, progress, alpha=0.2, color='green')
    
    ax.set_xlabel("Cycle")
    if total_expected:
        ax.set_ylabel("Completion (%)")
        ax.set_ylim(0, 105)
    else:
        ax.set_ylabel("Completed Transactions")
    
    ax.set_title("Transaction Progress", fontsize=14, fontweight='bold')
    
    if config.show_grid:
        ax.grid(True, alpha=config.grid_alpha)
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_router_comparison(
    collector: "MetricsCollector",
    metric: str = "flits",
    config: Optional[ChartConfig] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot per-router metric comparison as bar chart.
    
    Args:
        collector: MetricsCollector with captured data.
        metric: Which metric to compare ("flits" or "buffer").
        config: Chart configuration.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    if config is None:
        config = ChartConfig()
    
    stats = collector.get_per_router_stats()
    
    if not stats:
        fig, ax = plt.subplots(figsize=config.figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title("Per-Router Comparison")
        return fig
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    # Sort by coordinate
    sorted_coords = sorted(stats.keys())
    labels = [f"({x},{y})" for x, y in sorted_coords]
    
    if metric == "flits":
        values = [stats[c]['flits_forwarded'] for c in sorted_coords]
        ylabel = "Flits Forwarded"
        title = "Flits Forwarded per Router"
        color = 'steelblue'
    else:  # buffer
        values = [stats[c]['buffer_occupancy'] for c in sorted_coords]
        ylabel = "Buffer Occupancy"
        title = "Buffer Occupancy per Router"
        color = 'coral'
    
    bars = ax.bar(labels, values, color=color, alpha=0.8)
    
    ax.set_xlabel("Router Coordinate")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Rotate labels if many routers
    if len(labels) > 10:
        plt.xticks(rotation=45, ha='right')
    
    if config.show_grid:
        ax.grid(True, alpha=config.grid_alpha, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig




def plot_combined_dashboard(
    collector: "MetricsCollector",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create a combined dashboard with multiple charts.
    
    Args:
        collector: MetricsCollector with captured data.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=(16, 14))
    
    # 3x2 grid of subplots
    ax1 = fig.add_subplot(3, 2, 1) # Throughput
    ax2 = fig.add_subplot(3, 2, 2) # Transaction Progress
    ax3 = fig.add_subplot(3, 2, 3) # Flit Count (Volume)
    ax4 = fig.add_subplot(3, 2, 4) # Buffer Occupancy
    ax5 = fig.add_subplot(3, 1, 3) # Router Utilization (Wide Bar Chart)
    
    # 1. Throughput
    cycles, throughputs = collector.get_throughput_over_time()
    if cycles:
        ax1.plot(cycles, throughputs, color='steelblue', linewidth=1.5)
        ax1.set_title("Throughput over Time", fontweight='bold')
        ax1.set_xlabel("Cycle")
        ax1.set_ylabel("Bytes/cycle")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Throughput")
    
    # 2. Transaction Progress
    cycles, progress = collector.get_transaction_progress_over_time()
    if cycles:
        ax2.fill_between(cycles, progress, alpha=0.2, color='green')
        ax2.plot(cycles, progress, color='green', linewidth=1.5)
        ax2.set_title("Transaction Progress", fontweight='bold')
        ax2.set_xlabel("Cycle")
        ax2.set_ylabel("Completed Transcations")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Progress")
    
    # 3. Flit Count (Volume Curve)
    cycles, flits = collector.get_flit_count_over_time()
    if cycles:
        ax3.plot(cycles, flits, color='teal', linewidth=1.5)
        ax3.set_title("Traffic Volume (Cumulative Flits)", fontweight='bold')
        ax3.set_xlabel("Cycle")
        ax3.set_ylabel("Total Flits")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("Volume")
    
    # 4. Buffer Utilization (Occupancy)
    cycles, occupancies = collector.get_total_buffer_occupancy_over_time()
    if cycles:
        ax4.fill_between(cycles, occupancies, alpha=0.3, color='coral')
        ax4.plot(cycles, occupancies, color='coral', linewidth=1.5)
        ax4.set_title("Mesh Buffer Occupancy (Congestion)", fontweight='bold')
        ax4.set_xlabel("Cycle")
        ax4.set_ylabel("Flits in Buffers")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("Buffer Occupancy")
    
    # 5. Router Utilization (Bar Chart)
    stats = collector.get_per_router_stats()
    if stats:
        sorted_coords = sorted(stats.keys())
        labels = [f"({x},{y})" for x, y in sorted_coords]
        values = [stats[c]['flits_forwarded'] for c in sorted_coords]
        
        ax5.bar(labels, values, color='steelblue', alpha=0.8)
        ax5.set_title("Per-Router Total Utilization (Workload Distribution)", fontweight='bold')
        ax5.set_ylabel("Total Flits Forwarded")
        ax5.grid(True, alpha=0.3, axis='y')
        if len(labels) > 10:
            plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    else:
        ax5.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax5.transAxes)
    
    plt.suptitle("NoC Performance Analysis Dashboard (Simulation Results)", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
