"""
Heatmap Visualizations for NoC.

Includes buffer occupancy and router utilization heatmaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .metrics_collector import MetricsCollector


@dataclass
class BufferHeatmapConfig:
    """Configuration for buffer heatmap visualization."""
    
    title: str = "NoC Buffer Occupancy Heatmap"
    cmap: str = "YlOrRd"  # Yellow-Orange-Red
    figsize: Tuple[int, int] = (10, 8)
    # Range for color scale (auto if None)
    vmin: Optional[float] = 0
    vmax: Optional[float] = None
    # Grid lines
    show_grid: bool = True
    grid_color: str = 'gray'
    grid_alpha: float = 0.3


def plot_buffer_heatmap(
    collector: "MetricsCollector",
    config: Optional[BufferHeatmapConfig] = None,
    save_path: Optional[str] = None,
    snapshot_index: int = -1,
    use_average: bool = False,
) -> Figure:
    """
    Plot 2D heatmap of buffer occupancy.
    
    Args:
        collector: MetricsCollector with captured data.
        config: Visualization configuration.
        save_path: Optional path to save the figure.
        snapshot_index: Index of snapshot to plot (-1 for latest).
        use_average: If True, plots average occupancy across all snapshots.
    
    Returns:
        Matplotlib Figure object.
    """
    if config is None:
        config = BufferHeatmapConfig()
    
    # Get matrix
    if use_average:
        _, data = collector.get_buffer_occupancy_over_time()
        if data.size == 0:
            matrix = np.zeros((collector.mesh_rows, collector.mesh_cols))
        else:
            matrix = np.mean(data, axis=0)
    else:
        matrix = collector._get_raw_occupancy_matrix(snapshot_index)
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    im = ax.imshow(
        matrix,
        cmap=config.cmap,
        interpolation='nearest',
        origin='lower',
        vmin=config.vmin,
        vmax=config.vmax,
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Flits' if not use_average else 'Average Flits')
    
    # Add labels
    ax.set_title(config.title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Mesh X (Columns)')
    ax.set_ylabel('Mesh Y (Rows)')
    
    # Set ticks
    ax.set_xticks(np.arange(collector.mesh_cols))
    ax.set_yticks(np.arange(collector.mesh_rows))
    
    # Add text labels on each cell
    for y in range(collector.mesh_rows):
        for x in range(collector.mesh_cols):
            val = matrix[y, x]
            label = f"{val:.1f}" if use_average else f"{int(val)}"
            ax.text(x, y, label, ha='center', va='center', 
                    color='black' if val < (matrix.max() / 2) else 'white')
    
    if config.show_grid:
        ax.set_xticks(np.arange(-.5, collector.mesh_cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, collector.mesh_rows, 1), minor=True)
        ax.grid(which='minor', color=config.grid_color, linestyle='-', alpha=config.grid_alpha)
        ax.tick_params(which='minor', bottom=False, left=False)

    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_utilization_heatmap(
    collector: "MetricsCollector",
    save_path: Optional[str] = None,
    snapshot_index: int = -1,
) -> Figure:
    """
    Plot 2D heatmap of router utilization (flits forwarded).
    
    Args:
        collector: MetricsCollector with captured data.
        save_path: Optional path to save the figure.
        snapshot_index: Snapshot index to use.
    
    Returns:
        Matplotlib Figure object.
    """
    matrix = collector.get_utilization_matrix(snapshot_index)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        matrix,
        cmap="Blues",
        interpolation='nearest',
        origin='lower',
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Total Flits Forwarded')
    
    # Add labels
    ax.set_title("Router Utilization Heatmap (Cumulative Flit Count)", fontsize=14, fontweight='bold')
    ax.set_xlabel('Mesh X (Columns)')
    ax.set_ylabel('Mesh Y (Rows)')
    
    # Set ticks
    ax.set_xticks(np.arange(collector.mesh_cols))
    ax.set_yticks(np.arange(collector.mesh_rows))
    
    # Add text labels on each cell
    for y in range(collector.mesh_rows):
        for x in range(collector.mesh_cols):
            val = matrix[y, x]
            ax.text(x, y, f"{int(val)}", ha='center', va='center',
                    color='black' if val < (matrix.max() / 1.5) else 'white')
    
    # Grid
    ax.set_xticks(np.arange(-.5, collector.mesh_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, collector.mesh_rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', alpha=0.3)
    ax.tick_params(which='minor', bottom=False, left=False)

    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
