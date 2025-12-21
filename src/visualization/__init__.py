"""
NoC Visualization Module.

Provides metrics collection, data persistence, and visualization
for NoC simulation analysis.

Phase 1: Static Charts
- MetricsCollector: Collect simulation snapshots
- MetricsStore: Save/load metrics to/from files
- plot_buffer_heatmap: Buffer occupancy heatmap
- plot_latency_histogram: Latency distribution histogram
- plot_throughput_curve: Throughput over time

Phase 2: Additional Charts (charts.py)
Phase 3: Animation (animation.py)
Phase 4: Dashboard (dashboard/)

Note: Metrics collection classes don't require matplotlib.
      Chart/plot functions require matplotlib and are lazy-loaded.
"""

# Core metrics collection (no matplotlib dependency)
from .metrics_collector import (
    SimulationSnapshot,
    MetricsCollector,
)
from .metrics_store import (
    MetricsStore,
)


def __getattr__(name):
    """Lazy import for matplotlib-dependent modules."""
    # Heatmap functions
    if name in ("BufferHeatmapConfig", "plot_buffer_heatmap", "plot_utilization_heatmap"):
        from .heatmap import BufferHeatmapConfig, plot_buffer_heatmap, plot_utilization_heatmap
        return {"BufferHeatmapConfig": BufferHeatmapConfig,
                "plot_buffer_heatmap": plot_buffer_heatmap,
                "plot_utilization_heatmap": plot_utilization_heatmap}[name]

    # Histogram functions
    if name in ("LatencyHistogramConfig", "plot_latency_histogram"):
        from .histogram import LatencyHistogramConfig, plot_latency_histogram
        return {"LatencyHistogramConfig": LatencyHistogramConfig,
                "plot_latency_histogram": plot_latency_histogram}[name]

    # Throughput functions
    if name in ("ThroughputConfig", "plot_throughput_curve"):
        from .throughput import ThroughputConfig, plot_throughput_curve
        return {"ThroughputConfig": ThroughputConfig,
                "plot_throughput_curve": plot_throughput_curve}[name]

    # Chart functions
    if name in ("ChartConfig", "plot_flit_count_curve", "plot_buffer_utilization_curve",
                "plot_transaction_progress", "plot_router_comparison", "plot_combined_dashboard"):
        from .charts import (ChartConfig, plot_flit_count_curve, plot_buffer_utilization_curve,
                            plot_transaction_progress, plot_router_comparison, plot_combined_dashboard)
        return {"ChartConfig": ChartConfig,
                "plot_flit_count_curve": plot_flit_count_curve,
                "plot_buffer_utilization_curve": plot_buffer_utilization_curve,
                "plot_transaction_progress": plot_transaction_progress,
                "plot_router_comparison": plot_router_comparison,
                "plot_combined_dashboard": plot_combined_dashboard}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Metrics Collection (no matplotlib)
    "SimulationSnapshot",
    "MetricsCollector",
    # Storage (no matplotlib)
    "MetricsStore",
    # Heatmap (lazy loaded)
    "BufferHeatmapConfig",
    "plot_buffer_heatmap",
    "plot_utilization_heatmap",
    # Histogram (lazy loaded)
    "LatencyHistogramConfig",
    "plot_latency_histogram",
    # Throughput (lazy loaded)
    "ThroughputConfig",
    "plot_throughput_curve",
    # Charts (lazy loaded)
    "ChartConfig",
    "plot_flit_count_curve",
    "plot_buffer_utilization_curve",
    "plot_transaction_progress",
    "plot_router_comparison",
    "plot_combined_dashboard",
]
