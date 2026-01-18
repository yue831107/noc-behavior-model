#!/usr/bin/env python3
"""
Performance Trend Tracker for NoC Behavior Model.

Tracks performance metrics over time (commits) and detects anomalies.

Usage:
    # Record current performance
    python tools/perf_trend.py record --commit abc1234 --throughput 10.5 --latency 45

    # Show trend for a metric
    python tools/perf_trend.py plot --metric throughput --last 20

    # Detect anomalies
    python tools/perf_trend.py detect --threshold 2.0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics


@dataclass
class PerfRecord:
    """A single performance measurement record."""

    timestamp: str
    commit_hash: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerfRecord":
        return cls(
            timestamp=data["timestamp"],
            commit_hash=data["commit_hash"],
            metrics=data["metrics"],
            metadata=data.get("metadata", {}),
        )


class PerfTrendTracker:
    """
    Track performance metrics over time.

    Stores history in a JSON file and provides:
    - Recording new measurements
    - Plotting trends (text-based)
    - Detecting anomalies
    """

    def __init__(self, history_file: str | Path = "output/perf_history.json"):
        """
        Initialize tracker.

        Args:
            history_file: Path to JSON file storing performance history
        """
        self.history_file = Path(history_file)
        self.history: List[PerfRecord] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.history = [PerfRecord.from_dict(r) for r in data]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load history: {e}")
                self.history = []

    def _save_history(self) -> None:
        """Save history to file."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.history], f, indent=2)

    def record(
        self,
        metrics: Dict[str, float],
        commit_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerfRecord:
        """
        Record a new performance measurement.

        Args:
            metrics: Dictionary of metric name -> value
            commit_hash: Git commit hash (auto-detected if None)
            metadata: Additional metadata

        Returns:
            The created PerfRecord
        """
        if commit_hash is None:
            commit_hash = self._get_current_commit()

        record = PerfRecord(
            timestamp=datetime.now().isoformat(),
            commit_hash=commit_hash,
            metrics=metrics,
            metadata=metadata or {},
        )

        self.history.append(record)
        self._save_history()

        return record

    def get_metric_history(
        self,
        metric_name: str,
        last_n: Optional[int] = None,
    ) -> List[tuple[str, float]]:
        """
        Get history for a specific metric.

        Args:
            metric_name: Name of the metric
            last_n: Limit to last N records

        Returns:
            List of (commit_hash, value) tuples
        """
        history = []
        for record in self.history:
            if metric_name in record.metrics:
                history.append((record.commit_hash, record.metrics[metric_name]))

        if last_n is not None:
            history = history[-last_n:]

        return history

    def plot_trend(self, metric_name: str, last_n: int = 20) -> str:
        """
        Generate a text-based trend plot.

        Args:
            metric_name: Name of the metric to plot
            last_n: Number of recent records to include

        Returns:
            Text representation of the trend
        """
        history = self.get_metric_history(metric_name, last_n)

        if not history:
            return f"No data for metric '{metric_name}'"

        values = [v for _, v in history]
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val or 1  # Avoid division by zero

        # Chart dimensions
        width = 50
        height = 10

        lines = []
        lines.append(f"Performance Trend: {metric_name}")
        lines.append(f"Last {len(history)} measurements")
        lines.append("=" * (width + 15))

        # Generate sparkline
        for row in range(height - 1, -1, -1):
            threshold = min_val + (row / (height - 1)) * range_val
            line = f"{threshold:8.2f} |"

            for _, value in history:
                if value >= threshold:
                    line += "#"
                else:
                    line += " "

            lines.append(line)

        # X-axis
        lines.append(" " * 9 + "+" + "-" * len(history))

        # Commit labels (abbreviated)
        commit_line = " " * 10
        for commit, _ in history:
            commit_line += commit[:1] if commit else "?"
        lines.append(commit_line)

        # Statistics
        lines.append("")
        lines.append(f"Min: {min_val:.2f}  Max: {max_val:.2f}  "
                    f"Avg: {statistics.mean(values):.2f}  "
                    f"StdDev: {statistics.stdev(values) if len(values) > 1 else 0:.2f}")

        return "\n".join(lines)

    def detect_anomalies(
        self,
        threshold_std: float = 2.0,
    ) -> Dict[str, List[tuple[str, float, float]]]:
        """
        Detect performance anomalies.

        An anomaly is a measurement more than threshold_std standard
        deviations from the mean.

        Args:
            threshold_std: Number of standard deviations for anomaly

        Returns:
            Dict of metric_name -> list of (commit, value, z_score) tuples
        """
        anomalies: Dict[str, List[tuple[str, float, float]]] = {}

        # Get all metric names
        all_metrics = set()
        for record in self.history:
            all_metrics.update(record.metrics.keys())

        for metric_name in all_metrics:
            history = self.get_metric_history(metric_name)
            if len(history) < 3:  # Need at least 3 points
                continue

            values = [v for _, v in history]
            mean = statistics.mean(values)
            std = statistics.stdev(values)

            if std == 0:
                continue

            metric_anomalies = []
            for commit, value in history:
                z_score = (value - mean) / std
                if abs(z_score) > threshold_std:
                    metric_anomalies.append((commit, value, z_score))

            if metric_anomalies:
                anomalies[metric_name] = metric_anomalies

        return anomalies

    def get_latest(self) -> Optional[PerfRecord]:
        """Get the most recent record."""
        return self.history[-1] if self.history else None

    def compare_with_latest(
        self,
        metrics: Dict[str, float],
    ) -> Dict[str, tuple[float, float, float]]:
        """
        Compare metrics with the latest recorded values.

        Args:
            metrics: Current metrics to compare

        Returns:
            Dict of metric_name -> (current, previous, delta_percent)
        """
        latest = self.get_latest()
        if latest is None:
            return {}

        comparison = {}
        for name, current in metrics.items():
            if name in latest.metrics:
                previous = latest.metrics[name]
                if previous != 0:
                    delta = ((current - previous) / previous) * 100
                else:
                    delta = 0.0
                comparison[name] = (current, previous, delta)

        return comparison

    @staticmethod
    def _get_current_commit() -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Performance Trend Tracker for NoC Behavior Model"
    )
    parser.add_argument(
        "--history",
        default="output/perf_history.json",
        help="Path to history file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Record command
    record_parser = subparsers.add_parser("record", help="Record a measurement")
    record_parser.add_argument("--commit", help="Git commit hash")
    record_parser.add_argument("--throughput", type=float, help="Throughput (Gbps)")
    record_parser.add_argument("--latency", type=float, help="Latency (cycles)")
    record_parser.add_argument("--metric", action="append", nargs=2,
                              metavar=("NAME", "VALUE"),
                              help="Custom metric (can be repeated)")

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Plot trend")
    plot_parser.add_argument("--metric", default="throughput", help="Metric name")
    plot_parser.add_argument("--last", type=int, default=20, help="Last N records")

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect anomalies")
    detect_parser.add_argument("--threshold", type=float, default=2.0,
                              help="Standard deviation threshold")

    # List command
    subparsers.add_parser("list", help="List all records")

    args = parser.parse_args()

    tracker = PerfTrendTracker(args.history)

    if args.command == "record":
        metrics = {}
        if args.throughput is not None:
            metrics["throughput"] = args.throughput
        if args.latency is not None:
            metrics["latency"] = args.latency
        if args.metric:
            for name, value in args.metric:
                metrics[name] = float(value)

        if not metrics:
            print("Error: No metrics provided")
            sys.exit(1)

        record = tracker.record(metrics, args.commit)
        print(f"Recorded: {record.commit_hash} -> {record.metrics}")

    elif args.command == "plot":
        print(tracker.plot_trend(args.metric, args.last))

    elif args.command == "detect":
        anomalies = tracker.detect_anomalies(args.threshold)
        if anomalies:
            print(f"Anomalies detected (>{args.threshold} std):")
            for metric, items in anomalies.items():
                print(f"\n  {metric}:")
                for commit, value, z_score in items:
                    print(f"    {commit}: {value:.2f} (z={z_score:.2f})")
        else:
            print("No anomalies detected")

    elif args.command == "list":
        for record in tracker.history:
            print(f"{record.commit_hash} ({record.timestamp[:10]}): {record.metrics}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
