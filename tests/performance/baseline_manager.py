"""
Performance Baseline Manager for NoC Behavior Model.

Provides functionality for:
- Loading and saving performance baselines
- Comparing current results against baselines
- Detecting performance regressions
- Automatic baseline updates when performance improves
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List


@dataclass
class ComparisonResult:
    """Result of comparing current metrics against baseline."""

    regression_detected: bool
    throughput_delta: float  # Percentage change (positive = improvement)
    latency_delta: float     # Percentage change (negative = improvement)
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "REGRESSION" if self.regression_detected else "OK"
        return (
            f"[{status}] Throughput: {self.throughput_delta:+.1f}%, "
            f"Latency: {self.latency_delta:+.1f}%"
        )


@dataclass
class BaselineEntry:
    """A single baseline entry with metrics and metadata."""

    name: str
    metrics: Dict[str, float]
    timestamp: str
    commit_hash: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "commit_hash": self.commit_hash,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineEntry":
        return cls(
            name=data["name"],
            metrics=data["metrics"],
            timestamp=data["timestamp"],
            commit_hash=data.get("commit_hash"),
            description=data.get("description"),
        )


class BaselineManager:
    """
    Performance baseline manager.

    Manages performance baselines stored as JSON files in a directory structure:

        baselines/
        ├── host_to_noc/
        │   ├── broadcast_4kb.json
        │   └── scatter_1kb.json
        └── noc_to_noc/
            ├── neighbor_256b.json
            └── transpose_1kb.json

    Example usage:
        manager = BaselineManager("tests/performance/baselines")

        # Load baseline
        baseline = manager.load_baseline("host_to_noc/broadcast_4kb")

        # Compare current results
        current = {"throughput_gbps": 10.5, "avg_latency_cycles": 45}
        result = manager.compare(current, baseline.metrics)

        if result.regression_detected:
            print(f"Regression detected: {result.details}")

        # Save new baseline if better
        manager.update_if_better("host_to_noc/broadcast_4kb", current)
    """

    # Default regression thresholds
    THROUGHPUT_THRESHOLD = 0.95  # Must be >= 95% of baseline
    LATENCY_THRESHOLD = 1.10    # Must be <= 110% of baseline

    def __init__(
        self,
        baselines_dir: str | Path,
        throughput_threshold: float = THROUGHPUT_THRESHOLD,
        latency_threshold: float = LATENCY_THRESHOLD,
    ):
        """
        Initialize baseline manager.

        Args:
            baselines_dir: Directory containing baseline files
            throughput_threshold: Minimum throughput ratio vs baseline (default: 0.95)
            latency_threshold: Maximum latency ratio vs baseline (default: 1.10)
        """
        self.baselines_dir = Path(baselines_dir)
        self.throughput_threshold = throughput_threshold
        self.latency_threshold = latency_threshold

        # Create directory if it doesn't exist
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

    def _get_baseline_path(self, test_name: str) -> Path:
        """Get the file path for a baseline."""
        # Normalize path separators and add .json extension
        normalized = test_name.replace("\\", "/")
        if not normalized.endswith(".json"):
            normalized += ".json"
        return self.baselines_dir / normalized

    def load_baseline(self, test_name: str) -> Optional[BaselineEntry]:
        """
        Load a baseline from disk.

        Args:
            test_name: Name of the test (e.g., "host_to_noc/broadcast_4kb")

        Returns:
            BaselineEntry if found, None otherwise
        """
        path = self._get_baseline_path(test_name)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BaselineEntry.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load baseline {path}: {e}")
            return None

    def save_baseline(
        self,
        test_name: str,
        metrics: Dict[str, float],
        commit_hash: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Save a new baseline to disk.

        Args:
            test_name: Name of the test
            metrics: Performance metrics dict
            commit_hash: Optional git commit hash
            description: Optional description
        """
        path = self._get_baseline_path(test_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = BaselineEntry(
            name=test_name,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            commit_hash=commit_hash,
            description=description,
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry.to_dict(), f, indent=2)

    def compare(
        self,
        current: Dict[str, float],
        baseline: Dict[str, float],
    ) -> ComparisonResult:
        """
        Compare current metrics against baseline.

        Args:
            current: Current performance metrics
            baseline: Baseline metrics to compare against

        Returns:
            ComparisonResult with regression detection and deltas
        """
        # Extract key metrics (handle various naming conventions)
        current_throughput = self._get_throughput(current)
        baseline_throughput = self._get_throughput(baseline)
        current_latency = self._get_latency(current)
        baseline_latency = self._get_latency(baseline)

        # Calculate deltas as percentages
        if baseline_throughput > 0:
            throughput_ratio = current_throughput / baseline_throughput
            throughput_delta = (throughput_ratio - 1) * 100
        else:
            throughput_ratio = 1.0
            throughput_delta = 0.0

        if baseline_latency > 0:
            latency_ratio = current_latency / baseline_latency
            latency_delta = (latency_ratio - 1) * 100
        else:
            latency_ratio = 1.0
            latency_delta = 0.0

        # Detect regression
        regression = (
            throughput_ratio < self.throughput_threshold or
            latency_ratio > self.latency_threshold
        )

        details = {
            "current_throughput": current_throughput,
            "baseline_throughput": baseline_throughput,
            "throughput_ratio": throughput_ratio,
            "current_latency": current_latency,
            "baseline_latency": baseline_latency,
            "latency_ratio": latency_ratio,
            "thresholds": {
                "throughput": self.throughput_threshold,
                "latency": self.latency_threshold,
            },
        }

        return ComparisonResult(
            regression_detected=regression,
            throughput_delta=throughput_delta,
            latency_delta=latency_delta,
            details=details,
        )

    def update_if_better(
        self,
        test_name: str,
        current: Dict[str, float],
        commit_hash: Optional[str] = None,
    ) -> bool:
        """
        Update baseline if current metrics are better.

        Args:
            test_name: Name of the test
            current: Current performance metrics
            commit_hash: Optional git commit hash

        Returns:
            True if baseline was updated, False otherwise
        """
        existing = self.load_baseline(test_name)

        if existing is None:
            # No existing baseline, save current
            self.save_baseline(test_name, current, commit_hash, "Initial baseline")
            return True

        result = self.compare(current, existing.metrics)

        # Update if throughput improved and latency didn't get significantly worse
        # Or if both improved
        throughput_improved = result.throughput_delta > 1.0  # >1% improvement
        latency_improved = result.latency_delta < -1.0       # <-1% (lower is better)
        latency_acceptable = result.latency_delta < 5.0      # Not more than 5% worse

        should_update = (
            (throughput_improved and latency_acceptable) or
            (latency_improved and result.throughput_delta > -5.0)
        )

        if should_update:
            self.save_baseline(
                test_name,
                current,
                commit_hash,
                f"Updated: throughput {result.throughput_delta:+.1f}%, "
                f"latency {result.latency_delta:+.1f}%",
            )
            return True

        return False

    def list_baselines(self, category: Optional[str] = None) -> List[str]:
        """
        List all available baselines.

        Args:
            category: Optional category filter (e.g., "host_to_noc")

        Returns:
            List of baseline names
        """
        search_dir = self.baselines_dir
        if category:
            search_dir = search_dir / category

        if not search_dir.exists():
            return []

        baselines = []
        for path in search_dir.rglob("*.json"):
            relative = path.relative_to(self.baselines_dir)
            name = str(relative).replace("\\", "/").removesuffix(".json")
            baselines.append(name)

        return sorted(baselines)

    @staticmethod
    def _get_throughput(metrics: Dict[str, float]) -> float:
        """Extract throughput from metrics dict (handles various keys)."""
        for key in ["throughput_gbps", "throughput", "bw_gbps", "bandwidth"]:
            if key in metrics:
                return float(metrics[key])
        return 0.0

    @staticmethod
    def _get_latency(metrics: Dict[str, float]) -> float:
        """Extract latency from metrics dict (handles various keys)."""
        for key in ["avg_latency_cycles", "latency", "avg_latency", "latency_cycles"]:
            if key in metrics:
                return float(metrics[key])
        return 0.0
