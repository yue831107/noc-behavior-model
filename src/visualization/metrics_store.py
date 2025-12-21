"""
Metrics Store for NoC Simulation.

Provides persistent storage for simulation metrics in multiple formats:
- JSON: Human-readable, good for inspection
- CSV: Excel-compatible, good for analysis
- NPZ: NumPy compressed, efficient for large datasets
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .metrics_collector import MetricsCollector


@dataclass
class MetricsStore:
    """
    Persistent storage for simulation metrics.
    
    Usage:
        collector = MetricsCollector(system)
        # ... run simulation and capture ...
        
        store = MetricsStore()
        path = store.save_json(collector, "my_simulation")
        
        # Later, load back
        loaded = store.load(path)
    """
    
    base_dir: Path = None
    
    def __post_init__(self):
        if self.base_dir is None:
            self.base_dir = Path("output/metrics")
        else:
            self.base_dir = Path(self.base_dir)
        
        # Create directory if needed
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_filename(self, name: Optional[str], extension: str) -> Path:
        """Generate timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name:
            filename = f"{timestamp}_{name}.{extension}"
        else:
            filename = f"{timestamp}_metrics.{extension}"
        return self.base_dir / filename
    
    def save_json(
        self,
        collector: "MetricsCollector",
        name: Optional[str] = None,
    ) -> Path:
        """
        Save metrics to JSON format (human-readable).
        
        Args:
            collector: MetricsCollector with captured data.
            name: Optional name suffix for the file.
        
        Returns:
            Path to saved file.
        """
        filepath = self._generate_filename(name, "json")
        
        data = collector.to_dict()
        data['saved_at'] = datetime.now().isoformat()
        data['format_version'] = '1.0'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def save_csv(
        self,
        collector: "MetricsCollector",
        name: Optional[str] = None,
    ) -> Path:
        """
        Save metrics to CSV format (Excel-compatible).
        
        Saves time-series data with one row per snapshot.
        
        Args:
            collector: MetricsCollector with captured data.
            name: Optional name suffix for the file.
        
        Returns:
            Path to saved file.
        """
        filepath = self._generate_filename(name, "csv")
        
        # Create CSV with key metrics per snapshot
        headers = [
            'cycle',
            'flits_in_flight',
            'completed_transactions',
            'bytes_transferred',
            'total_buffer_occupancy',
            'total_flit_count',
        ]
        
        rows = []
        for snapshot in collector.snapshots:
            total_buffer = sum(snapshot.buffer_occupancy.values())
            total_flits = sum(snapshot.router_flit_counts.values())
            
            rows.append([
                snapshot.cycle,
                snapshot.flits_in_flight,
                snapshot.completed_transactions,
                snapshot.bytes_transferred,
                total_buffer,
                total_flits,
            ])
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(','.join(headers) + '\n')
            for row in rows:
                f.write(','.join(str(v) for v in row) + '\n')
        
        return filepath
    
    def save_npz(
        self,
        collector: "MetricsCollector",
        name: Optional[str] = None,
    ) -> Path:
        """
        Save metrics to NumPy compressed format (efficient).
        
        Good for large datasets and fast loading in Python.
        
        Args:
            collector: MetricsCollector with captured data.
            name: Optional name suffix for the file.
        
        Returns:
            Path to saved file.
        """
        filepath = self._generate_filename(name, "npz")
        
        # Extract arrays
        cycles = np.array([s.cycle for s in collector.snapshots])
        flits_in_flight = np.array([s.flits_in_flight for s in collector.snapshots])
        completed = np.array([s.completed_transactions for s in collector.snapshots])
        bytes_transferred = np.array([s.bytes_transferred for s in collector.snapshots])
        
        # Buffer occupancy as 3D array (snapshots, rows, cols)
        buffer_matrices = np.array([
            collector._get_raw_occupancy_matrix(i)
            for i in range(len(collector.snapshots))
        ])
        
        # All latencies
        all_latencies = np.array(collector.get_all_latencies())
        
        np.savez_compressed(
            filepath,
            cycles=cycles,
            flits_in_flight=flits_in_flight,
            completed_transactions=completed,
            bytes_transferred=bytes_transferred,
            buffer_matrices=buffer_matrices,
            latencies=all_latencies,
            mesh_cols=collector.mesh_cols,
            mesh_rows=collector.mesh_rows,
        )
        
        return filepath
    
    def load(self, filepath: Path) -> "MetricsCollector":
        """
        Load metrics from file.
        
        Automatically detects format from extension.
        
        Args:
            filepath: Path to saved metrics file.
        
        Returns:
            MetricsCollector with loaded data.
        
        Raises:
            ValueError: If file format is not supported.
        """
        from .metrics_collector import MetricsCollector
        
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            return self._load_json(filepath)
        elif filepath.suffix == '.npz':
            return self._load_npz(filepath)
        elif filepath.suffix == '.csv':
            return self._load_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def _load_json(self, filepath: Path) -> "MetricsCollector":
        """Load from JSON format."""
        from .metrics_collector import MetricsCollector
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return MetricsCollector.from_dict(data)
    
    def _load_csv(self, filepath: Path) -> "MetricsCollector":
        """Load from CSV format (limited data)."""
        from .metrics_collector import MetricsCollector, SimulationSnapshot
        
        # CSV has less data, create minimal collector
        collector = object.__new__(MetricsCollector)
        collector.system = None
        collector.capture_interval = 1
        collector._last_capture_cycle = -1
        collector._last_completed = 0
        collector._last_bytes = 0
        collector._mesh_cols = 5
        collector._mesh_rows = 4
        collector.snapshots = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            headers = f.readline().strip().split(',')
            
            for line in f:
                values = line.strip().split(',')
                if len(values) >= 4:
                    snapshot = SimulationSnapshot(
                        cycle=int(values[0]),
                        flits_in_flight=int(values[1]) if len(values) > 1 else 0,
                        completed_transactions=int(values[2]) if len(values) > 2 else 0,
                        bytes_transferred=int(values[3]) if len(values) > 3 else 0,
                    )
                    collector.snapshots.append(snapshot)
        
        return collector
    
    def _load_npz(self, filepath: Path) -> "MetricsCollector":
        """Load from NPZ format."""
        from .metrics_collector import MetricsCollector, SimulationSnapshot
        
        data = np.load(filepath, allow_pickle=True)
        
        collector = object.__new__(MetricsCollector)
        collector.system = None
        collector.capture_interval = 1
        collector._last_capture_cycle = -1
        collector._last_completed = 0
        collector._last_bytes = 0
        collector._mesh_cols = int(data.get('mesh_cols', 5))
        collector._mesh_rows = int(data.get('mesh_rows', 4))
        collector.snapshots = []
        
        cycles = data['cycles']
        flits = data['flits_in_flight']
        completed = data['completed_transactions']
        bytes_arr = data['bytes_transferred']
        buffer_matrices = data['buffer_matrices']
        
        for i in range(len(cycles)):
            # Convert buffer matrix back to dict
            buffer_occ = {}
            if i < len(buffer_matrices):
                matrix = buffer_matrices[i]
                for y in range(matrix.shape[0]):
                    for x in range(matrix.shape[1]):
                        if matrix[y, x] > 0:
                            buffer_occ[(x, y)] = int(matrix[y, x])
            
            snapshot = SimulationSnapshot(
                cycle=int(cycles[i]),
                buffer_occupancy=buffer_occ,
                flits_in_flight=int(flits[i]),
                completed_transactions=int(completed[i]),
                bytes_transferred=int(bytes_arr[i]),
            )
            collector.snapshots.append(snapshot)
        
        return collector
    
    def list_files(self, pattern: str = "*") -> list:
        """
        List saved metrics files.
        
        Args:
            pattern: Glob pattern for filtering.
        
        Returns:
            List of file paths.
        """
        return list(self.base_dir.glob(pattern))
