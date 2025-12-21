"""
Metrics Collector for NoC Simulation.

Collects time-series snapshots during simulation for later visualization.
Each snapshot captures the state of the mesh at a specific cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..core.routing_selector import V1System, NoCSystem
    from ..core.mesh import Mesh


@dataclass
class SimulationSnapshot:
    """Single point-in-time snapshot of simulation state."""
    
    cycle: int
    # Buffer occupancy per router: (x, y) -> total buffer occupancy
    buffer_occupancy: Dict[Tuple[int, int], int] = field(default_factory=dict)
    # Flits currently in flight (in router buffers)
    flits_in_flight: int = 0
    # Cumulative completed transactions
    completed_transactions: int = 0
    # Cumulative bytes transferred
    bytes_transferred: int = 0
    # Per-router flit counts
    router_flit_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    # Transaction latencies completed this snapshot window
    latencies: List[int] = field(default_factory=list)


class MetricsCollector:
    """
    Collect simulation metrics over time.
    
    Usage:
        system = NoCSystem(...)
        collector = MetricsCollector(system)
        
        for _ in range(1000):
            system.process_cycle()
            if system.current_cycle % 10 == 0:
                collector.capture()
        
        # Get data for visualization
        cycles, counts = collector.get_flit_count_over_time()
        latencies = collector.get_all_latencies()
    """
    
    def __init__(
        self,
        system: Union["V1System", "NoCSystem"],
        capture_interval: int = 1,
    ):
        """
        Initialize metrics collector.
        
        Args:
            system: V1System or NoCSystem instance to collect from.
            capture_interval: Minimum cycles between captures (default: 1).
        """
        self.system = system
        self.capture_interval = capture_interval
        self.snapshots: List[SimulationSnapshot] = []
        self._last_capture_cycle = -1
        self._last_completed = 0
        self._last_bytes = 0

        # Register packet arrival callback for latency tracking
        if hasattr(system, 'set_packet_arrival_callback'):
            system.set_packet_arrival_callback(self._on_packet_arrived)

    def _on_packet_arrived(self, packet_id: int, creation_time: int, arrival_time: int) -> None:
        """
        Callback when a packet arrives at destination.

        Calculates latency from packet creation to arrival.

        Args:
            packet_id: Unique packet identifier.
            creation_time: Cycle when packet was created at SlaveNI.
            arrival_time: Cycle when packet arrived at MasterNI.
        """
        latency = arrival_time - creation_time
        if latency >= 0:
            self.add_latency_sample(latency)

    @property
    def mesh(self) -> "Mesh":
        """Get mesh from system."""
        return self.system.mesh
    
    @property
    def mesh_cols(self) -> int:
        """Mesh columns."""
        if hasattr(self.system, 'mesh_cols'):
            return self.system.mesh_cols
        return self.system._mesh_cols
    
    @property
    def mesh_rows(self) -> int:
        """Mesh rows."""
        if hasattr(self.system, 'mesh_rows'):
            return self.system.mesh_rows
        return self.system._mesh_rows
    
    def capture(self) -> Optional[SimulationSnapshot]:
        """
        Capture current simulation state as a snapshot.
        
        Uses MetricsProvider protocol if available, falls back to legacy extraction.
        
        Returns:
            SimulationSnapshot if captured, None if skipped due to interval.
        """
        from ..core.metrics_provider import get_metrics_from_system
        
        # Get metrics via Protocol or fallback
        metrics = get_metrics_from_system(self.system)
        cycle = metrics['cycle']
        
        # Check capture interval
        if cycle - self._last_capture_cycle < self.capture_interval:
            return None
        
        self._last_capture_cycle = cycle
        
        # Calculate total flits in buffers
        total_flits = sum(metrics['buffer_occupancy'].values())
        
        # Create snapshot
        snapshot = SimulationSnapshot(
            cycle=cycle,
            buffer_occupancy=metrics['buffer_occupancy'],
            flits_in_flight=total_flits,
            completed_transactions=metrics['completed_transactions'],
            bytes_transferred=metrics['bytes_transferred'],
            router_flit_counts=metrics['flit_stats'],
            latencies=[],  # Filled by transaction completion callbacks
        )
        
        self.snapshots.append(snapshot)
        self._last_completed = metrics['completed_transactions']
        self._last_bytes = metrics['bytes_transferred']
        
        return snapshot
    
    def get_buffer_occupancy_over_time(self) -> Tuple[List[int], np.ndarray]:
        """
        Get buffer occupancy time series for all routers.
        
        Returns:
            Tuple of (cycles, data) where data is (num_snapshots, rows, cols).
        """
        if not self.snapshots:
            return [], np.array([])
        
        cycles = [s.cycle for s in self.snapshots]
        data = np.array([
            self._get_raw_occupancy_matrix(i)
            for i in range(len(self.snapshots))
        ])
        
        return cycles, data

    def _get_raw_occupancy_matrix(self, snapshot_index: int) -> np.ndarray:
        """Internal helper for occupancy data."""
        snapshot = self.snapshots[snapshot_index]
        matrix = np.zeros((self.mesh_rows, self.mesh_cols))
        for (x, y), occupancy in snapshot.buffer_occupancy.items():
            if 0 <= y < self.mesh_rows and 0 <= x < self.mesh_cols:
                matrix[y, x] = occupancy
        return matrix

    def get_utilization_matrix(self, snapshot_index: int = -1) -> np.ndarray:
        """
        Get flit forwarded counts (utilization) as a 2D matrix.
        
        Args:
            snapshot_index: Which snapshot to use (-1 = latest/cumulative).
        
        Returns:
            2D numpy array with flit counts.
        """
        if not self.snapshots:
            return np.zeros((self.mesh_rows, self.mesh_cols))
        
        snapshot = self.snapshots[snapshot_index]
        matrix = np.zeros((self.mesh_rows, self.mesh_cols))
        
        for (x, y), count in snapshot.router_flit_counts.items():
            if 0 <= y < self.mesh_rows and 0 <= x < self.mesh_cols:
                matrix[y, x] = count
        
        return matrix
    
    def get_total_buffer_occupancy_over_time(self) -> Tuple[List[int], List[int]]:
        """
        Get total buffer occupancy (sum across all routers) over time.
        
        Returns:
            Tuple of (cycles, occupancies).
        """
        cycles = [s.cycle for s in self.snapshots]
        occupancies = [s.flits_in_flight for s in self.snapshots]
        return cycles, occupancies
    
    def get_throughput_over_time(
        self,
        window_size: int = 10,
    ) -> Tuple[List[int], List[float]]:
        """
        Get throughput (bytes/cycle) over time using sliding window.
        
        Args:
            window_size: Number of snapshots for averaging.
        
        Returns:
            Tuple of (cycles, throughput_values).
        """
        if len(self.snapshots) < 2:
            return [], []
        
        cycles = []
        throughputs = []
        
        for i in range(1, len(self.snapshots)):
            curr = self.snapshots[i]
            prev_idx = max(0, i - window_size)
            prev = self.snapshots[prev_idx]
            
            delta_bytes = curr.bytes_transferred - prev.bytes_transferred
            delta_cycles = curr.cycle - prev.cycle
            
            if delta_cycles > 0:
                throughput = delta_bytes / delta_cycles
            else:
                throughput = 0.0
            
            cycles.append(curr.cycle)
            throughputs.append(throughput)
        
        return cycles, throughputs
    
    def get_flit_count_over_time(self) -> Tuple[List[int], List[int]]:
        """
        Get total flits forwarded over time.
        
        Returns:
            Tuple of (cycles, flit_counts).
        """
        cycles = []
        flit_counts = []
        
        for snapshot in self.snapshots:
            total = sum(snapshot.router_flit_counts.values())
            cycles.append(snapshot.cycle)
            flit_counts.append(total)
        
        return cycles, flit_counts
    
    def get_transaction_progress_over_time(
        self,
        total_expected: int = None,
    ) -> Tuple[List[int], List[float]]:
        """
        Get transaction completion progress over time.
        
        Args:
            total_expected: Expected total transactions for percentage calc.
        
        Returns:
            Tuple of (cycles, progress_percentage or absolute_count).
        """
        cycles = [s.cycle for s in self.snapshots]
        
        if total_expected and total_expected > 0:
            progress = [
                s.completed_transactions / total_expected * 100
                for s in self.snapshots
            ]
        else:
            progress = [s.completed_transactions for s in self.snapshots]
        
        return cycles, progress
    
    def get_all_latencies(self) -> List[int]:
        """
        Get all recorded latencies across all snapshots.
        
        Returns:
            List of latency values (in cycles).
        """
        latencies = []
        for snapshot in self.snapshots:
            latencies.extend(snapshot.latencies)
        return latencies
    
    def add_latency_sample(self, latency: int) -> None:
        """
        Add a latency sample to the current/latest snapshot.
        
        Args:
            latency: Transaction latency in cycles.
        """
        if self.snapshots:
            self.snapshots[-1].latencies.append(latency)
    
    def get_per_router_stats(self) -> Dict[Tuple[int, int], Dict]:
        """
        Get statistics per router from latest snapshot.
        
        Returns:
            Dict of coord -> {flits_forwarded, buffer_occupancy}.
        """
        if not self.snapshots:
            return {}
        
        latest = self.snapshots[-1]
        stats = {}
        
        for coord in latest.buffer_occupancy:
            stats[coord] = {
                'buffer_occupancy': latest.buffer_occupancy.get(coord, 0),
                'flits_forwarded': latest.router_flit_counts.get(coord, 0),
            }
        
        return stats
    
    def clear(self) -> None:
        """Clear all collected snapshots."""
        self.snapshots.clear()
        self._last_capture_cycle = -1
        self._last_completed = 0
        self._last_bytes = 0

    def __len__(self) -> int:
        """Number of snapshots collected."""
        return len(self.snapshots)
    
    def to_dict(self) -> Dict:
        """
        Convert collector data to dictionary for serialization.
        
        Returns:
            Dictionary representation of all snapshots.
        """
        return {
            'mesh_cols': self.mesh_cols,
            'mesh_rows': self.mesh_rows,
            'capture_interval': self.capture_interval,
            'snapshots': [
                {
                    'cycle': s.cycle,
                    'buffer_occupancy': {
                        f"{x},{y}": v for (x, y), v in s.buffer_occupancy.items()
                    },
                    'flits_in_flight': s.flits_in_flight,
                    'completed_transactions': s.completed_transactions,
                    'bytes_transferred': s.bytes_transferred,
                    'router_flit_counts': {
                        f"{x},{y}": v for (x, y), v in s.router_flit_counts.items()
                    },
                    'latencies': s.latencies,
                }
                for s in self.snapshots
            ],
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict,
        system: Optional[Union["V1System", "NoCSystem"]] = None,
    ) -> "MetricsCollector":
        """
        Create collector from dictionary (for loading saved metrics).
        
        Args:
            data: Dictionary from to_dict().
            system: Optional system reference (can be None for loaded data).
        
        Returns:
            MetricsCollector with loaded snapshots.
        """
        # Create minimal collector without real system
        collector = object.__new__(cls)
        collector.system = system
        collector.capture_interval = data.get('capture_interval', 1)
        collector._last_capture_cycle = -1
        collector._last_completed = 0
        collector._last_bytes = 0
        collector._mesh_cols = data.get('mesh_cols', 5)
        collector._mesh_rows = data.get('mesh_rows', 4)
        
        # Override mesh property for loaded data
        collector.snapshots = []
        
        for s_data in data.get('snapshots', []):
            buffer_occ = {}
            for key, val in s_data.get('buffer_occupancy', {}).items():
                x, y = map(int, key.split(','))
                buffer_occ[(x, y)] = val
            
            flit_counts = {}
            for key, val in s_data.get('router_flit_counts', {}).items():
                x, y = map(int, key.split(','))
                flit_counts[(x, y)] = val
            
            snapshot = SimulationSnapshot(
                cycle=s_data['cycle'],
                buffer_occupancy=buffer_occ,
                flits_in_flight=s_data.get('flits_in_flight', 0),
                completed_transactions=s_data.get('completed_transactions', 0),
                bytes_transferred=s_data.get('bytes_transferred', 0),
                router_flit_counts=flit_counts,
                latencies=s_data.get('latencies', []),
            )
            collector.snapshots.append(snapshot)
        
        return collector
