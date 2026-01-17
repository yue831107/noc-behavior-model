"""
Metrics Collector for NoC Simulation.

Collects time-series snapshots during simulation for later visualization.
Each snapshot captures the state of the mesh at a specific cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any, TYPE_CHECKING
import statistics
import numpy as np

from ..metrics.booksim_metrics import (
    BookSimMetrics,
    SaturationStatus,
    calculate_saturation_status,
)

if TYPE_CHECKING:
    from ..core.routing_selector import V1System, NoCSystem
    from ..core.mesh import Mesh
    from ..core.flit import AxiChannel


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

        # Latency tracking: key -> injection_cycle
        self._injection_log: Dict[Any, int] = {}

        # BookSim2-style: zero-load latency reference
        self._zero_load_latency: Optional[float] = None

        # Data bytes tracking (only W/R flits, not AW/AR/B)
        self._data_bytes_received: int = 0
        self._data_flits_received: int = 0

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
        from ..verification.metrics_provider import get_metrics_from_system
        
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
    
    def add_latency_sample(
        self,
        latency: int,
        axi_channel: Optional["AxiChannel"] = None,
        payload_bytes: int = 0
    ) -> None:
        """
        Add a latency sample to the current/latest snapshot.

        Args:
            latency: Transaction latency in cycles.
            axi_channel: Optional AXI channel type (for filtering).
            payload_bytes: Data bytes carried by this flit (W/R only).
        """
        if self.snapshots:
            self.snapshots[-1].latencies.append(latency)

        # Track data bytes (only W/R flits carry user data)
        if payload_bytes > 0:
            self._data_bytes_received += payload_bytes
            self._data_flits_received += 1

    # =========================================================================
    # Latency Tracking (Monitor-Based)
    # =========================================================================

    def record_injection(self, key: Any, cycle: int) -> None:
        """
        Record packet/transfer injection time.

        Call this when a packet or transfer is initiated (e.g., when submitting
        a write request or when a node starts its transfer).

        Args:
            key: Unique identifier for the transfer (e.g., axi_id, node_id).
            cycle: The cycle when injection occurred.
        """
        self._injection_log[key] = cycle

    def record_ejection(self, key: Any, cycle: int) -> None:
        """
        Record packet/transfer ejection time and calculate latency.

        Call this when a packet or transfer completes (e.g., when receiving
        a B response or when a node finishes its transfer).

        Automatically calculates latency = ejection_cycle - injection_cycle
        and adds it to the latency samples.

        Args:
            key: Unique identifier matching a previous record_injection call.
            cycle: The cycle when ejection occurred.
        """
        if key in self._injection_log:
            latency = cycle - self._injection_log[key]
            self.add_latency_sample(latency)
            del self._injection_log[key]

    # =========================================================================
    # Summary Statistics
    # =========================================================================

    def get_throughput(self, start_cycle: int = 0) -> float:
        """
        Calculate overall throughput (bytes/cycle).

        Args:
            start_cycle: Starting cycle for calculation (default: 0).

        Returns:
            Throughput in bytes/cycle.
        """
        if not self.snapshots:
            return 0.0

        end_snapshot = self.snapshots[-1]
        total_bytes = end_snapshot.bytes_transferred
        total_cycles = end_snapshot.cycle - start_cycle

        return total_bytes / total_cycles if total_cycles > 0 else 0.0

    def get_latency_stats(self) -> Dict[str, float]:
        """
        Calculate latency statistics.

        Returns:
            Dict with keys: min, max, avg, std, samples.
        """
        latencies = self.get_all_latencies()

        if not latencies:
            return {'min': 0, 'max': 0, 'avg': 0.0, 'std': 0.0, 'samples': 0}

        return {
            'min': min(latencies),
            'max': max(latencies),
            'avg': statistics.mean(latencies),
            'std': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            'samples': len(latencies),
        }

    def get_buffer_stats(self, total_capacity: int = 400) -> Dict[str, float]:
        """
        Calculate buffer utilization statistics.

        Args:
            total_capacity: Total buffer capacity in flits.
                           Default: 20 routers × 4 depth × 5 ports = 400.

        Returns:
            Dict with keys: peak, avg, utilization.
        """
        occupancies = [s.flits_in_flight for s in self.snapshots]

        if not occupancies:
            return {'peak': 0, 'avg': 0.0, 'utilization': 0.0}

        peak = max(occupancies)
        avg = statistics.mean(occupancies)

        return {
            'peak': peak,
            'avg': avg,
            'utilization': avg / total_capacity if total_capacity > 0 else 0.0,
        }

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
        """Clear all collected snapshots and latency tracking."""
        self.snapshots.clear()
        self._last_capture_cycle = -1
        self._last_completed = 0
        self._last_bytes = 0
        self._injection_log.clear()

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
        collector._injection_log = {}
        collector._zero_load_latency = None

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

    # =========================================================================
    # BookSim2-Style Metrics
    # =========================================================================

    def set_zero_load_latency(self, latency: float) -> None:
        """
        Set the zero-load latency (L₀) reference value.

        This should be calculated using TheoryValidator.calculate_zero_load_latency()
        based on the source and destination coordinates.

        Args:
            latency: Zero-load latency in cycles.
        """
        self._zero_load_latency = latency

    def get_booksim_metrics(self, bytes_per_flit: int = 32) -> BookSimMetrics:
        """
        Get BookSim2-style performance metrics.

        Returns a BookSimMetrics dataclass with:
        - throughput: bytes/cycle (measured, data flits only: W/R)
        - latency statistics
        - saturation detection (L > 3×L₀)

        Note: Call set_zero_load_latency() before using this method
        to enable saturation detection.

        Args:
            bytes_per_flit: Bytes per flit for display (default 32).
                Only used if no data bytes were tracked.

        Returns:
            BookSimMetrics dataclass with all metrics.
        """
        latency_stats = self.get_latency_stats()
        avg_latency = latency_stats.get("avg", 0.0)
        min_latency = latency_stats.get("min", 0)
        max_latency = latency_stats.get("max", 0)

        # Get cycle and packet counts
        total_cycles = self.snapshots[-1].cycle if self.snapshots else 1

        # Use tracked data bytes (W/R flits only) if available
        # Otherwise fall back to all flits * bytes_per_flit
        if self._data_bytes_received > 0:
            total_bytes = self._data_bytes_received
            total_packets = self._data_flits_received
        else:
            # Fallback for backward compatibility
            total_packets = latency_stats.get("samples", 0)
            total_bytes = total_packets * bytes_per_flit

        # Calculate throughput (bytes/cycle)
        throughput = total_bytes / total_cycles if total_cycles > 0 else 0.0

        # Determine zero-load latency
        zero_load = self._zero_load_latency
        if zero_load is None or zero_load <= 0:
            # Fallback: use minimum observed latency as approximation
            zero_load = float(min_latency) if min_latency > 0 else 1.0

        # Calculate saturation factor
        saturation_factor = avg_latency / zero_load if zero_load > 0 else 1.0
        saturation_status = calculate_saturation_status(saturation_factor)

        return BookSimMetrics(
            throughput=throughput,
            total_packets=total_packets,
            total_cycles=total_cycles,
            avg_latency=avg_latency,
            min_latency=float(min_latency),
            max_latency=float(max_latency),
            zero_load_latency=zero_load,
            saturation_factor=saturation_factor,
            saturation_status=saturation_status,
            bytes_per_flit=bytes_per_flit,
        )
