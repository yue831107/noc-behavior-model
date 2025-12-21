"""
Metrics Provider Protocol.

Defines a standard interface for systems to expose metrics data.
This decouples visualization from core implementation details.
"""

from __future__ import annotations

from typing import Protocol, Dict, Tuple, List, runtime_checkable


@runtime_checkable
class MetricsProvider(Protocol):
    """
    Protocol for systems that provide metrics for visualization.
    
    Systems (V1System, NoCSystem) should implement these methods
    to support MetricsCollector without tight coupling.
    """
    
    @property
    def current_cycle(self) -> int:
        """Current simulation cycle."""
        ...
    
    @property
    def mesh_dimensions(self) -> Tuple[int, int]:
        """Return (cols, rows) of the mesh."""
        ...
    
    def get_buffer_occupancy(self) -> Dict[Tuple[int, int], int]:
        """
        Get buffer occupancy for each router.
        
        Returns:
            Dict of (x, y) coordinate -> total buffer occupancy.
        """
        ...
    
    def get_flit_stats(self) -> Dict[Tuple[int, int], int]:
        """
        Get flit forwarding stats for each router.
        
        Returns:
            Dict of (x, y) coordinate -> flits forwarded count.
        """
        ...
    
    def get_transfer_stats(self) -> Tuple[int, int, int]:
        """
        Get transfer completion statistics.
        
        Returns:
            Tuple of (completed_transactions, bytes_transferred, transfer_size).
            transfer_size is per-transaction size (0 if unknown).
        """
        ...


def get_metrics_from_system(system) -> Dict:
    """
    Safely extract metrics from a system, regardless of implementation.
    
    Works with both Protocol-implementing systems and legacy systems.
    
    Args:
        system: V1System, NoCSystem, or any MetricsProvider.
    
    Returns:
        Dict with keys: cycle, buffer_occupancy, flit_stats, transfer_stats
    """
    # Try Protocol methods first
    if isinstance(system, MetricsProvider):
        cols, rows = system.mesh_dimensions
        completed, bytes_transferred, transfer_size = system.get_transfer_stats()
        return {
            'cycle': system.current_cycle,
            'mesh_cols': cols,
            'mesh_rows': rows,
            'buffer_occupancy': system.get_buffer_occupancy(),
            'flit_stats': system.get_flit_stats(),
            'completed_transactions': completed,
            'bytes_transferred': bytes_transferred,
            'transfer_size': transfer_size,
        }
    
    # Fallback for legacy systems
    return _extract_metrics_legacy(system)


def _extract_metrics_legacy(system) -> Dict:
    """
    Extract metrics from legacy systems without Protocol.
    
    This is the fallback path for backward compatibility.
    """
    # Get cycle
    if hasattr(system, 'current_cycle'):
        cycle = system.current_cycle
    elif hasattr(system, 'current_time'):
        cycle = system.current_time
    else:
        cycle = 0
    
    # Get mesh dimensions
    if hasattr(system, 'mesh_cols'):
        mesh_cols = system.mesh_cols
    elif hasattr(system, '_mesh_cols'):
        mesh_cols = system._mesh_cols
    else:
        mesh_cols = 5
    
    if hasattr(system, 'mesh_rows'):
        mesh_rows = system.mesh_rows
    elif hasattr(system, '_mesh_rows'):
        mesh_rows = system._mesh_rows
    else:
        mesh_rows = 4
    
    # Get buffer occupancy (flits in transit)
    buffer_occupancy = {}
    flit_stats = {}

    if hasattr(system, 'mesh') and hasattr(system.mesh, 'routers'):
        for coord, router in system.mesh.routers.items():
            occupancy = 0
            flit_count = 0

            if hasattr(router, 'req_router'):
                for port in router.req_router.ports.values():
                    if hasattr(port, 'input_buffer'):
                        occupancy += port.input_buffer.occupancy
                    # Count pending output signals
                    if hasattr(port, 'out_valid') and port.out_valid:
                        if hasattr(port, 'out_flit') and port.out_flit is not None:
                            occupancy += 1
                # Flits in pipeline stages
                if hasattr(router.req_router, 'flits_in_pipeline'):
                    occupancy += router.req_router.flits_in_pipeline
                if hasattr(router.req_router, 'stats'):
                    flit_count = router.req_router.stats.flits_forwarded

            if hasattr(router, 'resp_router'):
                for port in router.resp_router.ports.values():
                    if hasattr(port, 'input_buffer'):
                        occupancy += port.input_buffer.occupancy
                    if hasattr(port, 'out_valid') and port.out_valid:
                        if hasattr(port, 'out_flit') and port.out_flit is not None:
                            occupancy += 1
                if hasattr(router.resp_router, 'flits_in_pipeline'):
                    occupancy += router.resp_router.flits_in_pipeline
                if hasattr(router.resp_router, 'stats'):
                    flit_count += router.resp_router.stats.flits_forwarded

            buffer_occupancy[coord] = occupancy
            flit_stats[coord] = flit_count
    
    # Get transfer stats
    completed = 0
    bytes_transferred = 0
    transfer_size = 0
    
    if hasattr(system, 'host_axi_master') and system.host_axi_master:
        master = system.host_axi_master
        # Use controller_stats if available (more detailed)
        if hasattr(master, 'controller_stats'):
            c_stats = master.controller_stats
            completed = c_stats.completed_transactions + c_stats.read_completed
            bytes_transferred = c_stats.completed_bytes + c_stats.read_bytes_received
        else:
            # Fallback to HostAXIMasterStats
            stats = master.stats
            completed = getattr(stats, 'completed_transactions', 0)
            if not completed:
                completed = getattr(stats, 'b_received', 0) + getattr(stats, 'r_received', 0)
            bytes_transferred = getattr(stats, 'completed_bytes', 0)
    elif hasattr(system, 'node_controllers'):
        for controller in system.node_controllers.values():
            if hasattr(controller, 'stats'):
                completed += controller.stats.transfers_completed
        
        if hasattr(system, '_traffic_config') and system._traffic_config:
            transfer_size = system._traffic_config.transfer_size
            bytes_transferred = completed * transfer_size
    
    return {
        'cycle': cycle,
        'mesh_cols': mesh_cols,
        'mesh_rows': mesh_rows,
        'buffer_occupancy': buffer_occupancy,
        'flit_stats': flit_stats,
        'completed_transactions': completed,
        'bytes_transferred': bytes_transferred,
        'transfer_size': transfer_size,
    }
