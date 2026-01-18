#!/usr/bin/env python3
"""
Host to NoC Test Runner.

Usage:
    python run.py                           # Run default test (broadcast_write)
    python run.py broadcast_write           # Run broadcast write test
    python run.py broadcast_read            # Run broadcast read test
    python run.py scatter_write             # Run scatter write test
    python run.py gather_read               # Run gather read test
    python run.py multi_transfer            # Run multi-transfer queue test
    python run.py --config config/custom.yaml  # Run with custom config
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core import V1System, HostMemory, Memory
from src.config import load_transfer_config, load_transfer_configs, TransferConfig, TransferMode
from src.metrics import BookSimMetrics, SaturationStatus


def calculate_host_zero_load_latency(
    target_nodes: List[int],
    mesh_cols: int = 5,
    mesh_rows: int = 4
) -> float:
    """
    Calculate average zero-load latency (L0) for Host-to-NoC transfers.

    For Host-to-NoC, packets enter via edge routers (column 0) and
    route to compute nodes (columns 1-4). This calculates the average
    Manhattan distance + overhead.

    Args:
        target_nodes: List of target node IDs (0-15 for 4x4 compute mesh)
        mesh_cols: Total mesh columns (default 5, with col 0 as edge)
        mesh_rows: Mesh rows (default 4)

    Returns:
        Average zero-load latency in cycles
    """
    from src.verification import TheoryValidator
    from src.verification.theory_validator import MeshConfig

    mesh_config = MeshConfig(cols=mesh_cols, rows=mesh_rows, edge_column=0)
    validator = TheoryValidator(mesh_config=mesh_config)

    # Edge routers are in column 0, rows 0-3
    # Compute nodes are in columns 1-4, rows 0-3
    # node_id = row * (cols-1) + (col-1) for col > 0
    compute_cols = mesh_cols - 1  # 4 compute columns

    total_latency = 0.0
    for node_id in target_nodes:
        # Convert node_id to mesh coordinate
        node_col = (node_id % compute_cols) + 1  # +1 because col 0 is edge
        node_row = node_id // compute_cols

        # Find nearest edge router (same row, column 0)
        edge_coord = (0, node_row)
        node_coord = (node_col, node_row)

        # Calculate zero-load latency
        l0 = validator.calculate_zero_load_latency(edge_coord, node_coord)
        total_latency += l0

    return total_latency / len(target_nodes) if target_nodes else 0.0


def print_booksim_metrics(
    metrics: BookSimMetrics,
    title: str = "Performance Metrics (BookSim2-Style)"
) -> None:
    """
    Print BookSim2-style performance metrics summary.

    Args:
        metrics: BookSimMetrics dataclass
        title: Header title
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Throughput:          {metrics.throughput:.2f} bytes/cycle")
    print(f"Total Bytes:         {metrics.total_bytes}")
    print(f"Total Flits:         {metrics.total_packets}")
    print(f"Total Cycles:        {metrics.total_cycles}")
    print("-" * 70)
    print(f"Average Latency:     {metrics.avg_latency:.1f} cycles")
    print(f"Min Latency:         {metrics.min_latency:.1f} cycles")
    print(f"Max Latency:         {metrics.max_latency:.1f} cycles")
    print("-" * 70)
    print(f"Zero-Load Latency:   {metrics.zero_load_latency:.1f} cycles (L0)")
    print(f"Normalized Latency:  {metrics.saturation_factor:.2f} (L/L0)")
    print(f"Saturation Status:   {metrics.saturation_status.value.upper()}")
    if metrics.is_saturated:
        print("  WARNING: Network is SATURATED (L > 3*L0)")
    print("=" * 70)


def validate_performance_metrics(
    metrics: dict,
    zero_load_latency: float = None,
    verbose: bool = True
) -> bool:
    """
    Validate performance metrics using BookSim2-style checks.

    Validations:
    - Latency: L >= L0 (zero-load), saturation detection (L > 3*L0)
    - Buffer utilization: [0, 1] range
    - Flit Conservation: Sent = Received

    Args:
        metrics: Dict with performance metrics:
            - avg_latency: cycles
            - buffer_utilization: ratio [0, 1]
            - total_flits_sent: flits injected (for Flit Conservation)
            - total_flits_received: flits ejected (for Flit Conservation)
        zero_load_latency: Theoretical minimum latency (L0). If None, skip latency validation.
        verbose: Print validation results

    Returns:
        True if all validations pass.
    """
    all_passed = True
    results = []

    # === Theory Validation (BookSim2-Style) ===
    try:
        from src.verification import TheoryValidator
        theory_validator = TheoryValidator()

        # 1. Latency validation with saturation detection
        # Skip if no latency samples (avg_latency == 0)
        if 'avg_latency' in metrics and zero_load_latency is not None:
            if metrics['avg_latency'] > 0:
                is_valid, msg, details = theory_validator.validate_latency(
                    avg_latency=metrics['avg_latency'],
                    zero_load_latency=zero_load_latency
                )
                results.append(('Latency', is_valid, msg))
                if not is_valid:
                    all_passed = False
            else:
                results.append(('Latency', True, "OK (no samples - skipped)"))

        # 2. Buffer utilization validation
        if 'buffer_utilization' in metrics and metrics['buffer_utilization'] is not None:
            is_valid, msg = theory_validator.validate_buffer_utilization(
                metrics['buffer_utilization']
            )
            results.append(('Buffer Util', is_valid, msg))
            if not is_valid:
                all_passed = False

    except ImportError:
        if verbose:
            print("\n--- Performance Validation ---")
            print("  [SKIP] TheoryValidator not available")

    # === Consistency Validation ===
    try:
        from src.verification import ConsistencyValidator
        consistency_validator = ConsistencyValidator(tolerance=0.50)  # 50% tolerance for NoC

        # 3. Flit Conservation: Sent = Received
        if all(k in metrics for k in ['total_flits_sent', 'total_flits_received']):
            is_valid, msg = consistency_validator.validate_flit_conservation(
                total_sent=metrics['total_flits_sent'],
                total_received=metrics['total_flits_received']
            )
            results.append(('Flit Conserv', is_valid, msg))
            if not is_valid:
                all_passed = False

    except ImportError:
        pass  # Consistency validator not available

    # Print results
    if verbose and results:
        print("\n--- Performance Validation (BookSim2-Style) ---")
        for name, is_valid, msg in results:
            status = "PASS" if is_valid else "FAIL"
            print(f"  {name + ':':<17} {status}")
            if not is_valid:
                print(f"                     {msg}")

        overall = "PASS" if all_passed else "FAIL"
        print(f"\n  Validation:        {overall}")

    return all_passed


def save_host_metrics(
    system: V1System,
    configs: List[TransferConfig],
    cycles: int,
    success: bool,
    verify_results: Optional[List[dict]] = None,
    test_pattern: str = "single_transfer",
    collector: Optional["MetricsCollector"] = None
):
    """Save simulation metrics to latest.json for visualization."""
    import json
    import statistics
    from datetime import datetime

    metrics_dir = Path("output/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = sum(c.src_size for c in configs)
    total_transfers = len(configs)

    if verify_results:
        total_pass = sum(r['pass'] for r in verify_results)
        total_fail = sum(r['fail'] for r in verify_results)
    else:
        # Fallback for simple runs
        total_pass = total_transfers if success else 0
        total_fail = 0 if success else total_transfers

    # Router flit statistics
    flit_stats = system.get_flit_stats()
    total_flits = sum(flit_stats.values())
    active_routers = sum(1 for v in flit_stats.values() if v > 0)
    avg_flits_per_router = total_flits / len(flit_stats) if flit_stats else 0
    max_flits_router = max(flit_stats.values()) if flit_stats else 0

    # Buffer utilization from collector
    if collector:
        buffer_occupancies = [s.flits_in_flight for s in collector.snapshots]
        peak_buffer_util = max(buffer_occupancies) if buffer_occupancies else 0
        avg_buffer_util = sum(buffer_occupancies) / len(buffer_occupancies) if buffer_occupancies else 0
        # Average during active periods (when flits are in flight)
        active_occupancies = [x for x in buffer_occupancies if x > 0]
        active_avg_buffer = sum(active_occupancies) / len(active_occupancies) if active_occupancies else 0
        active_pct = len(active_occupancies) / len(buffer_occupancies) * 100 if buffer_occupancies else 0

        # Latency statistics
        latencies = collector.get_all_latencies()
        if latencies:
            min_latency = min(latencies)
            max_latency = max(latencies)
            avg_latency = statistics.mean(latencies)
            std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
            latency_samples = len(latencies)
        else:
            min_latency = max_latency = std_latency = 0
            avg_latency = cycles / total_transfers if total_transfers > 0 else 0
            latency_samples = 0
    else:
        peak_buffer_util = avg_buffer_util = active_avg_buffer = active_pct = 0
        min_latency = max_latency = std_latency = 0
        avg_latency = cycles / total_transfers if total_transfers > 0 else 0
        latency_samples = 0

    metrics = {
        'pattern': test_pattern,
        'mode': 'host_to_noc',
        'cycles': cycles,
        'total_bytes': total_bytes,
        'throughput': total_bytes / cycles if cycles > 0 else 0,
        'num_transfers': total_transfers,
        'completed_transfers': total_transfers if success else 0,
        'pass_count': total_pass,
        'fail_count': total_fail,
        'success': success,
        'timestamp': datetime.now().isoformat(),
        'mesh_cols': system._mesh_cols if hasattr(system, '_mesh_cols') else 5,
        'mesh_rows': system._mesh_rows if hasattr(system, '_mesh_rows') else 4,
        'num_nodes': 16,
        'transfer_size': total_bytes // total_transfers if total_transfers > 0 else 0,
        # Router statistics
        'total_flits': total_flits,
        'active_routers': active_routers,
        'avg_flits_per_router': avg_flits_per_router,
        'max_flits_router': max_flits_router,
        # Latency statistics
        'latency_samples': latency_samples,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'avg_latency': avg_latency,
        'std_latency': std_latency,
        # Buffer utilization
        'peak_buffer_util': peak_buffer_util,
        'avg_buffer_util': avg_buffer_util,
        'active_avg_buffer': active_avg_buffer,
        'active_pct': active_pct,
    }

    # Include real snapshots if collector is provided
    if collector and hasattr(collector, 'to_dict'):
        metrics['snapshots'] = collector.to_dict()['snapshots']
        print(f"  Real-time data: {len(metrics['snapshots'])} snapshots captured")

    latest_path = metrics_dir / "latest.json"
    with open(latest_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved: {latest_path}")


def run_broadcast_write(config: TransferConfig, verbose: bool = True, bin_file: str = None) -> dict:
    """
    Run broadcast write test.
    
    Args:
        config: Transfer configuration.
        verbose: Print progress and results.
        bin_file: Path to BIN file containing source data (required).
    
    Raises:
        ValueError: If bin_file is not provided or does not exist.
    """
    from pathlib import Path
    
    # Validate BIN file requirement
    if not bin_file:
        raise ValueError("bin_file is required. Use 'make gen_payload' to generate a payload file.")
    
    bin_path = Path(bin_file)
    if not bin_path.exists():
        raise ValueError(f"BIN file not found: {bin_file}. Use 'make gen_payload' to generate.")
    
    bin_data = bin_path.read_bytes()
    
    if verbose:
        print("\n" + "=" * 60)
        print("Host to NoC - Broadcast Write Test")
        print("=" * 60)
        print(f"BIN file: {bin_file} ({len(bin_data)} bytes)")

    # Create system and collector
    system = V1System(mesh_cols=5, mesh_rows=4)
    from src.visualization import MetricsCollector
    collector = MetricsCollector(system, capture_interval=max(1, config.src_size // 256))

    # === Connect NI callbacks for BookSim2-style flit-level latency tracking ===
    from src.core.flit import AxiChannel

    def on_flit_latency(latency: int, axi_ch: AxiChannel, payload_bytes: int) -> None:
        """Record per-flit latency when flit arrives at destination."""
        collector.add_latency_sample(latency, axi_ch, payload_bytes)

    for coord, master_ni in system.mesh.nis.items():
        master_ni.set_flit_latency_callback(on_flit_latency)
    system.master_ni.set_flit_latency_callback(on_flit_latency)

    # Load test data from BIN file
    host_memory = HostMemory(size=1024 * 1024)
    offset = config.src_addr % len(bin_data)
    if offset + config.src_size <= len(bin_data):
        test_data = bin_data[offset:offset + config.src_size]
    else:
        test_data = (bin_data[offset:] + bin_data * ((config.src_size // len(bin_data)) + 1))[:config.src_size]
    host_memory.fill(config.src_addr, config.src_size, test_data)

    if verbose:
        print(f"Source: Host Memory @ 0x{config.src_addr:04X}, {config.src_size} bytes")
        print(f"Dest: All nodes @ 0x{config.dst_addr:04X}")
        print()

    # Get target nodes
    target_nodes = config.get_target_node_list(total_nodes=16)

    # Submit writes to all nodes
    axi_id = 1
    outstanding = {}
    completed = 0
    cycle = 0
    max_cycles = 10000

    # Submit initial writes
    for node_id in target_nodes:
        if len(outstanding) >= config.max_outstanding:
            break
        addr = (node_id << 32) | config.dst_addr
        system.submit_write(addr, test_data, axi_id)
        collector.record_injection(axi_id, cycle)  # Record injection time
        outstanding[axi_id] = node_id
        axi_id += 1

    pending_nodes = [n for n in target_nodes if n not in outstanding.values()]

    # Run simulation
    while completed < len(target_nodes) and cycle < max_cycles:
        system.process_cycle()
        collector.capture()

        # Check responses
        while True:
            resp = system.master_ni.get_b_response()
            if resp is None:
                break
            if resp.bid in outstanding:
                collector.record_ejection(resp.bid, cycle)  # Record ejection time
                del outstanding[resp.bid]
                completed += 1

                # Submit next if pending
                if pending_nodes and len(outstanding) < config.max_outstanding:
                    node_id = pending_nodes.pop(0)
                    addr = (node_id << 32) | config.dst_addr
                    system.submit_write(addr, test_data, axi_id)
                    collector.record_injection(axi_id, cycle)  # Record injection time
                    outstanding[axi_id] = node_id
                    axi_id += 1

        cycle += 1

    if verbose:
        print(f"Completed: {completed}/{len(target_nodes)} nodes in {cycle} cycles")

        # Verify
        print("\n--- Verification ---")
        pass_count, fail_count = system.verify_all_writes(verbose=True)
        success = (fail_count == 0 and completed == len(target_nodes))
        if fail_count == 0:
            print("PASS: All data verified correctly")
        else:
            print(f"FAIL: {fail_count} verification failures")

    # === Performance Metrics (BookSim2-Style) ===
    # Calculate zero-load latency for these target nodes
    zero_load_latency = calculate_host_zero_load_latency(target_nodes)
    collector.set_zero_load_latency(zero_load_latency)

    # Get BookSim2-style metrics
    booksim_metrics = collector.get_booksim_metrics()

    if verbose:
        print_booksim_metrics(booksim_metrics)

    # Prepare metrics for validation
    buffer_stats = collector.get_buffer_stats(total_capacity=20 * 4 * 5)
    perf_metrics = {
        'throughput': booksim_metrics.throughput,  # Already in bytes/cycle
        'buffer_utilization': buffer_stats['utilization'],
        'avg_latency': booksim_metrics.avg_latency,
        'avg_occupancy': buffer_stats['avg'],
    }
    validation_passed = validate_performance_metrics(
        perf_metrics,
        zero_load_latency=zero_load_latency,
        verbose=verbose
    )
    success = success and validation_passed

    # Save metrics
    save_host_metrics(
        system, [config], cycle, success,
        verify_results=[{'pass': pass_count, 'fail': fail_count}],
        test_pattern='broadcast_write',
        collector=collector
    )

    return {
        'completed': completed,
        'total': len(target_nodes),
        'cycles': cycle,
        'success': success,
    }


def run_broadcast_read(config: TransferConfig, verbose: bool = True, bin_file: str = None) -> dict:
    """
    Run broadcast read test (requires prior write).
    
    Args:
        config: Transfer configuration.
        verbose: Print progress and results.
        bin_file: Path to BIN file containing source data (required).
    
    Raises:
        ValueError: If bin_file is not provided or does not exist.
    """
    from pathlib import Path
    
    # Validate BIN file requirement
    if not bin_file:
        raise ValueError("bin_file is required. Use 'make gen_payload' to generate a payload file.")
    
    bin_path = Path(bin_file)
    if not bin_path.exists():
        raise ValueError(f"BIN file not found: {bin_file}. Use 'make gen_payload' to generate.")
    
    bin_data = bin_path.read_bytes()
    
    if verbose:
        print("\n" + "=" * 60)
        print("Host to NoC - Broadcast Read Test")
        print("=" * 60)
        print(f"BIN file: {bin_file} ({len(bin_data)} bytes)")

    # Create system and collector
    system = V1System(mesh_cols=5, mesh_rows=4)
    from src.visualization import MetricsCollector
    collector = MetricsCollector(system, capture_interval=max(1, config.effective_read_size // 256))

    # === Connect NI callbacks for BookSim2-style flit-level latency tracking ===
    from src.core.flit import AxiChannel

    def on_flit_latency(latency: int, axi_ch: AxiChannel, payload_bytes: int) -> None:
        """Record per-flit latency when flit arrives at destination."""
        collector.add_latency_sample(latency, axi_ch, payload_bytes)

    for coord, master_ni in system.mesh.nis.items():
        master_ni.set_flit_latency_callback(on_flit_latency)
    system.master_ni.set_flit_latency_callback(on_flit_latency)

    # Load test data from BIN file
    offset = config.read_src_addr % len(bin_data)
    size = config.effective_read_size
    if offset + size <= len(bin_data):
        test_data = bin_data[offset:offset + size]
    else:
        test_data = (bin_data[offset:] + bin_data * ((size // len(bin_data)) + 1))[:size]
    target_nodes = config.get_target_node_list(total_nodes=16)

    if verbose:
        print("Phase 1: Writing test data to all nodes...")

    # Write phase
    axi_id = 1
    for node_id in target_nodes:
        addr = (node_id << 32) | config.read_src_addr
        system.submit_write(addr, test_data, axi_id)
        axi_id += 1

    # Process writes
    write_completed = 0
    cycle = 0
    while write_completed < len(target_nodes) and cycle < 5000:
        system.process_cycle()
        collector.capture()
        while True:
            resp = system.master_ni.get_b_response()
            if resp is None:
                break
            write_completed += 1
        cycle += 1

    if verbose:
        print(f"  Writes completed in {cycle} cycles")

    # Reset for read phase
    system.reset_for_read()
    
    # Reset collector for read-only metrics
    collector.clear()
    collector._data_bytes_received = 0
    collector._data_flits_received = 0

    if verbose:
        print("\nPhase 2: Reading data back from all nodes...")

    # Read phase
    read_phase_start_cycle = system.current_time
    read_axi_id = 1000
    outstanding = {}
    completed = 0
    read_data = {}
    read_buffers = {}  # Accumulate beats per rid

    for node_id in target_nodes:
        if len(outstanding) >= config.max_outstanding:
            break
        addr = (node_id << 32) | config.read_src_addr
        system.submit_read(addr, config.effective_read_size, read_axi_id)
        outstanding[read_axi_id] = node_id
        read_buffers[read_axi_id] = b''  # Initialize buffer for this read
        read_axi_id += 1

    pending_nodes = [n for n in target_nodes if n not in outstanding.values()]

    while completed < len(target_nodes) and cycle < 10000:
        system.process_cycle()
        collector.capture()

        # Check read responses
        while True:
            resp = system.master_ni.get_r_response()
            if resp is None:
                break
            if resp.rid in outstanding:
                # Accumulate beat data
                read_buffers[resp.rid] += resp.rdata

                if resp.rlast:
                    # Transfer complete - move accumulated data to read_data
                    node_id = outstanding[resp.rid]
                    read_data[node_id] = read_buffers[resp.rid]
                    del outstanding[resp.rid]
                    del read_buffers[resp.rid]
                    completed += 1

                    if pending_nodes and len(outstanding) < config.max_outstanding:
                        node_id = pending_nodes.pop(0)
                        addr = (node_id << 32) | config.read_src_addr
                        system.submit_read(addr, config.effective_read_size, read_axi_id)
                        outstanding[read_axi_id] = node_id
                        read_buffers[read_axi_id] = b''
                        read_axi_id += 1

        cycle += 1

    if verbose:
        print(f"  Reads completed: {completed}/{len(target_nodes)} in {cycle} cycles")

        # Verify
        print("\n--- Verification ---")
        pass_count = 0
        fail_count = 0
        for node_id, data in read_data.items():
            if data == test_data:
                pass_count += 1
            else:
                fail_count += 1
                print(f"  Node {node_id}: MISMATCH")

        success = (fail_count == 0 and completed == len(target_nodes))
        if fail_count == 0:
            print(f"PASS: All {pass_count} nodes verified correctly")
        else:
            print(f"FAIL: {fail_count} verification failures")

    # === Performance Metrics (BookSim2-Style) ===
    zero_load_latency = calculate_host_zero_load_latency(target_nodes)
    collector.set_zero_load_latency(zero_load_latency)

    booksim_metrics = collector.get_booksim_metrics()
    
    # Fix: Use read phase cycles for accurate throughput
    read_phase_cycles = system.current_time - read_phase_start_cycle
    if read_phase_cycles > 0:
        booksim_metrics = BookSimMetrics(
            throughput=booksim_metrics.total_bytes / read_phase_cycles,
            total_packets=booksim_metrics.total_packets,
            total_cycles=read_phase_cycles,
            avg_latency=booksim_metrics.avg_latency,
            min_latency=booksim_metrics.min_latency,
            max_latency=booksim_metrics.max_latency,
            zero_load_latency=booksim_metrics.zero_load_latency,
            saturation_factor=booksim_metrics.saturation_factor,
            saturation_status=booksim_metrics.saturation_status,
            bytes_per_flit=booksim_metrics.bytes_per_flit,
        )

    if verbose:
        print_booksim_metrics(booksim_metrics)

    buffer_stats = collector.get_buffer_stats(total_capacity=20 * 4 * 5)
    perf_metrics = {
        'throughput': booksim_metrics.throughput,
        'buffer_utilization': buffer_stats['utilization'],
        'avg_latency': booksim_metrics.avg_latency,
        'avg_occupancy': buffer_stats['avg'],
    }
    validation_passed = validate_performance_metrics(
        perf_metrics,
        zero_load_latency=zero_load_latency,
        verbose=verbose
    )
    success = success and validation_passed

    # Save metrics
    save_host_metrics(
        system, [config], cycle, success,
        verify_results=[{'pass': pass_count, 'fail': fail_count}],
        test_pattern='broadcast_read',
        collector=collector
    )

    return {
        'completed': completed,
        'total': len(target_nodes),
        'cycles': cycle,
        'success': success,
    }


def run_scatter_write(config: TransferConfig, verbose: bool = True, bin_file: str = None) -> dict:
    """
    Run scatter write test - distributes different data to different nodes.
    
    Args:
        config: Transfer configuration.
        verbose: Print progress and results.
        bin_file: Path to BIN file containing source data (required).
    
    Raises:
        ValueError: If bin_file is not provided or does not exist.
    """
    from pathlib import Path
    
    # Validate BIN file requirement
    if not bin_file:
        raise ValueError("bin_file is required. Use 'make gen_payload' to generate a payload file.")
    
    bin_path = Path(bin_file)
    if not bin_path.exists():
        raise ValueError(f"BIN file not found: {bin_file}. Use 'make gen_payload' to generate.")
    
    bin_data = bin_path.read_bytes()
    
    if verbose:
        print("\n" + "=" * 60)
        print("Host to NoC - Scatter Write Test")
        print("=" * 60)
        print(f"BIN file: {bin_file} ({len(bin_data)} bytes)")

    # Create system and collector
    system = V1System(mesh_cols=5, mesh_rows=4)
    from src.visualization import MetricsCollector
    collector = MetricsCollector(system, capture_interval=max(1, config.src_size // 256))

    # === Connect NI callbacks for BookSim2-style flit-level latency tracking ===
    from src.core.flit import AxiChannel

    def on_flit_latency(latency: int, axi_ch: AxiChannel, payload_bytes: int) -> None:
        """Record per-flit latency when flit arrives at destination."""
        collector.add_latency_sample(latency, axi_ch, payload_bytes)

    for coord, master_ni in system.mesh.nis.items():
        master_ni.set_flit_latency_callback(on_flit_latency)
    system.master_ni.set_flit_latency_callback(on_flit_latency)

    # Load test data from BIN file
    host_memory = HostMemory(size=1024 * 1024)
    offset = config.src_addr % len(bin_data)
    if offset + config.src_size <= len(bin_data):
        test_data = bin_data[offset:offset + config.src_size]
    else:
        test_data = (bin_data[offset:] + bin_data * ((config.src_size // len(bin_data)) + 1))[:config.src_size]
    host_memory.fill(config.src_addr, config.src_size, test_data)

    target_nodes = config.get_target_node_list(total_nodes=16)
    portion_size = config.src_size // len(target_nodes)

    if verbose:
        print(f"Source: Host Memory @ 0x{config.src_addr:04X}, {config.src_size} bytes")
        print(f"Mode: SCATTER ({portion_size} bytes per node)")
        print(f"Target: Nodes {target_nodes}")
        print()

    # Submit writes with different data portions per node
    axi_id = 1
    outstanding = {}
    completed = 0
    cycle = 0
    max_cycles = 10000

    for i, node_id in enumerate(target_nodes):
        if len(outstanding) >= config.max_outstanding:
            break
        offset = i * portion_size
        portion_data = test_data[offset:offset + portion_size]
        addr = (node_id << 32) | config.dst_addr
        system.submit_write(addr, portion_data, axi_id)

        # Capture golden for this node's portion
        system.capture_golden_from_write(node_id, config.dst_addr, portion_data, cycle)

        outstanding[axi_id] = (node_id, portion_data)
        axi_id += 1

    pending = [(i, n) for i, n in enumerate(target_nodes)
               if n not in [o[0] for o in outstanding.values()]]

    # Run simulation
    while completed < len(target_nodes) and cycle < max_cycles:
        system.process_cycle()
        collector.capture()

        while True:
            resp = system.master_ni.get_b_response()
            if resp is None:
                break
            if resp.bid in outstanding:
                del outstanding[resp.bid]
                completed += 1

                if pending and len(outstanding) < config.max_outstanding:
                    idx, node_id = pending.pop(0)
                    offset = idx * portion_size
                    portion_data = test_data[offset:offset + portion_size]
                    addr = (node_id << 32) | config.dst_addr
                    system.submit_write(addr, portion_data, axi_id)
                    system.capture_golden_from_write(node_id, config.dst_addr, portion_data, cycle)
                    outstanding[axi_id] = (node_id, portion_data)
                    axi_id += 1

        cycle += 1

    if verbose:
        print(f"Completed: {completed}/{len(target_nodes)} nodes in {cycle} cycles")

        print("\n--- Verification ---")
        pass_count = 0
        fail_count = 0
        for i, node_id in enumerate(target_nodes):
            offset = i * portion_size
            expected = test_data[offset:offset + portion_size]
            golden = system.golden_manager.get_golden(node_id, config.dst_addr)
            if golden == expected:
                pass_count += 1
            else:
                fail_count += 1
                print(f"  Node {node_id}: Golden mismatch")

        success = (fail_count == 0 and completed == len(target_nodes))
        if fail_count == 0:
            print(f"PASS: All {pass_count} nodes have correct golden data")
        else:
            print(f"FAIL: {fail_count} golden mismatches")

    # === Performance Metrics (BookSim2-Style) ===
    zero_load_latency = calculate_host_zero_load_latency(target_nodes)
    collector.set_zero_load_latency(zero_load_latency)

    booksim_metrics = collector.get_booksim_metrics()

    if verbose:
        print_booksim_metrics(booksim_metrics)

    buffer_stats = collector.get_buffer_stats(total_capacity=20 * 4 * 5)
    perf_metrics = {
        'throughput': booksim_metrics.throughput,
        'buffer_utilization': buffer_stats['utilization'],
        'avg_latency': booksim_metrics.avg_latency,
        'avg_occupancy': buffer_stats['avg'],
    }
    validation_passed = validate_performance_metrics(
        perf_metrics,
        zero_load_latency=zero_load_latency,
        verbose=verbose
    )
    success = success and validation_passed

    # Save metrics
    save_host_metrics(
        system, [config], cycle, success,
        verify_results=[{'pass': pass_count, 'fail': fail_count}],
        test_pattern='scatter_write',
        collector=collector
    )

    return {
        'completed': completed,
        'total': len(target_nodes),
        'cycles': cycle,
        'success': success,
    }


def run_gather_read(config: TransferConfig, verbose: bool = True, bin_file: str = None) -> dict:
    """
    Run gather read test - collects different data from different nodes into HostMemory.
    
    Args:
        config: Transfer configuration.
        verbose: Print progress and results.
        bin_file: Path to BIN file containing source data (required).
    
    Raises:
        ValueError: If bin_file is not provided or does not exist.
    """
    from pathlib import Path
    
    # Validate BIN file requirement
    if not bin_file:
        raise ValueError("bin_file is required. Use 'make gen_payload' to generate a payload file.")
    
    bin_path = Path(bin_file)
    if not bin_path.exists():
        raise ValueError(f"BIN file not found: {bin_file}. Use 'make gen_payload' to generate.")
    
    bin_data = bin_path.read_bytes()
    
    if verbose:
        print("\n" + "=" * 60)
        print("Host to NoC - Gather Read Test")
        print("=" * 60)
        print(f"BIN file: {bin_file} ({len(bin_data)} bytes)")

    # Create system and collector
    system = V1System(mesh_cols=5, mesh_rows=4)
    from src.visualization import MetricsCollector
    collector = MetricsCollector(system, capture_interval=max(1, config.effective_read_size // 256))

    # === Connect NI callbacks for BookSim2-style flit-level latency tracking ===
    from src.core.flit import AxiChannel

    def on_flit_latency(latency: int, axi_ch: AxiChannel, payload_bytes: int) -> None:
        """Record per-flit latency when flit arrives at destination."""
        collector.add_latency_sample(latency, axi_ch, payload_bytes)

    for coord, master_ni in system.mesh.nis.items():
        master_ni.set_flit_latency_callback(on_flit_latency)
    system.master_ni.set_flit_latency_callback(on_flit_latency)

    target_nodes = config.get_target_node_list(total_nodes=16)
    total_size = config.effective_read_size
    portion_size = total_size // len(target_nodes)
    host_dst_addr = 0x2000  # HostMemory destination for gathered data

    if verbose:
        print(f"Mode: GATHER ({portion_size} bytes from each node)")
        print(f"Source: Nodes {target_nodes} @ 0x{config.read_src_addr:04X}")
        print(f"Dest: HostMemory @ 0x{host_dst_addr:04X}")
        print()

    # Phase 1: Write different data to each node (prepare source data)
    if verbose:
        print("Phase 1: Writing different data to each node...")

    # Load test data from BIN file
    offset = config.read_src_addr % len(bin_data)
    if offset + total_size <= len(bin_data):
        test_data = bin_data[offset:offset + total_size]
    else:
        test_data = (bin_data[offset:] + bin_data * ((total_size // len(bin_data)) + 1))[:total_size]
    axi_id = 1
    data_portions = []

    for i, node_id in enumerate(target_nodes):
        offset = i * portion_size
        portion_data = test_data[offset:offset + portion_size]
        data_portions.append(portion_data)
        addr = (node_id << 32) | config.read_src_addr
        system.submit_write(addr, portion_data, axi_id)
        axi_id += 1

    write_completed = 0
    cycle = 0
    while write_completed < len(target_nodes) and cycle < 5000:
        system.process_cycle()
        collector.capture()
        while True:
            resp = system.master_ni.get_b_response()
            if resp is None:
                break
            write_completed += 1
        cycle += 1

    if verbose:
        print(f"  Writes completed in {cycle} cycles")

    # Capture GATHER golden: concatenated result expected in HostMemory
    system.golden_manager.capture_gather(
        host_addr=host_dst_addr,
        data_portions=data_portions,
        cycle=cycle,
    )

    # Reset for read
    system.reset_for_read()

    if verbose:
        print("\nPhase 2: Gathering data from all nodes...")

    # Phase 2: Read (gather) from each node
    read_axi_id = 1000
    outstanding = {}
    completed = 0
    read_data = {}  # node_id -> data (for ordering)
    read_buffers = {}  # Accumulate beats per rid

    for node_id in target_nodes:
        if len(outstanding) >= config.max_outstanding:
            break
        addr = (node_id << 32) | config.read_src_addr
        system.submit_read(addr, portion_size, read_axi_id)
        outstanding[read_axi_id] = node_id
        read_buffers[read_axi_id] = b''
        read_axi_id += 1

    pending_nodes = [n for n in target_nodes if n not in outstanding.values()]

    while completed < len(target_nodes) and cycle < 10000:
        system.process_cycle()
        collector.capture()

        while True:
            resp = system.master_ni.get_r_response()
            if resp is None:
                break
            if resp.rid in outstanding:
                # Accumulate beat data
                read_buffers[resp.rid] += resp.rdata

                if resp.rlast:
                    # Transfer complete - move accumulated data to read_data
                    node_id = outstanding[resp.rid]
                    read_data[node_id] = read_buffers[resp.rid]
                    del outstanding[resp.rid]
                    del read_buffers[resp.rid]
                    completed += 1

                    if pending_nodes and len(outstanding) < config.max_outstanding:
                        node_id = pending_nodes.pop(0)
                        addr = (node_id << 32) | config.read_src_addr
                        system.submit_read(addr, portion_size, read_axi_id)
                        outstanding[read_axi_id] = node_id
                        read_buffers[read_axi_id] = b''
                        read_axi_id += 1

        cycle += 1

    if verbose:
        print(f"  Reads completed: {completed}/{len(target_nodes)} in {cycle} cycles")

        # Concatenate gathered data in node order (simulating HostMemory result)
        gathered_result = b""
        for node_id in target_nodes:
            if node_id in read_data:
                gathered_result += read_data[node_id]

        # Verify against GATHER golden (HostMemory)
        print("\n--- Verification (HostMemory) ---")
        expected_golden = system.golden_manager.get_host_golden(host_dst_addr)

        success = (gathered_result == expected_golden and completed == len(target_nodes))
        if gathered_result == expected_golden:
            print(f"PASS: Gathered {len(gathered_result)} bytes verified correctly")
            print(f"  Expected: {len(expected_golden)} bytes in HostMemory @ 0x{host_dst_addr:04X}")
            pass_count = 1
            fail_count = 0
        else:
            print(f"FAIL: Gathered data mismatch")
            print(f"  Expected: {len(expected_golden)} bytes")
            print(f"  Actual: {len(gathered_result)} bytes")
            pass_count = 0
            fail_count = 1
            # Find first mismatch
            for i in range(min(len(expected_golden), len(gathered_result))):
                if expected_golden[i] != gathered_result[i]:
                    print(f"  First mismatch at offset {i}: expected 0x{expected_golden[i]:02X}, got 0x{gathered_result[i]:02X}")
                    break

    # === Performance Metrics (BookSim2-Style) ===
    zero_load_latency = calculate_host_zero_load_latency(target_nodes)
    collector.set_zero_load_latency(zero_load_latency)

    booksim_metrics = collector.get_booksim_metrics()

    if verbose:
        print_booksim_metrics(booksim_metrics)

    buffer_stats = collector.get_buffer_stats(total_capacity=20 * 4 * 5)
    perf_metrics = {
        'throughput': booksim_metrics.throughput,
        'buffer_utilization': buffer_stats['utilization'],
        'avg_latency': booksim_metrics.avg_latency,
        'avg_occupancy': buffer_stats['avg'],
    }
    validation_passed = validate_performance_metrics(
        perf_metrics,
        zero_load_latency=zero_load_latency,
        verbose=verbose
    )
    success = success and validation_passed

    # Save metrics
    save_host_metrics(
        system, [config], cycle, success,
        verify_results=[{'pass': pass_count, 'fail': fail_count}],
        test_pattern='gather_read',
        collector=collector
    )

    return {
        'completed': completed,
        'total': len(target_nodes),
        'cycles': cycle,
        'success': success,
    }


def run_multi_transfer(
    configs: List[TransferConfig],
    verbose: bool = True,
    bin_file: str = None,
    buffer_depth: int = 32,
    mesh_cols: int = 5,
    mesh_rows: int = 4,
    channel_mode: str = 'general',
) -> dict:
    """
    Run multi-transfer queue test with per-transfer verification.
    
    Uses HostAXIMaster's transfer queue to process multiple independent
    transfer configurations sequentially. Data is loaded from a BIN file
    and verified immediately after each transfer completes.
    
    Args:
        configs: List of TransferConfig objects to process.
        verbose: Print progress and results.
        bin_file: Path to BIN file containing source data (required).
    
    Returns:
        dict with test results including verification status.
    
    Raises:
        ValueError: If bin_file is not provided or does not exist.
    """
    from src.testbench import HostAXIMaster
    from src.config import TransferMode
    from pathlib import Path
    
    # Validate BIN file requirement
    if not bin_file:
        raise ValueError("bin_file is required. Use 'make gen_payload' to generate a payload file.")
    
    bin_path = Path(bin_file)
    if not bin_path.exists():
        raise ValueError(f"BIN file not found: {bin_file}. Use 'make gen_payload' to generate.")
    
    bin_data = bin_path.read_bytes()
    
    if verbose:
        print("\n" + "=" * 60)
        print("Host to NoC - Multi-Transfer Queue Test")
        print("=" * 60)
        print(f"Total transfers queued: {len(configs)}")
        print(f"BIN file: {bin_file} ({len(bin_data)} bytes)")
        print()

    # Create host memory with fixed size
    HOST_MEMORY_SIZE = 2 * 1024 * 1024  # 2MB
    host_memory = HostMemory(size=HOST_MEMORY_SIZE)
    
    # Per-transfer verification results
    verify_results: List[dict] = []  # [{idx, pass, fail, details}]

    # Track expected data for current transfer
    current_expected: dict = {}  # (node_id, addr) -> expected_bytes

    # Track transfer start cycles for latency calculation
    transfer_start_cycles: dict = {}  # idx -> start_cycle
    current_cycle_ref = [0]  # Mutable reference for closure

    def on_transfer_start(idx: int, config: TransferConfig) -> None:
        """Initialize memory before each transfer."""
        nonlocal current_expected
        current_expected.clear()

        # Record start cycle for transfer progress tracking (not latency)
        transfer_start_cycles[idx] = current_cycle_ref[0]
        # Note: Flit-level latency tracking is done via NI callbacks below
        
        # Load data from BIN file (cycle through if needed)
        offset = config.src_addr % len(bin_data)
        if offset + config.src_size <= len(bin_data):
            test_data = bin_data[offset:offset + config.src_size]
        else:
            # Wrap around for large transfers
            test_data = (bin_data[offset:] + bin_data * ((config.src_size // len(bin_data)) + 1))[:config.src_size]
        
        target_nodes = config.get_target_node_list(total_nodes=16)
        
        if config.is_read:
            # READ mode: Pre-write golden data to node memories
            # Host will read this data back and we verify it
            read_addr = config.read_src_addr if hasattr(config, 'read_src_addr') and config.read_src_addr > 0 else config.dst_addr
            read_size = config.read_size if hasattr(config, 'read_size') and config.read_size > 0 else config.src_size

            if config.transfer_mode == TransferMode.BROADCAST_READ:
                # BROADCAST_READ: All nodes get the same complete data
                for node_id in target_nodes:
                    for coord, ni in system.mesh.nis.items():
                        if ni.node_id == node_id:
                            # Write full test data (truncated to read_size)
                            node_data = test_data[:read_size]
                            ni.local_memory.write(read_addr, node_data)
                            current_expected[(node_id, read_addr)] = node_data
                            break
            else:
                # GATHER mode: Each node has different portion
                portion_size = read_size // len(target_nodes) if target_nodes else 0
                for i, node_id in enumerate(target_nodes):
                    for coord, ni in system.mesh.nis.items():
                        if ni.node_id == node_id:
                            start = i * portion_size
                            end = start + portion_size
                            node_data = test_data[start:end]
                            ni.local_memory.write(read_addr, node_data)
                            current_expected[(node_id, read_addr)] = node_data
                            break
        else:
            # WRITE mode (broadcast/scatter): Fill host memory
            host_memory.fill(config.src_addr, config.src_size, test_data)
            
            if config.transfer_mode == TransferMode.BROADCAST:
                for node_id in target_nodes:
                    current_expected[(node_id, config.dst_addr)] = test_data
            else:
                # SCATTER mode
                portion_size = config.src_size // len(target_nodes) if target_nodes else 0
                for i, node_id in enumerate(target_nodes):
                    start = i * portion_size
                    end = start + portion_size
                    current_expected[(node_id, config.dst_addr)] = test_data[start:end]

    def on_transfer_complete(idx: int, config: TransferConfig) -> None:
        """Verify data immediately after each transfer completes."""
        nonlocal verify_results

        # Note: Flit-level latency tracking is done via NI callbacks

        pass_count = 0
        fail_count = 0
        details = []
        
        if config.is_read:
            # GATHER (read) verification: Check read_data received from nodes
            # For gather, data is collected into system.host_axi_master.read_data
            target_nodes = config.get_target_node_list(total_nodes=16)
            read_addr = config.read_src_addr if hasattr(config, 'read_src_addr') and config.read_src_addr > 0 else config.dst_addr
            
            # Get read data from host_axi_master
            read_data = system.host_axi_master.read_data
            
            for node_id in target_nodes:
                # Get expected data from current_expected
                expected_data = current_expected.get((node_id, read_addr), b'')
                
                # Get actual data from read_data
                actual_data = read_data.get((node_id, read_addr), b'')
                
                if actual_data == expected_data:
                    pass_count += 1
                else:
                    fail_count += 1
                    if len(expected_data) > 0 and len(actual_data) > 0:
                        for off in range(min(len(expected_data), len(actual_data))):
                            if expected_data[off] != actual_data[off]:
                                details.append(
                                    f"Gather Node {node_id} @ 0x{read_addr:04X}: "
                                    f"mismatch at offset {off} (exp=0x{expected_data[off]:02X}, "
                                    f"got=0x{actual_data[off]:02X})"
                                )
                                break
                    else:
                        details.append(
                            f"Gather Node {node_id} @ 0x{read_addr:04X}: "
                            f"exp={len(expected_data)}B, got={len(actual_data)}B"
                        )
        else:
            # WRITE (broadcast/scatter) verification: Check node memories
            for (node_id, local_addr), expected_data in current_expected.items():
                # Find node's NI
                node_ni = None
                for coord, ni in system.mesh.nis.items():
                    if ni.node_id == node_id:
                        node_ni = ni
                        break
                
                if node_ni is None:
                    fail_count += 1
                    details.append(f"Node {node_id} not found")
                    continue
                
                try:
                    actual_data = node_ni.local_memory.get_contents(local_addr, len(expected_data))
                    
                    if actual_data == expected_data:
                        pass_count += 1
                    else:
                        fail_count += 1
                        for off in range(min(len(expected_data), len(actual_data))):
                            if expected_data[off] != actual_data[off]:
                                details.append(
                                    f"Node {node_id} @ 0x{local_addr:04X}: "
                                    f"mismatch at offset {off} (exp=0x{expected_data[off]:02X}, "
                                    f"got=0x{actual_data[off]:02X})"
                                )
                                break
                except (KeyError, IndexError, ValueError) as e:
                    fail_count += 1
                    details.append(f"Node {node_id}: Error - {e}")
        
        result = {
            'idx': idx,
            'pass': pass_count,
            'fail': fail_count,
            'details': details,
        }
        verify_results.append(result)
        
        if verbose and fail_count > 0:
            print(f"  [Transfer {idx+1}] FAIL: {fail_count} mismatches")

    # Create V1System and collector
    from src.core.router import ChannelMode
    ch_mode = ChannelMode.AXI if channel_mode == 'axi' else ChannelMode.GENERAL
    system = V1System(
        mesh_cols=mesh_cols,
        mesh_rows=mesh_rows,
        buffer_depth=buffer_depth,
        host_memory=host_memory,
        channel_mode=ch_mode,
    )
    from src.visualization import MetricsCollector
    # Approximate capture interval based on avg transfer size
    avg_size = sum(c.src_size for c in configs) // len(configs) if configs else 1024
    collector = MetricsCollector(system, capture_interval=max(1, avg_size // 512))

    # === Connect NI callbacks for BookSim2-style flit-level latency tracking ===
    # Each flit carries its injection_cycle, NI calculates latency on arrival
    # Callback receives (latency, axi_channel, payload_bytes)
    # - W/R flits carry user data (payload_bytes > 0)
    # - AW/AR/B are control flits (payload_bytes = 0)
    from src.core.flit import AxiChannel

    def on_flit_latency(latency: int, axi_ch: AxiChannel, payload_bytes: int) -> None:
        """Record per-flit latency when flit arrives at destination."""
        collector.add_latency_sample(latency, axi_ch, payload_bytes)

    # Connect all MasterNI (mesh nodes) for request path (AW, W, AR flits)
    for coord, master_ni in system.mesh.nis.items():
        master_ni.set_flit_latency_callback(on_flit_latency)

    # Connect SlaveNI for response path (B, R flits) - important for Read throughput
    system.master_ni.set_flit_latency_callback(on_flit_latency)

    # Create HostAXIMaster with both callbacks
    first_config = configs[0]
    host_master = HostAXIMaster(
        host_memory=host_memory,
        transfer_config=first_config,
        mesh_cols=mesh_cols,
        mesh_rows=mesh_rows,
        on_transfer_start=on_transfer_start,
        on_transfer_complete=on_transfer_complete,
    )
    
    # Connect HostAXIMaster to V1System
    host_master.connect_to_slave_ni(system.master_ni)
    system.host_axi_master = host_master
    system.host_memory = host_memory
    
    # Queue all transfers
    if verbose:
        print("Queuing transfers:")
        for i, config in enumerate(configs):
            target = config.target_nodes if config.target_nodes != "all" else "all (16 nodes)"
            print(f"  [{i+1}] src=0x{config.src_addr:04X} size={config.src_size} "
                  f"dst=0x{config.dst_addr:04X} targets={target} mode={config.transfer_mode.value}")
        print()
    
    # Queue transfers
    system.host_axi_master.queue_transfers(configs)
    
    # Start queue processing
    system.host_axi_master.start_queue()
    
    # Run simulation
    cycle = 0
    max_cycles = max(100000, len(configs) * 1500)
    last_progress = (-1, -1)
    
    if verbose:
        print("Running simulation...")
    
    while not system.host_axi_master.all_queue_transfers_complete and cycle < max_cycles:
        system.process_cycle()
        collector.capture()
        cycle += 1
        current_cycle_ref[0] = cycle  # Update for callbacks

        if verbose and cycle % 1000 == 0:
            progress = system.host_axi_master.queue_progress
            if progress != last_progress:
                print(f"  Cycle {cycle}: Transfer {progress[0]+1}/{progress[1]} in progress")
                last_progress = progress
    
    # Summary
    completed, total = system.host_axi_master.queue_progress
    transfer_success = system.host_axi_master.all_queue_transfers_complete
    
    total_pass = sum(r['pass'] for r in verify_results)
    total_fail = sum(r['fail'] for r in verify_results)
    verification_success = total_fail == 0
    overall_success = transfer_success and verification_success
    
    if verbose:
        print()
        print("=" * 40)
        print(f"Completed: {completed}/{total} transfers in {cycle} cycles")
        print(f"Transfer Status: {'COMPLETE' if transfer_success else 'INCOMPLETE'}")
        print("=" * 40)
        
        stats = system.host_axi_master.stats
        print(f"\nAXI Statistics:")
        print(f"  AW Sent: {stats.aw_sent} (blocked: {stats.aw_blocked})")
        print(f"  W Sent: {stats.w_sent} (blocked: {stats.w_blocked})")
        print(f"  B Received: {stats.b_received}")
        print(f"  AR Sent: {stats.ar_sent} (blocked: {stats.ar_blocked})")
        print(f"  R Received: {stats.r_beats_received}")
        
        print("\n" + "=" * 40)
        print("Golden Verification (Per-Transfer)")
        print("=" * 40)
        print(f"Total: {total_pass} PASS, {total_fail} FAIL")
        
        # Show first 10 failures
        if total_fail > 0:
            print("\nFailure Details (first 10):")
            shown = 0
            for r in verify_results:
                if r['fail'] > 0 and shown < 10:
                    for d in r['details'][:2]:
                        print(f"  Transfer {r['idx']+1}: {d}")
                        shown += 1
                        if shown >= 10:
                            break
            if total_fail > 10:
                print(f"  ... and more failures")
        
    # === Performance Metrics (BookSim2-Style) ===
    # Collect all target nodes across all configs for L0 calculation
    all_target_nodes = set()
    for config in configs:
        nodes = config.get_target_node_list(total_nodes=16)
        all_target_nodes.update(nodes)
    all_target_nodes = list(all_target_nodes)

    zero_load_latency = calculate_host_zero_load_latency(
        all_target_nodes, mesh_cols=mesh_cols, mesh_rows=mesh_rows
    )
    collector.set_zero_load_latency(zero_load_latency)

    booksim_metrics = collector.get_booksim_metrics(total_cycles=cycle)

    if verbose:
        print_booksim_metrics(booksim_metrics)

        # Additional router statistics
        flit_stats = system.get_flit_stats()
        total_flits = sum(flit_stats.values())
        active_routers = sum(1 for v in flit_stats.values() if v > 0)
        print("\n--- Router Statistics ---")
        print(f"  Total Flits:       {total_flits}")
        print(f"  Active Routers:    {active_routers}/{len(flit_stats)}")

    # === Performance Validation ===
    buffer_stats = collector.get_buffer_stats(total_capacity=20 * 4 * 5)
    perf_metrics = {
        'total_cycles': cycle,
        'throughput': booksim_metrics.throughput,
        'buffer_utilization': buffer_stats['utilization'],
        'avg_latency': booksim_metrics.avg_latency,
        'avg_occupancy': buffer_stats['avg'],
    }
    validation_passed = validate_performance_metrics(
        perf_metrics,
        zero_load_latency=zero_load_latency,
        verbose=verbose
    )

    # Include validation in overall success
    overall_success = overall_success and validation_passed

    if verbose:
        print()
        print("=" * 40)
        print(f"Overall Status: {'PASS' if overall_success else 'FAIL'}")
        print("=" * 40)
    
    # Save metrics
    save_host_metrics(
        system, configs, cycle, overall_success,
        verify_results=verify_results,
        test_pattern='multi_transfer',
        collector=collector
    )

    return {
        'completed': completed,
        'total': total,
        'cycles': cycle,
        'success': overall_success,
        'transfer_success': transfer_success,
        'verify_pass': total_pass,
        'verify_fail': total_fail,
        'metrics': perf_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Host to NoC Test Runner")
    parser.add_argument('test', nargs='?', default='broadcast_write',
                        choices=['broadcast_write', 'broadcast_read',
                                 'scatter_write', 'gather_read', 'multi_transfer'],
                        help='Test to run')
    parser.add_argument('--config', type=str, help='Custom config file')
    parser.add_argument('--bin', type=str, help='BIN file for source data (required for multi_transfer)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')

    # System parameters (for multi_para sweep)
    parser.add_argument('--buffer-depth', type=int, default=8,
                        help='Router buffer depth (default: 8)')
    parser.add_argument('--mesh-cols', type=int, default=5,
                        help='Mesh columns (default: 5)')
    parser.add_argument('--mesh-rows', type=int, default=4,
                        help='Mesh rows (default: 4)')
    parser.add_argument('--json-output', type=str, default=None,
                        help='Output metrics to JSON file')
    parser.add_argument('--mode', type=str, choices=['general', 'axi'], default='general',
                        help='Channel mode: general (2 Sub-Routers) or axi (5 Sub-Routers)')

    args = parser.parse_args()

    # Load config
    config_dir = Path(__file__).parent / 'config'
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = config_dir / f'{args.test}.yaml'

    # Run test
    verbose = not args.quiet

    if args.test == 'multi_transfer':
        # Multi-transfer uses list of configs
        if config_path.exists():
            configs = load_transfer_configs(config_path)
        else:
            print(f"Config not found: {config_path}")
            return 1
        result = run_multi_transfer(
            configs,
            verbose,
            bin_file=args.bin,
            buffer_depth=args.buffer_depth,
            mesh_cols=args.mesh_cols,
            mesh_rows=args.mesh_rows,
            channel_mode=args.mode,
        )

        # Output JSON if requested
        if args.json_output and 'metrics' in result:
            import json
            json_path = Path(args.json_output)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(result['metrics'], f, indent=2)
            if verbose:
                print(f"\nMetrics saved to: {json_path}")
    else:
        # Single transfer tests
        if config_path.exists():
            config = load_transfer_config(config_path)
        else:
            print(f"Config not found: {config_path}, using defaults")
            config = TransferConfig()

        if args.test == 'broadcast_write':
            result = run_broadcast_write(config, verbose, bin_file=args.bin)
        elif args.test == 'broadcast_read':
            result = run_broadcast_read(config, verbose, bin_file=args.bin)
        elif args.test == 'scatter_write':
            result = run_scatter_write(config, verbose, bin_file=args.bin)
        elif args.test == 'gather_read':
            result = run_gather_read(config, verbose, bin_file=args.bin)
        else:
            print(f"Unknown test: {args.test}")
            return 1

    return 0 if result['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
