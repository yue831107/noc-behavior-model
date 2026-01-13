"""
Parameter Sweep Runner.

Executes simulations with varying parameters and collects metrics.
"""

from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import sys

from .config import SweepConfig
from .results import SweepResults


class SweepRunner:
    """
    Executes parameter sweeps by running simulations with varying parameters.

    Can operate in two modes:
    1. Simulation mode: Actually runs simulations (requires proper setup)
    2. Manual mode: User provides metrics for each parameter combination
    """

    def __init__(
        self,
        system_type: str = "host_to_noc",
        project_root: Optional[Path] = None,
        verbose: bool = True,
        use_mock: bool = False,
    ):
        """
        Initialize sweep runner.

        Args:
            system_type: "host_to_noc" or "noc_to_noc"
            project_root: Project root path for imports
            verbose: Print progress during sweep
            use_mock: Force mock simulation (useful for testing visualization)
        """
        self.system_type = system_type
        self.verbose = verbose
        self.use_mock = use_mock

        # Setup project path if needed
        if project_root:
            sys.path.insert(0, str(project_root))
        else:
            # Try to find project root
            current = Path(__file__).resolve()
            for parent in current.parents:
                if (parent / "src").exists():
                    sys.path.insert(0, str(parent))
                    break

    def run_sweep(
        self,
        config: SweepConfig,
        simulation_func: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SweepResults:
        """
        Execute parameter sweep.

        Args:
            config: Sweep configuration
            simulation_func: Function that takes params dict and returns metrics dict
                            If None, uses built-in simulation (if available)
            progress_callback: Called with (current, total) for progress updates

        Returns:
            SweepResults with all collected data
        """
        results = SweepResults(system_type=config.system_type)
        combinations = list(config.generate_combinations())
        total = len(combinations)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Parameter Sweep: {config.parameter_names}")
            print(f"Total combinations: {total}")
            print(f"{'=' * 60}\n")

        for i, params in enumerate(combinations):
            if self.verbose:
                param_str = ", ".join(f"{k}={v}" for k, v in params.items()
                                      if k in config.parameter_names)
                print(f"[{i+1}/{total}] Running: {param_str}")

            # Run simulation
            if simulation_func:
                metrics = simulation_func(params)
            else:
                metrics = self._run_default_simulation(params)

            # Store results
            results.add_result(params, metrics)

            if progress_callback:
                progress_callback(i + 1, total)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Sweep complete: {len(results)} results collected")
            print(f"{'=' * 60}\n")

        return results

    def _run_default_simulation(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Run simulation with given parameters.

        This is a simplified version that works without external files.
        For full simulation, use a custom simulation_func.
        """
        # Force mock mode if requested
        if self.use_mock:
            return self._generate_mock_metrics(params)

        try:
            if self.system_type == "noc_to_noc":
                return self._run_noc_to_noc(params)
            else:
                return self._run_host_to_noc(params)
        except (ImportError, RuntimeError, ValueError, KeyError) as e:
            if self.verbose:
                print(f"  Warning: Simulation failed - {e}")
            return self._generate_mock_metrics(params)

    def _run_host_to_noc(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Run Host-to-NoC simulation."""
        try:
            from src.core import V1System
            from src.config import TransferConfig, TransferMode

            # Extract parameters
            transfer_size = params.get('transfer_size', 256)
            num_transfers = params.get('num_transfers', 1)
            inter_delay = params.get('inter_txn_delay', 0)
            buffer_depth = params.get('buffer_depth', 32)

            # Create system with configurable buffer depth
            system = V1System(mesh_cols=5, mesh_rows=4, buffer_depth=buffer_depth)

            # Simple simulation: write to single node
            test_data = bytes([i % 256 for i in range(transfer_size)])
            target_node = 0
            dst_addr = 0x1000

            total_cycles = 0
            completed = 0

            for t in range(num_transfers):
                addr = (target_node << 32) | dst_addr
                system.submit_write(addr, test_data, t + 1)

                # Run until complete
                cycle = 0
                max_cycles = 5000
                while cycle < max_cycles:
                    system.process_cycle()
                    resp = system.master_ni.get_b_response()
                    if resp is not None:
                        completed += 1
                        break
                    cycle += 1

                total_cycles += cycle + 1

                # Inter-transaction delay
                for _ in range(inter_delay):
                    system.process_cycle()
                    total_cycles += 1

            # Calculate metrics
            total_bytes = transfer_size * num_transfers
            throughput = total_bytes / total_cycles if total_cycles > 0 else 0
            avg_latency = total_cycles / num_transfers if num_transfers > 0 else 0

            return {
                'total_cycles': total_cycles,
                'throughput': throughput,
                'avg_latency': avg_latency,
                'completed': completed,
                'total_bytes': total_bytes,
            }

        except ImportError:
            return self._generate_mock_metrics(params)

    def _run_noc_to_noc(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Run NoC-to-NoC simulation."""
        try:
            from src.core.routing_selector import NoCSystem
            from src.config import NoCTrafficConfig, TrafficPattern
            from src.traffic.pattern_generator import TrafficPatternGenerator

            # Extract parameters
            transfer_size = params.get('transfer_size', 256)
            pattern_name = params.get('traffic_pattern', 'neighbor')

            # Create config
            pattern = TrafficPattern(pattern_name)
            traffic_config = NoCTrafficConfig(
                pattern=pattern,
                transfer_size=transfer_size,
            )

            # Generate node configs
            generator = TrafficPatternGenerator()
            node_configs = generator.generate(traffic_config)

            # Create system and run
            system = NoCSystem(mesh_cols=5, mesh_rows=4)

            # Configure all nodes
            for node_cfg in node_configs:
                node = system.get_node(node_cfg.src_node_id)
                if node:
                    test_data = bytes([i % 256 for i in range(transfer_size)])
                    node.initialize_memory(node_cfg.local_src_addr, test_data)
                    node.configure_transfer(node_cfg)

            # Start all transfers
            for node_cfg in node_configs:
                node = system.get_node(node_cfg.src_node_id)
                if node:
                    node.start_transfer()

            # Run until complete
            cycle = 0
            max_cycles = 10000
            while cycle < max_cycles:
                system.process_cycle(cycle)
                if system.all_transfers_complete():
                    break
                cycle += 1

            total_cycles = cycle + 1
            num_nodes = len(node_configs)
            total_bytes = transfer_size * num_nodes
            throughput = total_bytes / total_cycles if total_cycles > 0 else 0
            avg_latency = total_cycles / num_nodes if num_nodes > 0 else 0

            return {
                'total_cycles': total_cycles,
                'throughput': throughput,
                'avg_latency': avg_latency,
                'num_nodes': num_nodes,
                'total_bytes': total_bytes,
            }

        except ImportError:
            return self._generate_mock_metrics(params)

    def _generate_mock_metrics(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate plausible mock metrics for testing visualization.

        Based on theoretical NoC behavior models.
        Models buffer_depth effects on throughput and latency.
        """
        transfer_size = params.get('transfer_size', 256)
        num_transfers = params.get('num_transfers', 10)
        inter_delay = params.get('inter_txn_delay', 0)
        buffer_depth = params.get('buffer_depth', 4)

        # Simple model: cycles proportional to size + overhead
        base_latency = 5  # Minimum cycles
        bytes_per_cycle = 8  # Flit width

        # Buffer depth effect: deeper buffers reduce stall cycles
        # Model: latency reduction factor = log2(buffer_depth) / log2(16)
        import math
        buffer_efficiency = math.log2(max(2, buffer_depth)) / math.log2(16)

        # Base cycles per transfer
        base_cycles = base_latency + (transfer_size / bytes_per_cycle)

        # Contention overhead decreases with deeper buffers
        # For small transfers, contention is minimal
        # For large transfers, deeper buffers help more
        contention_factor = 1.0 + (transfer_size / 512) * (1.0 - buffer_efficiency)
        cycles_per_transfer = base_cycles * contention_factor

        total_cycles = int(num_transfers * (cycles_per_transfer + inter_delay))

        total_bytes = transfer_size * num_transfers
        throughput = total_bytes / total_cycles if total_cycles > 0 else 0

        # Buffer utilization: deeper buffers have lower utilization per flit
        base_util = 0.1
        load_factor = 1.0 / (1.0 + inter_delay * 0.1)
        size_factor = transfer_size / 256
        depth_factor = 4.0 / buffer_depth  # Normalized to buffer_depth=4
        buffer_util = min(1.0, base_util * load_factor * size_factor * depth_factor)

        return {
            'total_cycles': total_cycles,
            'throughput': throughput,
            'avg_latency': cycles_per_transfer,
            'buffer_utilization': buffer_util,
            'total_bytes': total_bytes,
        }


def run_quick_sweep(
    parameter: str,
    values: List[Any],
    system_type: str = "host_to_noc",
    **fixed_params
) -> SweepResults:
    """
    Convenience function to run a quick single-parameter sweep.

    Args:
        parameter: Parameter name to sweep
        values: List of values to test
        system_type: Type of simulation
        **fixed_params: Fixed parameter values

    Returns:
        SweepResults
    """
    config = SweepConfig(
        parameters={parameter: values},
        fixed_params=fixed_params,
        system_type=system_type,
    )

    runner = SweepRunner(system_type=system_type)
    return runner.run_sweep(config)
