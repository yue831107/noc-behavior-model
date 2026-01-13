"""
Parameter optimizer for regression testing.

ParameterOptimizer orchestrates the search process:
1. Prefilter infeasible combinations using theoretical bounds
2. Use search strategy to iterate through candidates
3. Run simulations and collect metrics
4. Track best solution and build results
"""

from typing import Dict, Any, Optional, Callable
from pathlib import Path

from .target import PerformanceTarget
from .parameter_space import ParameterSpace
from .search_strategy import SearchStrategy, GridSearch
from .result import OptimizationResult


class ParameterOptimizer:
    """
    Parameter optimization controller.

    Coordinates prefiltering, simulation, and result collection
    to find the best hardware configuration.
    """

    def __init__(
        self,
        target: PerformanceTarget,
        parameter_space: ParameterSpace,
        strategy: Optional[SearchStrategy] = None,
        system_type: str = "host_to_noc",
        verbose: bool = True,
    ):
        """
        Initialize optimizer.

        Args:
            target: Performance target to meet.
            parameter_space: Hardware parameters to search.
            strategy: Search strategy (default: GridSearch).
            system_type: "host_to_noc" or "noc_to_noc".
            verbose: Print progress messages.
        """
        self.target = target
        self.space = parameter_space
        self.strategy = strategy or GridSearch()
        self.system_type = system_type
        self.verbose = verbose

        # Results tracking
        self._all_results = []
        self._best_params = None
        self._best_metrics = None
        self._best_score = float("-inf")
        self._satisfied_count = 0

    def optimize(
        self,
        workload: Optional[Dict[str, Any]] = None,
        early_stop: bool = False,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
    ) -> OptimizationResult:
        """
        Run the optimization process.

        Args:
            workload: Workload configuration (transfer_size, num_transfers, etc).
            early_stop: Stop as soon as a satisfying solution is found.
            progress_callback: Called with (current, total, metrics) after each test.

        Returns:
            OptimizationResult with best parameters and statistics.
        """
        # Default workload
        if workload is None:
            workload = {
                "transfer_size": 1024,
                "num_transfers": 50,
            }

        # Prefilter
        total_combinations = self.space.total_combinations()
        feasible = self.space.prefilter(self.target, verbose=self.verbose)
        feasible_count = len(feasible)

        if self.verbose:
            print(f"\nStarting optimization with {feasible_count} feasible combinations")

        # Initialize search
        self.strategy.initialize(feasible)
        self._all_results = []
        self._best_params = None
        self._best_metrics = None
        self._best_score = float("-inf")
        self._satisfied_count = 0

        # Search loop
        tested = 0
        while not self.strategy.is_complete():
            batch = self.strategy.next_batch(batch_size=1)

            for params in batch:
                tested += 1

                # Run simulation
                metrics = self._run_simulation(params, workload)

                # Calculate score
                score = self.target.score(metrics)
                satisfied = self.target.is_satisfied(metrics)

                if satisfied:
                    self._satisfied_count += 1

                # Track result
                result_entry = {
                    "params": params,
                    "metrics": metrics,
                    "score": score,
                    "satisfied": satisfied,
                }
                self._all_results.append(result_entry)

                # Update best
                if score > self._best_score:
                    self._best_score = score
                    self._best_params = params
                    self._best_metrics = metrics

                # Progress callback
                if progress_callback:
                    progress_callback(tested, feasible_count, metrics)

                # Verbose output
                if self.verbose and tested % 10 == 0:
                    print(
                        f"  Progress: {tested}/{feasible_count} "
                        f"(best score: {self._best_score:.4f})"
                    )

                # Early stop
                if early_stop and satisfied:
                    if self.verbose:
                        print(f"\n  Early stop: found satisfying solution at {tested}/{feasible_count}")
                    break

            if early_stop and self._satisfied_count > 0:
                break

        # Build result
        return self._build_result(
            total_combinations=total_combinations,
            feasible_count=feasible_count,
            tested_count=tested,
        )

    def _run_simulation(
        self,
        params: Dict[str, Any],
        workload: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Run a single simulation with given parameters.

        Dispatches to appropriate simulation based on system_type.

        Args:
            params: Hardware parameters.
            workload: Workload configuration.

        Returns:
            Metrics dictionary with throughput, avg_latency, etc.
        """
        if self.system_type == "noc_to_noc":
            return self._run_noc_to_noc(params, workload)
        else:
            return self._run_host_to_noc(params, workload)

    def _run_host_to_noc(
        self,
        params: Dict[str, Any],
        workload: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Run Host-to-NoC simulation via V1System.

        Measures end-to-end latency from submit_write() to get_b_response().

        Args:
            params: Hardware parameters.
            workload: Workload configuration.

        Returns:
            Metrics dictionary.
        """
        try:
            from src.core import V1System

            # Extract parameters
            mesh_rows = params.get("mesh_rows", 4)
            mesh_cols = params.get("mesh_cols", 5)
            buffer_depth = params.get("buffer_depth", 4)
            transfer_size = workload.get("transfer_size", 1024)
            num_transfers = workload.get("num_transfers", 10)

            # Create system
            system = V1System(
                mesh_cols=mesh_cols,
                mesh_rows=mesh_rows,
                buffer_depth=buffer_depth,
            )

            # Run simple simulation
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

            # Calculate metrics
            total_bytes = transfer_size * num_transfers
            throughput = total_bytes / total_cycles if total_cycles > 0 else 0
            avg_latency = total_cycles / num_transfers if num_transfers > 0 else 0

            return {
                "throughput": throughput,
                "avg_latency": avg_latency,
                "total_cycles": total_cycles,
                "total_bytes": total_bytes,
            }

        except (ImportError, RuntimeError, ValueError, KeyError) as e:
            # Fallback to theoretical mock when simulation unavailable
            return self._mock_simulation(params, workload)

    def _run_noc_to_noc(
        self,
        params: Dict[str, Any],
        workload: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Run NoC-to-NoC simulation via NoCSystem.

        Measures node-to-node communication latency using specified traffic pattern.

        Args:
            params: Hardware parameters.
            workload: Workload configuration.

        Returns:
            Metrics dictionary.
        """
        try:
            from src.core.routing_selector import NoCSystem
            from src.config import NoCTrafficConfig, TrafficPattern
            from src.traffic.pattern_generator import TrafficPatternGenerator

            # Extract parameters
            mesh_rows = params.get("mesh_rows", 4)
            mesh_cols = params.get("mesh_cols", 5)
            buffer_depth = params.get("buffer_depth", 4)
            transfer_size = workload.get("transfer_size", 256)
            traffic_pattern = workload.get("traffic_pattern", "neighbor")

            # Create traffic config
            pattern = TrafficPattern(traffic_pattern)
            traffic_config = NoCTrafficConfig(
                pattern=pattern,
                transfer_size=transfer_size,
                mesh_rows=mesh_rows,
                mesh_cols=mesh_cols,
            )

            # Generate node configs
            generator = TrafficPatternGenerator()
            node_configs = generator.generate(traffic_config)

            # Create system
            system = NoCSystem(
                mesh_cols=mesh_cols,
                mesh_rows=mesh_rows,
                buffer_depth=buffer_depth,
            )

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
            avg_latency = total_cycles  # Total time for all parallel transfers

            return {
                "throughput": throughput,
                "avg_latency": avg_latency,
                "total_cycles": total_cycles,
                "total_bytes": total_bytes,
                "num_nodes": num_nodes,
            }

        except (ImportError, RuntimeError, ValueError, KeyError) as e:
            # Fallback to theoretical mock when simulation unavailable
            return self._mock_simulation(params, workload)

    def _mock_simulation(
        self,
        params: Dict[str, Any],
        workload: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Generate mock metrics based on theoretical models.

        Used when actual simulation is not available.

        Args:
            params: Hardware parameters.
            workload: Workload configuration.

        Returns:
            Estimated metrics.
        """
        rows = params.get("mesh_rows", 4)
        cols = params.get("mesh_cols", 5)
        buffer_depth = params.get("buffer_depth", 4)
        transfer_size = workload.get("transfer_size", 1024)

        # Theoretical bounds
        bounds = self.space.estimate_theoretical_bounds(params)
        t_max = bounds["t_max"]
        l_min = bounds["l_min"]
        max_hops = bounds["max_hops"]

        # Estimate actual performance (with some degradation)
        # Throughput: ~80% of theoretical max
        throughput = t_max * 0.8

        # Latency: l_min + some queuing based on buffer depth
        # Lower buffer = more queuing
        queuing_factor = max(1.0, 4.0 / buffer_depth)
        avg_latency = l_min + max_hops * queuing_factor

        # Total cycles
        total_bytes = transfer_size * (rows - 1) * (cols - 1)  # Rough estimate
        total_cycles = total_bytes / throughput + avg_latency

        # Buffer utilization
        buffer_utilization = 0.3 + 0.4 * (1.0 / buffer_depth)

        return {
            "throughput": throughput,
            "avg_latency": avg_latency,
            "total_cycles": total_cycles,
            "buffer_utilization": buffer_utilization,
            "total_bytes": total_bytes,
        }

    def _build_result(
        self,
        total_combinations: int,
        feasible_count: int,
        tested_count: int,
    ) -> OptimizationResult:
        """Build the final OptimizationResult."""
        target_satisfied = self._satisfied_count > 0

        # If no valid result, create empty best
        if self._best_params is None:
            self._best_params = {}
            self._best_metrics = {}
            self._best_score = 0.0

        return OptimizationResult(
            best_params=self._best_params,
            best_metrics=self._best_metrics,
            best_score=self._best_score,
            target_satisfied=target_satisfied,
            target=self.target,
            total_combinations=total_combinations,
            feasible_count=feasible_count,
            tested_count=tested_count,
            satisfied_count=self._satisfied_count,
            all_results=self._all_results,
        )
