"""
Baseline Economic Dispatch Solver using Classical Optimization
Uses PyPSA (Python for Power System Analysis) for static optimization at each timestep.

This module implements the classical approach to solving the Economic Dispatch problem.
It formulates the problem as a Linear Programming (LP) task where the objective is to
minimize generation costs subject to power balance and capacity constraints.
"""

import numpy as np
import pandas as pd
import pypsa
import time
from typing import Dict, Tuple
import yaml


class ClassicalEDSolver:
    """
    Classical Economic Dispatch solver using PyPSA's Linear Programming.

    This class serves as the baseline for comparison with the Reinforcement Learning (RL) agent.
    It solves the static optimization problem at each timestep independently, assuming
    perfect knowledge of the current load and generator parameters.

    The optimization problem is:
        Minimize Sum(Cost_i(P_i))
        Subject to:
            Sum(P_i) = Net_Load (Power Balance)
            P_min_i <= P_i <= P_max_i (Capacity Limits)
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the classical solver.

        Args:
            config_path: Path to the YAML configuration file containing system parameters.
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.n_generators = self.config['system']['n_generators']
        self.solver_name = self.config['baseline']['solver']

        # Extract generator parameters (alpha, beta, gamma, limits)
        self.gen_params = self._extract_generator_params()

        # Performance tracking lists
        self.solve_times = []
        self.total_costs = []

    def _extract_generator_params(self) -> Dict[str, np.ndarray]:
        """
        Extract generator parameters from the configuration dictionary.

        Returns:
            A dictionary containing numpy arrays for each parameter type:
            - alpha: Fixed cost ($/h)
            - beta: Linear cost coefficient ($/MWh)
            - gamma: Quadratic cost coefficient ($/MW^2h)
            - p_min: Minimum power output (MW)
            - p_max: Maximum power output (MW)
        """
        params = {
            'alpha': [],
            'beta': [],
            'gamma': [],
            'p_min': [],
            'p_max': [],
            'names': []
        }

        for i in range(1, self.n_generators + 1):
            gen_key = f'gen_{i}'
            gen_config = self.config['generators'][gen_key]

            params['alpha'].append(gen_config['alpha'])
            params['beta'].append(gen_config['beta'])
            params['gamma'].append(gen_config['gamma'])
            params['p_min'].append(gen_config['p_min'])
            params['p_max'].append(gen_config['p_max'])
            params['names'].append(gen_config['name'])

        # Convert lists to numpy arrays for vectorized operations
        # type: ignore
        return {k: np.array(v) if k != 'names' else v for k, v in params.items()}

    def _create_network(self, net_load: float) -> pypsa.Network:
        """
        Create a PyPSA network model for the current timestep.

        This function builds a fresh network object for each optimization step.
        It defines the bus, generators, and load based on the current system state.

        Args:
            net_load: The target net load (Demand - Renewables) to meet (MW).

        Returns:
            A configured PyPSA network object ready for optimization.
        """
        # Check feasibility BEFORE creating network to avoid solver errors
        total_capacity = np.sum(self.gen_params['p_max'])
        min_capacity = np.sum(self.gen_params['p_min'])

        # Handle infeasible high load
        if net_load > total_capacity:
            print(
                f"WARNING: Load {net_load:.2f} MW exceeds total capacity {total_capacity:.2f} MW")
            # Cap at 95% of max capacity to ensure feasibility
            net_load = total_capacity * 0.95

        # Handle infeasible low load
        if net_load < min_capacity:
            print(
                f"WARNING: Load {net_load:.2f} MW below minimum capacity {min_capacity:.2f} MW")
            net_load = min_capacity * 1.05  # Set to 105% of min capacity

        # Create network with explicit snapshot
        network = pypsa.Network()
        network.set_snapshots([0])  # Single timestep optimization

        # Add a single electrical bus (copper plate assumption, no transmission constraints)
        network.add("Bus", "bus_0", carrier="AC")

        # Add generators to the bus
        for i in range(self.n_generators):
            network.add(
                "Generator",
                f"gen_{i}",
                bus="bus_0",
                p_nom=self.gen_params['p_max'][i],  # Nominal capacity (MW)
                # Min output as per unit
                p_min_pu=self.gen_params['p_min'][i] /
                    self.gen_params['p_max'][i],
                p_max_pu=1.0,  # Max output as per unit (always 1.0 of p_nom)
                # Linear cost term ($/MWh)
                marginal_cost=self.gen_params['beta'][i],
                # No unit commitment (generators are always online)
                committable=False,
                carrier="thermal"
            )

        # Add the load to the bus
        network.add(
            "Load",
            "load_0",
            bus="bus_0",
            p_set=net_load  # Fixed load value to be met
        )

        return network

    def solve_single_timestep(self, net_load: float) -> Tuple[np.ndarray, float, float]:
        """
        Solve the economic dispatch optimization for a single timestep.

        Args:
            net_load: Target net load to meet (MW).

        Returns:
            Tuple containing:
            - generator_powers: Array of optimal power outputs (MW).
            - total_cost: Total generation cost ($).
            - solve_time: Time taken by the solver (seconds).
        """
        # Create the network model
        network = self._create_network(net_load)

        # Start timer
        start_time = time.time()

        try:
            # Solve using PyPSA's optimizer (Linear Optimal Power Flow)
            # Note: PyPSA minimizes linear costs. Quadratic costs are handled by
            # the solver if configured, but here we use linear approximation for dispatch
            # and calculate full quadratic cost in post-processing.
            status, condition = network.optimize(
                solver_name=self.solver_name
            )

            # Check if optimization succeeded
            if status == 'warning' and condition == 'infeasible':
                raise Exception(f"Solver returned status: {condition}")

            solve_time = time.time() - start_time

            # Extract solution (generator power outputs)
            try:
                generator_powers = network.generators_t.p.iloc[0].values
            except (IndexError, AttributeError):
                generator_powers = network.generators.p.values

            # Verify we got valid powers
            if len(generator_powers) == 0 or np.any(np.isnan(generator_powers)):
                raise ValueError(
                    "Failed to extract generator powers from network")

            # Calculate total cost using the full quadratic cost function
            # This ensures fair comparison even if the solver optimized a linear relaxation
            cost = self._calculate_cost(generator_powers)

            return generator_powers, cost, solve_time

        except Exception as e:
            print(f"Solver failed: {e}")
            # Fallback strategy: Return a feasible but suboptimal solution
            solve_time = time.time() - start_time

            # Proportional distribution based on max capacity
            # This guarantees power balance if total capacity is sufficient
            total_capacity = np.sum(self.gen_params['p_max'])
            generator_powers = (net_load / total_capacity) * \
                self.gen_params['p_max']

            # Clip to respect limits
            generator_powers = np.clip(
                generator_powers,
                self.gen_params['p_min'],
                self.gen_params['p_max']
            )

            cost = self._calculate_cost(generator_powers)

            return generator_powers, cost, solve_time

    def _calculate_cost(self, powers: np.ndarray) -> float:
        """
        Calculate total generation cost using the full quadratic cost function.

        Cost_i = alpha_i + beta_i * P_i + gamma_i * P_i^2

        Args:
            powers: Array of generator power outputs (MW).

        Returns:
            Total system cost ($/hour).
        """
        alpha = self.gen_params['alpha']
        beta = self.gen_params['beta']
        gamma = self.gen_params['gamma']

        # Vectorized calculation of quadratic cost
        cost = np.sum(alpha + beta * powers + gamma * powers ** 2)

        # Debug output for detailed inspection
        # print(f"    Cost breakdown: Fixed={np.sum(alpha):.2f}, Linear={np.sum(beta * powers):.2f}, Quadratic={np.sum(gamma * powers**2):.2f}")

        return float(cost)

    def solve_full_simulation(self, data_path: str) -> pd.DataFrame:
        """
        Solve the optimization problem for the entire simulation period.

        Iterates through the dataset, solving for each timestep sequentially.

        Args:
            data_path: Path to the CSV file containing simulation data (net_load).

        Returns:
            DataFrame containing results for each timestep (powers, costs, etc.).
        """
        # Load simulation data
        data = pd.read_csv(data_path)

        results = []

        print(f"Solving classical ED for {len(data)} timesteps...")
        print(f"Using solver: {self.solver_name}")

        # Iterate through each timestep
        for idx, row in enumerate(data.itertuples(index=False), start=0):
            net_load = row.net_load

            # Solve for this specific timestep
            powers, cost, solve_time = self.solve_single_timestep(
                net_load)  # type: ignore

            # Store results
            result = {
                'timestamp': row.timestamp,
                'net_load': net_load,
                'total_cost': cost,
                'solve_time': solve_time,
                'total_generation': np.sum(powers)
            }

            # Add individual generator powers to result
            for i, power in enumerate(powers):
                result[f'gen_{i}_power'] = power

            results.append(result)

            # Progress update
            if (idx + 1) % 50 == 0:
                print(f"  Completed {idx + 1}/{len(data)} timesteps")

        # Convert list of results to DataFrame
        results_df = pd.DataFrame(results)

        # Calculate and print summary statistics
        print("\n" + "="*60)
        print("CLASSICAL OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Total cost: ${results_df['total_cost'].sum():,.2f}")
        print(
            f"Average cost per timestep: ${results_df['total_cost'].mean():,.2f}")
        print(
            f"Average solve time: {results_df['solve_time'].mean()*1000:.2f} ms")
        print(f"Max solve time: {results_df['solve_time'].max()*1000:.2f} ms")
        print(
            f"Total solve time: {results_df['solve_time'].sum():.2f} seconds")

        # Check power balance (Supply - Demand)
        balance_error = np.abs(
            results_df['total_generation'] - results_df['net_load'])
        print(f"\nPower balance:")
        print(f"  Average error: {balance_error.mean():.6f} MW")
        print(f"  Max error: {balance_error.max():.6f} MW")
        print("="*60)

        return results_df


if __name__ == "__main__":
    # Test the classical solver
    print("Testing Classical Economic Dispatch Solver...")

    solver = ClassicalEDSolver()

    # Test single timestep
    print("\nTesting single timestep optimization...")
    net_load = 500.0  # 500 MW
    powers, cost, solve_time = solver.solve_single_timestep(net_load)

    print(f"\nNet load: {net_load} MW")
    print(f"Generator powers: {powers}")
    print(f"Total generation: {np.sum(powers):.2f} MW")
    print(f"Total cost: ${cost:.2f}")
    print(f"Solve time: {solve_time*1000:.2f} ms")

    print("\nClassical solver test completed!")
