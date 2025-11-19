"""
Baseline Economic Dispatch Solver using Classical Optimization
Uses PyPSA for static optimization at each timestep
"""

import numpy as np
import pandas as pd
import pypsa
import time
from typing import Dict, Tuple
import yaml


class ClassicalEDSolver:
    """
    Classical Economic Dispatch solver using PyPSA's Linear Programming

    This serves as the baseline for comparison with the RL agent.
    It solves the static optimization problem at each timestep.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the classical solver

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.n_generators = self.config['system']['n_generators']
        self.solver_name = self.config['baseline']['solver']

        # Extract generator parameters
        self.gen_params = self._extract_generator_params()

        # Performance tracking
        self.solve_times = []
        self.total_costs = []

    def _extract_generator_params(self) -> Dict[str, np.ndarray]:
        """Extract generator parameters from config"""
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

        return {k: np.array(v) if k != 'names' else v for k, v in params.items()} #type: ignore

    def _create_network(self, net_load: float) -> pypsa.Network:
        """Create a PyPSA network for the current timestep
        
        Args:
            net_load: Target net load to meet (MW)
            
        Returns:
            Configured PyPSA network
        """
        # Check feasibility BEFORE creating network
        total_capacity = np.sum(self.gen_params['p_max'])
        min_capacity = np.sum(self.gen_params['p_min'])
        
        if net_load > total_capacity:
            print(f"WARNING: Load {net_load:.2f} MW exceeds total capacity {total_capacity:.2f} MW")
            net_load = total_capacity * 0.95  # Cap at 95% of max capacity
        
        if net_load < min_capacity:
            print(f"WARNING: Load {net_load:.2f} MW below minimum capacity {min_capacity:.2f} MW")
            net_load = min_capacity * 1.05  # Set to 105% of min capacity
        
        # Create network with explicit snapshot
        network = pypsa.Network()
        network.set_snapshots([0])  # Single timestep optimization
        
        # Add a single bus with carrier
        network.add("Bus", "bus_0", carrier="AC")
        
        # Add generators
        for i in range(self.n_generators):
            network.add(
                "Generator",
                f"gen_{i}",
                bus="bus_0",
                p_nom=self.gen_params['p_max'][i],  # Nominal capacity
                p_min_pu=self.gen_params['p_min'][i] / self.gen_params['p_max'][i],  # Min as fraction
                p_max_pu=1.0,  # Max as fraction (always 1.0)
                marginal_cost=self.gen_params['beta'][i],  # $/MWh
                committable=False,  # No unit commitment
                carrier="thermal"
            )
        
        # Add load at the snapshot
        network.add(
            "Load",
            "load_0",
            bus="bus_0",
            p_set=net_load  # Fixed load value
        )
        
        return network


    def solve_single_timestep(self, net_load: float) -> Tuple[np.ndarray, float, float]:
        """Solve optimization for a single timestep

        Args:
            net_load: Target net load to meet (MW)

        Returns:
            Tuple of (generator_powers, total_cost, solve_time)
        """
        # Create network
        network = self._create_network(net_load)

        # Solve using PyPSA's optimizer
        start_time = time.time()

        try:
            # Use linear power flow (LOPF)
            status, condition = network.optimize(
                solver_name=self.solver_name
            )
            
            # Check if optimization succeeded
            if status == 'warning' and condition == 'infeasible':
                raise Exception(f"Solver returned status: {condition}")

            solve_time = time.time() - start_time

            # Extract solution
            try:
                generator_powers = network.generators_t.p.iloc[0].values
            except (IndexError, AttributeError):
                generator_powers = network.generators.p.values
            
            # Verify we got valid powers
            if len(generator_powers) == 0 or np.any(np.isnan(generator_powers)):
                raise ValueError("Failed to extract generator powers from network")
            

            # Calculate total cost using the full quadratic cost function
            cost = self._calculate_cost(generator_powers)

            return generator_powers, cost, solve_time

        except Exception as e:
            print(f"Solver failed: {e}")
            # Return a feasible but suboptimal solution
            solve_time = time.time() - start_time

            # Proportional distribution based on max capacity
            total_capacity = np.sum(self.gen_params['p_max'])
            generator_powers = (net_load / total_capacity) * self.gen_params['p_max']
            generator_powers = np.clip(
                generator_powers,
                self.gen_params['p_min'],
                self.gen_params['p_max']
            )

            cost = self._calculate_cost(generator_powers)

            return generator_powers, cost, solve_time

    def _calculate_cost(self, powers: np.ndarray) -> float:
        """Calculate total generation cost using quadratic cost function
        
        Args:
            powers: Generator power outputs
            
        Returns:
            Total cost in $/hour
        """
        alpha = self.gen_params['alpha']
        beta = self.gen_params['beta']
        gamma = self.gen_params['gamma']
        
        # Calculate quadratic cost: C_i = alpha_i + beta_i * P_i + gamma_i * P_i^2
        cost = np.sum(alpha + beta * powers + gamma * powers ** 2)
        
        # Debug output
        print(f"    Cost breakdown: Fixed={np.sum(alpha):.2f}, Linear={np.sum(beta * powers):.2f}, Quadratic={np.sum(gamma * powers**2):.2f}")
        
        return float(cost)


    def solve_full_simulation(self, data_path: str) -> pd.DataFrame:
        """Solve optimization for entire simulation period

        Args:
            data_path: Path to simulation data

        Returns:
            DataFrame with results for each timestep
        """
        # Load data
        data = pd.read_csv(data_path)

        results = []

        print(f"Solving classical ED for {len(data)} timesteps...")
        print(f"Using solver: {self.solver_name}")

        for idx, row in enumerate(data.itertuples(index=False), start=0):
            net_load = row.net_load

            # Solve for this timestep
            powers, cost, solve_time = self.solve_single_timestep(net_load) #type: ignore

            # Store results
            result = {
                'timestamp': row.timestamp,
                'net_load': net_load,
                'total_cost': cost,
                'solve_time': solve_time,
                'total_generation': np.sum(powers)
            }

            # Add individual generator powers
            for i, power in enumerate(powers):
                result[f'gen_{i}_power'] = power

            results.append(result)

            # Progress update
            if (idx + 1) % 50 == 0:
                print(f"  Completed {idx + 1}/{len(data)} timesteps")

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Calculate statistics
        print("\n" + "="*60)
        print("CLASSICAL OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Total cost: ${results_df['total_cost'].sum():,.2f}")
        print(f"Average cost per timestep: ${results_df['total_cost'].mean():,.2f}")
        print(f"Average solve time: {results_df['solve_time'].mean()*1000:.2f} ms")
        print(f"Max solve time: {results_df['solve_time'].max()*1000:.2f} ms")
        print(f"Total solve time: {results_df['solve_time'].sum():.2f} seconds")

        # Check power balance
        balance_error = np.abs(results_df['total_generation'] - results_df['net_load'])
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
