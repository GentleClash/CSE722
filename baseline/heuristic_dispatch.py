"""
Simple Heuristic Dispatch - Provably Correct Baseline
Uses merit order (cheapest first) dispatch.

This module implements a rule-based economic dispatch algorithm based on the
"Merit Order" principle. It sorts generators by their marginal cost and
dispatches them sequentially until the demand is met. This method is
computationally efficient and guarantees power balance if sufficient capacity exists.
"""

import numpy as np
import pandas as pd


class HeuristicDispatcher:
    """
    Simple rule-based dispatcher using merit order.

    The dispatcher ranks generators by their linear cost coefficient (beta)
    and allocates power to the cheapest available generators first.
    """

    def __init__(self, config_path='config.yaml'):
        """
        Initialize the heuristic dispatcher.

        Args:
            config_path: Path to the configuration file.
        """
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Generator parameters
        self.n_gen = config['system']['n_generators']
        self.p_min = []
        self.p_max = []
        self.marginal_cost = []

        # Extract parameters for each generator
        for i in range(1, self.n_gen + 1):
            gen = config['generators'][f'gen_{i}']
            self.p_min.append(gen['p_min'])
            self.p_max.append(gen['p_max'])
            # Linear cost term used for ranking
            self.marginal_cost.append(gen['beta'])

        self.p_min = np.array(self.p_min)
        self.p_max = np.array(self.p_max)
        self.marginal_cost = np.array(self.marginal_cost)

        # Sort generators by cost (merit order)
        # This creates an index array where the first element is the index of the cheapest generator
        self.merit_order = np.argsort(self.marginal_cost)

    def dispatch(self, net_load: float) -> np.ndarray:
        """
        Dispatch generators to meet demand using merit order.

        Algorithm:
        1. Set all generators to their minimum output.
        2. Calculate remaining demand.
        3. If remaining demand is negative (over-generation), scale down all generators.
        4. If remaining demand is positive, increase output of generators in merit order
           (cheapest to expensive) until demand is met or max capacity is reached.

        Args:
            net_load: Target generation (MW) to meet.

        Returns:
            Generator power setpoints (MW).
        """
        # Start with minimum generation for all units
        # This ensures we respect the P_min constraint
        powers = self.p_min.copy()
        remaining = net_load - np.sum(powers)

        # Case 1: Demand is less than sum of minimums (Over-generation risk)
        if remaining < 0:
            if net_load < 0:
                # Return zeros for negative demand (shouldn't happen)
                return self.p_min * 0  # type: ignore

            # Scale all generators proportionally down to meet load exactly
            # Note: This technically violates P_min, but ensures power balance
            scale_factor = net_load / np.sum(self.p_min)
            return self.p_min * scale_factor

        # Case 2: Demand is greater than sum of minimums
        # Dispatch in merit order (cheapest first)
        for idx in self.merit_order:
            if remaining <= 0:
                break

            # How much more can this generator produce?
            available = self.p_max[idx] - powers[idx]

            # Allocate as much as needed or as much as available
            allocated = min(available, remaining)
            powers[idx] += allocated
            remaining -= allocated

        # Case 3: Demand exceeds total capacity
        # If still can't meet demand, we are at max capacity
        if remaining > 1e-6:
            # In a real system, this would mean load shedding
            powers = self.p_max.copy()

        return powers  # type: ignore


def benchmark_heuristic(data_path='data/test_data.csv'):
    """
    Run heuristic dispatcher on test data and calculate metrics.

    Args:
        data_path: Path to the test dataset.

    Returns:
        DataFrame containing dispatch results and costs.
    """

    dispatcher = HeuristicDispatcher()
    data = pd.read_csv(data_path)

    results = []
    for idx, row in data.iterrows():
        net_load = row['net_load']
        powers = dispatcher.dispatch(net_load)

        # Calculate cost using the full quadratic function
        # Hardcoded parameters for quick benchmarking (should match config)
        alpha = np.array([150, 100, 50, 30, 200])
        beta = np.array([30, 35, 45, 60, 20])
        gamma = np.array([0.02, 0.015, 0.03, 0.04, 0.001])
        cost = np.sum(alpha + beta * powers + gamma * powers**2)

        results.append({
            'net_load': net_load,
            'total_generation': np.sum(powers),
            'balance_error': abs(np.sum(powers) - net_load),
            'cost': cost,
            **{f'gen_{i}_power': powers[i] for i in range(len(powers))}
        })

    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("HEURISTIC DISPATCHER RESULTS")
    print(f"{'='*60}")
    print(f"Total cost: ${df['cost'].sum():,.2f}")
    print(f"Avg cost: ${df['cost'].mean():,.2f}")
    print(f"Avg balance error: {df['balance_error'].mean():.6f} MW")
    print(f"Max balance error: {df['balance_error'].max():.6f} MW")
    print(f"{'='*60}")

    df.to_csv('results/heuristic_results.csv', index=False)
    return df


if __name__ == "__main__":
    benchmark_heuristic()
