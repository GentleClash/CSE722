"""
Simple Heuristic Dispatch - Provably Correct Baseline
Uses merit order (cheapest first) dispatch
"""

import numpy as np
import pandas as pd

class HeuristicDispatcher:
    """Simple rule-based dispatcher using merit order"""
    
    def __init__(self, config_path='config.yaml'):
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Generator parameters
        self.n_gen = config['system']['n_generators']
        self.p_min = []
        self.p_max = []
        self.marginal_cost = []
        
        for i in range(1, self.n_gen + 1):
            gen = config['generators'][f'gen_{i}']
            self.p_min.append(gen['p_min'])
            self.p_max.append(gen['p_max'])
            self.marginal_cost.append(gen['beta'])  # Linear cost term
        
        self.p_min = np.array(self.p_min)
        self.p_max = np.array(self.p_max)
        self.marginal_cost = np.array(self.marginal_cost)
        
        # Sort generators by cost (merit order)
        self.merit_order = np.argsort(self.marginal_cost)
        
    def dispatch(self, net_load: float) -> np.ndarray:
        """Dispatch generators to meet demand using merit order
        
        Args:
            net_load: Target generation (MW)
            
        Returns:
            Generator power setpoints (MW)
        """
        # Start with minimum generation
        powers = self.p_min.copy()
        remaining = net_load - np.sum(powers)
        
        # If demand < sum of minimums, scale down proportionally
        if remaining < 0:
            if net_load < 0:
                return self.p_min * 0  # Return zeros for negative demand #type: ignore
            # Scale all generators proportionally down
            scale_factor = net_load / np.sum(self.p_min)
            return self.p_min * scale_factor
        
        # Dispatch in merit order (cheapest first)
        for idx in self.merit_order:
            if remaining <= 0:
                break
            
            # How much can this generator increase?
            available = self.p_max[idx] - powers[idx]
            
            # Allocate up to its maximum
            allocated = min(available, remaining)
            powers[idx] += allocated
            remaining -= allocated
        
        # If still can't meet demand, scale up to maximum
        if remaining > 1e-6:
            powers = self.p_max.copy()
        
        return powers #type: ignore


def benchmark_heuristic(data_path='data/test_data.csv'):
    """Run heuristic dispatcher on test data"""
    
    dispatcher = HeuristicDispatcher()
    data = pd.read_csv(data_path)
    
    results = []
    for idx, row in data.iterrows():
        net_load = row['net_load']
        powers = dispatcher.dispatch(net_load)
        
        # Calculate cost
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
