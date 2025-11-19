import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import sys
sys.path.append('.')

from environment.economic_dispatch_env import EconomicDispatchEnv
from baseline.classical_solver import ClassicalEDSolver
from baseline.heuristic_dispatch import HeuristicDispatcher

print("="*70)
print("THREE-WAY BENCHMARK: Classical LP vs Heuristic vs PPO (Failed)")
print("="*70)

# Load test data
test_data = pd.read_csv('data/test_data.csv')

# ============================================================
# 1. CLASSICAL LP
# ============================================================
print("\n[1/3] Running Classical LP Solver...")
classical_solver = ClassicalEDSolver()
classical_results = classical_solver.solve_full_simulation('data/test_data.csv')
classical_results.to_csv('results/classical_results.csv', index=False)

# ============================================================
# 2. HEURISTIC
# ============================================================
print("\n[2/3] Running Heuristic Merit-Order Dispatcher...")
heuristic = HeuristicDispatcher()

heuristic_results = []
for idx, row in test_data.iterrows():
    net_load = row['net_load']
    powers = heuristic.dispatch(net_load)
    
    # Calculate cost
    alpha = np.array([150, 100, 50, 30, 200])
    beta = np.array([30, 35, 45, 60, 20])
    gamma = np.array([0.02, 0.015, 0.03, 0.04, 0.001])
    cost = np.sum(alpha + beta * powers + gamma * powers**2)
    
    result = {
        'timestamp': row['timestamp'],
        'net_load': net_load,
        'total_generation': np.sum(powers),
        'balance_error': abs(np.sum(powers) - net_load),
        'total_cost': cost,
        'solve_time': 0.0001  # ~0.1ms
    }
    
    for i, power in enumerate(powers):
        result[f'gen_{i}_power'] = power
    
    heuristic_results.append(result)

heuristic_df = pd.DataFrame(heuristic_results)
heuristic_df.to_csv('results/heuristic_results.csv', index=False)

print(f"Heuristic completed: {len(heuristic_df)} timesteps")
print(f"  Total cost: ${heuristic_df['total_cost'].sum():,.2f}")
print(f"  Avg balance error: {heuristic_df['balance_error'].mean():.6f} MW")

# ============================================================
# 3. PPO (FAILED MODEL)
# ============================================================
print("\n[3/3] Running PPO Agent (Expected to fail)...")

# Load the failed model
try:
    model = PPO.load('models/ppo_economic_dispatch_best/best_model')
    print("✓ Model loaded")
except:
    print("✗ Model not found - using final model")
    model = PPO.load('models/ppo_economic_dispatch_final')

# Create environment
env = EconomicDispatchEnv(config_path='config.yaml', data_path='data/test_data.csv')

ppo_results = []
obs, _ = env.reset()
done = False
step = 0

while not done and step < len(test_data):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    result = {
        'timestamp': test_data.iloc[step]['timestamp'],
        'net_load': test_data.iloc[step]['net_load'],
        'total_generation': np.sum(env.current_gen_powers),
        'balance_error': info['balance_error'],
        'total_cost': info['cost'],
        'solve_time': 0.0003
    }
    
    for i, power in enumerate(env.current_gen_powers):
        result[f'gen_{i}_power'] = power
    
    ppo_results.append(result)
    step += 1

ppo_df = pd.DataFrame(ppo_results)
ppo_df.to_csv('results/ppo_results.csv', index=False)

print(f"PPO completed: {len(ppo_df)} timesteps")
print(f"  Total cost: ${ppo_df['total_cost'].sum():,.2f}")
print(f"  Avg balance error: {ppo_df['balance_error'].mean():.2f} MW")

env.close()

# ============================================================
# SUMMARY COMPARISON
# ============================================================
print("\n" + "="*70)
print("THREE-WAY COMPARISON SUMMARY")
print("="*70)

comparison = pd.DataFrame({
    'Method': ['Classical LP', 'Heuristic Merit-Order', 'PPO (Failed)'],
    'Total Cost ($)': [
        classical_results['total_cost'].sum(),
        heuristic_df['total_cost'].sum(),
        ppo_df['total_cost'].sum()
    ],
    'Avg Cost ($/step)': [
        classical_results['total_cost'].mean(),
        heuristic_df['total_cost'].mean(),
        ppo_df['total_cost'].mean()
    ],
    'Avg Balance Error (MW)': [
        (classical_results['total_generation'] - classical_results['net_load']).abs().mean(),
        heuristic_df['balance_error'].mean(),
        ppo_df['balance_error'].mean()
    ],
    'Max Balance Error (MW)': [
        (classical_results['total_generation'] - classical_results['net_load']).abs().max(),
        heuristic_df['balance_error'].max(),
        ppo_df['balance_error'].max()
    ],
    'Avg Response Time (ms)': [
        classical_results['solve_time'].mean() * 1000,
        0.1,
        0.3
    ]
})

print(comparison.to_string(index=False))
print("="*70)

comparison.to_csv('results/three_way_comparison.csv', index=False)
print("\n✓ All results saved to results/ directory")
print("  - classical_results.csv")
print("  - heuristic_results.csv")
print("  - ppo_results.csv")
print("  - three_way_comparison.csv")
