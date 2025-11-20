"""
Comprehensive Benchmark Runner
Executes all three economic dispatch methods (Classical, Heuristic, RL) and compares them.

This script serves as the master benchmarking tool. It:
1. Runs the Classical Optimization solver (PyPSA).
2. Runs the Heuristic Dispatcher (Merit Order).
3. Runs the trained RL Agent (PPO).
4. Aggregates results into a unified format.
5. Generates a summary comparison table.
"""

from benchmarks.compare_methods import Benchmarker
from baseline.heuristic_dispatch import HeuristicDispatcher
from baseline.classical_solver import ClassicalEDSolver
import os
import sys
import pandas as pd
import numpy as np
import yaml
import argparse

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_methods(rl_model_path, test_data_path='data/test_data.csv'):
    """
    Execute all three dispatch methods on the same test dataset.

    Args:
        rl_model_path: Path to the trained RL model file.
        test_data_path: Path to the test data CSV file.

    Returns:
        Three DataFrames corresponding to the results of each method:
        (classical_df, heuristic_df, rl_df)
    """
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE BENCHMARK: CLASSICAL vs HEURISTIC vs RL")
    print("="*80)

    # 1. Run Classical Solver
    print("\n" + "-"*40)
    print("1. Running Classical Optimization (PyPSA)...")
    print("-"*40)
    classical_solver = ClassicalEDSolver()
    classical_results = classical_solver.solve_full_simulation(test_data_path)
    classical_results['method'] = 'Classical (LP)'

    # 2. Run Heuristic Dispatcher
    print("\n" + "-"*40)
    print("2. Running Heuristic Dispatch (Merit Order)...")
    print("-"*40)
    heuristic_dispatcher = HeuristicDispatcher()
    data = pd.read_csv(test_data_path)

    heuristic_data = []
    alpha = np.array([150, 100, 50, 30, 200])
    beta = np.array([30, 35, 45, 60, 20])
    gamma = np.array([0.02, 0.015, 0.03, 0.04, 0.001])

    for idx, row in data.iterrows():
        net_load = row['net_load']
        powers = heuristic_dispatcher.dispatch(net_load)

        # Calculate cost
        cost = np.sum(alpha + beta * powers + gamma * powers**2)

        heuristic_data.append({
            'timestamp': row['timestamp'],
            'net_load': net_load,
            'total_cost': cost,
            'solve_time': 0.0001,
            'total_generation': np.sum(powers),
            'balance_error': abs(np.sum(powers) - net_load),
            'method': 'Heuristic'
        })

        # Add individual generator powers
        for i, p in enumerate(powers):
            heuristic_data[-1][f'gen_{i}_power'] = p

    heuristic_results = pd.DataFrame(heuristic_data)
    print(
        f"Heuristic run completed. Total cost: ${heuristic_results['total_cost'].sum():,.2f}")

    # 3. Run RL Agent
    print("\n" + "-"*40)
    print("3. Running RL Agent (PPO)...")
    print("-"*40)
    benchmarker = Benchmarker(test_data_path=test_data_path)
    rl_results = benchmarker.benchmark_rl(rl_model_path)
    rl_results['method'] = 'RL Agent (PPO)'

    return classical_results, heuristic_results, rl_results


def print_summary_table(classical, heuristic, rl):
    """
    Print a formatted summary table comparing the three methods.

    Args:
        classical: DataFrame of classical results.
        heuristic: DataFrame of heuristic results.
        rl: DataFrame of RL results.
    """
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    # Define metrics to compare
    metrics = {
        'Total Cost ($)': [
            classical['total_cost'].sum(),
            heuristic['total_cost'].sum(),
            rl['total_cost'].sum()
        ],
        'Avg Cost ($/h)': [
            classical['total_cost'].mean(),
            heuristic['total_cost'].mean(),
            rl['total_cost'].mean()
        ],
        'Avg Balance Error (MW)': [
            (classical['total_generation'] -
             classical['net_load']).abs().mean(),
            heuristic['balance_error'].mean(),
            rl['balance_error'].mean()
        ],
        'Avg Solve Time (ms)': [
            classical['solve_time'].mean() * 1000,
            heuristic['solve_time'].mean() * 1000,
            rl['solve_time'].mean() * 1000
        ]
    }

    # Create summary DataFrame
    summary = pd.DataFrame(
        metrics, index=['Classical', 'Heuristic', 'RL Agent'])

    # Calculate savings relative to Heuristic (Baseline)
    heuristic_cost = metrics['Total Cost ($)'][1]
    summary['Savings vs Heuristic (%)'] = [
        (heuristic_cost - x) / heuristic_cost * 100 for x in metrics['Total Cost ($)']
    ]

    # Format and print
    pd.options.display.float_format = '{:,.2f}'.format
    print(summary)
    print("="*80)

    # Save to CSV
    os.makedirs('results', exist_ok=True)
    summary.to_csv('results/three_way_summary.csv')
    print("Summary saved to results/three_way_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run all three economic dispatch methods')
    parser.add_argument('--rl-model', type=str, required=True,
                        help='Path to trained RL model')
    parser.add_argument('--data', type=str,
                        default='data/test_data.csv', help='Path to test data')

    args = parser.parse_args()

    # Execute comparison
    c_res, h_res, rl_res = run_all_methods(args.rl_model, args.data)

    # Save detailed results
    c_res.to_csv('results/classical_detailed.csv', index=False)
    h_res.to_csv('results/heuristic_detailed.csv', index=False)
    rl_res.to_csv('results/rl_detailed.csv', index=False)

    # Print summary
    print_summary_table(c_res, h_res, rl_res)
