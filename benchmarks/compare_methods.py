"""
Benchmarking Script - Classical Optimization vs RL Agent
Compares performance on key metrics
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml
import time

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.classical_solver import ClassicalEDSolver
from environment.economic_dispatch_env import EconomicDispatchEnv
from stable_baselines3 import PPO


class Benchmarker:
    """Benchmark classical and RL approaches"""

    def __init__(self, 
                 config_path: str = 'config.yaml',
                 test_data_path: str = 'data/test_data.csv'):
        """Initialize benchmarker

        Args:
            config_path: Path to configuration file
            test_data_path: Path to test data
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.test_data_path = test_data_path
        self.test_data = pd.read_csv(test_data_path)

        # Create results directory
        self.results_dir = self.config['paths']['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)

    def benchmark_classical(self) -> pd.DataFrame:
        """Run classical solver on test data

        Returns:
            DataFrame with classical solver results
        """
        print("\n" + "="*60)
        print("BENCHMARKING CLASSICAL SOLVER")
        print("="*60)

        solver = ClassicalEDSolver()
        results = solver.solve_full_simulation(self.test_data_path)

        # Save results
        results.to_csv(f"{self.results_dir}/classical_results.csv", index=False)

        return results

    def benchmark_rl(self, model_path: str) -> pd.DataFrame:
        """Run RL agent on test data

        Args:
            model_path: Path to trained RL model

        Returns:
            DataFrame with RL agent results
        """
        print("\n" + "="*60)
        print("BENCHMARKING RL AGENT")
        print("="*60)

        # Load model
        from pathlib import Path
        model_file = Path(model_path)
        if model_file.suffix != '.zip':
            model_path = str(model_file) + '.zip'  # Add .zip extension
        
        if not Path(model_path).exists():
            print(f"ERROR: Model file not found: {model_path}")
            print("Available models:")
            import glob
            for f in glob.glob("models/*.zip"):
                print(f"  - {f}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        print(f"Loading model from: {model_path}")
        try:
            model = PPO.load(model_path)
            print(f"✓ Model loaded successfully")
            print(f"  Policy architecture: {model.policy}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

        # Create environment
        env = EconomicDispatchEnv(
            config_path='config.yaml',
            data_path=self.test_data_path
        )

        if model.num_timesteps == 0:
            print("WARNING: Model appears untrained (num_timesteps=0)!")
            print("   This will produce constant/random outputs.")
            print("   Make sure you trained the model before benchmarking.")
        else:
            print(f"✓ Model has been trained for {model.num_timesteps:,} timesteps")

        # Run simulation
        results = []
        obs, info = env.reset()
        done = False
        step = 0

        print(f"Running simulation for {len(self.test_data)} timesteps...")

        inference_times = []
        actions_list = []

        while not done and step < len(self.test_data):
            # Predict action
            start_time = time.time()
            action, _states = model.predict(obs, deterministic=False)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            actions_list.append(action.copy())

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store results
            result = {
                'timestamp': self.test_data.iloc[step]['timestamp'],
                'net_load': self.test_data.iloc[step]['net_load'],
                'total_cost': info['cost'],
                'solve_time': inference_time,
                'balance_error': info['balance_error'],
                'limit_violations': info['limit_violations'],
                'ramp_violations': info['ramp_violations'],
                'total_generation': np.sum(action)
            }

            # Add individual generator powers
            for i, power in enumerate(action):
                result[f'gen_{i}_power'] = power

            results.append(result)
            step += 1

            # Progress update
            if (step) % 50 == 0:
                print(f"  Completed {step}/{len(self.test_data)} timesteps")
        
        actions_array = np.array(actions_list)
        action_variance = np.var(actions_array, axis=0)
        
        print("\\n" + "="*60)
        print("ACTION VERIFICATION")
        print("="*60)
        print(f"Action variance per generator: {action_variance}")
        
        if np.all(action_variance < 1e-6):
            print("❌ WARNING: All actions are constant!")
            print("   This means the model is not trained or using wrong policy.")
            print("   Generator outputs:")
            for i in range(actions_array.shape[1]):
                print(f"     Gen {i}: {actions_array[0, i]:.4f} MW (constant)")
        else:
            print("✓ Actions vary across timesteps (model is working)")
            for i in range(actions_array.shape[1]):
                print(f"  Gen {i}: μ={np.mean(actions_array[:, i]):.2f}, σ={np.std(actions_array[:, i]):.2f} MW")

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Calculate statistics
        print("\n" + "="*60)
        print("RL AGENT RESULTS")
        print("="*60)
        print(f"Total cost: ${results_df['total_cost'].sum():,.2f}")
        print(f"Average cost per timestep: ${results_df['total_cost'].mean():,.2f}")
        print(f"Average inference time: {results_df['solve_time'].mean()*1000:.2f} ms")
        print(f"Max inference time: {results_df['solve_time'].max()*1000:.2f} ms")
        print(f"Total inference time: {results_df['solve_time'].sum():.2f} seconds")
        print(f"\nAverage balance error: {results_df['balance_error'].mean():.6f} MW")
        print(f"Max balance error: {results_df['balance_error'].max():.6f} MW")
        print(f"Total limit violations: {results_df['limit_violations'].sum():.2f}")
        print(f"Total ramp violations: {results_df['ramp_violations'].sum():.2f}")
        print("="*60)

        # Save results
        results_df.to_csv(f"{self.results_dir}/rl_results.csv", index=False)

        env.close()

        return results_df

    def compare_results(self, 
                       classical_results: pd.DataFrame,
                       rl_results: pd.DataFrame) -> dict:
        """Compare classical and RL results

        Args:
            classical_results: Classical solver results
            rl_results: RL agent results

        Returns:
            Dictionary of comparison metrics
        """
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS")
        print("="*60)

        comparison = {
            'classical': {
                'total_cost': classical_results['total_cost'].sum(),
                'avg_cost': classical_results['total_cost'].mean(),
                'avg_solve_time_ms': classical_results['solve_time'].mean() * 1000,
                'max_solve_time_ms': classical_results['solve_time'].max() * 1000,
                'total_time_s': classical_results['solve_time'].sum(),
                'avg_balance_error': (classical_results['total_generation'] - 
                                     classical_results['net_load']).abs().mean(),
                'max_balance_error': (classical_results['total_generation'] - 
                                     classical_results['net_load']).abs().max()
            },
            'rl': {
                'total_cost': rl_results['total_cost'].sum(),
                'avg_cost': rl_results['total_cost'].mean(),
                'avg_solve_time_ms': rl_results['solve_time'].mean() * 1000,
                'max_solve_time_ms': rl_results['solve_time'].max() * 1000,
                'total_time_s': rl_results['solve_time'].sum(),
                'avg_balance_error': rl_results['balance_error'].mean(),
                'max_balance_error': rl_results['balance_error'].max(),
                'total_limit_violations': rl_results['limit_violations'].sum(),
                'total_ramp_violations': rl_results['ramp_violations'].sum()
            }
        }

        # Calculate improvements
        cost_saving = comparison['classical']['total_cost'] - comparison['rl']['total_cost']
        cost_saving_pct = (cost_saving / comparison['classical']['total_cost']) * 100

        speedup = (comparison['classical']['avg_solve_time_ms'] / 
                  comparison['rl']['avg_solve_time_ms'])

        print("\n📊 COST COMPARISON:")
        print(f"  Classical Total Cost: ${comparison['classical']['total_cost']:,.2f}")
        print(f"  RL Total Cost:        ${comparison['rl']['total_cost']:,.2f}")
        print(f"  Cost Savings:         ${cost_saving:,.2f} ({cost_saving_pct:+.2f}%)")

        print("\n⚡ RESPONSE TIME COMPARISON:")
        print(f"  Classical Avg Time: {comparison['classical']['avg_solve_time_ms']:.2f} ms")
        print(f"  RL Avg Time:        {comparison['rl']['avg_solve_time_ms']:.2f} ms")
        print(f"  Speedup:            {speedup:.1f}x faster")

        print("\n✓ RELIABILITY (Power Balance):")
        print(f"  Classical Avg Error: {comparison['classical']['avg_balance_error']:.6f} MW")
        print(f"  RL Avg Error:        {comparison['rl']['avg_balance_error']:.6f} MW")

        if 'total_limit_violations' in comparison['rl']:
            print(f"\n⚠️  RL CONSTRAINT VIOLATIONS:")
            print(f"  Limit Violations: {comparison['rl']['total_limit_violations']:.2f}")
            print(f"  Ramp Violations:  {comparison['rl']['total_ramp_violations']:.2f}")

        print("="*60)

        # Save comparison
        comparison_df = pd.DataFrame(comparison).T
        comparison_df.to_csv(f"{self.results_dir}/comparison_summary.csv")

        return comparison

    def plot_results(self, 
                    classical_results: pd.DataFrame,
                    rl_results: pd.DataFrame):
        """Create visualization plots

        Args:
            classical_results: Classical solver results
            rl_results: RL agent results
        """
        print("\nCreating visualization plots...")

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Cost comparison over time
        ax = axes[0, 0]
        timesteps = range(len(classical_results))
        ax.plot(timesteps, classical_results['total_cost'], 
               label='Classical', alpha=0.7, linewidth=1.5)
        ax.plot(timesteps, rl_results['total_cost'], 
               label='RL Agent', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Cost ($/hour)')
        ax.set_title('Generation Cost Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Cumulative cost
        ax = axes[0, 1]
        ax.plot(timesteps, classical_results['total_cost'].cumsum(), 
               label='Classical', linewidth=2)
        ax.plot(timesteps, rl_results['total_cost'].cumsum(), 
               label='RL Agent', linewidth=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Cumulative Cost ($)')
        ax.set_title('Cumulative Generation Cost')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Solve time comparison
        ax = axes[1, 0]
        methods = ['Classical', 'RL Agent']
        avg_times = [
            classical_results['solve_time'].mean() * 1000,
            rl_results['solve_time'].mean() * 1000
        ]
        colors = ['#3498db', '#e74c3c']
        ax.bar(methods, avg_times, color=colors, alpha=0.7)
        ax.set_ylabel('Average Time (ms)')
        ax.set_title('Response Time Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(avg_times):
            ax.text(i, v + max(avg_times)*0.02, f'{v:.2f} ms', 
                   ha='center', va='bottom', fontweight='bold')

        # Plot 4: Power balance errors
        ax = axes[1, 1]
        classical_error = (classical_results['total_generation'] - 
                          classical_results['net_load']).abs()
        rl_error = rl_results['balance_error']

        ax.plot(timesteps, classical_error, 
               label='Classical', alpha=0.7, linewidth=1.5)
        ax.plot(timesteps, rl_error, 
               label='RL Agent', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Balance Error (MW)')
        ax.set_title('Power Balance Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = f"{self.results_dir}/comparison_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_path}")

        plt.close()

    def run_full_benchmark(self, rl_model_path: str):
        """Run complete benchmark suite

        Args:
            rl_model_path: Path to trained RL model
        """
        print("\n" + "="*70)
        print(" "*15 + "ECONOMIC DISPATCH BENCHMARK SUITE")
        print("="*70)

        # Run classical solver
        classical_results = self.benchmark_classical()

        # Run RL agent
        rl_results = self.benchmark_rl(rl_model_path)

        # Compare results
        comparison = self.compare_results(classical_results, rl_results)

        # Create visualizations
        self.plot_results(classical_results, rl_results)

        print("\n" + "="*70)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print(f"Results saved to: {self.results_dir}/")
        print("="*70)

        return comparison


def main():
    """Main benchmarking function"""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark Economic Dispatch solvers')
    parser.add_argument('--rl-model', type=str, required=True,
                       help='Path to trained RL model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--test-data', type=str, default='data/test_data.csv',
                       help='Path to test data')

    args = parser.parse_args()

    # Run benchmark
    benchmarker = Benchmarker(
        config_path=args.config,
        test_data_path=args.test_data
    )

    comparison = benchmarker.run_full_benchmark(args.rl_model)

    return comparison


if __name__ == "__main__":
    main()
