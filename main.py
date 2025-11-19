"""
Main Project Runner
Orchestrates the complete Economic Dispatch project workflow
"""

import os
import sys
import argparse
from datetime import datetime

# Add project directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_generator import DataGenerator
from baseline.classical_solver import ClassicalEDSolver
from rl_agent.train_ppo import RLTrainer
from benchmarks.compare_methods import Benchmarker


def setup_project():
    """Setup project directories"""
    directories = ['data', 'models', 'results', 'logs']

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("✓ Project directories created")


def generate_data(train_days: int = 7, test_days: int = 1):
    """Generate simulation data

    Args:
        train_days: Number of days for training data
        test_days: Number of days for test data
    """
    print("\n" + "="*60)
    print("STEP 1: DATA GENERATION")
    print("="*60)

    generator = DataGenerator()

    # Generate training data
    print(f"\nGenerating {train_days} days of training data...")
    train_data = generator.generate_complete_dataset(days=train_days, seed=42)
    generator.save_dataset(train_data, 'train_data.csv')

    # Generate test data
    print(f"\nGenerating {test_days} day(s) of test data...")
    test_data = generator.generate_complete_dataset(days=test_days, seed=100)
    generator.save_dataset(test_data, 'test_data.csv')

    print("\n✓ Data generation completed")


def run_baseline():
    """Run classical baseline solver"""
    print("\n" + "="*60)
    print("STEP 2: BASELINE - CLASSICAL OPTIMIZATION")
    print("="*60)

    solver = ClassicalEDSolver()

    # Test on a single timestep first
    print("\nTesting single timestep...")
    powers, cost, solve_time = solver.solve_single_timestep(500.0)
    print(f"  ✓ Single timestep test passed")
    print(f"    Cost: ${cost:.2f}, Time: {solve_time*1000:.2f} ms")

    print("\n✓ Baseline solver ready")


def train_rl_agent(total_timesteps: int = 1000000):
    """Train RL agent

    Args:
        total_timesteps: Total training timesteps
    """
    print("\n" + "="*60)
    print("STEP 3: RL AGENT TRAINING")
    print("="*60)

    trainer = RLTrainer(
        config_path='config.yaml',
        train_data_path='data/train_data.csv',
        test_data_path='data/test_data.csv'
    )

    # Update training timesteps if specified
    if total_timesteps:
        trainer.total_timesteps = total_timesteps

    # Train model
    model, model_path = trainer.train(
        model_name="ppo_economic_dispatch",
        n_envs=4,
        eval_freq=10000,
        save_freq=50000
    )

    print(f"\n✓ RL agent trained and saved to: {model_path}")

    return model_path


def run_benchmark(rl_model_path: str):
    """Run benchmark comparison

    Args:
        rl_model_path: Path to trained RL model
    """
    print("\n" + "="*60)
    print("STEP 4: BENCHMARKING")
    print("="*60)

    benchmarker = Benchmarker(
        config_path='config.yaml',
        test_data_path='data/test_data.csv'
    )

    comparison = benchmarker.run_full_benchmark(rl_model_path)

    print("\n✓ Benchmark completed")

    return comparison


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Economic Dispatch Optimization Project'
    )

    parser.add_argument('--mode', type=str, 
                       choices=['full', 'data', 'baseline', 'train', 'benchmark'],
                       default='full',
                       help='Execution mode')

    parser.add_argument('--train-days', type=int, default=7,
                       help='Days of training data')

    parser.add_argument('--test-days', type=int, default=1,
                       help='Days of test data')

    parser.add_argument('--timesteps', type=int, default=1000000,
                       help='RL training timesteps')

    parser.add_argument('--rl-model', type=str,
                       help='Path to RL model (for benchmark mode)')

    args = parser.parse_args()

    # Print header
    print("\n" + "="*70)
    print(" "*10 + "ECONOMIC DISPATCH OPTIMIZATION PROJECT")
    print(" "*15 + "Classical Optimization vs RL")
    print("="*70)
    print(f"\nExecution Mode: {args.mode.upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Setup project
    setup_project()

    # Execute based on mode
    if args.mode == 'full':
        # Run complete pipeline
        generate_data(args.train_days, args.test_days)
        run_baseline()
        model_path = train_rl_agent(args.timesteps)
        run_benchmark(model_path)

    elif args.mode == 'data':
        generate_data(args.train_days, args.test_days)

    elif args.mode == 'baseline':
        run_baseline()

    elif args.mode == 'train':
        model_path = train_rl_agent(args.timesteps)

    elif args.mode == 'benchmark':
        if not args.rl_model:
            print("\nError: --rl-model required for benchmark mode")
            return
        run_benchmark(args.rl_model)

    print("\n" + "="*70)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY ✓")
    print("="*70)
    print("\nNext steps:")
    print("  1. Check results/ directory for benchmark results")
    print("  2. View logs/ directory for TensorBoard logs")
    print("  3. Examine models/ directory for saved models")
    print("\nTo view TensorBoard:")
    print("  tensorboard --logdir=logs/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
