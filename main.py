"""
Main Project Runner
Orchestrates the complete Economic Dispatch project workflow.

This script serves as the entry point for the project, allowing the user to:
1. Generate synthetic training and test data.
2. Run the classical optimization baseline (PyPSA).
3. Train the Reinforcement Learning (PPO) agent.
4. Benchmark the performance of different methods.

Usage:
    python main.py --mode [full|data|baseline|train|benchmark]
"""

from benchmarks.compare_methods import Benchmarker
from rl_agent.train_ppo import RLTrainer
from baseline.classical_solver import ClassicalEDSolver
from utils.data_generator import DataGenerator
import os
import sys
import argparse
from datetime import datetime

# Add project directories to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def setup_project():
    """
    Setup project directories.

    Creates the necessary folder structure for storing data, models, results, and logs
    if they do not already exist.
    """
    directories = ['data', 'models', 'results', 'logs']

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("✓ Project directories created")


def generate_data(train_days: int = 7, test_days: int = 1):
    """
    Generate simulation data for training and testing.

    Args:
        train_days: Number of days for training data (default: 7).
        test_days: Number of days for test data (default: 1).

    This function uses the DataGenerator class to create synthetic demand and renewable
    generation profiles based on the models described in the project report.
    """
    print("\n" + "="*60)
    print("STEP 1: DATA GENERATION")
    print("="*60)

    generator = DataGenerator()

    # Generate training data
    # The training dataset is used to train the RL agent.
    print(f"\nGenerating {train_days} days of training data...")
    train_data = generator.generate_complete_dataset(days=train_days, seed=42)
    generator.save_dataset(train_data, 'train_data.csv')

    # Generate test data
    # The test dataset is used for out-of-sample evaluation and benchmarking.
    # A different seed is used to ensure the test data is unseen.
    print(f"\nGenerating {test_days} day(s) of test data...")
    test_data = generator.generate_complete_dataset(days=test_days, seed=100)
    generator.save_dataset(test_data, 'test_data.csv')

    print("\n✓ Data generation completed")


def run_baseline():
    """
    Run the classical baseline solver (PyPSA).

    This function initializes the ClassicalEDSolver and runs a quick test on a single
    timestep to verify that the optimization environment is set up correctly.
    """
    print("\n" + "="*60)
    print("STEP 2: BASELINE - CLASSICAL OPTIMIZATION")
    print("="*60)

    solver = ClassicalEDSolver()

    # Test on a single timestep first to ensure the solver is working
    print("\nTesting single timestep...")
    # Solve for a hypothetical load of 500 MW
    powers, cost, solve_time = solver.solve_single_timestep(500.0)
    print(f"  ✓ Single timestep test passed")
    print(f"    Cost: ${cost:.2f}, Time: {solve_time*1000:.2f} ms")

    print("\n✓ Baseline solver ready")


def train_rl_agent(total_timesteps: int = 1000000):
    """
    Train the Reinforcement Learning (RL) agent.

    Args:
        total_timesteps: Total number of timesteps to train the agent.

    This function initializes the RLTrainer and starts the training process using
    Proximal Policy Optimization (PPO). The trained model is saved to the 'models' directory.
    """
    print("\n" + "="*60)
    print("STEP 3: RL AGENT TRAINING")
    print("="*60)

    trainer = RLTrainer(
        config_path='config.yaml',
        train_data_path='data/train_data.csv',
        test_data_path='data/test_data.csv'
    )

    # Update training timesteps if specified in arguments
    if total_timesteps:
        trainer.total_timesteps = total_timesteps

    # Train model
    # n_envs=4 allows parallel data collection for faster training
    model, model_path = trainer.train(
        model_name="ppo_economic_dispatch",
        n_envs=4,
        eval_freq=10000,
        save_freq=50000
    )

    print(f"\n✓ RL agent trained and saved to: {model_path}")

    return model_path


def run_benchmark(rl_model_path: str):
    """
    Run the benchmark comparison.

    Args:
        rl_model_path: Path to the trained RL model to evaluate.

    This function compares the performance of the trained RL agent against the
    classical baseline using the test dataset. It generates metrics and plots.
    """
    print("\n" + "="*60)
    print("STEP 4: BENCHMARKING")
    print("="*60)

    benchmarker = Benchmarker(
        config_path='config.yaml',
        test_data_path='data/test_data.csv'
    )

    # Run the full benchmark suite
    comparison = benchmarker.run_full_benchmark(rl_model_path)

    print("\n✓ Benchmark completed")

    return comparison


def main():
    """
    Main execution function.

    Parses command-line arguments and executes the requested mode.
    """
    parser = argparse.ArgumentParser(
        description='Economic Dispatch Optimization Project'
    )

    # Define command-line arguments
    parser.add_argument('--mode', type=str,
                        choices=['full', 'data', 'baseline',
                                 'train', 'benchmark'],
                        default='full',
                        help='Execution mode: full pipeline or specific step')

    parser.add_argument('--train-days', type=int, default=7,
                        help='Number of days of training data to generate')

    parser.add_argument('--test-days', type=int, default=1,
                        help='Number of days of test data to generate')

    parser.add_argument('--timesteps', type=int, default=1000000,
                        help='Total timesteps for RL training')

    parser.add_argument('--rl-model', type=str,
                        help='Path to RL model file (required for benchmark mode)')

    args = parser.parse_args()

    # Print project header
    print("\n" + "="*70)
    print(" "*10 + "ECONOMIC DISPATCH OPTIMIZATION PROJECT")
    print(" "*15 + "Classical Optimization vs RL")
    print("="*70)
    print(f"\nExecution Mode: {args.mode.upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Setup project structure
    setup_project()

    # Execute logic based on selected mode
    if args.mode == 'full':
        # Run the complete pipeline in order
        generate_data(args.train_days, args.test_days)
        run_baseline()
        model_path = train_rl_agent(args.timesteps)
        run_benchmark(model_path)

    elif args.mode == 'data':
        # Only generate data
        generate_data(args.train_days, args.test_days)

    elif args.mode == 'baseline':
        # Only run baseline test
        run_baseline()

    elif args.mode == 'train':
        # Only train the agent
        model_path = train_rl_agent(args.timesteps)

    elif args.mode == 'benchmark':
        # Only run benchmarking (requires a pre-trained model)
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
