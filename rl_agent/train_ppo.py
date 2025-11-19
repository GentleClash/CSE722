"""
Reinforcement Learning Training Script
Trains PPO agent for Economic Dispatch using Stable-Baselines3
"""

import os
import sys
from typing import Optional
import yaml
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.economic_dispatch_env import EconomicDispatchEnv


class RLTrainer:
    """Trainer for PPO-based Economic Dispatch agent"""

    def __init__(self, 
                 config_path: str = 'config.yaml',
                 train_data_path: str = 'data/train_data.csv',
                 test_data_path: str = 'data/test_data.csv'):
        """Initialize the trainer

        Args:
            config_path: Path to configuration file
            train_data_path: Path to training data
            test_data_path: Path to test/evaluation data
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

        # Create directories
        self.models_dir = self.config['paths']['models_dir']
        self.logs_dir = self.config['paths']['logs_dir']
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Training parameters
        self.total_timesteps = self.config['rl_training']['total_timesteps']
        self.learning_rate = self.config['rl_training']['learning_rate']
        self.n_steps = self.config['rl_training']['n_steps']
        self.batch_size = self.config['rl_training']['batch_size']
        self.n_epochs = self.config['rl_training']['n_epochs']
        self.gamma = self.config['rl_training']['gamma']
        self.gae_lambda = self.config['rl_training']['gae_lambda']
        self.clip_range = self.config['rl_training']['clip_range']
        self.ent_coef = self.config['rl_training']['ent_coef']
        self.vf_coef = self.config['rl_training']['vf_coef']
        self.max_grad_norm = self.config['rl_training']['max_grad_norm']

        print("="*60)
        print("RL TRAINER INITIALIZED")
        print("="*60)
        print(f"Training timesteps: {self.total_timesteps:,}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        print("="*60)

    def create_training_env(self, n_envs: int = 4):
        """Create vectorized training environment

        Args:
            n_envs: Number of parallel environments

        Returns:
            Vectorized environment
        """
        def make_env():
            env = EconomicDispatchEnv(
                config_path='config.yaml',
                data_path=self.train_data_path
            )
            env = Monitor(env)
            return env

        # Create vectorized environment
        env = make_vec_env(make_env, n_envs=n_envs)

        return env

    def create_eval_env(self):
        """Create evaluation environment

        Returns:
            Evaluation environment
        """
        env = EconomicDispatchEnv(
            config_path='config.yaml',
            data_path=self.test_data_path
        )
        env = Monitor(env)

        return env

    def train(self, 
              model_name: Optional[str] = None,
              n_envs: int = 4,
              eval_freq: int = 10000,
              save_freq: int = 50000):
        """Train the PPO agent

        Args:
            model_name: Name for saved model (default: timestamp-based)
            n_envs: Number of parallel training environments
            eval_freq: Frequency of evaluation episodes
            save_freq: Frequency of model checkpoints
        """
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"ppo_ed_{timestamp}"

        print(f"\nTraining model: {model_name}")
        print("Creating training environment...")

        # Create environments
        train_env = self.create_training_env(n_envs=n_envs)
        eval_env = self.create_eval_env()

        print("Initializing PPO agent...")

        # Create PPO model
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            verbose=1,
            tensorboard_log=f"{self.logs_dir}/{model_name}",
            device='auto'  # Automatically use GPU if available
        )

        print("\nModel architecture:")
        print(model.policy)

        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.models_dir}/{model_name}_best",
            log_path=f"{self.logs_dir}/{model_name}_eval",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=f"{self.models_dir}/{model_name}_checkpoints",
            name_prefix=model_name
        )

        callbacks = [eval_callback, checkpoint_callback]

        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)

        # Train the model
        try:
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                progress_bar=True
            )

            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("="*60)

            # Save final model
            final_model_path = f"{self.models_dir}/{model_name}_final"
            model.save(final_model_path)
            print(f"Final model saved to: {final_model_path}")

            return model, final_model_path

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            # Save current progress
            interrupt_path = f"{self.models_dir}/{model_name}_interrupted"
            model.save(interrupt_path)
            print(f"Progress saved to: {interrupt_path}")
            return model, interrupt_path

        finally:
            # Clean up
            train_env.close()
            eval_env.close()

    def evaluate_model(self, 
                      model_path: str,
                      n_episodes: int = 10,
                      deterministic: bool = True):
        """Evaluate a trained model

        Args:
            model_path: Path to saved model
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating model: {model_path}")

        # Load model
        model = PPO.load(model_path)

        # Create evaluation environment
        env = self.create_eval_env()

        episode_rewards = []
        episode_costs = []
        episode_violations = []

        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            if 'episode' in info:
                episode_costs.append(info['episode']['total_cost'])
                episode_violations.append(info['episode']['total_violations'])

            print(f"  Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}")

        # Calculate statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_cost': np.mean(episode_costs) if episode_costs else 0,
            'std_cost': np.std(episode_costs) if episode_costs else 0,
            'mean_violations': np.mean(episode_violations) if episode_violations else 0
        }

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean cost: ${results['mean_cost']:,.2f} ± ${results['std_cost']:,.2f}")
        print(f"Mean violations: {results['mean_violations']:.4f}")
        print("="*60)

        env.close()

        return results


def main():
    """Main training function"""
    # Initialize trainer
    trainer = RLTrainer(
        config_path='config.yaml',
        train_data_path='data/train_data.csv',
        test_data_path='data/test_data.csv'
    )

    # Train model
    model, model_path = trainer.train(
        model_name="ppo_economic_dispatch",
        n_envs=4,
        eval_freq=10000,
        save_freq=50000
    )

    # Evaluate final model
    results = trainer.evaluate_model(
        model_path=model_path,
        n_episodes=10,
        deterministic=True
    )

    print("\nTraining and evaluation completed successfully!")
    return model, results


if __name__ == "__main__":
    main()
