"""
Custom Gymnasium Environment for Economic Dispatch
Wraps PyPSA simulation in a Gymnasium-compatible interface
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yaml
from typing import Dict, Tuple, Optional


class EconomicDispatchEnv(gym.Env):
    """
    Custom Gymnasium environment for Economic Dispatch optimization

    This environment simulates a power grid with multiple generators
    and stochastic renewable energy sources. The agent must learn to
    dispatch generators optimally to minimize cost while maintaining
    power balance.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, 
                 config_path: str = 'config.yaml',
                 data_path: str = 'data/train_data.csv',
                 render_mode: Optional[str] = None):
        """Initialize the environment

        Args:
            config_path: Path to configuration file
            data_path: Path to demand/renewable data
            render_mode: Rendering mode for visualization
        """
        super().__init__()

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load data
        self.data = pd.read_csv(data_path)
        self.max_steps = len(self.data)

        # Environment parameters
        self.n_generators = self.config['system']['n_generators']
        self.timestep_duration = self.config['system']['timestep_duration']

        # Extract generator parameters
        self.gen_params = self._extract_generator_params()

        # Reward function weights
        self.cost_weight = self.config['reward']['cost_weight']
        self.balance_penalty = self.config['reward']['balance_penalty']
        self.limit_penalty = self.config['reward']['limit_penalty']
        self.ramp_penalty = self.config['reward']['ramp_penalty']

        # Define action space (continuous power setpoints for each generator)
        self.p_min_true = self.gen_params['p_min'].copy()
        self.p_max_true = self.gen_params['p_max'].copy()
        
        # Define NORMALIZED action space [-1, 1] for better learning
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_generators,),
            dtype=np.float32
        )

        # Define observation space
        # State: [net_load, gen_powers (t-1), hour_of_day, ramp_capabilities]
        obs_dim = 1 + self.n_generators + 1 + self.n_generators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Initialize state
        self.current_step = 0
        self.current_gen_powers = None
        self.render_mode = render_mode
        self._step_count = 0

        # Tracking
        self.episode_costs = []
        self.episode_violations = []

    def _extract_generator_params(self) -> Dict[str, np.ndarray]:
        """Extract generator parameters from config

        Returns:
            Dictionary of generator parameter arrays
        """
        params = {
            'alpha': [],
            'beta': [],
            'gamma': [],
            'p_min': [],
            'p_max': [],
            'emission_rate': [],
            'ramp_rate': []
        }

        for i in range(1, self.n_generators + 1):
            gen_key = f'gen_{i}'
            gen_config = self.config['generators'][gen_key]

            params['alpha'].append(gen_config['alpha'])
            params['beta'].append(gen_config['beta'])
            params['gamma'].append(gen_config['gamma'])
            params['p_min'].append(gen_config['p_min'])
            params['p_max'].append(gen_config['p_max'])
            params['emission_rate'].append(gen_config['emission_rate'])
            params['ramp_rate'].append(gen_config['ramp_rate'])

        # Convert to numpy arrays
        return {k: np.array(v, dtype=np.float32) for k, v in params.items()}

    def _get_observation(self) -> np.ndarray:
        """Construct observation for the CURRENT timestep"""
        
        # Use current step (which points to the NEXT demand to handle)
        step_idx = self.current_step % len(self.data)
        
        # Get net load for the timestep we're ABOUT TO act on
        net_load = self.data.iloc[step_idx]['net_load']
        
        # Get time information
        timestamp = pd.to_datetime(self.data.iloc[step_idx]['timestamp'])
        hour_of_day = (timestamp.hour + timestamp.minute / 60) / 24
        
        # Initialize generator powers if first step
        if self.current_gen_powers is None:
            total_capacity = np.sum(self.gen_params['p_max'])
            self.current_gen_powers = (net_load / total_capacity) * self.gen_params['p_max']
            self.current_gen_powers = np.clip(
                self.current_gen_powers,
                self.gen_params['p_min'],
                self.gen_params['p_max']
            )
        
        # Construct state: [demand, prev_powers, hour, ramp_rates]
        state = np.concatenate([
            [net_load],                    # What demand to meet
            self.current_gen_powers,       # What we did last time
            [hour_of_day],                 # When it is
            self.gen_params['ramp_rate']   # How fast we can change
        ])
        
        return state.astype(np.float32)


    def _calculate_cost(self, powers: np.ndarray) -> float:
        """Calculate total generation cost

        Args:
            powers: Generator power outputs

        Returns:
            Total cost in $/hour
        """
        alpha = self.gen_params['alpha']
        beta = self.gen_params['beta']
        gamma = self.gen_params['gamma']

        # Quadratic cost function: C_i = alpha_i + beta_i * P_i + gamma_i * P_i^2
        cost = np.sum(alpha + beta * powers + gamma * powers ** 2)

        return float(cost)

    def _calculate_reward(self, powers: np.ndarray, net_load: float) -> Tuple[float, Dict]:
        """Reward function with exponential scaling for fast learning
        
        Key insight: Small errors should get BIG positive rewards,
        large errors should get EXPONENTIALLY worse penalties
        """
        
        # Metrics
        cost = self._calculate_cost(powers)
        total_generation = np.sum(powers)
        balance_error = abs(total_generation - net_load)
        
        # Constraint violations
        lower_viols = np.sum(np.maximum(0, self.gen_params['p_min'] - powers))
        upper_viols = np.sum(np.maximum(0, powers - self.gen_params['p_max']))
        total_viols = lower_viols + upper_viols
        
        # EXPONENTIAL REWARD STRUCTURE FOR FAST LEARNING
        if total_viols > 0.01:
            # Hard constraint - instant death
            reward = -100000.0
        elif balance_error < 1.0:
            # PERFECT: Within 1 MW → HUGE positive reward
            reward = 50000.0 - (cost / 5.0)
        elif balance_error < 5.0:
            # EXCELLENT: Within 5 MW → Large positive reward
            # Linear decay from 50000 to 20000
            reward = 50000.0 - (balance_error - 1.0) * 7500.0 - (cost / 5.0)
        elif balance_error < 10.0:
            # GOOD: Within 10 MW → Positive reward
            reward = 20000.0 - (balance_error - 5.0) * 3000.0 - (cost / 5.0)
        elif balance_error < 20.0:
            # OK: Within 20 MW → Small positive/negative
            reward = 5000.0 - (balance_error - 10.0) * 400.0 - (cost / 5.0)
        elif balance_error < 50.0:
            # BAD: 20-50 MW error → Increasing penalty
            reward = -((balance_error - 20.0) ** 2) * 10.0  # Quadratic penalty
        else:
            # TERRIBLE: >50 MW error → Exponential penalty
            reward = -((balance_error - 50.0) ** 2) * 50.0 - 10000.0
        
        # Info
        info = {
            'cost': cost,
            'balance_error': balance_error,
            'limit_violations': total_viols,
            'ramp_violations': 0
        }
        
        return reward, info
    
    def _enforce_balance(self, powers: np.ndarray, target_load: float) -> np.ndarray:
        """Post-process RL action to enforce exact power balance"""
        total_gen = np.sum(powers)
        error = target_load - total_gen
        
        # Distribute error proportionally to available headroom
        if error > 0:  # Need more power
            headroom = self.gen_params['p_max'] - powers
        else:  # Need less power
            headroom = powers - self.gen_params['p_min']
        
        # Avoid division by zero
        if np.sum(headroom) > 1e-6:
            correction = error * (headroom / np.sum(headroom))
            powers_corrected = powers + correction
            return np.clip(powers_corrected, self.gen_params['p_min'], self.gen_params['p_max'])
        
        return powers


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (initial_observation, info)
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Reset to start of episode
        self.current_step = 0
        self.current_gen_powers = None
        self.episode_costs = []
        self.episode_violations = []

        # Get initial observation
        observation = self._get_observation()
        info = {}

        return observation, info
    
    def _get_current_net_load(self) -> float:
        """Get net load for current timestep"""
        step_idx = self.current_step % len(self.data)
        return self.data.iloc[step_idx]['net_load']


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment
        
        Args:
            action: Normalized actions in [-1, 1] range from PPO
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Scale normalized actions [-1, 1] to [p_min, p_max]
        # action = -1 → p_min
        # action =  0 → (p_min + p_max) / 2
        # action = +1 → p_max
        action_scaled = self.p_min_true + (action + 1.0) / 2.0 * (self.p_max_true - self.p_min_true)
        
        # Clip to bounds for safety
        action_scaled = np.clip(action_scaled, self.p_min_true, self.p_max_true)
        
        # Get current net load (BEFORE incrementing step)
        current_step_idx = self.current_step % len(self.data)
        current_net_load = self.data.iloc[current_step_idx]['net_load']
        
        # Print SCALED actions
        if self.current_step % 50 == 0:
            print(f"Step {self.current_step}: net_load={current_net_load:.2f} MW, "
                f"action_scaled_sum={np.sum(action_scaled):.2f} MW, "
                f"balance_error={abs(np.sum(action_scaled) - current_net_load):.2f} MW")
        
        # Calculate reward using SCALED actions
        reward, info = self._calculate_reward(action_scaled, current_net_load)
        
        # Update state with SCALED actions
        self.current_gen_powers = action_scaled.copy()
        self.episode_costs.append(info['cost'])
        self.episode_violations.append(info['balance_error'])
        
        # Move to next timestep
        self.current_step += 1
        
        # Check episode termination
        terminated = (self.current_step % len(self.data)) == 0
        truncated = False
        
        # Get next observation
        observation = self._get_observation()
        
        # Add episode summary if done
        if terminated:
            info['episode'] = {
                'total_cost': np.sum(self.episode_costs),
                'avg_cost': np.mean(self.episode_costs) if self.episode_costs else 0,
                'total_violations': np.sum(self.episode_violations),
                'avg_violation': np.mean(self.episode_violations) if self.episode_violations else 0
            }
        
        return observation, reward, terminated, truncated, info




    def render(self):
        """Render the environment (optional)"""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Net Load: {self.data.iloc[self.current_step]['net_load']:.2f} MW")
            print(f"Generator Powers: {self.current_gen_powers}")

    def close(self):
        """Clean up resources"""
        pass


if __name__ == "__main__":
    # Test the environment
    print("Testing Economic Dispatch Environment...")

    env = EconomicDispatchEnv()
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Run a few random steps
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}: Reward = {reward:.2f}, Cost = {info['cost']:.2f}")

        if terminated:
            break

    print("\nEnvironment test completed successfully!")
