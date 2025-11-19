"""
Data Generation Module for Economic Dispatch Optimization
Generates synthetic demand and renewable generation profiles
"""

from typing import Optional
import numpy as np
import pandas as pd
import yaml


class DataGenerator:
    """Generate synthetic load and renewable generation profiles"""

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the data generator with configuration

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.n_timesteps = self.config['system']['simulation_timesteps']
        self.base_demand = self.config['system']['base_demand']
        self.solar_capacity = self.config['renewable']['solar_capacity']
        self.wind_capacity = self.config['renewable']['wind_capacity']
        self.variability = self.config['renewable']['variability_factor']

    def generate_demand_profile(self, 
                                days: int = 1, 
                                seed: int = None) -> np.ndarray:
        """Generate realistic demand profile with daily patterns

        Args:
            days: Number of days to simulate
            seed: Random seed for reproducibility

        Returns:
            Array of demand values (MW)
        """
        if seed is not None:
            np.random.seed(seed)

        n_steps = self.n_timesteps * days
        time = np.linspace(0, 24 * days, n_steps)

        # Base daily pattern (higher during day, lower at night)
        daily_pattern = (
            self.base_demand * 0.7 +  # Base load
            self.base_demand * 0.3 * np.sin(2 * np.pi * (time / 24 - 0.25))  # Daily cycle
        )

        # Add weekly variation
        weekly_pattern = 1.0 + 0.1 * np.sin(2 * np.pi * time / (24 * 7))

        # Add random noise
        noise = np.random.normal(0, self.base_demand * 0.15, n_steps)
        spike_indices = np.random.choice(n_steps, size=int(n_steps * 0.05), replace=False)

        demand = daily_pattern * weekly_pattern + noise
        demand[spike_indices] *= 1.4  # 40% spikes in 5% of timesteps

        # Ensure demand is positive and realistic
        MIN_GENERATION = 210.0  # 50+30+20+10+100 = sum of p_min
        demand = np.maximum(demand, MIN_GENERATION * 1.1)  # At least 10% above min

        return demand

    def generate_solar_profile(self, 
                               days: int = 1, 
                               seed: Optional[int] = None) -> np.ndarray:
        """Generate solar generation profile

        Args:
            days: Number of days to simulate
            seed: Random seed for reproducibility

        Returns:
            Array of solar generation values (MW)
        """
        if seed is not None:
            np.random.seed(seed + 1)

        n_steps = self.n_timesteps * days
        time = np.linspace(0, 24 * days, n_steps)

        # Solar only generates during daytime (6 AM to 6 PM)
        hour_of_day = time % 24

        # Bell curve for solar generation
        solar_curve = np.where(
            (hour_of_day >= 6) & (hour_of_day <= 18),
            np.sin(np.pi * (hour_of_day - 6) / 12),
            0
        )

        # Add cloud cover variability
        cloud_factor = 1.0 + self.variability * np.random.randn(n_steps)
        cloud_factor = np.clip(cloud_factor, 0.3, 1.0)

        solar = self.solar_capacity * solar_curve * cloud_factor

        return np.maximum(solar, 0)

    def generate_wind_profile(self, 
                             days: int = 1, 
                             seed: Optional[int] = None) -> np.ndarray:
        """Generate wind generation profile

        Args:
            days: Number of days to simulate
            seed: Random seed for reproducibility

        Returns:
            Array of wind generation values (MW)
        """
        if seed is not None:
            np.random.seed(seed + 2)

        n_steps = self.n_timesteps * days

        # Wind has more variability and can occur at any time
        base_wind = np.random.uniform(0.3, 0.8, n_steps)

        # Add temporal correlation (wind speeds don't change instantly)
        for i in range(1, n_steps):
            base_wind[i] = 0.9 * base_wind[i-1] + 0.1 * base_wind[i]

        # Add random gusts and lulls
        variability = 1.0 + self.variability * np.random.randn(n_steps)
        variability = np.clip(variability, 0.1, 1.2)

        wind = self.wind_capacity * base_wind * variability

        return np.maximum(wind, 0)

    def generate_complete_dataset(self, 
                                 days: int = 1, 
                                 seed: int = 42) -> pd.DataFrame:
        """Generate complete dataset with demand and renewables

        Args:
            days: Number of days to simulate
            seed: Random seed for reproducibility

        Returns:
            DataFrame with time, demand, solar, wind, and net_load
        """
        # Generate profiles
        demand = self.generate_demand_profile(days, seed)
        solar = self.generate_solar_profile(days, seed)
        wind = self.generate_wind_profile(days, seed)

        # Calculate net load (demand - renewables)
        net_load = demand - solar - wind
        net_load = np.maximum(net_load, 0)  # Ensure non-negative

        # Create time index
        timestep_hours = self.config['system']['timestep_duration'] / 60
        time_index = pd.date_range(
            start='2025-01-01', 
            periods=len(demand), 
            freq=f'{self.config["system"]["timestep_duration"]}min'
        )

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': time_index,
            'demand': demand,
            'solar_generation': solar,
            'wind_generation': wind,
            'total_renewable': solar + wind,
            'net_load': net_load
        })

        return df

    def save_dataset(self, 
                    df: pd.DataFrame, 
                    filename: str = 'simulation_data.csv'):
        """Save dataset to CSV file

        Args:
            df: DataFrame to save
            filename: Output filename
        """
        filepath = f"data/{filename}"
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        print(f"Dataset shape: {df.shape}")
        print(f"\nDataset statistics:\n{df.describe()}")

        return filepath


if __name__ == "__main__":
    # Test the data generator
    generator = DataGenerator()

    # Generate 7 days of data for training
    train_data = generator.generate_complete_dataset(days=7, seed=42)
    generator.save_dataset(train_data, 'train_data.csv')

    # Generate 1 day of test data
    test_data = generator.generate_complete_dataset(days=1, seed=100)
    generator.save_dataset(test_data, 'test_data.csv')

    print("\nData generation completed successfully!")
