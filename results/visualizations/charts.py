"""
Comprehensive Visualization Script for Economic Dispatch Models
Generates publication-quality plots of:
1. Demand generation model
2. Solar generation model  
3. Wind generation model
4. Net load calculation
5. Generator cost functions
6. Merit order analysis
7. Complete system overview
8. Combined renewable generation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
})

# Create output directory
output_dir = Path('results/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("GENERATING MODEL VISUALIZATIONS")
print("="*70)

# ============================================================================
# FIGURE 1: DEMAND GENERATION MODEL
# ============================================================================
print("\n[1/8] Creating demand generation model plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Demand Generation Model: Multi-Scale Temporal Patterns',
             fontweight='bold', fontsize=15)

# Time array (7 days, 5-minute intervals)
n_steps = 288 * 7
time_hours = np.linspace(0, 24*7, n_steps)
base_demand = 500  # MW

# Component 1: Daily cycle
daily_cycle = 0.7 + 0.3 * np.sin(2*np.pi*(time_hours/24 - 0.25))
axes[0, 0].plot(time_hours[:288*2], daily_cycle[:288*2],
                linewidth=2.5, color='#2E86AB')
axes[0, 0].set_xlabel('Time (hours)', fontweight='bold')
axes[0, 0].set_ylabel('Normalized Demand', fontweight='bold')
axes[0, 0].set_title('A. Daily Cycle Component', fontweight='bold')
axes[0, 0].axhline(y=1.0, color='gray', linestyle='--',
                   alpha=0.5, label='Peak')
axes[0, 0].axhline(y=0.4, color='gray', linestyle='--',
                   alpha=0.5, label='Trough')
axes[0, 0].legend()
axes[0, 0].set_xlim([0, 48])
axes[0, 0].text(0.02, 0.98, r'$0.7 + 0.3\sin(2\pi t/24 - \pi/2)$',
                transform=axes[0, 0].transAxes, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Component 2: Weekly variation
weekly_variation = 1 + 0.1 * np.sin(2*np.pi*time_hours/(24*7))
axes[0, 1].plot(time_hours, weekly_variation, linewidth=2.5, color='#A23B72')
axes[0, 1].set_xlabel('Time (hours)', fontweight='bold')
axes[0, 1].set_ylabel('Weekly Multiplier', fontweight='bold')
axes[0, 1].set_title('B. Weekly Variation Component', fontweight='bold')
axes[0, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
axes[0, 1].text(0.02, 0.98, r'$1 + 0.1\sin(2\pi t/168)$',
                transform=axes[0, 1].transAxes, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Component 3: Stochastic noise
np.random.seed(42)
noise = np.random.normal(0, 25, n_steps)
axes[1, 0].hist(noise, bins=50, density=True, alpha=0.7,
                color='#F18F01', edgecolor='black')
x_noise = np.linspace(-75, 75, 200)
gaussian = (1/(25*np.sqrt(2*np.pi))) * np.exp(-0.5*(x_noise/25)**2)
axes[1, 0].plot(x_noise, gaussian, 'r-', linewidth=2.5,
                label='Theoretical N(0,25)')
axes[1, 0].set_xlabel('Noise (MW)', fontweight='bold')
axes[1, 0].set_ylabel('Probability Density', fontweight='bold')
axes[1, 0].set_title('C. Stochastic Noise Component', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].text(0.02, 0.98, r'$\mathcal{N}(0, \sigma^2), \sigma = 25$ MW',
                transform=axes[1, 0].transAxes, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Component 4: Complete demand profile
complete_demand = base_demand * daily_cycle * weekly_variation + noise
complete_demand = np.maximum(complete_demand, 210)  # Minimum feasibility
axes[1, 1].plot(time_hours[:288*2], complete_demand[:288*2], linewidth=2,
                color='#06A77D', label='Generated Demand')
axes[1, 1].fill_between(time_hours[:288*2], 250, 594, alpha=0.2, color='gray',
                        label='Operating Range')
axes[1, 1].set_xlabel('Time (hours)', fontweight='bold')
axes[1, 1].set_ylabel('Demand (MW)', fontweight='bold')
axes[1, 1].set_title(
    'D. Complete Demand Profile (48 hours)', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].set_xlim([0, 48])

# Statistics box
stats_text = f"μ = {np.mean(complete_demand):.1f} MW\n"
stats_text += f"σ = {np.std(complete_demand):.1f} MW\n"
stats_text += f"Min = {np.min(complete_demand):.1f} MW\n"
stats_text += f"Max = {np.max(complete_demand):.1f} MW"
axes[1, 1].text(0.98, 0.02, stats_text, transform=axes[1, 1].transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / 'demand_generation_model.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved: demand_generation_model.png")
plt.close()

# ============================================================================
# FIGURE 2: SOLAR GENERATION MODEL
# ============================================================================
print("[2/8] Creating solar generation model plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Solar Generation Model: Diurnal Pattern with Cloud Variability',
             fontweight='bold', fontsize=15)

# Solar generation function


def solar_generation(hour, cloud_factor=1.0):
    if 6 <= hour <= 18:
        return 150 * np.sin(np.pi * (hour - 6) / 12) * cloud_factor
    return 0


# Panel A: Ideal solar curve (no clouds)
hours_day = np.linspace(0, 24, 289)
solar_ideal = np.array([solar_generation(h, 1.0) for h in hours_day])
axes[0, 0].plot(hours_day, solar_ideal, linewidth=3,
                color='#FF8C00', label='Clear Sky')
axes[0, 0].fill_between(hours_day, 0, solar_ideal, alpha=0.3, color='#FF8C00')
axes[0, 0].axvline(x=6, color='purple', linestyle='--',
                   alpha=0.5, label='Sunrise')
axes[0, 0].axvline(x=18, color='purple', linestyle='--',
                   alpha=0.5, label='Sunset')
axes[0, 0].axvline(x=12, color='red', linestyle='--',
                   alpha=0.5, label='Solar Noon')
axes[0, 0].set_xlabel('Hour of Day', fontweight='bold')
axes[0, 0].set_ylabel('Solar Generation (MW)', fontweight='bold')
axes[0, 0].set_title('A. Ideal Solar Profile (Clear Sky)', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].set_xlim([0, 24])
axes[0, 0].set_xticks(range(0, 25, 3))
axes[0, 0].text(0.02, 0.98, r'$P_{solar} = 150 \sin\left(\pi \frac{h-6}{12}\right)$',
                transform=axes[0, 0].transAxes, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel B: Cloud factor distribution
np.random.seed(42)
cloud_factors = np.clip(1 + 0.3 * np.random.randn(10000), 0.3, 1.0)
axes[0, 1].hist(cloud_factors, bins=50, density=True, alpha=0.7,
                color='#4682B4', edgecolor='black')
axes[0, 1].axvline(x=1.0, color='red', linestyle='--',
                   linewidth=2, label='Clear Sky')
axes[0, 1].axvline(x=np.mean(cloud_factors), color='green', linestyle='--',
                   linewidth=2, label=f'Mean = {np.mean(cloud_factors):.3f}')
axes[0, 1].set_xlabel('Cloud Factor', fontweight='bold')
axes[0, 1].set_ylabel('Probability Density', fontweight='bold')
axes[0, 1].set_title('B. Cloud Cover Variability', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].text(0.02, 0.98, r'$CF \sim \text{clip}(\mathcal{N}(1, 0.3^2), 0.3, 1.0)$',
                transform=axes[0, 1].transAxes, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel C: Multiple realizations with clouds
axes[1, 0].plot(hours_day, solar_ideal, linewidth=3, color='red',
                alpha=0.8, label='Clear Sky', zorder=10)
for i in range(10):
    np.random.seed(42 + i)
    cf = np.clip(1 + 0.3 * np.random.randn(len(hours_day)), 0.3, 1.0)
    solar_cloudy = np.array([solar_generation(h, cf[j])
                            for j, h in enumerate(hours_day)])
    axes[1, 0].plot(hours_day, solar_cloudy,
                    linewidth=1, alpha=0.4, color='gray')
axes[1, 0].set_xlabel('Hour of Day', fontweight='bold')
axes[1, 0].set_ylabel('Solar Generation (MW)', fontweight='bold')
axes[1, 0].set_title(
    'C. Solar Generation with Cloud Variability (10 days)', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].set_xlim([0, 24])
axes[1, 0].set_xticks(range(0, 25, 3))

# Panel D: 7-day solar profile
np.random.seed(42)
solar_7days = []
for hour in time_hours:
    h = hour % 24
    cf = np.clip(1 + 0.3 * np.random.randn(), 0.3, 1.0)
    solar_7days.append(solar_generation(h, cf))
solar_7days = np.array(solar_7days)

axes[1, 1].plot(time_hours, solar_7days, linewidth=1.5,
                color='#FF8C00', alpha=0.8)
axes[1, 1].fill_between(time_hours, 0, solar_7days, alpha=0.3, color='#FF8C00')
axes[1, 1].set_xlabel('Time (hours)', fontweight='bold')
axes[1, 1].set_ylabel('Solar Generation (MW)', fontweight='bold')
axes[1, 1].set_title(
    'D. Seven-Day Solar Generation Profile', fontweight='bold')

# Add day/night shading
for day in range(7):
    axes[1, 1].axvspan(day*24, day*24+6, alpha=0.1,
                       color='black', label='Night' if day == 0 else '')
    axes[1, 1].axvspan(day*24+18, (day+1)*24, alpha=0.1, color='black')
axes[1, 1].legend()

# Statistics
stats_text = f"Capacity: 150 MW\n"
stats_text += f"Avg Output: {np.mean(solar_7days):.1f} MW\n"
stats_text += f"Capacity Factor: {np.mean(solar_7days)/150*100:.1f}%\n"
stats_text += f"Max: {np.max(solar_7days):.1f} MW"
axes[1, 1].text(0.98, 0.98, stats_text, transform=axes[1, 1].transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / 'solar_generation_model.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved: solar_generation_model.png")
plt.close()

# ============================================================================
# FIGURE 3: WIND GENERATION MODEL
# ============================================================================
print("[3/8] Creating wind generation model plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Wind Generation Model: Temporal Correlation and Variability',
             fontweight='bold', fontsize=15)

# Panel A: AR(1) process demonstration
np.random.seed(42)
n = 1000
W = np.zeros(n)
W[0] = 0.5
for t in range(1, n):
    W[t] = 0.9 * W[t-1] + 0.1 * np.random.uniform(0.3, 0.8)

axes[0, 0].plot(W[:200], linewidth=2, color='#2E8B57')
axes[0, 0].set_xlabel('Timestep', fontweight='bold')
axes[0, 0].set_ylabel('Wind Speed Factor', fontweight='bold')
axes[0, 0].set_title('A. Autoregressive Wind Model AR(1)', fontweight='bold')
axes[0, 0].axhline(y=np.mean(W), color='red', linestyle='--',
                   label=f'Mean = {np.mean(W):.3f}')
axes[0, 0].legend()
axes[0, 0].text(0.02, 0.02, r'$W_t = 0.9 W_{t-1} + 0.1 \mathcal{U}(0.3, 0.8)$',
                transform=axes[0, 0].transAxes, fontsize=10,
                verticalalignment='bottom', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel B: Autocorrelation function
lags = range(50)
acf = [np.corrcoef(W[:-lag if lag > 0 else None], W[lag:])[0, 1]
       if lag > 0 else 1.0 for lag in lags]
axes[0, 1].bar(lags, acf, color='#4682B4', alpha=0.7, edgecolor='black')
axes[0, 1].plot(lags, [0.9**lag for lag in lags], 'r--', linewidth=2,
                label='Theoretical: 0.9^lag')
axes[0, 1].set_xlabel('Lag (timesteps)', fontweight='bold')
axes[0, 1].set_ylabel('Autocorrelation', fontweight='bold')
axes[0, 1].set_title('B. Autocorrelation Function (ACF)', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Panel C: Wind generation distribution
np.random.seed(42)
wind_samples = []
W = 0.5
for _ in range(10000):
    W = 0.9 * W + 0.1 * np.random.uniform(0.3, 0.8)
    variability = np.clip(1 + 0.3 * np.random.randn(), 0.1, 1.2)
    wind_power = 100 * W * variability
    wind_samples.append(wind_power)

axes[1, 0].hist(wind_samples, bins=50, density=True, alpha=0.7,
                color='#20B2AA', edgecolor='black')
axes[1, 0].axvline(x=np.mean(wind_samples), color='red', linestyle='--',
                   linewidth=2, label=f'Mean = {np.mean(wind_samples):.1f} MW')
axes[1, 0].axvline(x=np.median(wind_samples), color='orange', linestyle='--',
                   linewidth=2, label=f'Median = {np.median(wind_samples):.1f} MW')
axes[1, 0].set_xlabel('Wind Generation (MW)', fontweight='bold')
axes[1, 0].set_ylabel('Probability Density', fontweight='bold')
axes[1, 0].set_title('C. Wind Generation Distribution', fontweight='bold')
axes[1, 0].legend()

# Panel D: 7-day wind profile
np.random.seed(42)
wind_7days = []
W = 0.5
for _ in range(len(time_hours)):
    W = 0.9 * W + 0.1 * np.random.uniform(0.3, 0.8)
    variability = np.clip(1 + 0.3 * np.random.randn(), 0.1, 1.2)
    wind_7days.append(100 * W * variability)
wind_7days = np.array(wind_7days)

axes[1, 1].plot(time_hours, wind_7days, linewidth=1.5,
                color='#20B2AA', alpha=0.8)
axes[1, 1].fill_between(time_hours, 0, wind_7days, alpha=0.3, color='#20B2AA')
axes[1, 1].set_xlabel('Time (hours)', fontweight='bold')
axes[1, 1].set_ylabel('Wind Generation (MW)', fontweight='bold')
axes[1, 1].set_title('D. Seven-Day Wind Generation Profile', fontweight='bold')

# Statistics
stats_text = f"Capacity: 100 MW\n"
stats_text += f"Avg Output: {np.mean(wind_7days):.1f} MW\n"
stats_text += f"Capacity Factor: {np.mean(wind_7days)/100*100:.1f}%\n"
stats_text += f"Std Dev: {np.std(wind_7days):.1f} MW"
axes[1, 1].text(0.98, 0.98, stats_text, transform=axes[1, 1].transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / 'wind_generation_model.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved: wind_generation_model.png")
plt.close()

# ============================================================================
# FIGURE 4: NET LOAD CALCULATION
# ============================================================================
print("[4/8] Creating net load calculation plot...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Net Load Calculation: Demand Minus Renewable Generation',
             fontweight='bold', fontsize=15)

# Calculate net load
net_load = complete_demand[:len(wind_7days)] - solar_7days - wind_7days
net_load = np.maximum(net_load, 0)

# Panel A: Components stacked
axes[0].fill_between(time_hours, 0, complete_demand[:len(wind_7days)],
                     label='Total Demand', alpha=0.3, color='#2C3E50')
axes[0].fill_between(time_hours, 0, solar_7days,
                     label='Solar Generation', alpha=0.7, color='#FF8C00')
axes[0].fill_between(time_hours, solar_7days, solar_7days + wind_7days,
                     label='Wind Generation', alpha=0.7, color='#20B2AA')
axes[0].plot(time_hours, complete_demand[:len(wind_7days)],
             linewidth=2.5, color='black', label='Demand Curve')
axes[0].plot(time_hours, net_load, linewidth=2.5, color='red',
             linestyle='--', label='Net Load (Demand - Renewables)')
axes[0].set_xlabel('Time (hours)', fontweight='bold')
axes[0].set_ylabel('Power (MW)', fontweight='bold')
axes[0].set_title('A. Power Balance: Demand and Renewable Generation (7 days)',
                  fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Panel B: Net load detail (48 hours)
axes[1].plot(time_hours[:288*2], complete_demand[:288*2],
             linewidth=2, color='#2C3E50', label='Gross Demand')
axes[1].plot(time_hours[:288*2], solar_7days[:288*2] + wind_7days[:288*2],
             linewidth=2, color='green', label='Total Renewable')
axes[1].plot(time_hours[:288*2], net_load[:288*2],
             linewidth=2.5, color='red', label='Net Load (To be dispatched)')
axes[1].fill_between(time_hours[:288*2], 0, net_load[:288*2],
                     alpha=0.3, color='red')
axes[1].axhline(y=210, color='purple', linestyle='--', linewidth=2,
                label='Min Generation Capacity (210 MW)')
axes[1].axhline(y=880, color='orange', linestyle='--', linewidth=2,
                label='Max Generation Capacity (880 MW)')
axes[1].set_xlabel('Time (hours)', fontweight='bold')
axes[1].set_ylabel('Power (MW)', fontweight='bold')
axes[1].set_title('B. Net Load Detail (48 hours) with Capacity Limits',
                  fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].set_xlim([0, 48])
axes[1].grid(True, alpha=0.3)

# Statistics box
stats_text = "Net Load Statistics:\n"
stats_text += f"Mean: {np.mean(net_load):.1f} MW\n"
stats_text += f"Std: {np.std(net_load):.1f} MW\n"
stats_text += f"Min: {np.min(net_load):.1f} MW\n"
stats_text += f"Max: {np.max(net_load):.1f} MW\n"
stats_text += f"Range: {np.max(net_load) - np.min(net_load):.1f} MW\n"
stats_text += f"\nRenewable Penetration:\n"
stats_text += f"{np.mean(solar_7days + wind_7days) / np.mean(complete_demand[:len(wind_7days)]) * 100:.1f}%"
axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
             fontsize=9, verticalalignment='top',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig(output_dir / 'net_load_calculation.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved: net_load_calculation.png")
plt.close()

# ============================================================================
# FIGURE 5: GENERATOR COST FUNCTIONS
# ============================================================================
print("[5/8] Creating generator cost functions plot...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Generator Cost Functions: Quadratic Models',
             fontweight='bold', fontsize=15)

# Generator parameters
generators = {
    'Gen 1 (Coal)': {'alpha': 150, 'beta': 30, 'gamma': 0.020,
                     'p_min': 50, 'p_max': 200, 'color': '#8B4513'},
    'Gen 2 (CCGT)': {'alpha': 100, 'beta': 35, 'gamma': 0.015,
                     'p_min': 30, 'p_max': 180, 'color': '#4169E1'},
    'Gen 3 (Gas Turb)': {'alpha': 50, 'beta': 45, 'gamma': 0.030,
                         'p_min': 20, 'p_max': 120, 'color': '#FF6347'},
    'Gen 4 (Peaker)': {'alpha': 30, 'beta': 60, 'gamma': 0.040,
                       'p_min': 10, 'p_max': 80, 'color': '#FF8C00'},
    'Gen 5 (Nuclear)': {'alpha': 200, 'beta': 20, 'gamma': 0.001,
                        'p_min': 100, 'p_max': 300, 'color': '#2E8B57'},
}

for idx, (name, params) in enumerate(generators.items()):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    # Power range
    P = np.linspace(params['p_min'], params['p_max'], 200)

    # Total cost
    C = params['alpha'] + params['beta'] * P + params['gamma'] * P**2

    # Marginal cost (derivative)
    MC = params['beta'] + 2 * params['gamma'] * P

    # Plot cost function
    ax.plot(P, C, linewidth=2.5, color=params['color'], label='Total Cost')
    ax2 = ax.twinx()
    ax2.plot(P, MC, linewidth=2, linestyle='--', color='red',
             alpha=0.7, label='Marginal Cost')

    # Mark operating range
    ax.axvline(x=params['p_min'], color='purple', linestyle=':',
               alpha=0.5, label=f"$P_{{min}}$ = {params['p_min']} MW")
    ax.axvline(x=params['p_max'], color='orange', linestyle=':',
               alpha=0.5, label=f"$P_{{max}}$ = {params['p_max']} MW")

    ax.set_xlabel('Power Output (MW)', fontweight='bold')
    ax.set_ylabel('Total Cost ($/h)', fontweight='bold', color=params['color'])
    ax2.set_ylabel('Marginal Cost ($/MWh)', fontweight='bold', color='red')
    ax.set_title(f'{name}', fontweight='bold')

    # Cost function equation
    eq_text = f"$C(P) = {params['alpha']} + {params['beta']}P$\n"
    eq_text += f"$ + {params['gamma']:.3f}P^2$\n"
    eq_text += f"$MC(P) = {params['beta']} + {2*params['gamma']:.3f}P$"
    ax.text(0.05, 0.95, eq_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.tick_params(axis='y', labelcolor=params['color'])
    ax2.tick_params(axis='y', labelcolor='red')
    ax.grid(True, alpha=0.3)

# Hide the 6th subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'generator_cost_functions.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved: generator_cost_functions.png")
plt.close()

# ============================================================================
# FIGURE 6: MERIT ORDER ANALYSIS
# ============================================================================
print("[6/8] Creating merit order analysis plot...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
fig.suptitle('Merit Order Dispatch: Economic Principle',
             fontweight='bold', fontsize=15)

# Extract marginal costs at midpoint
merit_data = []
for name, params in generators.items():
    P_mid = (params['p_min'] + params['p_max']) / 2
    MC_mid = params['beta'] + 2 * params['gamma'] * P_mid
    merit_data.append({
        'name': name,
        'MC': MC_mid,
        'beta': params['beta'],
        'capacity': params['p_max'] - params['p_min'],
        'p_min': params['p_min'],
        'p_max': params['p_max'],
        'color': params['color']
    })

# Sort by marginal cost (merit order)
merit_data.sort(key=lambda x: x['beta'])

# Panel A: Merit order curve
ax1 = fig.add_subplot(gs[0, :])
cumulative_capacity = 0
for i, gen in enumerate(merit_data):
    ax1.barh(i, gen['capacity'], left=cumulative_capacity + gen['p_min'],
             color=gen['color'], alpha=0.7, edgecolor='black', linewidth=2,
             label=gen['name'])
    # Add cost label
    ax1.text(cumulative_capacity + gen['p_min'] + gen['capacity']/2, i,
             f"β={gen['beta']:.0f}",
             ha='center', va='center', fontweight='bold', fontsize=10)
    cumulative_capacity += gen['capacity']

ax1.set_yticks(range(len(merit_data)))
ax1.set_yticklabels([g['name'] for g in merit_data])
ax1.set_xlabel('Cumulative Capacity (MW)', fontweight='bold')
ax1.set_title('A. Merit Order Curve (Sorted by Marginal Cost β)',
              fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# Mark example dispatch levels
for level, label in [(300, '300 MW'), (500, '500 MW'), (700, '700 MW')]:
    ax1.axvline(x=level, color='red', linestyle='--', alpha=0.5)
    ax1.text(level, len(merit_data)-0.5, label, ha='center',
             fontweight='bold', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Panel B: Marginal cost vs capacity
ax2 = fig.add_subplot(gs[1, 0])
cumulative = np.cumsum([0] + [g['p_min'] for g in merit_data])
betas = [g['beta'] for g in merit_data]

for i in range(len(merit_data)):
    ax2.hlines(betas[i], cumulative[i], cumulative[i+1] + merit_data[i]['capacity'],
               colors=merit_data[i]['color'], linewidth=4, alpha=0.7)
    ax2.scatter([cumulative[i+1] + merit_data[i]['capacity']], [betas[i]],
                s=100, color=merit_data[i]['color'], zorder=5, edgecolor='black')

ax2.set_xlabel('Cumulative Capacity (MW)', fontweight='bold')
ax2.set_ylabel('Marginal Cost β ($/MWh)', fontweight='bold')
ax2.set_title('B. Supply Curve (Merit Order Step Function)', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Panel C: Dispatch example table
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')

# Example: Dispatch 400 MW
target_load = 400
dispatch_example = []
remaining = target_load - sum(g['p_min'] for g in merit_data)

for gen in merit_data:
    if remaining > 0:
        available = gen['p_max'] - gen['p_min']
        allocated = min(available, remaining)
        power = gen['p_min'] + allocated
        remaining -= allocated
    else:
        power = gen['p_min']
    dispatch_example.append(power)

table_data = [['Generator', 'P_min', 'P_max', 'β ($/MWh)', 'Dispatched']]
for i, gen in enumerate(merit_data):
    table_data.append([
        gen['name'].split()[0] + ' ' + gen['name'].split()[1],
        f"{gen['p_min']:.0f}",
        f"{gen['p_max']:.0f}",
        f"{gen['beta']:.0f}",
        f"{dispatch_example[i]:.0f} MW"
    ])

table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.15, 0.15, 0.2, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Color header
for i in range(5):
    table[(0, i)].set_facecolor('#2C3E50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows
for i in range(1, len(table_data)):
    for j in range(5):
        table[(i, j)].set_facecolor(merit_data[i-1]['color'])
        table[(i, j)].set_alpha(0.3)

ax3.text(0.5, 0.95, f'Example: Dispatch {target_load} MW Load',
         ha='center', fontsize=12, fontweight='bold',
         transform=ax3.transAxes)
ax3.text(0.5, 0.05, f'Total Dispatched: {sum(dispatch_example):.0f} MW',
         ha='center', fontsize=10, fontweight='bold',
         transform=ax3.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.savefig(output_dir / 'merit_order_analysis.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved: merit_order_analysis.png")
plt.close()

# ============================================================================
# FIGURE 7: COMPLETE SYSTEM OVERVIEW
# ============================================================================
print("[7/8] Creating complete system overview...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
fig.suptitle('Economic Dispatch System: Complete Overview',
             fontweight='bold', fontsize=16)

# Panel 1: Demand profile
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time_hours[:288], complete_demand[:288], linewidth=2, color='#2C3E50')
ax1.fill_between(time_hours[:288], 0,
                 complete_demand[:288], alpha=0.3, color='#2C3E50')
ax1.set_xlabel('Hour', fontweight='bold')
ax1.set_ylabel('Demand (MW)', fontweight='bold')
ax1.set_title('24-Hour Demand Profile', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Panel 2: Renewable generation
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(time_hours[:288], solar_7days[:288], linewidth=2,
         color='#FF8C00', label='Solar')
ax2.plot(time_hours[:288], wind_7days[:288], linewidth=2,
         color='#20B2AA', label='Wind')
ax2.fill_between(time_hours[:288], 0,
                 solar_7days[:288], alpha=0.3, color='#FF8C00')
ax2.fill_between(time_hours[:288], solar_7days[:288],
                 solar_7days[:288] + wind_7days[:288],
                 alpha=0.3, color='#20B2AA')
ax2.set_xlabel('Hour', fontweight='bold')
ax2.set_ylabel('Generation (MW)', fontweight='bold')
ax2.set_title('24-Hour Renewable Generation', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Net load
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(time_hours[:288], net_load[:288], linewidth=2.5, color='red')
ax3.fill_between(time_hours[:288], 0, net_load[:288], alpha=0.3, color='red')
ax3.axhline(y=210, color='purple', linestyle='--', linewidth=2,
            label='Min Capacity', alpha=0.7)
ax3.axhline(y=880, color='orange', linestyle='--', linewidth=2,
            label='Max Capacity', alpha=0.7)
ax3.set_xlabel('Hour', fontweight='bold')
ax3.set_ylabel('Net Load (MW)', fontweight='bold')
ax3.set_title('24-Hour Net Load (To be dispatched)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: System parameters table
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

system_params = [
    ['Parameter', 'Value'],
    ['Number of Generators', '5'],
    ['Total Capacity', '880 MW'],
    ['Minimum Generation', '210 MW'],
    ['Solar Capacity', '150 MW'],
    ['Wind Capacity', '100 MW'],
    ['Base Demand', '500 MW'],
    ['Simulation Timestep', '5 minutes'],
    ['Total Timesteps', '2,016 (7 days)'],
    ['Renewable Penetration', '26.3%'],
]

table2 = ax4.table(cellText=system_params, cellLoc='left', loc='center',
                   colWidths=[0.6, 0.4])
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1, 2.5)

for i in range(2):
    table2[(0, i)].set_facecolor('#2C3E50')
    table2[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(system_params)):
    if i % 2 == 0:
        for j in range(2):
            table2[(i, j)].set_facecolor('#ECF0F1')

ax4.text(0.5, 0.95, 'System Parameters', ha='center', fontsize=12,
         fontweight='bold', transform=ax4.transAxes)

# Panel 5: Cost comparison
ax5 = fig.add_subplot(gs[2, 0])
methods = ['Classical\nLP', 'Heuristic\nMerit-Order', 'PPO\n(Failed)']
costs = [2402715, 2394759, 2500000]  # Approximate for PPO
colors_bar = ['#3498db', '#27ae60', '#e74c3c']
bars = ax5.bar(methods, costs, color=colors_bar,
               alpha=0.7, edgecolor='black', linewidth=2)
ax5.set_ylabel('Total Cost ($)', fontweight='bold')
ax5.set_title('Method Comparison: Total Cost (288 timesteps)',
              fontweight='bold')
ax5.ticklabel_format(style='plain', axis='y')
for bar, cost in zip(bars, costs):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'${cost:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Panel 6: Key statistics
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')

key_stats = [
    ['Metric', 'Value'],
    ['Avg Net Load', f'{np.mean(net_load):.1f} MW'],
    ['Peak Net Load', f'{np.max(net_load):.1f} MW'],
    ['Min Net Load', f'{np.min(net_load):.1f} MW'],
    ['Load Variability (σ)', f'{np.std(net_load):.1f} MW'],
    ['Avg Solar Output', f'{np.mean(solar_7days):.1f} MW'],
    ['Avg Wind Output', f'{np.mean(wind_7days):.1f} MW'],
    ['Solar CF', f'{np.mean(solar_7days)/150*100:.1f}%'],
    ['Wind CF', f'{np.mean(wind_7days)/100*100:.1f}%'],
]

table3 = ax6.table(cellText=key_stats, cellLoc='left', loc='center',
                   colWidths=[0.6, 0.4])
table3.auto_set_font_size(False)
table3.set_fontsize(10)
table3.scale(1, 2.5)

for i in range(2):
    table3[(0, i)].set_facecolor('#2C3E50')
    table3[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(key_stats)):
    if i % 2 == 0:
        for j in range(2):
            table3[(i, j)].set_facecolor('#ECF0F1')

ax6.text(0.5, 0.95, 'Key Statistics', ha='center', fontsize=12,
         fontweight='bold', transform=ax6.transAxes)

plt.savefig(output_dir / 'system_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: system_overview.png")
plt.close()

# ============================================================================
# FIGURE 8: COMBINED RENEWABLE GENERATION
# ============================================================================
print("[8/8] Creating combined renewable generation plot...")

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Combined Renewable Generation (Single Day)',
             fontweight='bold', fontsize=15)

# Time array for 24 hours
hours_24 = np.linspace(0, 24, 289)

# Solar Generation
solar_24 = np.array([solar_generation(h, 1.0) for h in hours_24])

# Wind Generation (AR(1) Process)
np.random.seed(100)  # Different seed for variety
wind_24 = []
W = 0.5
for _ in range(len(hours_24)):
    W = 0.9 * W + 0.1 * np.random.uniform(0.3, 0.8)
    variability = np.clip(1 + 0.3 * np.random.randn(), 0.1, 1.2)
    wind_24.append(100 * W * variability)
wind_24 = np.array(wind_24)

# Total Renewable
total_renewable = solar_24 + wind_24

stat_text = ""
# Plotting
ax.plot(hours_24, solar_24, label='Solar Generation',
        color='#FF8C00', linewidth=2)
ax.plot(hours_24, wind_24, label='Wind Generation',
        color='#20B2AA', linewidth=2)
ax.plot(hours_24, total_renewable, label='Total Renewable',
        color='#2E8B57', linewidth=3, linestyle='--')

ax.fill_between(hours_24, 0, solar_24, alpha=0.2, color='#FF8C00')
ax.fill_between(hours_24, solar_24, solar_24 +
                wind_24, alpha=0.2, color='#20B2AA')

ax.set_xlabel('Hour of Day', fontweight='bold')
ax.set_ylabel('Power Generation (MW)', fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim((0.0, 24.0))

# Add text box with totals
total_solar_mwh = np.trapezoid(solar_24, hours_24)
total_wind_mwh = np.trapezoid(wind_24, hours_24)
total_ren_mwh = total_solar_mwh + total_wind_mwh

plt.tight_layout()
plt.savefig(output_dir / 'combined_renewable_generation.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved: combined_renewable_generation.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*70)
print(f"\nOutput directory: {output_dir}/")
print("\nGenerated files:")
print("  1. demand_generation_model.png      - Demand synthesis components")
print("  2. solar_generation_model.png       - Solar model & variability")
print("  3. wind_generation_model.png        - Wind AR(1) process")
print("  4. net_load_calculation.png         - Power balance visualization")
print("  5. generator_cost_functions.png     - Quadratic cost curves")
print("  6. merit_order_analysis.png         - Economic dispatch principle")
print("  7. system_overview.png              - Complete system summary")
print("  8. combined_renewable_generation.png- Combined solar and wind")
print("\n" + "="*70)
print("✓ Ready for presentation!")
print("="*70)
