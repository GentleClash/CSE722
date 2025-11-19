import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300

# Colors
COLOR_CLASSICAL = '#3498db'  # Blue
COLOR_HEURISTIC = '#27ae60'  # Green
COLOR_PPO = '#e74c3c'        # Red (for failed)

# Create output directory
output_dir = Path('results/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading results...")
classical = pd.read_csv('results/classical_results.csv')
heuristic = pd.read_csv('results/heuristic_results.csv')
ppo = pd.read_csv('results/ppo_results.csv')

# ==============================================================================
# FIGURE 1: THREE-WAY COMPARISON (2x2 GRID)
# ==============================================================================
print("Creating Figure 1: Main comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Economic Dispatch: Three-Method Comparison', fontsize=16, fontweight='bold')

# Panel A: Total Cost
ax = axes[0, 0]
methods = ['Classical\nLP', 'Heuristic\nMerit-Order', 'PPO\n(Failed)']
costs = [classical['total_cost'].sum(), heuristic['total_cost'].sum(), ppo['total_cost'].sum()]
colors = [COLOR_CLASSICAL, COLOR_HEURISTIC, COLOR_PPO]
bars = ax.bar(methods, costs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Total Cost ($)', fontweight='bold')
ax.set_title('A. Total Generation Cost', fontweight='bold')
ax.ticklabel_format(style='plain', axis='y')
for bar, cost in zip(bars, costs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${cost:,.0f}', ha='center', va='bottom', fontweight='bold')

# Panel B: Balance Error (log scale)
ax = axes[0, 1]
classical_err = (classical['total_generation'] - classical['net_load']).abs()
heuristic_err = heuristic['balance_error']
ppo_err = ppo['balance_error']

errors = [classical_err.mean(), heuristic_err.mean(), ppo_err.mean()]
bars = ax.bar(methods, errors, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Avg Balance Error (MW, log scale)', fontweight='bold')
ax.set_title('B. Power Balance Accuracy', fontweight='bold')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3, which='both')
for bar, err in zip(bars, errors):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height*2,
            f'{err:.2f}', ha='center', va='bottom', fontweight='bold')

# Panel C: Cost Over Time
ax = axes[1, 0]
timesteps = range(len(classical))
ax.plot(timesteps, classical['total_cost'].cumsum()/1e6, 
        label='Classical LP', color=COLOR_CLASSICAL, linewidth=2.5, alpha=0.8)
ax.plot(timesteps, heuristic['total_cost'].cumsum()/1e6,
        label='Heuristic', color=COLOR_HEURISTIC, linewidth=2.5, alpha=0.8)
ax.plot(timesteps, ppo['total_cost'].cumsum()/1e6,
        label='PPO (Failed)', color=COLOR_PPO, linewidth=2.5, alpha=0.8, linestyle='--')
ax.set_xlabel('Timestep', fontweight='bold')
ax.set_ylabel('Cumulative Cost (Million $)', fontweight='bold')
ax.set_title('C. Cumulative Cost Comparison', fontweight='bold')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)

# Panel D: Balance Error Over Time
ax = axes[1, 1]
ax.plot(timesteps, classical_err, label='Classical LP', 
        color=COLOR_CLASSICAL, linewidth=1.5, alpha=0.7)
ax.plot(timesteps, heuristic_err, label='Heuristic',
        color=COLOR_HEURISTIC, linewidth=1.5, alpha=0.7)
ax.plot(timesteps, ppo_err, label='PPO (Failed)',
        color=COLOR_PPO, linewidth=1.5, alpha=0.7)
ax.set_xlabel('Timestep', fontweight='bold')
ax.set_ylabel('Balance Error (MW)', fontweight='bold')
ax.set_title('D. Balance Error Over Time', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'three_way_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: three_way_comparison.png")
plt.close()

# ==============================================================================
# FIGURE 2: GENERATOR DISPATCH COMPARISON (3x1 GRID)
# ==============================================================================
print("Creating Figure 2: Dispatch schedules...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Generator Dispatch Schedule Comparison (First 48 timesteps)', 
             fontsize=16, fontweight='bold')

n_steps = min(48, len(classical))
timesteps_plot = range(n_steps)
gen_cols = [f'gen_{i}_power' for i in range(5)]

# Classical
dispatch_classical = classical[gen_cols].iloc[:n_steps].values.T
axes[0].stackplot(timesteps_plot, dispatch_classical, 
                  labels=[f'Gen {i}' for i in range(5)], alpha=0.8)
axes[0].set_ylabel('Power (MW)', fontweight='bold')
axes[0].set_title('Classical LP Solver', fontweight='bold')
axes[0].legend(loc='upper left', ncol=5)
axes[0].grid(alpha=0.3, axis='y')

# Heuristic
dispatch_heuristic = heuristic[gen_cols].iloc[:n_steps].values.T
axes[1].stackplot(timesteps_plot, dispatch_heuristic,
                  labels=[f'Gen {i}' for i in range(5)], alpha=0.8)
axes[1].set_ylabel('Power (MW)', fontweight='bold')
axes[1].set_title('Heuristic Merit-Order Dispatcher', fontweight='bold')
axes[1].legend(loc='upper left', ncol=5)
axes[1].grid(alpha=0.3, axis='y')

# PPO (Failed)
dispatch_ppo = ppo[gen_cols].iloc[:n_steps].values.T
axes[2].stackplot(timesteps_plot, dispatch_ppo,
                  labels=[f'Gen {i}' for i in range(5)], alpha=0.8)
axes[2].set_xlabel('Timestep (5-minute intervals)', fontweight='bold')
axes[2].set_ylabel('Power (MW)', fontweight='bold')
axes[2].set_title('PPO Agent (Failed - Did Not Converge)', fontweight='bold', color=COLOR_PPO)
axes[2].legend(loc='upper left', ncol=5)
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'dispatch_schedules_three_way.png', dpi=300, bbox_inches='tight')
print("✓ Saved: dispatch_schedules_three_way.png")
plt.close()

# ==============================================================================
# FIGURE 3: STATISTICAL SUMMARY TABLE
# ==============================================================================
print("Creating Figure 3: Summary table...")

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('tight')
ax.axis('off')

table_data = [
    ['Metric', 'Classical LP', 'Heuristic Merit-Order', 'PPO (Failed)', 'Winner'],
    ['Total Cost',
     f"${classical['total_cost'].sum():,.0f}",
     f"${heuristic['total_cost'].sum():,.0f}",
     f"${ppo['total_cost'].sum():,.0f}",
     'Heuristic'],
    ['Avg Cost/Step',
     f"${classical['total_cost'].mean():,.2f}",
     f"${heuristic['total_cost'].mean():,.2f}",
     f"${ppo['total_cost'].mean():,.2f}",
     'Heuristic'],
    ['Avg Balance Error',
     f"{classical_err.mean():.4f} MW",
     f"{heuristic_err.mean():.6f} MW",
     f"{ppo_err.mean():.2f} MW",
     'Heuristic'],
    ['Max Balance Error',
     f"{classical_err.max():.4f} MW",
     f"{heuristic_err.max():.6f} MW",
     f"{ppo_err.max():.2f} MW",
     'Heuristic'],
    ['Convergence',
     '✓ Optimal',
     '✓ Guaranteed',
     '✗ Failed',
     'Classical/Heuristic'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.18, 0.18, 0.18, 0.21])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

plt.title('Three-Method Performance Comparison', fontsize=16, fontweight='bold', pad=20)
plt.savefig(output_dir / 'comparison_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_table.png")
plt.close()

print("\n" + "="*70)
print("ALL VISUALIZATIONS GENERATED!")
print("="*70)
print(f"Location: {output_dir}/")
print("Files:")
print("  - three_way_comparison.png (main figure)")
print("  - dispatch_schedules_three_way.png")
print("  - comparison_table.png")
print("="*70)
