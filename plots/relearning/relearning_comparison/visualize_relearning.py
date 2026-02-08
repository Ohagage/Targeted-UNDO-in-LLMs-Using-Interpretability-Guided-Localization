import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Load the data
snmf_df = pd.read_csv('snmf-releaning.csv')
oracle_df = pd.read_csv('oracal-relearning.csv')
unlearned_df = pd.read_csv('unlearned_model_relearning.csv')
global_noise_df = pd.read_csv('global_noise_relearn_group_by_alpha.csv')

# Extract SNMF mean accuracy columns for each alpha
snmf_alpha_columns = {
    'α = 1.0': 'alpha: 1.0 - val/multiplication_equation_acc',
    'α = 0.9': 'alpha: 0.9 - val/multiplication_equation_acc',
    'α = 0.6': 'alpha: 0.6 - val/multiplication_equation_acc',
    'α = 0.3': 'alpha: 0.3 - val/multiplication_equation_acc',
    'α = 0.1': 'alpha: 0.1 - val/multiplication_equation_acc',
}

# Extract Global Noise mean accuracy columns for each alpha
global_noise_alpha_columns = {
    'α = 0.9': 'alpha: 0.9 - val/multiplication_equation_acc',
    'α = 0.6': 'alpha: 0.6 - val/multiplication_equation_acc',
    'α = 0.3': 'alpha: 0.3 - val/multiplication_equation_acc',
    'α = 0.1': 'alpha: 0.1 - val/multiplication_equation_acc',
}

# Create the plot
plt.figure(figsize=(14, 8))

# Plot SNMF data for each alpha (solid lines)
snmf_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for (label, col), color in zip(snmf_alpha_columns.items(), snmf_colors):
    plt.plot(snmf_df['Step'], snmf_df[col], label=f'SNMF {label}', color=color, linewidth=2)

# Plot Global Noise data for each alpha (dashed lines with same colors)
global_noise_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Same colors for matching alphas
for (label, col), color in zip(global_noise_alpha_columns.items(), global_noise_colors):
    plt.plot(global_noise_df['Step'], global_noise_df[col], label=f'Global Noise {label}', 
             color=color, linewidth=2, linestyle='--', alpha=0.7)

# Plot Oracle data
plt.plot(oracle_df['Step'], oracle_df['Grouped runs - val/multiplication_equation_acc'], 
         label='Oracle', color='black', linewidth=2.5, linestyle='--')

# Plot Unlearned model data
plt.plot(unlearned_df['Step'], unlearned_df['Grouped runs - val/multiplication_equation_acc'], 
         label='Unlearned Model', color='gray', linewidth=2.5, linestyle=':')

# Customize the plot
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Mean Accuracy', fontsize=12)
plt.title('Relearning Comparison: SNMF vs Global Noise vs Oracle vs Unlearned', fontsize=14)
plt.legend(loc='lower right', fontsize=9, ncol=2)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.xlim(0, 600)

plt.tight_layout()
plt.savefig('relearning_comparison.png', dpi=150, bbox_inches='tight')

print("Plot saved as 'relearning_comparison.png'")
