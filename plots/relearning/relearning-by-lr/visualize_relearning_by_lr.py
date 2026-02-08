import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the data
snmf_df = pd.read_csv('relearn_snmf_noise_relearn_per_LR.csv')
global_noise_df = pd.read_csv('relearn_global_noise_per_LR.csv')
unlearned_df = pd.read_csv('relearn_unlearn_noise_per_LR.csv')

# Define colors for different alphas
alpha_colors = {
    '1.0': '#1f77b4',  # blue
    '0.9': '#ff7f0e',  # orange
    '0.6': '#2ca02c',  # green
    '0.3': '#d62728',  # red
    '0.1': '#9467bd',  # purple
}

def create_plot_for_lr(lr, filename):
    """Create a plot for a specific learning rate."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # ============ SNMF Data ============
    snmf_alphas = ['1.0', '0.9', '0.6', '0.3', '0.1']
    for alpha in snmf_alphas:
        col = f'PartialDistill_alpha_{alpha}_mask_snmf-mlp-supervised-label-0.1_Relearn_LR_{lr} - val/multiplication_equation_acc'
        if col in snmf_df.columns:
            ax.plot(snmf_df['Step'], snmf_df[col], 
                    label=f'SNMF α={alpha}',
                    color=alpha_colors[alpha], 
                    linestyle='-',
                    linewidth=2.5)
    
    # ============ Global Noise Data ============
    global_noise_alphas = ['0.9', '0.6', '0.3', '0.1']
    for alpha in global_noise_alphas:
        col = f'PartialDistill_alpha_{alpha}_mask_none_Relearn_LR_{lr} - val/multiplication_equation_acc'
        if col in global_noise_df.columns:
            ax.plot(global_noise_df['Step'], global_noise_df[col], 
                    label=f'GlobalNoise α={alpha}',
                    color=alpha_colors[alpha], 
                    linestyle='--',
                    linewidth=2.5)
    
    # ============ Unlearned Data ============
    unlearned_col = f'Unlearned_MaxEnt_Relearn_LR_{lr} - val/multiplication_equation_acc'
    if unlearned_col in unlearned_df.columns:
        ax.plot(unlearned_df['Step'], unlearned_df[unlearned_col], 
                label='Unlearned',
                color='black', 
                linestyle='-',
                linewidth=3)
    
    # Customize the plot
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Mean Accuracy', fontsize=14)
    ax.set_title(f'Relearning Comparison (LR = {lr})', fontsize=16)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 600)
    ax.grid(True, alpha=0.3)
    
    # Create legend
    ax.legend(loc='lower right', fontsize=10, ncol=2, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as '{filename}'")

# Create combined plot (all LRs)
fig, ax = plt.subplots(figsize=(16, 10))

lr_styles = {
    '0.0004': '-',
    '0.0007': '--',
}

snmf_configs = [
    ('1.0', '0.0004'), ('1.0', '0.0007'),
    ('0.9', '0.0004'), ('0.9', '0.0007'),
    ('0.6', '0.0004'), ('0.6', '0.0007'),
    ('0.3', '0.0004'), ('0.3', '0.0007'),
    ('0.1', '0.0004'), ('0.1', '0.0007'),
]

for alpha, lr in snmf_configs:
    col = f'PartialDistill_alpha_{alpha}_mask_snmf-mlp-supervised-label-0.1_Relearn_LR_{lr} - val/multiplication_equation_acc'
    if col in snmf_df.columns:
        ax.plot(snmf_df['Step'], snmf_df[col], 
                label=f'SNMF α={alpha} LR={lr}',
                color=alpha_colors[alpha], 
                linestyle=lr_styles[lr],
                linewidth=2, alpha=0.9)

global_noise_configs = [
    ('0.9', '0.0004'), ('0.9', '0.0007'),
    ('0.6', '0.0004'), ('0.6', '0.0007'),
    ('0.3', '0.0004'), ('0.3', '0.0007'),
    ('0.1', '0.0004'), ('0.1', '0.0007'),
]

for alpha, lr in global_noise_configs:
    col = f'PartialDistill_alpha_{alpha}_mask_none_Relearn_LR_{lr} - val/multiplication_equation_acc'
    if col in global_noise_df.columns:
        ax.plot(global_noise_df['Step'], global_noise_df[col], 
                label=f'GlobalNoise α={alpha} LR={lr}',
                color=alpha_colors[alpha], 
                linestyle=lr_styles[lr],
                linewidth=2, alpha=0.6,
                marker='o', markersize=3, markevery=10)

unlearned_configs = [
    ('0.0004', 'Unlearned_MaxEnt_Relearn_LR_0.0004 - val/multiplication_equation_acc'),
    ('0.0007', 'Unlearned_MaxEnt_Relearn_LR_0.0007 - val/multiplication_equation_acc'),
]

for lr, col in unlearned_configs:
    if col in unlearned_df.columns:
        ax.plot(unlearned_df['Step'], unlearned_df[col], 
                label=f'Unlearned LR={lr}',
                color='black', 
                linestyle=lr_styles[lr],
                linewidth=2.5)

ax.set_xlabel('Training Step', fontsize=14)
ax.set_ylabel('Mean Accuracy', fontsize=14)
ax.set_title('Relearning by Learning Rate: SNMF vs Global Noise vs Unlearned', fontsize=16)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 600)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=8, ncol=3, framealpha=0.9)

plt.tight_layout()
plt.savefig('relearning_by_lr.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved as 'relearning_by_lr.png'")

# Create separate plots for each learning rate
create_plot_for_lr('0.0004', 'relearning_lr_0.0004.png')
create_plot_for_lr('0.0007', 'relearning_lr_0.0007.png')

print("\nNote: Oracle data contains wiki_eval_time (not accuracy) and was not included.")
