import json
import numpy as np
import matplotlib.pyplot as plt


def generate_diagnostic_plot(log_path):
    # Load the data
    with open(log_path, 'r') as f:
        data = json.load(f)

    # Metric Groups
    forget_metrics = [
        'val/multiplication_equation_acc', 'val/division_equation_acc',
        'val/multiplication_word_problem_acc', 'val/division_word_problem_acc'
    ]
    retain_metrics = [
        'val/addition_equation_acc', 'val/subtraction_equation_acc',
        'val/addition_word_problem_acc', 'val/subtraction_word_problem_acc'
    ]

    def get_agg_scores(suite):
        if not suite or "targeted" not in suite:
            return (0, 0, 0), (0, 0, 0), (0, 0, 0)

        # Targeted scores
        f_target = np.mean([suite["targeted"].get(m, 0) for m in forget_metrics])
        r_target = np.mean([suite["targeted"].get(m, 0) for m in retain_metrics])
        l_target = suite["targeted"].get('val/eng_ce_loss', 0)

        # Random trials aggregation
        trials = suite.get("random_trials", [])
        if not trials:
            return (f_target, r_target, l_target), (0, 0, 0), (0, 0, 0)

        f_rand = [np.mean([t.get(m, 0) for m in forget_metrics]) for t in trials]
        r_rand = [np.mean([t.get(m, 0) for m in retain_metrics]) for t in trials]
        l_rand = [t.get('val/eng_ce_loss', 0) for t in trials]

        return (f_target, r_target, l_target), \
            (np.mean(f_rand), np.mean(r_rand), np.mean(l_rand)), \
            (np.std(f_rand), np.std(r_rand), np.std(l_rand))

    # Extract Baseline
    base_f = np.mean([data["Baseline"].get(m, 0) for m in forget_metrics])
    base_r = np.mean([data["Baseline"].get(m, 0) for m in retain_metrics])
    base_l = data["Baseline"].get('val/eng_ce_loss', 0)

    # Extract Suites
    snmf_scores = get_agg_scores(data.get("SNMF"))
    delta_scores = get_agg_scores(data.get("Delta"))

    # Mapping for plotting
    (snmf_f, snmf_r, snmf_l), (snmf_rf, snmf_rr, snmf_rl), (snmf_sf, snmf_sr, snmf_sl) = snmf_scores
    (delta_f, delta_r, delta_l), (delta_rf, delta_rr, delta_rl), (delta_sf, delta_sr, delta_sl) = delta_scores

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Accuracy (Targeted vs Random)
    labels = ['Forget Set (Target)', 'Retain Set (Control)']
    x = np.arange(len(labels))
    width = 0.2

    ax1.bar(x - 1.5 * width, [base_f, base_r], width, label='Pretrained Baseline', color='lightgray', hatch='..')

    # SNMF
    ax1.bar(x - 0.5 * width, [snmf_f, snmf_r], width, label='SNMF Targeted', color='orange', edgecolor='black')
    ax1.errorbar(x - 0.5 * width, [snmf_rf, snmf_rr], yerr=[snmf_sf, snmf_sr], fmt='o', color='brown',
                 label='SNMF Random Baseline', capsize=5)

    # Delta
    ax1.bar(x + 0.5 * width, [delta_f, delta_r], width, label='Delta Targeted', color='blue', edgecolor='black')
    ax1.errorbar(x + 0.5 * width, [delta_rf, delta_rr], yerr=[delta_sf, delta_sr], fmt='o', color='darkblue',
                 label='Delta Random Baseline', capsize=5)

    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Localization Precision: Targeted vs. Random Erasure', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight='bold')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Subplot 2: Language Utility (English CE Loss)
    methods = ['Baseline', 'SNMF Targeted', 'SNMF Random', 'Delta Targeted', 'Delta Random']
    losses = [base_l, snmf_l, snmf_rl, delta_l, delta_rl]
    errors = [0, 0, snmf_sl, 0, delta_sl]
    colors = ['lightgray', 'orange', 'wheat', 'blue', 'lightblue']

    bars = ax2.bar(methods, losses, yerr=errors, color=colors, capsize=8, edgecolor='black')
    ax2.set_ylabel('English CE Loss (Lower is Better)', fontweight='bold')
    ax2.set_title('Collateral Damage to General Utility', fontsize=12, fontweight='bold')
    ax2.set_ylim(min(losses) * 0.95, max(losses) * 1.05)

    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=10,
                 fontweight='bold')

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('localization_diagnostic_plot.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    generate_diagnostic_plot("comparison_logs.json")