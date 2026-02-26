import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# ================= CONSTANTS =================
wandb_key = os.getenv("WANDB_API_KEY_2")
if not wandb_key:
    raise ValueError("WANDB_API_KEY_2 not found. Please create a .env file with your API key.")
RELEARN_PROJECT = "rashkovits-tel-aviv-university/gemma-2-0.1B_relearn_eng_forget"
DISTILL_PROJECT = "rashkovits-tel-aviv-university/gemma-2-0.1B_MaxEnt_lr_7e-05_partial_distill"

ORACLE_RUN_NAME = "Oracle_Relearn_LR_0.001"
UNLEARN_ONLY_RUN_NAME = "Unlearned_MaxEnt_Relearn_LR_0.001"
S_DATA_FILTERING = 1000

RUN_GROUPS = [
    {"label": "Minimalist", "relearn_name": "PD_a1.0_p0.05_t1.0_b0.0_Trace_Minimalist_Relearn_LR_0.001"},
    {"label": "Hybrid Surgical", "relearn_name": "PD_a1.0_p0.1_t1.0_b0.1_Trace_Hybrid_Surgical_Relearn_LR_0.001"},
    {"label": "Hybrid Aggressive", "relearn_name": "PD_a1.0_p0.3_t1.0_b0.2_Trace_Hybrid_Aggressive_Relearn_LR_0.001"},
    {"label": "Stochastic Erasure",
     "relearn_name": "PD_a1.0_p0.5_t0.8_b0.3_Distributed_Stochastic_Erasure_Relearn_LR_0.001"},
    {"label": "Global Parity", "relearn_name": "PD_a1.0_p0.9_t0.5_b0.5_Global_Parity_Baseline_Relearn_LR_0.001"},
]

FORGET_COLS = ["val/multiplication_equation_acc", "val/multiplication_word_problem_acc",
               "val/division_equation_acc", "val/division_word_problem_acc"]
# =============================================

wandb.login(key=wandb_key)
api = wandb.Api()


def get_final_forget_acc(project, run_name):
    runs = api.runs(project, filters={"config.wandb_run_name": run_name})
    if not runs: return None
    hist = runs[0].history(keys=FORGET_COLS)
    return hist[FORGET_COLS].iloc[-1].mean()


p_data_filtering = get_final_forget_acc(RELEARN_PROJECT, ORACLE_RUN_NAME)
p_unlearn_only = get_final_forget_acc(RELEARN_PROJECT, UNLEARN_ONLY_RUN_NAME)

plot_data = []
for group in RUN_GROUPS:
    p_run = get_final_forget_acc(RELEARN_PROJECT, group["relearn_name"])
    if p_run is None: continue

    # Robustness calculation (Paper Normalized)
    robustness_score = (p_run - p_unlearn_only) / (p_data_filtering - p_unlearn_only)

    distill_runs = api.runs(DISTILL_PROJECT)
    s_run = 0
    for dr in distill_runs:
        search_key = group["label"].replace(" ", "_")
        if search_key in dr.name or group["label"] in dr.name:
            s_run = dr.summary.get("_step", 0)
            break

    compute_pct = (s_run / S_DATA_FILTERING) * 100
    if s_run > 0:
        plot_data.append({"Label": group["label"], "ComputePct": compute_pct, "Robustness": robustness_score})

df = pd.DataFrame(plot_data).sort_values("ComputePct")

# --- FINAL PAPER STYLE VISUALIZATION ---
fig, ax = plt.subplots(figsize=(10, 10))

# 1. Plot the trend line (dashed black)
ax.plot(df["ComputePct"], df["Robustness"], color='black', linestyle='--', linewidth=2.5, zorder=1)

# 2. Define a color map for the different strategies
# Using tab10 or a similar qualitative map to distinguish models
colors = plt.cm.tab10(np.linspace(0, 0.5, len(df)))

# 3. Plot each strategy as a separate scatter call for the legend
for (i, row), color in zip(df.iterrows(), colors):
    ax.scatter(row["ComputePct"], row["Robustness"], s=550, color=color,
               edgecolors='black', linewidths=3, label=row["Label"], zorder=2)

# 4. Plot Gold Standard Star
ax.scatter(100, 1.0, marker='*', s=1600, color='black', label="Data Filtering (Gold Standard)", zorder=3)

# --- STYLING ---
# Title with extra padding to prevent clipping
ax.set_title("Compute Trade Off: Arithmetic", fontsize=28, fontweight='bold', loc='center', pad=50)
ax.set_xlabel("Compute (% of Data Filtering)", fontsize=22, labelpad=15)
ax.set_ylabel("Robustness", fontsize=22, labelpad=15)

# Axis Ticks and Percentage formatting
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=18)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%', '120%'], fontsize=18)

# Right-side Y-axis labels
ax_right = ax.twinx()
ax_right.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
ax_right.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%', '120%'], fontsize=18)
ax_right.set_ylim(-0.1, 1.4)

# Grid and Limits
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.set_xlim(-10, 125)
ax.set_ylim(-0.1, 1.4)

# 5. The Legend
# Positioning it inside the plot where there is empty space
ax.legend(fontsize=16, frameon=True, edgecolor='black', loc='lower right', borderpad=1)

# Ensure the title and labels fit perfectly
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save as PDF for Overleaf
output_filename = "arithmetic_tradeoff_final.pdf"
plt.savefig(output_filename, format='pdf', bbox_inches='tight')
print(f"Final plot saved successfully as {output_filename}")
plt.show()