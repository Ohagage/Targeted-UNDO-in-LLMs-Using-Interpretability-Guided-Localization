import wandb
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib.cm as cm
import os
from dotenv import load_dotenv

load_dotenv()

# ================= CONSTANTS =================
wandb_key = os.getenv("WANDB_API_KEY_2")
if not wandb_key:
    raise ValueError("WANDB_API_KEY_2 not found. Please create a .env file with your API key.")
PROJECT_PATH = "rashkovits-tel-aviv-university/gemma-2-0.1B_relearn_eng_forget"

# Grouping runs to apply the gradient correctly
LOCALIZED_RUNS = [
    "PD_a1.0_p0.05_t1.0_b0.0_Trace_Minimalist_Relearn_LR_0.001",
    "PD_a1.0_p0.1_t1.0_b0.1_Trace_Hybrid_Surgical_Relearn_LR_0.001",
    "PD_a1.0_p0.3_t1.0_b0.2_Trace_Hybrid_Aggressive_Relearn_LR_0.001",
    "PD_a1.0_p0.5_t0.8_b0.3_Distributed_Stochastic_Erasure_Relearn_LR_0.001",
    "PD_a1.0_p0.9_t0.5_b0.5_Global_Parity_Baseline_Relearn_LR_0.001",
]
BASELINES = ["Oracle_Relearn_LR_0.001", "Unlearned_MaxEnt_Relearn_LR_0.001"]

FORGET_COLS = ["val/multiplication_equation_acc", "val/multiplication_word_problem_acc",
               "val/division_equation_acc", "val/division_word_problem_acc"]
# =============================================

wandb.login(key=wandb_key)
api = wandb.Api()
plt.figure(figsize=(11, 7))

# 1. Plot Baselines with distinct colors
baseline_styles = {"Oracle": {"color": "orange", "ls": "-"}, "Unlearned": {"color": "black", "ls": "--"}}

for b_name in BASELINES:
    runs = api.runs(PROJECT_PATH, filters={"config.wandb_run_name": b_name})
    if not runs: continue
    history = runs[0].history(keys=["train/step"] + FORGET_COLS, samples=2000)
    acc = history[FORGET_COLS].mean(axis=1)
    label = "Oracle (Gold Standard)" if "Oracle" in b_name else "Unlearned Baseline (MaxEnt)"
    style = baseline_styles["Oracle"] if "Oracle" in b_name else baseline_styles["Unlearned"]
    plt.plot(history["train/step"], acc, label=label, color=style["color"], linestyle=style["ls"], linewidth=2)

# 2. Plot Localized Masks with a Gradient (Heatmap effect)
# Using 'viridis' or 'PuBu' for a nice scientific gradient
colors = cm.PuBu(np.linspace(0.4, 1.0, len(LOCALIZED_RUNS)))

for i, target_name in enumerate(LOCALIZED_RUNS):
    runs = api.runs(PROJECT_PATH, filters={"config.wandb_run_name": target_name})
    if not runs: continue
    history = runs[0].history(keys=["train/step"] + FORGET_COLS, samples=2000)
    acc = history[FORGET_COLS].mean(axis=1)

    # Extraction of parameters from the run name using Regex
    p = re.search(r'p(\d+\.\d+)', target_name).group(1)
    t = re.search(r't(\d+\.\d+)', target_name).group(1)
    b = re.search(r'b(\d+\.\d+)', target_name).group(1)

    # Extract the descriptive label (e.g., Trace Minimalist)
    # This takes the part after the last underscore before "Relearn"
    label_match = re.search(r'b\d+\.\d+_(.*)_Relearn', target_name)
    strategy_label = label_match.group(1).replace("_", " ") if label_match else "Localized"

    # Updated Legend Label
    display_label = (f"Localized-UNDO ({strategy_label}):\n"
                     f"p={p}, selective noise={t}, background noise={b}")

    plt.plot(history["train/step"], acc,
             label=display_label,
             color=colors[i], linewidth=2.5)

# --- STYLING ---
plt.title("Relearning Attack Robustness (LR: 0.001)\n", fontsize=14)
plt.xlabel("Relearning Steps", fontsize=12)
plt.ylabel("Forget Set Accuracy", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.3)

# Inside legend to remove white gap
plt.legend(loc='upper left', fontsize=8, frameon=True, edgecolor='gray')
plt.ylim(-0.02, 1.02)
plt.xlim(0, 500)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()

# Save the figure in PDF format for academic use (Overleaf)
# bbox_inches='tight' ensures the legend isn't cut off
output_filename = "relearning_robustness_results.pdf"
plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
print(f"Plot saved successfully as {output_filename}")

plt.show()