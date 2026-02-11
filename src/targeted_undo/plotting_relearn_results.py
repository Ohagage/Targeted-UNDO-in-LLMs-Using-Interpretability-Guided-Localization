import wandb
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import json

# ================= CONSTANTS =================
WANDB_KEY = "8b80f738391c946f3c8b26d878a282cbf763ff78"
PROJECT_PATH = "hagage-tel-aviv-university/gemma-2-0.1B_relearn_only_forget"

# List of mixing coefficients to iterate through
ALPHAS_TO_PROCESS = [0.1, 0.3, 0.6, 0.9]

# Transparency and style settings
INDIVIDUAL_OPACITY = 0.10  # Faint lines for LR attacks
MEAN_LINE_STYLE = "--"  # Dashed for the mean trend
WORST_CASE_STYLE = "-"  # Solid for the bold worst-case adversary
# =============================================

# Authenticate
wandb.login(key=WANDB_KEY)
api = wandb.Api()

for ALPHA in ALPHAS_TO_PROCESS:
    print(f"\n" + "=" * 50)
    print(f"STARTING PROCESS FOR ALPHA: {ALPHA}")
    print("=" * 50)

    # Define Run Groups dynamically for the current ALPHA
    groups = {
        f"UNDO (alpha={ALPHA}, No Mask)": {
            "filters": {"config.wandb_run_name": {"$regex": f"PartialDistill_alpha_{ALPHA}_mask_none.*"},
                        "state": "finished"},
            "color": "#1f77b4"
        },
        f"Localized-UNDO (alpha={ALPHA}, Binary Mask)": {
            "filters": {"config.wandb_run_name": {"$regex": f"PartialDistill_alpha_{ALPHA}_mask_binary.*"},
                        "state": "finished"},
            "color": "#2ca02c"
        },
        f"Localized-UNDO (alpha={ALPHA}, SNMF Mask)": {
            "filters": {"config.wandb_run_name": {"$regex": f"PartialDistill_alpha_{ALPHA}_mask_snmf.*"},
                        "state": "finished"},
            "color": "#d62728"
        },
        "Unlearn Only (MaxEnt)": {
            "filters": {"config.wandb_run_name": {"$regex": "Unlearned_MaxEnt_Relearn.*"}, "state": "finished"},
            "color": "#9467bd"
        },
        "Oracle (Gold Standard)": {
            "filters": {"config.wandb_run_name": {"$regex": "Oracle_Relearn.*"}, "state": "finished"},
            "color": "#ff7f0e"
        }
    }

    plt.figure(figsize=(12, 7))

    for label, params in groups.items():
        runs = api.runs(PROJECT_PATH, filters=params["filters"])
        all_curves = []

        print(f"Processing {label}: Found {len(runs)} finished runs.")

        for run in runs:
            # Fetch history for arithmetic Forget Set metrics
            history = run.history(keys=[
                "train/step",
                "val/multiplication_equation_acc", "val/multiplication_word_problem_acc",
                "val/division_equation_acc", "val/division_word_problem_acc"
            ], samples=1000)

            if history.empty: continue

            forget_cols = [
                "val/multiplication_equation_acc", "val/multiplication_word_problem_acc",
                "val/division_equation_acc", "val/division_word_problem_acc"
            ]

            if all(col in history.columns for col in forget_cols):
                # Calculate combined performance P
                history["combined_forget_acc"] = history[forget_cols].mean(axis=1)
                history["run_id"] = run.id
                all_curves.append(history)

        if all_curves:
            df = pd.concat(all_curves)

            # 1. Plot individual LR trajectories
            for run_id, run_df in df.groupby("run_id"):
                plt.plot(run_df["train/step"], run_df["combined_forget_acc"],
                         color=params["color"], alpha=INDIVIDUAL_OPACITY, linewidth=0.8)

            # 2. Compute and plot Mean (Dashed) and Worst-Case (Solid Bold)
            stats = df.groupby("train/step")["combined_forget_acc"].agg(['mean', 'max']).reset_index()

            plt.plot(stats["train/step"], stats["mean"],
                     color=params["color"], linestyle=MEAN_LINE_STYLE, linewidth=1.5,
                     label=f"{label} (Mean)")

            plt.plot(stats["train/step"], stats["max"],
                     color=params["color"], linestyle=WORST_CASE_STYLE, linewidth=3,
                     label=f"{label} (Worst-Case)")

    # Plot Finalization
    plt.xlabel("Training Steps on Forget Domain", fontsize=12)
    plt.ylabel("Accuracy (Forget Set Average)", fontsize=12)
    plt.title(f"Comparison of Masking Strategies vs. Baselines (alpha={ALPHA})", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylim(-0.02, 1.02)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()

    # ================= SAVE RESULTS =================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"relearning_robustness_alpha_{ALPHA}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved plot to: {filename}")
    # ================================================

    # Close current figure to free memory before next iteration
    plt.close()

print("\nAll alpha values have been processed successfully.")