import wandb
import pandas as pd
import matplotlib.pyplot as plt
import json

# Authenticate
wandb.login(key="8b80f738391c946f3c8b26d878a282cbf763ff78")
api = wandb.Api()
project_path = "hagage-tel-aviv-university/gemma-2-0.1B_relearn_only_forget"

# 1. Define Run Groups
# Ensure these regex patterns match your wandb_run_name logic in run_relearn_arithmetic.py
groups = {
    "Localized-UNDO (alpha=0.6)": {
        "filters": {"config.wandb_run_name": {"$regex": "PartialDistill_alpha_0.6_mask_none.*"}},
        "color": "#1f77b4"
    },
    "Unlearn Only (MaxEnt)": {
        "filters": {"config.wandb_run_name": {"$regex": "Unlearned_MaxEnt_Relearn.*"}},
        "color": "#9467bd"
    },
    "Oracle (Gold Standard)": {
        "filters": {"config.wandb_run_name": {"$regex": "Oracle_Relearn.*"}},
        "color": "#ff7f0e"
    }
}

plt.figure(figsize=(10, 6))

for label, params in groups.items():
    runs = api.runs(project_path, filters=params["filters"])
    all_curves = []

    print(f"\nProcessing {label}: Found {len(runs)} runs.")

    for run in runs:
        # Get LR from config for identification
        config = run.config
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except:
                continue

        lr = config.get("learning_rate", "unknown")
        print(f" - Found run: {run.name} | LR: {lr}")

        history = run.history(keys=[
            "train/step",
            "val/multiplication_equation_acc", "val/multiplication_word_problem_acc",
            "val/division_equation_acc", "val/division_word_problem_acc"
        ], samples=1000)

        if history.empty:
            continue

        forget_cols = [
            "val/multiplication_equation_acc", "val/multiplication_word_problem_acc",
            "val/division_equation_acc", "val/division_word_problem_acc"
        ]

        if all(col in history.columns for col in forget_cols):
            # Calculate the average performance (P) for this specific LR run
            history["combined_forget_acc"] = history[forget_cols].mean(axis=1)
            history["run_id"] = run.id  # Unique ID to separate curves
            all_curves.append(history)

    if all_curves:
        df = pd.concat(all_curves)

        # Plot individual LR curves with partial opacity
        # Grouping by run_id ensures each LR attack is drawn as a separate thin line
        for run_id, run_df in df.groupby("run_id"):
            plt.plot(run_df["train/step"], run_df["combined_forget_acc"],
                     color=params["color"], alpha=0.45, linewidth=1)

        # Compute and plot the Upper Envelope (Worst-case)
        worst_case = df.groupby("train/step")["combined_forget_acc"].max().reset_index()
        plt.plot(worst_case["train/step"], worst_case["combined_forget_acc"],
                 color=params["color"], linewidth=3, label=label)

# Final Plot Styling

plt.xlabel("Training Steps on Forget Domain", fontsize=12)
plt.ylabel("Accuracy (Forget Set)", fontsize=12)
plt.title("Localized-UNDO vs. Baselines: Robustness to Relearning", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=False)
plt.ylim(-0.02, 1.02)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()