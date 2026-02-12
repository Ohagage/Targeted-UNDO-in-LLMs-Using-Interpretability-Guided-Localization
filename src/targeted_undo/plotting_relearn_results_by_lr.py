import wandb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import json
import os
from pathlib import Path
from collections import defaultdict

# Get the script name (without .py) for output directory
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = Path(__file__).parent / SCRIPT_NAME

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= CONSTANTS =================
WANDB_KEY = "8b80f738391c946f3c8b26d878a282cbf763ff78"
PROJECT_PATH = "hagage-tel-aviv-university/gemma-2-0.1B_relearn_only_forget"

# List of mixing coefficients to iterate through
ALPHAS_TO_PROCESS = [0.1, 0.3, 0.6, 0.9]

# Line style settings
LINE_WIDTH = 2.5
# =============================================

# Authenticate
wandb.login(key=WANDB_KEY)
api = wandb.Api()

# Colors for different alpha values
alpha_colors = {
    0.1: '#1f77b4',  # blue
    0.3: '#ff7f0e',  # orange
    0.6: '#2ca02c',  # green
    0.9: '#d62728',  # red
    1.0: '#9467bd',  # purple
}

# Define group configurations (without alpha - we'll add it dynamically)
group_configs = {
    "UNDO (No Mask)": {
        "pattern": "PartialDistill_alpha_{alpha}_mask_none",
        "linestyle": "-",
    },
    "Localized-UNDO (Binary Mask)": {
        "pattern": "PartialDistill_alpha_{alpha}_mask_binary",
        "linestyle": "--",
    },
    "Localized-UNDO (SNMF Mask)": {
        "pattern": "PartialDistill_alpha_{alpha}_mask_snmf",
        "linestyle": "-.",
    },
}

# Baseline groups (no alpha dependency)
baseline_configs = {
    "Unlearn Only (MaxEnt)": {
        "pattern": "Unlearned_MaxEnt_Relearn",
        "color": "#7f7f7f",  # gray
        "linestyle": ":",
    },
    "Oracle (Gold Standard)": {
        "pattern": "Oracle_Relearn",
        "color": "#000000",  # black
        "linestyle": "--",
    },
}


def extract_lr_from_run(run):
    """Extract learning rate from run config or name."""
    # Try to get LR from config
    if 'learning_rate' in run.config:
        return run.config['learning_rate']
    if 'lr' in run.config:
        return run.config['lr']
    
    # Try to extract from run name
    run_name = run.config.get('wandb_run_name', run.name)
    if '_LR_' in run_name:
        lr_part = run_name.split('_LR_')[-1].split('_')[0]
        try:
            return float(lr_part)
        except:
            pass
    
    # Try relearn_lr from config
    if 'relearn_lr' in run.config:
        return run.config['relearn_lr']
    
    return None


def fetch_all_runs_with_lr():
    """Fetch all runs and organize them by learning rate."""
    runs_by_lr = defaultdict(lambda: defaultdict(list))
    
    # Fetch alpha-dependent groups
    for alpha in ALPHAS_TO_PROCESS:
        for group_name, config in group_configs.items():
            pattern = config["pattern"].format(alpha=alpha)
            filters = {
                "config.wandb_run_name": {"$regex": f"{pattern}.*"},
                "state": "finished"
            }
            
            runs = api.runs(PROJECT_PATH, filters=filters)
            print(f"Processing {group_name} (alpha={alpha}): Found {len(runs)} runs")
            
            for run in runs:
                lr = extract_lr_from_run(run)
                if lr is None:
                    print(f"  Warning: Could not extract LR from run {run.name}")
                    continue
                
                # Fetch history
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
                    history["combined_forget_acc"] = history[forget_cols].mean(axis=1)
                    runs_by_lr[lr][(group_name, alpha)].append(history)
    
    # Fetch baseline groups (not alpha-dependent)
    for group_name, config in baseline_configs.items():
        filters = {
            "config.wandb_run_name": {"$regex": f"{config['pattern']}.*"},
            "state": "finished"
        }
        
        runs = api.runs(PROJECT_PATH, filters=filters)
        print(f"Processing {group_name}: Found {len(runs)} runs")
        
        for run in runs:
            lr = extract_lr_from_run(run)
            if lr is None:
                print(f"  Warning: Could not extract LR from run {run.name}")
                continue
            
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
                history["combined_forget_acc"] = history[forget_cols].mean(axis=1)
                runs_by_lr[lr][(group_name, None)].append(history)
    
    return runs_by_lr


def create_plot_for_lr(lr, groups_data):
    """Create a single plot for a specific learning rate."""
    plt.figure(figsize=(14, 8))
    
    # Plot alpha-dependent groups
    for (group_name, alpha), histories in groups_data.items():
        if alpha is None:
            continue  # Skip baselines for now
        
        if not histories:
            continue
        
        df = pd.concat(histories)
        stats = df.groupby("train/step")["combined_forget_acc"].agg(['mean']).reset_index()
        
        config = group_configs[group_name]
        color = alpha_colors.get(alpha, '#333333')
        
        plt.plot(stats["train/step"], stats["mean"],
                 color=color, linestyle=config["linestyle"], linewidth=LINE_WIDTH,
                 label=f"{group_name} (α={alpha})")
    
    # Plot baselines
    for (group_name, alpha), histories in groups_data.items():
        if alpha is not None:
            continue  # Skip non-baselines
        
        if not histories:
            continue
        
        df = pd.concat(histories)
        stats = df.groupby("train/step")["combined_forget_acc"].agg(['mean']).reset_index()
        
        config = baseline_configs[group_name]
        
        plt.plot(stats["train/step"], stats["mean"],
                 color=config["color"], linestyle=config["linestyle"], linewidth=LINE_WIDTH,
                 label=group_name)
    
    # Plot finalization
    plt.xlabel("Training Steps on Forget Domain", fontsize=14)
    plt.ylabel("Accuracy (Forget Set Average)", fontsize=14)
    plt.title(f"Relearning Comparison (Learning Rate = {lr})", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.ylim(-0.02, 1.02)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    # Save
    lr_str = f"{lr:.0e}".replace("+", "").replace("-0", "-")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"relearning_LR_{lr_str}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved plot to: {filename}")
    plt.close()


def main():
    print(f"Output directory: {OUTPUT_DIR}")
    print("Fetching runs from W&B...")
    runs_by_lr = fetch_all_runs_with_lr()
    
    print(f"\nFound {len(runs_by_lr)} unique learning rates:")
    for lr in sorted(runs_by_lr.keys()):
        print(f"  LR = {lr}")
    
    print("\nGenerating individual plots for each learning rate...")
    for lr in sorted(runs_by_lr.keys()):
        print(f"\nCreating plot for LR = {lr}")
        create_plot_for_lr(lr, runs_by_lr[lr])
    
    print("\nAll plots have been generated successfully!")


if __name__ == "__main__":
    main()
