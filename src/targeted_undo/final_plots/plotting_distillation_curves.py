"""
Distillation Training Curves

For each mask configuration, generates a figure with two subplots:
  - Left:  Accuracy on the Retain set (addition + subtraction)
  - Right: Accuracy on the Forget set (multiplication + division)

Each subplot shows one line per alpha value, plotted against training steps.
Data is fetched from the partial distillation wandb project.
"""

import wandb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ========================= CONFIGURATION =========================
wandb_key = os.getenv("WANDB_API_KEY")
if not wandb_key:
    raise ValueError("WANDB_API_KEY not found. Please create a .env file with your API key.")
WANDB_ENTITY = "hagage-tel-aviv-university"
DISTILL_PROJECT = f"{WANDB_ENTITY}/gemma-2-0.1B_MaxEnt_lr_7e-05_partial_distill"

ALPHAS = [0.1, 0.3, 0.6, 0.9, 1.0]

MASK_CONFIGS = {
    "none":     "UNDO (global mask)",
    "binary":   "Localized-UNDO (Delta-Masking via Weight Discrepancy)",
    "relative": "Localized-UNDO (SNMF mask)",
}

FORGET_COLS = [
    "val/multiplication_equation_acc", "val/multiplication_word_problem_acc",
    "val/division_equation_acc", "val/division_word_problem_acc",
]
RETAIN_COLS = [
    "val/addition_equation_acc", "val/addition_word_problem_acc",
    "val/subtraction_equation_acc", "val/subtraction_word_problem_acc",
]
ALL_METRIC_COLS = FORGET_COLS + RETAIN_COLS

LINE_WIDTH = 2.5

ALPHA_COLORS = {
    0.1: "#636EFA",
    0.3: "#EF553B",
    0.6: "#00CC96",
    0.9: "#AB63FA",
    1.0: "#FFA15A",
}

OUTPUT_DIR = Path(__file__).parent / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# =================================================================


def fetch_distill_runs(api):
    """Fetch all finished distillation runs and organize by (mask_type, alpha)."""
    all_keys = ["train/step"] + ALL_METRIC_COLS

    runs_by_config = {}
    try:
        runs = list(api.runs(DISTILL_PROJECT, filters={"state": "finished"}))
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return runs_by_config

    print(f"Found {len(runs)} finished distillation runs")

    for run in runs:
        alpha = run.config.get("noise_alpha")
        mask_type = run.config.get("mask_type", "none")
        if alpha is None:
            continue
        alpha = round(alpha, 2)
        if alpha not in ALPHAS:
            continue

        history = run.history(keys=all_keys, samples=5000)
        if history.empty:
            print(f"  mask={mask_type}, alpha={alpha}: empty history – skipped")
            continue

        if all(c in history.columns for c in FORGET_COLS):
            history["combined_forget_acc"] = history[FORGET_COLS].mean(axis=1)
        if all(c in history.columns for c in RETAIN_COLS):
            history["combined_retain_acc"] = history[RETAIN_COLS].mean(axis=1)

        key = (mask_type, alpha)
        if key not in runs_by_config:
            runs_by_config[key] = []
        runs_by_config[key].append(history)

        n_val = history["train/step"].dropna().shape[0]
        print(f"  mask={mask_type}, alpha={alpha}: {n_val} validation points")

    # Summary
    print("\n--- Data availability ---")
    for mask_type in MASK_CONFIGS:
        for alpha in ALPHAS:
            key = (mask_type, alpha)
            n = len(runs_by_config.get(key, []))
            status = "OK" if n > 0 else "MISSING"
            print(f"  [{status}] mask={mask_type}, alpha={alpha}: {n} run(s)")

    return runs_by_config


def create_config_figure(mask_type, display_name, runs_by_config, ts):
    """Create a figure with two subplots (retain + forget) for one mask config."""
    fig, (ax_retain, ax_forget) = plt.subplots(1, 2, figsize=(16, 6))

    for alpha in ALPHAS:
        key = (mask_type, alpha)
        histories = runs_by_config.get(key, [])
        if not histories:
            print(f"  MISSING: alpha={alpha}")
            continue

        df = pd.concat(histories)
        color = ALPHA_COLORS.get(alpha, "#888888")

        # Retain subplot (left)
        if "combined_retain_acc" in df.columns:
            stats = (
                df.dropna(subset=["combined_retain_acc"])
                .groupby("train/step")["combined_retain_acc"]
                .mean()
                .reset_index()
                .sort_values("train/step")
            )
            ax_retain.plot(
                stats["train/step"], stats["combined_retain_acc"],
                color=color, linewidth=LINE_WIDTH, label=f"α = {alpha}",
            )

        # Forget subplot (right)
        if "combined_forget_acc" in df.columns:
            stats = (
                df.dropna(subset=["combined_forget_acc"])
                .groupby("train/step")["combined_forget_acc"]
                .mean()
                .reset_index()
                .sort_values("train/step")
            )
            ax_forget.plot(
                stats["train/step"], stats["combined_forget_acc"],
                color=color, linewidth=LINE_WIDTH, label=f"α = {alpha}",
            )

    for ax, ylabel, subtitle in [
        (ax_retain, "Accuracy (Retain)", "Accuracy on Retain Set"),
        (ax_forget, "Accuracy (Forget)", "Accuracy on Forget Set"),
    ]:
        ax.set_xlabel("Training Steps", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(subtitle, fontsize=14)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title="Alpha", fontsize=10, title_fontsize=11,
                  frameon=True, fancybox=True, edgecolor="0.8")

    fig.suptitle(f"Distillation Training Curves: {display_name}",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    safe_mask = mask_type.replace("/", "_")
    png_path = OUTPUT_DIR / f"distill_curves_{safe_mask}_{ts}.png"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  [SUCCESS] Saved → {png_path}")
    plt.close()


def main():
    wandb.login(key=wandb_key)
    api = wandb.Api()

    print("Fetching distillation runs...")
    runs_by_config = fetch_distill_runs(api)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nGenerating figures...")
    for mask_type, display_name in MASK_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"{display_name}")
        print(f"{'='*60}")
        create_config_figure(mask_type, display_name, runs_by_config, ts)

    print("\nDone!")


if __name__ == "__main__":
    main()
