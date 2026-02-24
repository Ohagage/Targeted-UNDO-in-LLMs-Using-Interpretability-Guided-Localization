"""
Distillation Training Curves

Generates a single figure with two subplots:
  - Left:  Accuracy on the Retain set (addition + subtraction)
  - Right: Accuracy on the Forget set (multiplication + division)

All mask configurations are shown together.  Each mask type gets a
distinct colour (matching the relearn plot) and linestyle, while the
distillation alpha controls line opacity.
"""

import wandb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import datetime
from pathlib import Path

# ========================= CONFIGURATION =========================
WANDB_KEY = "8b80f738391c946f3c8b26d878a282cbf763ff78"
WANDB_ENTITY = "hagage-tel-aviv-university"
DISTILL_PROJECT = f"{WANDB_ENTITY}/gemma-2-0.1B_MaxEnt_lr_7e-05_partial_distill"

ALPHAS = [0.1, 0.3, 0.6, 0.9, 1.0]

MASK_CONFIGS = {
    "none": {
        "label": "UNDO (Global mask)",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "binary": {
        "label": "Localized-UNDO (Delta mask)",
        "color": "#2ca02c",
        "linestyle": "--",
    },
    "relative": {
        "label": "Localized-UNDO (SNMF mask)",
        "color": "#d62728",
        "linestyle": "-.",
    },
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

LINE_WIDTH = 1.5

OUTPUT_DIR = Path(__file__).parent / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# =================================================================


def alpha_to_opacity(alpha_val):
    a_min, a_max = min(ALPHAS), max(ALPHAS)
    if a_max == a_min:
        return 1.0
    t = (alpha_val - a_min) / (a_max - a_min)
    return 0.1 + 0.9 * t


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


def create_combined_figure(runs_by_config, ts):
    """Create a single figure with retain (left) and forget (right) subplots."""
    fig, (ax_retain, ax_forget) = plt.subplots(1, 2, figsize=(18, 7))

    seen_masks = set()

    for mask_type, cfg in MASK_CONFIGS.items():
        for alpha in ALPHAS:
            key = (mask_type, alpha)
            histories = runs_by_config.get(key, [])
            if not histories:
                continue

            df = pd.concat(histories)
            opacity = alpha_to_opacity(alpha)

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
                    color=cfg["color"], linestyle=cfg["linestyle"],
                    linewidth=LINE_WIDTH, alpha=opacity,
                )

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
                    color=cfg["color"], linestyle=cfg["linestyle"],
                    linewidth=LINE_WIDTH, alpha=opacity,
                )

            seen_masks.add(mask_type)

    for ax, ylabel, subtitle in [
        (ax_retain, "Accuracy on Retain Domain", "Accuracy on Retain Set"),
        (ax_forget, "Accuracy on Forget Domain", "Accuracy on Forget Set"),
    ]:
        ax.set_xlabel("Training Steps", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(subtitle, fontsize=14)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = [], []
    for mask_type in MASK_CONFIGS:
        if mask_type not in seen_masks:
            continue
        cfg = MASK_CONFIGS[mask_type]
        proxy = plt.Line2D([], [], color=cfg["color"], linestyle=cfg["linestyle"],
                           linewidth=LINE_WIDTH, alpha=1.0)
        handles.append(proxy)
        labels.append(cfg["label"])

    fig.suptitle("Distillation Training Curves", fontsize=16)
    fig.subplots_adjust(bottom=0.18, right=0.88, top=0.90)

    fig.legend(handles, labels, frameon=False, fontsize=12,
               loc="lower center", ncol=len(handles))

    a_min, a_max = min(ALPHAS), max(ALPHAS)
    norm = mcolors.Normalize(vmin=a_min, vmax=a_max)
    sm = cm.ScalarMappable(cmap=cm.Blues, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.91, 0.18, 0.02, 0.72])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("α\n(line color opacity)", fontsize=14)
    cbar.ax.tick_params(labelsize=10)

    png_path = OUTPUT_DIR / f"distill_curves_{ts}.png"
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    print(f"  [SUCCESS] Saved → {png_path}")
    plt.close()


def main():
    wandb.login(key=WANDB_KEY)
    api = wandb.Api()

    print("Fetching distillation runs...")
    runs_by_config = fetch_distill_runs(api)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nGenerating figure...")
    create_combined_figure(runs_by_config, ts)

    print("\nDone!")


if __name__ == "__main__":
    main()
