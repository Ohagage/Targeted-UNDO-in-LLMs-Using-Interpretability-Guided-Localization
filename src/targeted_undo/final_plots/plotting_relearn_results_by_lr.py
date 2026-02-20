import wandb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import numpy as np
from pathlib import Path
from collections import defaultdict

# Output lives alongside this script
OUTPUT_DIR = Path(__file__).parent / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= CONSTANTS =================
WANDB_KEY = "8b80f738391c946f3c8b26d878a282cbf763ff78"
PROJECT_PATH = "hagage-tel-aviv-university/gemma-2-0.1B_relearn_only_forget"

ALPHAS_TO_PROCESS = [0.1, 0.3, 0.6, 0.9, 1.0]

# Only generate the plot for this LR (set to None to generate for all LRs)
TARGET_LR = 4e-6

LINE_WIDTH = 2.5

FORGET_COLS = [
    "val/multiplication_equation_acc", "val/multiplication_word_problem_acc",
    "val/division_equation_acc", "val/division_word_problem_acc",
]
RETAIN_COLS = [
    "val/addition_equation_acc", "val/addition_word_problem_acc",
    "val/subtraction_equation_acc", "val/subtraction_word_problem_acc",
]
ALL_METRIC_COLS = FORGET_COLS + RETAIN_COLS
# =============================================

wandb.login(key=WANDB_KEY)
api = wandb.Api()

# One base colour per group config; alpha-value controls opacity
group_configs = {
    "UNDO (No Mask)": {
        "pattern": "PartialDistill_alpha_{alpha}_mask_none",
        "linestyle": "-",
        "color": "#1f77b4",   # blue
    },
    "Localized-UNDO (Binary Mask)": {
        "pattern": "PartialDistill_alpha_{alpha}_mask_binary",
        "linestyle": "--",
        "color": "#2ca02c",   # green
    },
    "Localized-UNDO (SNMF Mask)": {
        "pattern": "PartialDistill_alpha_{alpha}_mask_snmf",
        "linestyle": "-.",
        "color": "#d62728",   # red
    },
}

baseline_configs = {
    "Unlearn Only (MaxEnt)": {
        "pattern": "Unlearned_MaxEnt_Relearn",
        "color": "#7f7f7f",
        "linestyle": ":",
    },
    "Oracle (Gold Standard)": {
        "pattern": "Oracle_Relearn",
        "color": "#000000",
        "linestyle": "--",
    },
}


def alpha_to_opacity(alpha_val):
    """Map the distillation alpha to line opacity.

    Larger alpha → higher opacity (closer to 1.0).
    Smallest alpha → 0.5 (never below 50 %).
    """
    a_min, a_max = min(ALPHAS_TO_PROCESS), max(ALPHAS_TO_PROCESS)
    if a_max == a_min:
        return 1.0
    t = (alpha_val - a_min) / (a_max - a_min)
    return 0.1 + 0.9 * t


def extract_lr_from_run(run):
    """Extract learning rate from run config or name."""
    if 'learning_rate' in run.config:
        return run.config['learning_rate']
    if 'lr' in run.config:
        return run.config['lr']
    run_name = run.config.get('wandb_run_name', run.name)
    if '_LR_' in run_name:
        lr_part = run_name.split('_LR_')[-1].split('_')[0]
        try:
            return float(lr_part)
        except ValueError:
            pass
    if 'relearn_lr' in run.config:
        return run.config['relearn_lr']
    return None


def _lr_matches(lr):
    if TARGET_LR is None:
        return True
    return lr is not None and np.isclose(lr, TARGET_LR, rtol=1e-3)


# ------------------------------------------------------------------ #
# Data fetching
# ------------------------------------------------------------------ #
def fetch_all_runs_with_lr():
    """Fetch all runs and organise them by learning rate.

    Each history dataframe now includes both forget and retain combined cols.
    """
    runs_by_lr = defaultdict(lambda: defaultdict(list))

    all_keys = ["train/step"] + ALL_METRIC_COLS

    for alpha in ALPHAS_TO_PROCESS:
        for group_name, cfg in group_configs.items():
            pattern = cfg["pattern"].format(alpha=alpha)
            filters = {
                "config.wandb_run_name": {"$regex": f"{pattern}.*"},
                "state": "finished",
            }
            runs = list(api.runs(PROJECT_PATH, filters=filters))
            total_runs = len(runs)
            matched = 0
            skipped_lr = 0
            skipped_empty = 0

            for run in runs:
                lr = extract_lr_from_run(run)
                if lr is None or not _lr_matches(lr):
                    skipped_lr += 1
                    continue
                history = run.history(keys=all_keys, samples=1000)
                if history.empty:
                    skipped_empty += 1
                    continue
                if all(c in history.columns for c in FORGET_COLS):
                    history["combined_forget_acc"] = history[FORGET_COLS].mean(axis=1)
                if all(c in history.columns for c in RETAIN_COLS):
                    history["combined_retain_acc"] = history[RETAIN_COLS].mean(axis=1)
                runs_by_lr[lr][(group_name, alpha)].append(history)
                matched += 1

            status = "OK" if matched > 0 else "MISSING"
            print(f"[{status}] {group_name} (alpha={alpha}): "
                  f"{total_runs} total, {matched} matched LR, "
                  f"{skipped_lr} wrong LR, {skipped_empty} empty history")

    for group_name, cfg in baseline_configs.items():
        filters = {
            "config.wandb_run_name": {"$regex": f"{cfg['pattern']}.*"},
            "state": "finished",
        }
        runs = list(api.runs(PROJECT_PATH, filters=filters))
        total_runs = len(runs)
        matched = 0
        skipped_lr = 0
        skipped_empty = 0

        for run in runs:
            lr = extract_lr_from_run(run)
            if lr is None or not _lr_matches(lr):
                skipped_lr += 1
                continue
            history = run.history(keys=all_keys, samples=1000)
            if history.empty:
                skipped_empty += 1
                continue
            if all(c in history.columns for c in FORGET_COLS):
                history["combined_forget_acc"] = history[FORGET_COLS].mean(axis=1)
            if all(c in history.columns for c in RETAIN_COLS):
                history["combined_retain_acc"] = history[RETAIN_COLS].mean(axis=1)
            runs_by_lr[lr][(group_name, None)].append(history)
            matched += 1

        status = "OK" if matched > 0 else "MISSING"
        print(f"[{status}] {group_name}: "
              f"{total_runs} total, {matched} matched LR, "
              f"{skipped_lr} wrong LR, {skipped_empty} empty history")

    # Summary of missing combinations
    print("\n--- Data availability summary ---")
    for alpha in ALPHAS_TO_PROCESS:
        for group_name in group_configs:
            found = any(
                len(runs_by_lr[lr].get((group_name, alpha), [])) > 0
                for lr in runs_by_lr
            )
            if not found:
                print(f"  MISSING: {group_name} (alpha={alpha})")
    for group_name in baseline_configs:
        found = any(
            len(runs_by_lr[lr].get((group_name, None), [])) > 0
            for lr in runs_by_lr
        )
        if not found:
            print(f"  MISSING: {group_name}")

    return runs_by_lr


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #
def _plot_metric(groups_data, metric_col, ylabel, title, out_path):
    """Generic plotter used for both forget and retain figures."""
    plt.figure(figsize=(14, 8))

    # Alpha-dependent groups
    for (group_name, alpha_val), histories in groups_data.items():
        if alpha_val is None or not histories:
            continue
        df = pd.concat(histories)
        if metric_col not in df.columns:
            continue
        stats = df.groupby("train/step")[metric_col].agg(['mean']).reset_index()

        cfg = group_configs[group_name]
        opacity = alpha_to_opacity(alpha_val)

        plt.plot(
            stats["train/step"], stats["mean"],
            color=cfg["color"], linestyle=cfg["linestyle"],
            linewidth=LINE_WIDTH, alpha=opacity,
            label=f"{group_name} (α={alpha_val})",
        )

    # Baselines (full opacity)
    for (group_name, alpha_val), histories in groups_data.items():
        if alpha_val is not None or not histories:
            continue
        df = pd.concat(histories)
        if metric_col not in df.columns:
            continue
        stats = df.groupby("train/step")[metric_col].agg(['mean']).reset_index()
        cfg = baseline_configs[group_name]

        plt.plot(
            stats["train/step"], stats["mean"],
            color=cfg["color"], linestyle=cfg["linestyle"],
            linewidth=LINE_WIDTH, label=group_name,
        )

    plt.xlabel("Training Steps on Forget Domain", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.ylim(-0.02, 1.02)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved plot → {out_path}")
    plt.close()


# ------------------------------------------------------------------ #
# Tables
# ------------------------------------------------------------------ #
def _build_accuracy_table(groups_data, metric_col):
    """Return a DataFrame with start / final accuracy per config + alpha."""
    rows = []

    for (group_name, alpha_val), histories in groups_data.items():
        if not histories:
            continue
        df = pd.concat(histories)
        if metric_col not in df.columns:
            continue
        stats = df.groupby("train/step")[metric_col].mean().reset_index()
        stats = stats.sort_values("train/step")

        start_acc = stats[metric_col].iloc[0]
        final_acc = stats[metric_col].iloc[-1]

        label = group_name if alpha_val is None else f"{group_name} (α={alpha_val})"
        rows.append({
            "Config": label,
            "Alpha": alpha_val if alpha_val is not None else "-",
            "Start Accuracy": f"{start_acc:.4f}",
            "Final Accuracy": f"{final_acc:.4f}",
            "Delta": f"{final_acc - start_acc:+.4f}",
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Main per-LR routine
# ------------------------------------------------------------------ #
def create_plots_for_lr(lr, groups_data):
    lr_str = f"{lr:.0e}".replace("+", "").replace("-0", "-")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Forget figure ---
    _plot_metric(
        groups_data,
        metric_col="combined_forget_acc",
        ylabel="Accuracy (Forget Set Average)",
        title="Relearning: Accuracy on Forget Domain",
        out_path=OUTPUT_DIR / f"forget_LR_{lr_str}_{ts}.png",
    )

    # --- Retain figure ---
    _plot_metric(
        groups_data,
        metric_col="combined_retain_acc",
        ylabel="Accuracy (Retain Set Average)",
        title="Relearning: Accuracy on Retain Domain",
        out_path=OUTPUT_DIR / f"retain_LR_{lr_str}_{ts}.png",
    )

    # --- Tables ---
    forget_table = _build_accuracy_table(groups_data, "combined_forget_acc")
    retain_table = _build_accuracy_table(groups_data, "combined_retain_acc")

    forget_csv = OUTPUT_DIR / f"forget_table_LR_{lr_str}_{ts}.csv"
    retain_csv = OUTPUT_DIR / f"retain_table_LR_{lr_str}_{ts}.csv"

    forget_table.to_csv(forget_csv, index=False)
    retain_table.to_csv(retain_csv, index=False)

    print(f"[SUCCESS] Saved forget table → {forget_csv}")
    print(f"[SUCCESS] Saved retain table → {retain_csv}")

    print("\n--- Forget Accuracy Table ---")
    print(forget_table.to_string(index=False))
    print("\n--- Retain Accuracy Table ---")
    print(retain_table.to_string(index=False))


def main():
    print(f"Output directory: {OUTPUT_DIR}")
    if TARGET_LR is not None:
        print(f"Filtering to TARGET_LR = {TARGET_LR}")
    print("Fetching runs from W&B...")
    runs_by_lr = fetch_all_runs_with_lr()

    print(f"\nFound {len(runs_by_lr)} unique learning rates:")
    for lr in sorted(runs_by_lr.keys()):
        print(f"  LR = {lr}")

    print("\nGenerating plots and tables...")
    for lr in sorted(runs_by_lr.keys()):
        print(f"\n{'='*60}")
        print(f"LR = {lr}")
        print('='*60)
        create_plots_for_lr(lr, runs_by_lr[lr])

    print("\nDone!")


if __name__ == "__main__":
    main()
