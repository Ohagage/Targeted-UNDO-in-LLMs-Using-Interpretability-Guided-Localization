"""
Compute Trade Off: Arithmetic

Generates a scatter plot showing the trade-off between:
  - Robustness: (P_UNDO - P_UnlearnOnly) / (P_DataFiltering - P_UnlearnOnly)
    where P = worst-case forget accuracy after relearning (max across LR attacks)
  - Compute: S_UNDO / S_DataFiltering (as %)
    where S = training steps of partial distillation (UNDO) vs oracle pretraining (Data Filtering)

Data is fetched from three wandb projects:
  1. Relearn project       – worst-case forget performance for each model
  2. Partial distill project – actual training steps per (alpha, mask_type)
  3. Oracle pretrain project – total pretraining steps for data filtering baseline
"""

import wandb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import numpy as np
import datetime
from pathlib import Path

# ========================= CONFIGURATION =========================
WANDB_KEY = "8b80f738391c946f3c8b26d878a282cbf763ff78"
WANDB_ENTITY = "hagage-tel-aviv-university"

RELEARN_PROJECT = f"{WANDB_ENTITY}/gemma-2-0.1B_relearn_only_forget"
DISTILL_PROJECT = f"{WANDB_ENTITY}/gemma-2-0.1B_MaxEnt_lr_7e-05_partial_distill"
PRETRAIN_ORACLE_PROJECT = f"{WANDB_ENTITY}/gemma-2-0.1B_addition_subtraction+eng"

ALPHAS = [0.1, 0.3, 0.6, 0.9, 1.0]
RELEARN_MASK_TYPES = ["none", "binary", "snmf"]

RELEARN_LR = 4e-6

FORGET_COLS = [
    "val/multiplication_equation_acc", "val/multiplication_word_problem_acc",
    "val/division_equation_acc", "val/division_word_problem_acc",
]

OUTPUT_DIR = Path(__file__).parent / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASK_RELEARN_TO_DISTILL = {"none": "none", "binary": "binary", "snmf": "relative"}

# Colours per mask type – same as plotting_relearn_results_by_lr
MASK_STYLES = {
    "none":   {"color": "#1f77b4", "marker": "o", "label": "UNDO (global mask)"},
    "binary": {"color": "#2ca02c", "marker": "s", "label": "Localized-UNDO (Delta-Masking via Weight Discrepancy)"},
    "snmf":   {"color": "#d62728", "marker": "^", "label": "Localized-UNDO (SNMF mask)"},
}
# =================================================================


def alpha_to_opacity(alpha_val):
    """Map distillation alpha to marker opacity (same range as relearn plot)."""
    a_min, a_max = min(ALPHAS), max(ALPHAS)
    if a_max == a_min:
        return 1.0
    t = (alpha_val - a_min) / (a_max - a_min)
    return 0.1 + 0.9 * t


def extract_lr_from_run(run):
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


def get_per_run_max_forget(api, project, filters, target_lr=None):
    try:
        runs = list(api.runs(project, filters=filters))
    except Exception as e:
        print(f"    Error fetching runs: {e}")
        return []

    max_accs = []
    for run in runs:
        if target_lr is not None:
            lr = extract_lr_from_run(run)
            if lr is None or not np.isclose(lr, target_lr, rtol=1e-3):
                continue
        history = run.history(keys=["train/step"] + FORGET_COLS, samples=1000)
        if history.empty:
            continue
        if not all(col in history.columns for col in FORGET_COLS):
            continue
        combined = history[FORGET_COLS].mean(axis=1)
        val = combined.max()
        if not np.isnan(val):
            max_accs.append(val)
    return max_accs


def get_distill_steps(api, project):
    steps = {}
    try:
        runs = list(api.runs(project, filters={"state": "finished"}))
    except Exception as e:
        print(f"  Error fetching distill runs: {e}")
        return steps
    for run in runs:
        alpha = run.config.get("noise_alpha")
        mask_type = run.config.get("mask_type", "none")
        if alpha is None:
            continue
        history = run.history(keys=["train/step"], samples=10000)
        if history.empty:
            continue
        last_step = int(history["train/step"].dropna().max())
        key = (round(alpha, 2), mask_type)
        if key not in steps or last_step > steps[key]:
            steps[key] = last_step
    return steps


def get_oracle_pretrain_steps(api, project):
    try:
        runs = list(api.runs(project, filters={"state": "finished"}))
    except Exception as e:
        print(f"  Error: {e}")
        return None
    for run in runs:
        ms = run.config.get("max_steps")
        if ms and ms > 0:
            return ms
        history = run.history(keys=["train/step"], samples=10000)
        if not history.empty:
            return int(history["train/step"].dropna().max())
    return None


# ========================= MAIN =========================
def main():
    wandb.login(key=WANDB_KEY)
    api = wandb.Api()

    # 1. Oracle pretraining steps
    print("=" * 60)
    print("STEP 1: Oracle pretraining steps (S_DataFiltering)")
    print("=" * 60)
    S_df = get_oracle_pretrain_steps(api, PRETRAIN_ORACLE_PROJECT)
    if S_df is None:
        S_df = 1000
        print(f"  Could not fetch – using default {S_df}")
    else:
        print(f"  S_DataFiltering = {S_df}")

    # 2. Partial-distillation steps
    print("\n" + "=" * 60)
    print("STEP 2: Partial-distillation steps (S_UNDO)")
    print("=" * 60)
    distill_steps = get_distill_steps(api, DISTILL_PROJECT)
    for k, v in sorted(distill_steps.items()):
        print(f"  alpha={k[0]}, mask={k[1]}: {v} steps")

    # 3. Relearning performances
    print("\n" + "=" * 60)
    print("STEP 3: Relearning performances")
    print("=" * 60)

    print("\n  [Unlearn Only (MaxEnt)]")
    unlearn_accs = get_per_run_max_forget(api, RELEARN_PROJECT, {
        "config.wandb_run_name": {"$regex": "Unlearned_MaxEnt_Relearn.*"},
        "state": "finished",
    }, target_lr=RELEARN_LR)
    if not unlearn_accs:
        print("  ERROR: no Unlearn-Only relearn runs found"); return
    P_uo = max(unlearn_accs)
    print(f"    worst-case P_UO = {P_uo:.4f}  (mean={np.mean(unlearn_accs):.4f}, n={len(unlearn_accs)})")

    print("\n  [Oracle (Data Filtering)]")
    oracle_accs = get_per_run_max_forget(api, RELEARN_PROJECT, {
        "config.wandb_run_name": {"$regex": "Oracle_Relearn.*"},
        "state": "finished",
    }, target_lr=RELEARN_LR)
    if not oracle_accs:
        print("  ERROR: no Oracle relearn runs found"); return
    P_df = max(oracle_accs)
    print(f"    worst-case P_DF = {P_df:.4f}  (mean={np.mean(oracle_accs):.4f}, n={len(oracle_accs)})")

    denom = P_df - P_uo
    print(f"\n  Denominator (P_DF - P_UO) = {denom:.4f}")
    if abs(denom) < 1e-6:
        print("  ERROR: P_DF ≈ P_UO – cannot compute robustness"); return

    print("\n  [UNDO Models]")
    results = []
    for alpha in ALPHAS:
        for mask in RELEARN_MASK_TYPES:
            regex = f"PartialDistill_alpha_{alpha}_mask_{mask}.*"
            accs = get_per_run_max_forget(api, RELEARN_PROJECT, {
                "config.wandb_run_name": {"$regex": regex},
                "state": "finished",
            }, target_lr=RELEARN_LR)
            if not accs:
                continue

            distill_mask = MASK_RELEARN_TO_DISTILL.get(mask, mask)
            S_undo = distill_steps.get((alpha, distill_mask))
            if S_undo is None:
                S_undo = distill_steps.get((alpha, mask))
            if S_undo is None:
                print(f"    alpha={alpha}, mask={mask}: {len(accs)} runs but no distill steps – skipped")
                continue

            P_worst = max(accs)
            P_mean  = np.mean(accs)
            P_std   = np.std(accs)

            compute   = (S_undo / S_df) * 100
            rob_worst = ((P_worst - P_uo) / denom) * 100
            rob_mean  = ((P_mean  - P_uo) / denom) * 100
            rob_std   = (P_std / abs(denom)) * 100

            results.append({
                "alpha": alpha, "mask_type": mask,
                "compute": compute,
                "robustness": rob_worst,
                "robustness_mean": rob_mean,
                "robustness_std": rob_std,
                "P_worst": P_worst, "P_mean": P_mean,
                "S_undo": S_undo, "n_runs": len(accs),
            })
            print(f"    alpha={alpha}, mask={mask}: P={P_worst:.4f}, S={S_undo}, "
                  f"Compute={compute:.1f}%, Robustness={rob_worst:.1f}% ({len(accs)} runs)")

    if not results:
        print("\n  ERROR: no UNDO data collected"); return

    df = pd.DataFrame(results)

    # ------ 4. Plot ------
    print("\n" + "=" * 60)
    print("STEP 4: Plotting")
    print("=" * 60)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Plot each point individually so opacity can vary per alpha
    seen_masks = set()
    for _, row in df.iterrows():
        mask = row["mask_type"]
        st = MASK_STYLES.get(mask, {"color": "gray", "marker": "D", "label": f"UNDO ({mask})"})
        opacity = alpha_to_opacity(row["alpha"])

        ax1.errorbar(
            row["compute"], row["robustness"],
            yerr=row["robustness_std"],
            fmt=st["marker"], color=st["color"],
            markersize=10, capsize=5, capthick=1.5, linewidth=1.5,
            alpha=opacity, zorder=5,
        )
        seen_masks.add(mask)

    # Data Filtering reference star
    star = ax1.plot(100, 100, marker="*", color="black", markersize=20,
                    zorder=10, label="Oracle (data filtering)")

    # Dashed trend-line
    all_x = np.concatenate([df["compute"].values, [100]])
    all_y = np.concatenate([df["robustness"].values, [100]])
    coeffs = np.polyfit(all_x, all_y, 1)
    trend = np.poly1d(coeffs)
    x_line = np.linspace(0, 110, 200)
    ax1.plot(x_line, trend(x_line), "k--", alpha=0.4, linewidth=1.5)

    ax1.set_xlabel("Compute (% of Data Filtering)", fontsize=14)
    ax1.set_ylabel("Robustness", fontsize=14)
    ax1.set_title("Compute Trade Off: Arithmetic", fontsize=16, fontweight="bold")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.set_ylabel("")

    # Build legend with full-opacity proxy handles
    handles, labels = [], []
    for mask in seen_masks:
        st = MASK_STYLES.get(mask, {"color": "gray", "marker": "D", "label": f"UNDO ({mask})"})
        proxy = plt.Line2D([], [], color=st["color"], marker=st["marker"],
                           linestyle="None", markersize=10, alpha=1.0)
        handles.append(proxy)
        labels.append(st["label"])
    handles.append(star[0])
    labels.append("Oracle (data filtering)")
    ax1.legend(handles, labels, frameon=False, fontsize=10, loc="upper left")

    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.spines["top"].set_visible(False)

    # Alpha colorbar on the right side
    a_min, a_max = min(ALPHAS), max(ALPHAS)
    norm = mcolors.Normalize(vmin=a_min, vmax=a_max)
    sm = cm.ScalarMappable(cmap=cm.Blues, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, pad=0.12, aspect=30, shrink=0.8)
    cbar.set_label("α", fontsize=14)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = OUTPUT_DIR / f"compute_tradeoff_arithmetic_{ts}.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"\n[SUCCESS] Saved plot → {outpath}")
    plt.close()

    # ------ Summary table ------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    hdr = f"{'Alpha':<8}{'Mask':<10}{'Compute%':<10}{'Robust%':<10}{'P_worst':<10}{'S_undo':<8}{'#Runs':<6}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.sort_values(["mask_type", "alpha"]).iterrows():
        print(f"{r['alpha']:<8.1f}{r['mask_type']:<10}{r['compute']:<10.1f}"
              f"{r['robustness']:<10.1f}{r['P_worst']:<10.4f}{r['S_undo']:<8}{r['n_runs']:<6}")
    print("-" * len(hdr))
    print(f"P_UnlearnOnly   = {P_uo:.4f}")
    print(f"P_DataFiltering = {P_df:.4f}")
    print(f"S_DataFiltering = {S_df}")


if __name__ == "__main__":
    main()
