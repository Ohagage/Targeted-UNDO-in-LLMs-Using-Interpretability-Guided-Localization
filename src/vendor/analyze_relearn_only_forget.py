#!/usr/bin/env python3
"""
Analyze relearning local records for only_forget_data: for each SNMF model,
find best, average (median), and worst run across all learning rates and output MD.
Also generates a plot per model showing best / average / worst accuracy over steps.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same as run_relearn_arithmetic / paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "non-wmdp"
LOCAL_RECORDS = MODEL_DIR / "local_records" / "relearned_models"

VAL_ACC_KEYS = [
    "val/addition_equation_acc",
    "val/addition_word_problem_acc",
    "val/subtraction_equation_acc",
    "val/subtraction_word_problem_acc",
    "val/multiplication_equation_acc",
    "val/multiplication_word_problem_acc",
    "val/division_equation_acc",
    "val/division_word_problem_acc",
]

# Forget data = multiplication + division only
FORGET_ACC_KEYS = [
    "val/multiplication_equation_acc",
    "val/multiplication_word_problem_acc",
    "val/division_equation_acc",
    "val/division_word_problem_acc",
]


def get_all_val_records(filepath: Path) -> list[dict]:
    """Read JSONL file and return ALL validation records (those containing val acc keys)."""
    val_records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val/addition_equation_acc" in rec:
                val_records.append(rec)
    return val_records


def get_final_val_metrics(filepath: Path) -> dict | None:
    """Read JSONL file and return the last validation record (final run metrics)."""
    records = get_all_val_records(filepath)
    return records[-1] if records else None


def mean_val_accuracy(rec: dict) -> float:
    """Average of the 8 validation accuracy metrics."""
    vals = [rec[k] for k in VAL_ACC_KEYS if k in rec]
    return sum(vals) / len(vals) if vals else 0.0


def mean_forget_accuracy(rec: dict) -> float:
    """Average of the 4 forget-data (multiplication + division) accuracy metrics."""
    vals = [rec[k] for k in FORGET_ACC_KEYS if k in rec]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    if not LOCAL_RECORDS.exists():
        print(f"Local records dir not found: {LOCAL_RECORDS}")
        return

    # Only SNMF models: directory names contain mask_snmf
    snmf_dirs = [d for d in LOCAL_RECORDS.iterdir() if d.is_dir() and "mask_snmf" in d.name]
    snmf_dirs.sort(key=lambda x: x.name)

    results = []

    for model_dir in snmf_dirs:
        # Model display name: e.g. gemma-2-0.1B_MaxEnt_lr_7.0e-05-partial_distill-alpha_0.1-beta_0.1-mask_snmf
        model_display = model_dir.name.replace("relearned_partial_distill_models_arith_", "")

        only_forget_files = list(model_dir.glob("*_only_forget_data.txt"))
        if not only_forget_files:
            results.append((model_display, [], "No only_forget_data files found"))
            continue

        runs = []  # (lr_str, mean_acc, rec)
        for fp in only_forget_files:
            lr_str = fp.stem.replace("_only_forget_data", "")
            rec = get_final_val_metrics(fp)
            if rec is None:
                continue
            mean_acc = mean_val_accuracy(rec)
            runs.append((lr_str, mean_acc, rec))

        if not runs:
            results.append((model_display, [], "No valid validation records"))
            continue

        runs.sort(key=lambda x: x[1], reverse=True)  # best first
        n = len(runs)
        best = runs[0]
        worst = runs[-1]

        # "Average" run = the run whose score is closest to the arithmetic mean
        mean_of_all = sum(r[1] for r in runs) / n
        average_run = min(runs, key=lambda r: abs(r[1] - mean_of_all))

        results.append((model_display, runs, {
            "best": best,
            "average": average_run,
            "worst": worst,
        }))

    # Build markdown
    lines = [
        "# Relearning (only_forget data): Best, Average, and Worst Runs per SNMF Model",
        "",
        "For each SNMF model, runs over all relearning learning rates are ranked by mean validation accuracy "
        "(average of the 8 arithmetic validation accuracies). Reported: **best**, **average** (run closest to the arithmetic mean of all runs), and **worst** run with their learning rate and metrics.",
        "",
    ]

    for model_display, runs, triple in results:
        if isinstance(triple, str):
            lines.append(f"## {model_display}")
            lines.append("")
            lines.append(triple)
            lines.append("")
            continue

        best_lr, best_acc, best_rec = triple["best"]
        avg_lr, avg_acc, avg_rec = triple["average"]
        worst_lr, worst_acc, worst_rec = triple["worst"]

        lines.append(f"## {model_display}")
        lines.append("")
        lines.append(f"Total runs (learning rates): {len(runs)}")
        lines.append("")
        lines.append("| Run type | Learning rate | Mean val acc | add_eq | add_wp | sub_eq | sub_wp | mult_eq | mult_wp | div_eq | div_wp | eng_ce |")
        lines.append("|----------|----------------|--------------|--------|--------|--------|--------|---------|---------|--------|--------|-------|")

        def row(label, lr, acc, rec):
            add_eq = rec.get("val/addition_equation_acc", "")
            add_wp = rec.get("val/addition_word_problem_acc", "")
            sub_eq = rec.get("val/subtraction_equation_acc", "")
            sub_wp = rec.get("val/subtraction_word_problem_acc", "")
            mult_eq = rec.get("val/multiplication_equation_acc", "")
            mult_wp = rec.get("val/multiplication_word_problem_acc", "")
            div_eq = rec.get("val/division_equation_acc", "")
            div_wp = rec.get("val/division_word_problem_acc", "")
            eng_ce = rec.get("val/eng_ce_loss", "")
            return f"| {label} | {lr} | {acc:.4f} | {add_eq} | {add_wp} | {sub_eq} | {sub_wp} | {mult_eq} | {mult_wp} | {div_eq} | {div_wp} | {eng_ce} |"

        lines.append(row("**Best**", best_lr, best_acc, best_rec))
        lines.append(row("**Average** (closest to mean)", avg_lr, avg_acc, avg_rec))
        lines.append(row("**Worst**", worst_lr, worst_acc, worst_rec))
        lines.append("")

    out_path = PROJECT_ROOT / "relearn_only_forget_best_avg_worst.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")

    # ----------------------------------------------------------------
    # Generate one plot per model: best / average / worst over steps
    # ----------------------------------------------------------------
    plots_dir = PROJECT_ROOT / "relearn_only_forget_plots"
    plots_dir.mkdir(exist_ok=True)

    for model_display, runs, triple in results:
        if isinstance(triple, str):
            continue

        best_lr = triple["best"][0]
        avg_lr = triple["average"][0]
        worst_lr = triple["worst"][0]

        # Find the corresponding files for each of the three LRs
        # Reconstruct the model directory path from model_display
        model_dir_name = "relearned_partial_distill_models_arith_" + model_display
        model_dir = LOCAL_RECORDS / model_dir_name

        fig, ax = plt.subplots(figsize=(10, 6))

        for lr_str, label, color, linestyle in [
            (best_lr, f"Best (LR={best_lr})", "#2ecc71", "-"),
            (avg_lr, f"Average (LR={avg_lr})", "#3498db", "--"),
            (worst_lr, f"Worst (LR={worst_lr})", "#e74c3c", "-."),
        ]:
            fp = model_dir / f"{lr_str}_only_forget_data.txt"
            if not fp.exists():
                continue
            val_records = get_all_val_records(fp)
            steps = [r.get("train/step", i) for i, r in enumerate(val_records)]
            accs = [mean_forget_accuracy(r) for r in val_records]
            ax.plot(steps, accs, label=label, color=color, linestyle=linestyle, linewidth=2)

        # Extract alpha from model name for a cleaner title
        alpha_str = model_display.split("alpha_")[1].split("-")[0] if "alpha_" in model_display else "?"
        ax.set_title(f"Relearning (only forget data) — SNMF alpha={alpha_str}\n(Forget accuracy: multiplication + division)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Forget Accuracy (mult + div)", fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=11, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = plots_dir / f"{model_display}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
