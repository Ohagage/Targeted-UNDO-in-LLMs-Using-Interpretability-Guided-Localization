"""
Training Steps Table

Fetches the actual training steps (S) for each configuration from wandb and
produces a CSV table showing, for every (config, alpha) combination:
  - Training steps of the partial distillation (S_UNDO)
  - Training steps of the oracle pretraining (S_DataFiltering)
  - Compute ratio  S_UNDO / S_DataFiltering  (%)
"""

import wandb
import pandas as pd
import datetime
from pathlib import Path

# ========================= CONFIGURATION =========================
WANDB_KEY = "8b80f738391c946f3c8b26d878a282cbf763ff78"
WANDB_ENTITY = "hagage-tel-aviv-university"

DISTILL_PROJECT = f"{WANDB_ENTITY}/gemma-2-0.1B_MaxEnt_lr_7e-05_partial_distill"
PRETRAIN_ORACLE_PROJECT = f"{WANDB_ENTITY}/gemma-2-0.1B_addition_subtraction+eng"
PRETRAIN_REFERENCE_PROJECT = f"{WANDB_ENTITY}/gemma-2-0.1B_all_arithmetic+eng"

ALPHAS = [0.1, 0.3, 0.6, 0.9, 1.0]

# Maps the mask_type value stored in distill wandb config to a display name
MASK_DISPLAY = {
    "none":     "UNDO (global mask)",
    "binary":   "Localized-UNDO (Delta-Masking via Weight Discrepancy)",
    "relative": "Localized-UNDO (SNMF mask)",
}

OUTPUT_DIR = Path(__file__).parent / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# =================================================================


def get_distill_steps(api, project):
    """Return dict {(alpha, mask_type): training_steps}."""
    steps = {}
    try:
        runs = list(api.runs(project, filters={"state": "finished"}))
    except Exception as e:
        print(f"  Error: {e}")
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


def get_pretrain_steps(api, project):
    """Return max training step from a pretraining project."""
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


def main():
    wandb.login(key=WANDB_KEY)
    api = wandb.Api()

    # --- Oracle (Data Filtering) pretraining steps ---
    print("Fetching Oracle pretraining steps...")
    S_oracle = get_pretrain_steps(api, PRETRAIN_ORACLE_PROJECT)
    if S_oracle is None:
        S_oracle = 1000
        print(f"  Could not fetch – using default {S_oracle}")
    else:
        print(f"  Oracle (Data Filtering): {S_oracle} steps")

    # --- Reference model pretraining steps ---
    print("Fetching Reference pretraining steps...")
    S_reference = get_pretrain_steps(api, PRETRAIN_REFERENCE_PROJECT)
    if S_reference is None:
        S_reference = 1000
        print(f"  Could not fetch – using default {S_reference}")
    else:
        print(f"  Reference (Full Pretrain): {S_reference} steps")

    # --- Partial distillation steps ---
    print("Fetching Partial Distillation steps...")
    distill_steps = get_distill_steps(api, DISTILL_PROJECT)

    # --- Build table ---
    rows = []

    # Baselines
    rows.append({
        "Config": "Oracle (data filtering)",
        "Alpha": "-",
        "Training Steps": S_oracle,
        "Compute (% of Data Filtering)": "100.0%",
    })
    rows.append({
        "Config": "Reference (Full Pretrain)",
        "Alpha": "-",
        "Training Steps": S_reference,
        "Compute (% of Data Filtering)": f"{S_reference / S_oracle * 100:.1f}%",
    })

    # UNDO configs
    for alpha in ALPHAS:
        for mask_type, display_name in MASK_DISPLAY.items():
            key = (alpha, mask_type)
            if key not in distill_steps:
                continue
            s = distill_steps[key]
            pct = s / S_oracle * 100
            rows.append({
                "Config": display_name,
                "Alpha": alpha,
                "Training Steps": s,
                "Compute (% of Data Filtering)": f"{pct:.1f}%",
            })

    df = pd.DataFrame(rows)

    # --- Save ---
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"training_steps_table_{ts}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n[SUCCESS] Saved table → {csv_path}")
    print(f"\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
