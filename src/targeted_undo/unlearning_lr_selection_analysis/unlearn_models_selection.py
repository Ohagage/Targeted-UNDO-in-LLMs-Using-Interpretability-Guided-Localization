import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

# ================= CONSTANTS =================
wandb_key = os.getenv("WANDB_API_KEY")
if not wandb_key:
    raise ValueError("WANDB_API_KEY not found. Please create a .env file with your API key.")
PROJECT_PATH = "hagage-tel-aviv-university/gemma-2-0.1B_all_arithmetic+eng_unlearn_MaxEnt"

RETAIN_COLS = ["val/addition_equation_acc", "val/addition_word_problem_acc",
               "val/subtraction_equation_acc", "val/subtraction_word_problem_acc"]
FORGET_COLS = ["val/multiplication_equation_acc", "val/multiplication_word_problem_acc",
               "val/division_equation_acc", "val/division_word_problem_acc"]
# =============================================

wandb.login(key=wandb_key)
api = wandb.Api()
runs = api.runs(PROJECT_PATH)

results = []
for run in runs:
    if run.state != "finished" or "learning_rate" not in run.config:
        continue

    lr = run.config["learning_rate"]
    summary = run.summary

    retain_acc = np.mean([summary.get(col, 0) for col in RETAIN_COLS])
    forget_acc = np.mean([summary.get(col, 0) for col in FORGET_COLS])

    # Calculate Unlearning Score: Higher is better
    # Formula: Retain * (1 - Forget). This rewards high retain and low forget.
    score = retain_acc * (1 - forget_acc)

    results.append({"lr": lr, "retain_acc": retain_acc, "forget_acc": forget_acc, "score": score})

df = pd.DataFrame(results).sort_values(by="lr")

# Find Best and Worst based on the Score
best_run = df.loc[df['score'].idxmax()]
worst_run = df.loc[df['score'].idxmin()]

# --- PLOTTING ---
plt.figure(figsize=(12, 7))

# 1. Plot Accuracy lines
plt.semilogx(df["lr"], df["retain_acc"], marker='o', label="Retain Utility (Keep High)",
             color="#2c7fb8", linewidth=2.5, alpha=0.8)
plt.semilogx(df["lr"], df["forget_acc"], marker='s', label="Forget Suppression (Keep Low)",
             color="#e31a1c", linewidth=2.5, alpha=0.8)

# 2. Add Score visual as a filled area (Optional but helpful)
plt.fill_between(df["lr"], df["retain_acc"], df["forget_acc"], alpha=0.1, color='gray', label="Optimization Margin")

# 3. Highlight Best Choice
plt.axvline(x=best_run['lr'], color='green', linestyle='--', linewidth=2, alpha=0.8)
plt.scatter(best_run['lr'], best_run['retain_acc'], color='green', s=150, edgecolors='black', zorder=5)
plt.scatter(best_run['lr'], best_run['forget_acc'], color='green', s=150, edgecolors='black', zorder=5)

# 4. Annotate Best and Worst
plt.text(best_run['lr'], 0.5, f"  ★ BEST CHOICE\n  LR: {best_run['lr']:.1e}\n  Score: {best_run['score']:.2f}",
         color='green', fontweight='bold', verticalalignment='center')

plt.text(worst_run['lr'], 0.1, f"  WORST\n  LR: {worst_run['lr']:.1e}",
         color='maroon', fontsize=9, verticalalignment='bottom')

# Labels and Styling
plt.title(
    "Automated Hyperparameter Selection: MaxEnt LR Sensitivity\nEvaluation based on Maximizing Utility-Suppression Margin",
    fontsize=14)
plt.xlabel("Learning Rate (Log Scale)", fontsize=12)
plt.ylabel("Final Accuracy", fontsize=12)
plt.legend(loc='lower left', fontsize=10)
plt.ylim(-0.05, 1.1)
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("automated_lr_selection.pdf", format='pdf', bbox_inches='tight')

# Print Order to terminal
print("\n--- RANKING OF LEARNING RATES (Best to Worst) ---")
ranked_df = df.sort_values(by="score", ascending=False)
print(ranked_df[['lr', 'score', 'retain_acc', 'forget_acc']].to_string(index=False))

plt.show()