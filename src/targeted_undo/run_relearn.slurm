#!/bin/bash

#SBATCH --job-name=relearn_arith_shir
#SBATCH --output=logs/relearn_%j.out
#SBATCH --error=logs/relearn_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:8
#SBATCH --time=1440
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# ============================================================
# Environment Setup
# ============================================================
PROJECT_ROOT="/home/ADV_2526a/rashkovits/distillation-robustify-unlearning-copy"

cd "$PROJECT_ROOT" || exit

# Load environment variables from .env file
# This replaces the hardcoded WANDB_API_KEY
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Successfully loaded environment variables from .env"
else
    echo "Error: .env file not found. Please create one with your WANDB_API_KEY."
    exit 1
fi

# Set environment variables
export HF_HOME="$PROJECT_ROOT/.hf-cache"
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# Create necessary directories for logs, cache, and results
mkdir -p logs .hf-cache
mkdir -p models/non-wmdp/relearned_models
mkdir -p models/non-wmdp/local_records/relearned_models

# ============================================================
# Execution
# ============================================================

echo "Starting Parallel Relearning Experiment..."
echo "Working Directory: $(pwd)"
echo "Start time: $(date)"

"$PROJECT_ROOT/env/bin/python" run_relearn_arithmetic.py

echo "Relearning experiment completed at $(date)"