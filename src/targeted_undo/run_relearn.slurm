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
# Define the project root for consistency
PROJECT_ROOT="/home/ADV_2526a/rashkovits/distillation-robustify-unlearning-copy"

export HF_HOME="$PROJECT_ROOT/.hf-cache"
export WANDB_API_KEY=8b80f738391c946f3c8b26d878a282cbf763ff78
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# Navigate to the project directory so Python finds your local modules
cd $PROJECT_ROOT

# Create necessary directories for logs and results
mkdir -p logs .hf-cache
mkdir -p models/non-wmdp/relearned_models
mkdir -p models/non-wmdp/local_records/relearned_models

# ============================================================
# Execution
# ============================================================

echo "Starting Parallel Relearning Experiment..."
echo "Working Directory: $(pwd)"
echo "Start time: $(date)"

# Execute using the specific environment's python
$PROJECT_ROOT/env/bin/python run_relearn_arithmetic.py

echo "Relearning experiment completed at $(date)"