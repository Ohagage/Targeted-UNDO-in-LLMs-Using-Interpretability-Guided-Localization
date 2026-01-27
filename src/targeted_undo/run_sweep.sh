#!/bin/bash

#SBATCH --job-name=distill_sweep_shir
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:8
#SBATCH --time=1440
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# ============================================================
# Environment Setup
# ============================================================
export HF_HOME="/home/ADV_2526a/rashkovits/distillation-robustify-unlearning-copy/.hf-cache"
export WANDB_API_KEY=8b80f738391c946f3c8b26d878a282cbf763ff78
export PYTHONUNBUFFERED=1

export PYTHONPATH=$PYTHONPATH:/home/ADV_2526a/rashkovits/distillation-robustify-unlearning-copy

mkdir -p logs .hf-cache

# ============================================================
# Execution
# ============================================================

echo "Starting Parallel Distillation Sweep..."
echo "Mask Type: ${1:-relative}"

/home/ADV_2526a/rashkovits/distillation-robustify-unlearning-copy/env/bin/python run_partial_distill_arithmetic.py --run_all --mask_type ${1:-relative}

echo "Sweep completed at $(date)"