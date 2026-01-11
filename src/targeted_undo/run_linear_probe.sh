#!/bin/bash
# =============================================================================
# Linear Probe Training Pipeline
# =============================================================================
# This script:
#   1. Collects sample-level activations (with labels)
#   2. Trains linear probes on each layer
#   3. Outputs accuracy per layer to identify where concepts are encoded
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Model settings
MODEL_NAME="gpt2-small"
LAYERS="0-11"                          # All layers for GPT-2 small
MODE="residual"                             # mlp, residual, mlp_out, attn, attn_out
AGGREGATION="mean"                     # mean, last, first, max

# Data settings
DATA_PATH="../../../snmf-mlp-decomposition/data/wmdp_dataset.json"
MAX_SAMPLES=""                         # Empty = use all samples
MAX_LENGTH="512"

# Device settings
MODEL_DEVICE="mps"                     # cuda, mps, or cpu
DATA_DEVICE="cpu"

# Output settings
OUTPUT_DIR="outputs/probe_experiment"
ACTIVATIONS_PATH="${OUTPUT_DIR}/sample_activations"
RESULTS_PATH="${OUTPUT_DIR}/probe_results.json"

# Training settings
BATCH_SIZE="4"
PROBE_EPOCHS="100"
PROBE_LR="0.01"
TEST_SIZE="0.2"
SEED="42"

# =============================================================================
# Script Execution
# =============================================================================

echo "============================================================"
echo "        Linear Probe Training Pipeline"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model:        ${MODEL_NAME}"
echo "  Layers:       ${LAYERS}"
echo "  Mode:         ${MODE}"
echo "  Aggregation:  ${AGGREGATION}"
echo "  Data Path:    ${DATA_PATH}"
echo "  Output Dir:   ${OUTPUT_DIR}"
echo ""
echo "============================================================"

# Change to script directory
cd "$(dirname "$0")"

# =============================================================================
# Step 1: Collect Sample-Level Activations
# =============================================================================
echo ""
echo "=== Step 1: Collect Sample-Level Activations ==="

CMD_COLLECT="PYTHONPATH=.. python -m targeted_undo.collect_sample_activations \
    --model-name ${MODEL_NAME} \
    --layers ${LAYERS} \
    --mode ${MODE} \
    --aggregation ${AGGREGATION} \
    --data-path ${DATA_PATH} \
    --save-path ${ACTIVATIONS_PATH} \
    --model-device ${MODEL_DEVICE} \
    --data-device ${DATA_DEVICE} \
    --batch-size ${BATCH_SIZE} \
    --seed ${SEED}"

if [ -n "${MAX_SAMPLES}" ]; then
    CMD_COLLECT="${CMD_COLLECT} --max-samples ${MAX_SAMPLES}"
fi

if [ -n "${MAX_LENGTH}" ]; then
    CMD_COLLECT="${CMD_COLLECT} --max-length ${MAX_LENGTH}"
fi

echo "Running: ${CMD_COLLECT}"
echo ""
eval ${CMD_COLLECT}

# =============================================================================
# Step 2: Train Linear Probes
# =============================================================================
echo ""
echo "=== Step 2: Train Linear Probes ==="

CMD_TRAIN="PYTHONPATH=.. python -m targeted_undo.train_linear_probe \
    --activations-path ${ACTIVATIONS_PATH} \
    --mode ${MODE} \
    --layers ${LAYERS} \
    --output-path ${RESULTS_PATH} \
    --epochs ${PROBE_EPOCHS} \
    --learning-rate ${PROBE_LR} \
    --test-size ${TEST_SIZE} \
    --batch-size 32 \
    --device ${MODEL_DEVICE} \
    --seed ${SEED} \
    --save-models"

echo "Running: ${CMD_TRAIN}"
echo ""
eval ${CMD_TRAIN}

# =============================================================================
# Done
# =============================================================================
echo ""
echo "============================================================"
echo "        Pipeline Complete!"
echo "============================================================"
echo ""
echo "Results saved to: ${RESULTS_PATH}"
echo "Probe models saved to: ${OUTPUT_DIR}/probe_models/"
echo ""
echo "To view results:"
echo "  cat ${RESULTS_PATH} | python -m json.tool"
echo ""
echo "============================================================"

