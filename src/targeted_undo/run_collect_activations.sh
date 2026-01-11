#!/bin/bash
# =============================================================================
# Collect Activations from LLM Layers
# =============================================================================
# This script collects activations from specified layers of an LLM and saves
# them organized by layer and mode for downstream analysis.
#
# Usage:
#   ./run_collect_activations.sh
#
# Configuration can be modified in the "Configuration" section below.
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Model settings
MODEL_NAME="gpt2-small"
LAYERS="3-5" # "0-25"                          # Format: "0-31" or "0,4,10-12"
MODE="mlp"                             # Options: mlp, residual, mlp_out, attn, attn_out

# Data settings
DATA_PATH="../../../snmf-mlp-decomposition/data/wmdp_dataset.json"
MAX_SAMPLES=""                         # Empty = use all samples (e.g., "1000")
MAX_LENGTH="512"                       # Max sequence length

# Device settings
MODEL_DEVICE="mps"                     # cuda, mps, or cpu
DATA_DEVICE="cpu"                      # Device for storing activations

# Output settings
OUTPUT_DIR="outputs/activations"
SAVE_PATH="${OUTPUT_DIR}/${MODEL_NAME##*/}_${MODE}"

# Processing settings
BATCH_SIZE="4"
SEED="42"

# =============================================================================
# Script Execution
# =============================================================================

echo "============================================================"
echo "           Activation Collection Pipeline"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model:        ${MODEL_NAME}"
echo "  Layers:       ${LAYERS}"
echo "  Mode:         ${MODE}"
echo "  Data Path:    ${DATA_PATH}"
echo "  Save Path:    ${SAVE_PATH}"
echo "  Model Device: ${MODEL_DEVICE}"
echo "  Data Device:  ${DATA_DEVICE}"
echo "  Batch Size:   ${BATCH_SIZE}"
echo "  Seed:         ${SEED}"
echo ""
echo "============================================================"

# Change to script directory
cd "$(dirname "$0")"

echo ""
echo "=== Step 1: Verify Dataset Exists ==="
if [ -f "${DATA_PATH}" ]; then
    echo "Dataset found at ${DATA_PATH}"
else
    echo "ERROR: Dataset not found at ${DATA_PATH}"
    echo "Please ensure the dataset exists or update DATA_PATH in this script."
    exit 1
fi

echo ""
echo "=== Step 2: Create Output Directory ==="
mkdir -p "${SAVE_PATH}"
echo "Output directory: ${SAVE_PATH}"

echo ""
echo "=== Step 3: Collect Activations ==="

# Build the command
CMD="PYTHONPATH=.. python -m targeted_undo.collect_activations \
    --model-name ${MODEL_NAME} \
    --layers ${LAYERS} \
    --mode ${MODE} \
    --data-path ${DATA_PATH} \
    --save-path ${SAVE_PATH} \
    --model-device ${MODEL_DEVICE} \
    --data-device ${DATA_DEVICE} \
    --batch-size ${BATCH_SIZE} \
    --seed ${SEED}"

# Add optional arguments
if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max-samples ${MAX_SAMPLES}"
fi

if [ -n "${MAX_LENGTH}" ]; then
    CMD="${CMD} --max-length ${MAX_LENGTH}"
fi

echo "Running command:"
echo "${CMD}"
echo ""

eval ${CMD}

echo ""
echo "============================================================"
echo "           Activation Collection Complete!"
echo "============================================================"
echo ""
echo "Output saved to: ${SAVE_PATH}"
echo ""
echo "Output structure:"
echo "  ${SAVE_PATH}/"
echo "  ├── ${MODE}/"
echo "  │   ├── layer_X/"
echo "  │   │   └── activations.pt"
echo "  │   └── ..."
echo "  ├── metadata.pt"
echo "  ├── vocab_freq.json"
echo "  └── config.json"
echo ""
echo "============================================================"

