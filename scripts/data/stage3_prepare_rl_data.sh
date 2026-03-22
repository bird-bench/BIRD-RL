#!/bin/bash
# Stage 3: Prepare single-turn reasoning RL data (critic).
#
# Creates parquet files in VERL RL format for GRPO/PPO training.
# Generates both train and dev splits.
#
# Usage:
#   bash scripts/data/stage3_prepare_rl_data.sh \
#     --train_data <train.jsonl> \
#     --train_schema <train_schema.jsonl> \
#     --dev_data <prompts.jsonl> \
#     --dev_schema <dev_schemas.jsonl> \
#     --output_dir <output_directory> \
#     [--max_samples 200]

set -euo pipefail

# Defaults
MAX_SAMPLES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train_data) TRAIN_DATA="$2"; shift 2 ;;
        --train_schema) TRAIN_SCHEMA="$2"; shift 2 ;;
        --dev_data) DEV_DATA="$2"; shift 2 ;;
        --dev_schema) DEV_SCHEMA="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate required arguments
for var in TRAIN_DATA TRAIN_SCHEMA DEV_DATA DEV_SCHEMA OUTPUT_DIR; do
    if [ -z "${!var:-}" ]; then
        echo "Error: --$(echo $var | tr '[:upper:]' '[:lower:]') is required"
        exit 1
    fi
done

MAX_SAMPLES_ARG=""
if [ -n "${MAX_SAMPLES}" ]; then
    MAX_SAMPLES_ARG="--max-samples ${MAX_SAMPLES}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Stage 3: Prepare Reasoning RL Data"
echo "============================================================"
echo "Train data: ${TRAIN_DATA}"
echo "Train schema: ${TRAIN_SCHEMA}"
echo "Dev data: ${DEV_DATA}"
echo "Dev schema: ${DEV_SCHEMA}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "============================================================"

# Generate train parquet
echo ""
echo "--- Generating train split ---"
python -m bird_rl.data.prepare_reasoning_rl_data \
    --data "${TRAIN_DATA}" \
    --schema "${TRAIN_SCHEMA}" \
    --output "${OUTPUT_DIR}/train_sql_debug.parquet" \
    --data-source "bird_critic/sqlite_train" \
    --split "train" \
    ${MAX_SAMPLES_ARG}

# Generate dev parquet
echo ""
echo "--- Generating dev split ---"
python -m bird_rl.data.prepare_reasoning_rl_data \
    --data "${DEV_DATA}" \
    --schema "${DEV_SCHEMA}" \
    --output "${OUTPUT_DIR}/dev_sql_debug.parquet" \
    --data-source "bird_critic/sqlite_dev" \
    --split "dev" \
    ${MAX_SAMPLES_ARG}

echo ""
echo "============================================================"
echo "Done. RL parquets saved to: ${OUTPUT_DIR}"
echo "  - ${OUTPUT_DIR}/train_sql_debug.parquet"
echo "  - ${OUTPUT_DIR}/dev_sql_debug.parquet"
echo "Use these for Stage 3 training (single-turn reasoning RL)."
echo "============================================================"
