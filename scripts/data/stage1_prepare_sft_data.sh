#!/bin/bash
# Stage 1, Step 3: Prepare single-turn reasoning SFT data (critic).
#
# Extracts <thought> from LLM responses, pairs with ground-truth SQL,
# and creates a parquet file for VERL SFT training.
#
# Usage:
#   bash scripts/data/stage1_prepare_sft_data.sh \
#     --response_data <thought_responses.jsonl> \
#     --train_data <train.jsonl> \
#     --schema_data <train_schema.jsonl> \
#     --output_path <sft_train.parquet>

set -euo pipefail

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --response_data) RESPONSE_DATA="$2"; shift 2 ;;
        --train_data) TRAIN_DATA="$2"; shift 2 ;;
        --schema_data) SCHEMA_DATA="$2"; shift 2 ;;
        --output_path) OUTPUT_PATH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate required arguments
for var in RESPONSE_DATA TRAIN_DATA SCHEMA_DATA OUTPUT_PATH; do
    if [ -z "${!var:-}" ]; then
        echo "Error: --$(echo $var | tr '[:upper:]' '[:lower:]') is required"
        exit 1
    fi
done

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "============================================================"
echo "Stage 1, Step 3: Prepare Reasoning SFT Data"
echo "============================================================"
echo "Response data: ${RESPONSE_DATA}"
echo "Train data: ${TRAIN_DATA}"
echo "Schema data: ${SCHEMA_DATA}"
echo "Output: ${OUTPUT_PATH}"
echo "============================================================"

python -m bird_rl.data.prepare_reasoning_sft_data \
    --response_data "${RESPONSE_DATA}" \
    --train_data "${TRAIN_DATA}" \
    --schema_data "${SCHEMA_DATA}" \
    --output_path "${OUTPUT_PATH}"

echo ""
echo "Done. SFT parquet saved to: ${OUTPUT_PATH}"
echo "Use this for Stage 1 training (single-turn reasoning SFT)."
