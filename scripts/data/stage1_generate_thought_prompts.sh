#!/bin/bash
# Stage 1, Step 1: Generate prompts for thought generation (critic).
#
# Creates JSONL prompts that include the ground-truth solution so a strong LLM
# (e.g., DeepSeek-R1) can generate realistic debugging thought processes.
#
# Usage:
#   bash scripts/data/stage1_generate_thought_prompts.sh \
#     --train_data <train.jsonl> \
#     --schema_data <train_schema.jsonl> \
#     --output_dir <output_directory>

set -euo pipefail

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train_data) TRAIN_DATA="$2"; shift 2 ;;
        --schema_data) SCHEMA_DATA="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate required arguments
for var in TRAIN_DATA SCHEMA_DATA OUTPUT_DIR; do
    if [ -z "${!var:-}" ]; then
        echo "Error: --$(echo $var | tr '[:upper:]' '[:lower:]') is required"
        exit 1
    fi
done

LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARG="--limit ${LIMIT}"
fi

OUTPUT_FILE="${OUTPUT_DIR}/thought_prompts.jsonl"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Stage 1, Step 1: Generate Thought Prompts"
echo "============================================================"
echo "Train data: ${TRAIN_DATA}"
echo "Schema data: ${SCHEMA_DATA}"
echo "Output: ${OUTPUT_FILE}"
echo "============================================================"

python -m bird_rl.data.generate_thought_prompts \
    --train_data "${TRAIN_DATA}" \
    --schema_data "${SCHEMA_DATA}" \
    --output_path "${OUTPUT_FILE}" \
    ${LIMIT_ARG}

echo ""
echo "Done. Prompts saved to: ${OUTPUT_FILE}"
echo "Next: run stage1_call_api.sh to generate thoughts via LLM."
