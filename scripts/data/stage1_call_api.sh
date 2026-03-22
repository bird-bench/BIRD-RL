#!/bin/bash
# Stage 1, Step 2: Call LLM API to generate debugging thoughts.
#
# Sends the thought prompts to a strong LLM (e.g., DeepSeek-R1) and collects
# responses containing <thought> blocks.
#
# Usage:
#   bash scripts/data/stage1_call_api.sh \
#     --input <thought_prompts.jsonl> \
#     --output_dir <output_directory> \
#     --api_key <your_api_key> \
#     [--model_name deepseek-r1] [--num_threads 8]

set -euo pipefail

# Defaults
MODEL_NAME="deepseek-r1"
NUM_THREADS=8
MAX_TOKENS=15000
TEMPERATURE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input) INPUT="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --api_key) API_KEY="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --num_threads) NUM_THREADS="$2"; shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate required arguments
for var in INPUT OUTPUT_DIR API_KEY; do
    if [ -z "${!var:-}" ]; then
        echo "Error: --$(echo $var | tr '[:upper:]' '[:lower:]') is required"
        exit 1
    fi
done

LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARG="--limit ${LIMIT}"
fi

OUTPUT_FILE="${OUTPUT_DIR}/thought_responses.jsonl"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Stage 1, Step 2: Call LLM API for Thought Generation"
echo "============================================================"
echo "Input: ${INPUT}"
echo "Output: ${OUTPUT_FILE}"
echo "Model: ${MODEL_NAME}"
echo "Threads: ${NUM_THREADS}"
echo "============================================================"

python -m bird_rl.data.call_api \
    --input_path "${INPUT}" \
    --output_path "${OUTPUT_FILE}" \
    --model_name "${MODEL_NAME}" \
    --api_key "${API_KEY}" \
    --num_threads "${NUM_THREADS}" \
    --max_tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    ${LIMIT_ARG}

echo ""
echo "Done. Responses saved to: ${OUTPUT_FILE}"
echo "Next: run stage1_prepare_sft_data.sh to build the SFT parquet."
