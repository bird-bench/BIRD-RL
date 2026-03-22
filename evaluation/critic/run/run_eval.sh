#!/bin/bash
# Critic SQLite evaluation script.
#
# Runs test cases against predicted SQL on ephemeral SQLite database copies.
#
# Usage:
#   bash evaluation/critic/run/run_eval.sh \
#     --jsonl_file <path_to_predictions.jsonl> \
#     --db_dir <path_to_sqlite_databases> \
#     [--num_threads 4] [--batch_size 10] [--mode pred]

set -euo pipefail

# Defaults
NUM_THREADS=4
BATCH_SIZE=10
MODE="pred"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --jsonl_file) JSONL_FILE="$2"; shift 2 ;;
        --db_dir) DB_DIR="$2"; shift 2 ;;
        --num_threads) NUM_THREADS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate required arguments
if [ -z "${JSONL_FILE:-}" ] || [ -z "${DB_DIR:-}" ]; then
    echo "Usage: bash run_eval.sh --jsonl_file <JSONL> --db_dir <DB_DIR> [--num_threads N] [--batch_size N] [--mode pred|gold]"
    exit 1
fi

EVAL_DIR="$(cd "$(dirname "$0")/../src" && pwd)"

echo "============================================================"
echo "BIRD-Critic SQLite Evaluation"
echo "============================================================"
echo "JSONL file: ${JSONL_FILE}"
echo "DB dir: ${DB_DIR}"
echo "Threads: ${NUM_THREADS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Mode: ${MODE}"
echo "============================================================"

python "${EVAL_DIR}/evaluate.py" \
    --jsonl_file "${JSONL_FILE}" \
    --num_threads "${NUM_THREADS}" \
    --batch_size "${BATCH_SIZE}" \
    --mode "${MODE}" \
    --db_dir "${DB_DIR}" \
    --logging "true"

echo "Evaluation completed!"
