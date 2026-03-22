#!/bin/bash
# Multi-turn inference pipeline for the Hybrid model on BIRD task.
# submit_solution uses sql_list (array) format.
#
# Usage:
#   bash scripts/infer/run_hybrid_bird_inference.sh \
#     --model_path <path_to_model> \
#     --dev_data <path_to_mini_dev.json> \
#     --db_dir <path_to_dev_databases> \
#     --column_meaning <path_to_column_meaning.json> \
#     --output_dir <output_directory> \
#     [--gpu "0,1"] [--max_turns 5] [--max_tokens 3000]

set -euo pipefail

# Defaults
GPU="0"
MAX_TURNS=5
MAX_TOKENS=3000
MAX_MODEL_LEN=20000
TEMPERATURE=0.0
BATCH_SIZE=100
LIMIT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --dev_data) DEV_DATA="$2"; shift 2 ;;
        --db_dir) DB_DIR="$2"; shift 2 ;;
        --column_meaning) COLUMN_MEANING="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --max_turns) MAX_TURNS="$2"; shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --limit) LIMIT="--limit $2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate required arguments
for var in MODEL_PATH DEV_DATA DB_DIR COLUMN_MEANING OUTPUT_DIR; do
    if [ -z "${!var:-}" ]; then
        echo "Error: --$(echo $var | tr '[:upper:]' '[:lower:]') is required"
        exit 1
    fi
done

TRAJ_DIR="${OUTPUT_DIR}/trajectories"
mkdir -p "${TRAJ_DIR}"

echo "============================================================"
echo "Hybrid Model BIRD Multi-Turn Inference Pipeline"
echo "============================================================"
echo "Model: ${MODEL_PATH}"
echo "Dev data: ${DEV_DATA}"
echo "DB dir: ${DB_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Max turns: ${MAX_TURNS}"
echo "GPU: ${GPU}"
echo "============================================================"

for TURN in $(seq 0 $((MAX_TURNS - 1))); do
    echo ""
    echo "==================== Turn ${TURN} ===================="

    PROMPT_FILE="${OUTPUT_DIR}/prompts_turn_${TURN}.jsonl"
    RESPONSE_FILE="${OUTPUT_DIR}/responses_turn_${TURN}.jsonl"
    PARSED_FILE="${OUTPUT_DIR}/parsed_turn_${TURN}.jsonl"
    OBS_FILE="${OUTPUT_DIR}/observations_turn_${TURN}.jsonl"
    TRAJ_FILE="${TRAJ_DIR}/traj_${TURN}.jsonl"

    # Step 1: Generate prompts (hybrid format)
    echo "[Step 1] Generating prompts (hybrid format)..."
    python -m bird_rl.inference.bird.generate_prompts_hybrid \
        --turn "${TURN}" \
        --max-turns "${MAX_TURNS}" \
        --dev-data "${DEV_DATA}" \
        --db-dir "${DB_DIR}" \
        --column-meaning "${COLUMN_MEANING}" \
        --traj-dir "${TRAJ_DIR}" \
        --output "${PROMPT_FILE}" \
        ${LIMIT}

    # Check if any prompts were generated
    if [ ! -s "${PROMPT_FILE}" ]; then
        echo "No prompts generated (all instances finished). Stopping."
        break
    fi

    # Step 2: Run vLLM inference
    echo "[Step 2] Running vLLM inference..."
    python -m bird_rl.inference.vllm_infer \
        --model_path "${MODEL_PATH}" \
        --prompt_path "${PROMPT_FILE}" \
        --output_path "${RESPONSE_FILE}" \
        --gpu "${GPU}" \
        --batch_size "${BATCH_SIZE}" \
        --max_model_len "${MAX_MODEL_LEN}" \
        --max_tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}"

    # Step 3: Parse responses
    echo "[Step 3] Parsing responses..."
    python -m bird_rl.inference.parse_responses \
        --input "${RESPONSE_FILE}" \
        --output "${PARSED_FILE}"

    # Step 4: Execute SQL and collect observations
    echo "[Step 4] Executing SQL..."
    python -m bird_rl.inference.execute_sql_observations \
        --input "${PARSED_FILE}" \
        --output "${OBS_FILE}" \
        --db-dir "${DB_DIR}"

    # Step 5: Build trajectory (sql_list format for hybrid)
    echo "[Step 5] Building trajectory..."
    python -m bird_rl.inference.build_trajectory \
        --turn "${TURN}" \
        --traj-dir "${TRAJ_DIR}" \
        --observations "${OBS_FILE}" \
        --output "${TRAJ_FILE}" \
        --submit-format sql_list

done

# Final trajectory file
FINAL_TRAJ=$(ls -t "${TRAJ_DIR}"/traj_*.jsonl 2>/dev/null | head -1)

if [ -n "${FINAL_TRAJ}" ]; then
    echo ""
    echo "==================== Evaluation ===================="
    echo "[Step 6] Evaluating with EX metric..."
    python -m bird_rl.inference.bird.evaluate \
        --trajectory "${FINAL_TRAJ}" \
        --gold "${DEV_DATA}" \
        --db-dir "${DB_DIR}" \
        --output "${OUTPUT_DIR}/eval_results.json"
fi

echo ""
echo "============================================================"
echo "Pipeline complete. Results in: ${OUTPUT_DIR}"
echo "============================================================"
