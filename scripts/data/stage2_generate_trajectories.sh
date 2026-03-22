#!/bin/bash
# Stage 2: Generate multi-turn debugging trajectories via LLM.
#
# Runs the full turn loop: generate_prompts → call_api → parse_responses →
# execute_sql → build_trajectory, for each turn from 0 to MAX_TURNS-1.
#
# Usage:
#   bash scripts/data/stage2_generate_trajectories.sh \
#     --train_data <train.jsonl> \
#     --schema_data <train_schema.jsonl> \
#     --db_dir <database_directory> \
#     --output_dir <output_directory> \
#     --api_key <your_api_key> \
#     [--model_name claude-sonnet-4] [--max_turns 5] [--num_threads 8]

set -euo pipefail

# Defaults
MAX_TURNS=5
NUM_THREADS=8
MODEL_NAME="claude-sonnet-4"
MAX_TOKENS=20000
TEMPERATURE=0
LIMIT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train_data) TRAIN_DATA="$2"; shift 2 ;;
        --schema_data) SCHEMA_DATA="$2"; shift 2 ;;
        --db_dir) DB_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --api_key) API_KEY="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --max_turns) MAX_TURNS="$2"; shift 2 ;;
        --num_threads) NUM_THREADS="$2"; shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate required arguments
for var in TRAIN_DATA SCHEMA_DATA DB_DIR OUTPUT_DIR API_KEY; do
    if [ -z "${!var:-}" ]; then
        echo "Error: --$(echo $var | tr '[:upper:]' '[:lower:]') is required"
        exit 1
    fi
done

LIMIT_ARG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_ARG="--limit ${LIMIT}"
fi

# Output subdirectories
PROMPTS_DIR="$OUTPUT_DIR/prompts"
RESPONSES_DIR="$OUTPUT_DIR/responses"
PARSED_DIR="$OUTPUT_DIR/parsed"
OBSERVATIONS_DIR="$OUTPUT_DIR/observations"
TRAJECTORIES_DIR="$OUTPUT_DIR/trajectories"

mkdir -p "$PROMPTS_DIR" "$RESPONSES_DIR" "$PARSED_DIR" "$OBSERVATIONS_DIR" "$TRAJECTORIES_DIR"

echo "============================================================"
echo "Stage 2: Generate Multi-Turn Trajectories"
echo "============================================================"
echo "Train data: ${TRAIN_DATA}"
echo "Schema data: ${SCHEMA_DATA}"
echo "Database dir: ${DB_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Model: ${MODEL_NAME}"
echo "Max turns: ${MAX_TURNS}"
echo "Threads: ${NUM_THREADS}"
echo "Limit: ${LIMIT:-all}"
echo "============================================================"
echo ""

# Loop through all turns
for TURN in $(seq 0 $((MAX_TURNS - 1))); do
    echo ""
    echo "######################################"
    echo "# Turn ${TURN}"
    echo "######################################"
    echo ""

    # Step 1: Generate prompts
    echo "[Turn ${TURN} - Step 1/5] Generating prompts..."
    python -m bird_rl.data.generate_turn_prompts \
        --turn ${TURN} \
        --max-turns ${MAX_TURNS} \
        --train-data "${TRAIN_DATA}" \
        --schema-data "${SCHEMA_DATA}" \
        --traj-dir "${TRAJECTORIES_DIR}" \
        --output "${PROMPTS_DIR}/turn_${TURN}_prompts.jsonl" \
        ${LIMIT_ARG}

    # Check if any prompts were generated
    PROMPT_COUNT=$(wc -l < "${PROMPTS_DIR}/turn_${TURN}_prompts.jsonl" 2>/dev/null || echo "0")
    if [ "$PROMPT_COUNT" -eq 0 ]; then
        echo "All instances finished before turn ${TURN}. Stopping."
        break
    fi

    # Step 2: Call LLM API
    echo "[Turn ${TURN} - Step 2/5] Calling LLM API..."
    python -m bird_rl.data.call_api \
        --input_path "${PROMPTS_DIR}/turn_${TURN}_prompts.jsonl" \
        --output_path "${RESPONSES_DIR}/turn_${TURN}_response.jsonl" \
        --model_name "${MODEL_NAME}" \
        --api_key "${API_KEY}" \
        --num_threads "${NUM_THREADS}" \
        --max_tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}"

    # Step 3: Parse responses
    echo "[Turn ${TURN} - Step 3/5] Parsing responses..."
    python -m bird_rl.data.parse_turn_responses \
        --turn ${TURN} \
        --input "${RESPONSES_DIR}/turn_${TURN}_response.jsonl" \
        --output "${PARSED_DIR}/turn_${TURN}_parsed.jsonl"

    # Step 4: Execute SQL
    echo "[Turn ${TURN} - Step 4/5] Executing SQL..."
    python -m bird_rl.data.execute_sql_observations \
        --turn ${TURN} \
        --input "${PARSED_DIR}/turn_${TURN}_parsed.jsonl" \
        --output "${OBSERVATIONS_DIR}/turn_${TURN}_observations.jsonl" \
        --train-data "${TRAIN_DATA}" \
        --db-dir "${DB_DIR}" \
        --threads "${NUM_THREADS}"

    # Step 5: Build trajectory
    echo "[Turn ${TURN} - Step 5/5] Building trajectory..."
    python -m bird_rl.data.build_trajectory \
        --turn ${TURN} \
        --traj-dir "${TRAJECTORIES_DIR}" \
        --observations "${OBSERVATIONS_DIR}/turn_${TURN}_observations.jsonl" \
        --output "${TRAJECTORIES_DIR}/traj_${TURN}.jsonl"

    echo "Turn ${TURN} completed."
    LAST_TURN=${TURN}
done

echo ""
echo "============================================================"
echo "Trajectory generation completed!"
echo "============================================================"
echo "Final trajectory: ${TRAJECTORIES_DIR}/traj_${LAST_TURN:-$((MAX_TURNS - 1))}.jsonl"
echo ""
echo "Next steps:"
echo "  1. Postprocess trajectories to extract SQL for evaluation"
echo "  2. Run evaluation to validate trajectories"
echo "  3. Run stage2_prepare_sft_data.sh to create final SFT parquet"
