#!/bin/bash
# Stage 2: Prepare multi-turn tool-calling SFT data.
#
# This script:
# 1. Postprocesses trajectories to extract SQL predictions
# 2. (User should run evaluation separately to get status file)
# 3. Creates final multi-turn SFT parquet from validated trajectories
#
# Usage:
#   # Step 1: Postprocess trajectories
#   bash scripts/data/stage2_prepare_sft_data.sh postprocess \
#     --trajectory_file <traj_4.jsonl> \
#     --data_file <train.jsonl> \
#     --output_file <pred_output.jsonl>
#
#   # Step 2: Run evaluation (see evaluation/critic/README.md)
#   # This produces a status file with 'success'/'fail' for each instance
#
#   # Step 3: Create SFT parquet
#   bash scripts/data/stage2_prepare_sft_data.sh create_sft \
#     --status_file <eval_status.jsonl> \
#     --trajectory_file <traj_4.jsonl> \
#     --train_data <train.jsonl> \
#     --schema_data <train_schema.jsonl> \
#     --output_path <multi_turn_sft.parquet>

set -euo pipefail

COMMAND="${1:-}"
shift || true

case "${COMMAND}" in
    postprocess)
        # Parse arguments
        while [[ $# -gt 0 ]]; do
            case $1 in
                --trajectory_file) TRAJECTORY_FILE="$2"; shift 2 ;;
                --data_file) DATA_FILE="$2"; shift 2 ;;
                --output_file) OUTPUT_FILE="$2"; shift 2 ;;
                *) echo "Unknown argument: $1"; exit 1 ;;
            esac
        done

        for var in TRAJECTORY_FILE DATA_FILE OUTPUT_FILE; do
            if [ -z "${!var:-}" ]; then
                echo "Error: --$(echo $var | tr '[:upper:]' '[:lower:]') is required"
                exit 1
            fi
        done

        echo "============================================================"
        echo "Stage 2: Postprocess Trajectories"
        echo "============================================================"

        python -m bird_rl.data.postprocess_trajectories \
            --trajectory-file "${TRAJECTORY_FILE}" \
            --data-file "${DATA_FILE}" \
            --output-file "${OUTPUT_FILE}"

        echo ""
        echo "Done. Now run evaluation on the output to get a status file."
        echo "Then use 'create_sft' command to create the final parquet."
        ;;

    create_sft)
        # Parse arguments
        MAX_TURNS=5
        USE_THINK_TAGS="--use-think-tags"

        while [[ $# -gt 0 ]]; do
            case $1 in
                --status_file) STATUS_FILE="$2"; shift 2 ;;
                --trajectory_file) TRAJECTORY_FILE="$2"; shift 2 ;;
                --train_data) TRAIN_DATA="$2"; shift 2 ;;
                --schema_data) SCHEMA_DATA="$2"; shift 2 ;;
                --output_path) OUTPUT_PATH="$2"; shift 2 ;;
                --max_turns) MAX_TURNS="$2"; shift 2 ;;
                --no_think_tags) USE_THINK_TAGS="--no-think-tags"; shift ;;
                *) echo "Unknown argument: $1"; exit 1 ;;
            esac
        done

        for var in STATUS_FILE TRAJECTORY_FILE TRAIN_DATA SCHEMA_DATA OUTPUT_PATH; do
            if [ -z "${!var:-}" ]; then
                echo "Error: --$(echo $var | tr '[:upper:]' '[:lower:]') is required"
                exit 1
            fi
        done

        mkdir -p "$(dirname "${OUTPUT_PATH}")"

        echo "============================================================"
        echo "Stage 2: Create Multi-Turn SFT Parquet"
        echo "============================================================"
        echo "Status file: ${STATUS_FILE}"
        echo "Trajectory: ${TRAJECTORY_FILE}"
        echo "Train data: ${TRAIN_DATA}"
        echo "Schema data: ${SCHEMA_DATA}"
        echo "Output: ${OUTPUT_PATH}"
        echo "Max turns: ${MAX_TURNS}"
        echo "============================================================"

        python -m bird_rl.data.prepare_multi_turn_sft_data \
            --status-file "${STATUS_FILE}" \
            --trajectory-file "${TRAJECTORY_FILE}" \
            --train-data "${TRAIN_DATA}" \
            --schema-data "${SCHEMA_DATA}" \
            --output-path "${OUTPUT_PATH}" \
            --max-turns "${MAX_TURNS}" \
            ${USE_THINK_TAGS}

        echo ""
        echo "Done. Multi-turn SFT parquet saved to: ${OUTPUT_PATH}"
        echo "Use this for Stage 2 training (multi-turn tool-calling SFT)."
        ;;

    *)
        echo "Usage: $0 {postprocess|create_sft} [options]"
        echo ""
        echo "Commands:"
        echo "  postprocess  - Extract SQL from trajectories for evaluation"
        echo "  create_sft   - Create final SFT parquet from validated trajectories"
        echo ""
        echo "Run '$0 <command>' without options to see required arguments."
        exit 1
        ;;
esac
