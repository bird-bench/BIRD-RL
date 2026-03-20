#!/bin/bash
#
# Stage 4: Agentic RL
# Optimizes multi-turn tool exploration with persistent database sessions.
#
# Input:  Stage 3 checkpoint
# Output: Final model with agentic SQL debugging capability
#
# Usage:
#   bash scripts/train/stage4_agentic_rl.sh
#
# Required environment variables:
#   MODEL_PATH        - Path to Stage 3 checkpoint
#   TRAIN_DATA        - Path to agentic training prompts (.parquet)
#   DEV_DATA          - Path to agentic validation prompts (.parquet)
#   SQL_REWARD_DB_DIR - Path to SQLite database directory
#   TOOL_CONFIG       - Path to tool config YAML (e.g., configs/critic_tools.yaml)
#   AGENT_LOOP_CONFIG - Path to agent loop YAML (e.g., configs/agent_loop.yaml)
#
# Optional:
#   NPROC_PER_NODE        - Number of GPUs (default: 4)
#   OUTPUT_DIR            - Output checkpoint directory
#   SQL_EXECUTION_TIMEOUT - Per-query timeout in seconds (default: 30)
#   TRAJECTORY_TIMEOUT    - Per-trajectory timeout in seconds (default: 500)
#   POOL_SIZE             - Number of DB copies per database (default: 160)

set -x

# ==========================================
# Configuration
# ==========================================
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MODEL_PATH=${MODEL_PATH:?"Set MODEL_PATH to Stage 3 checkpoint"}
TRAIN_DATA=${TRAIN_DATA:?"Set TRAIN_DATA to agentic training parquet file"}
DEV_DATA=${DEV_DATA:?"Set DEV_DATA to agentic validation parquet file"}
SQL_REWARD_DB_DIR=${SQL_REWARD_DB_DIR:?"Set SQL_REWARD_DB_DIR to SQLite database directory"}
TOOL_CONFIG=${TOOL_CONFIG:?"Set TOOL_CONFIG to tool config YAML path"}
AGENT_LOOP_CONFIG=${AGENT_LOOP_CONFIG:?"Set AGENT_LOOP_CONFIG to agent loop YAML path"}

export SQL_REWARD_DB_DIR
export SQL_EXECUTION_TIMEOUT=${SQL_EXECUTION_TIMEOUT:-30}
export TRAJECTORY_TIMEOUT=${TRAJECTORY_TIMEOUT:-500}
export VLLM_USE_V1=1

# Add bird_rl to Python path for tool imports
export PYTHONPATH="$(pwd):$PYTHONPATH"

RUN_DATE=$(date +%Y%m%d_%H%M%S)
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/stage4_agentic_rl/${MODEL_NAME}_${RUN_DATE}"}
REWARD_FN_PATH="bird_rl/rewards/critic_reward_agentic.py"

export REWARD_OUTPUT_DIR="$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
ROLLOUT_DUMP_DIR="${OUTPUT_DIR}/rollout_jsonl/train"
VAL_DUMP_DIR="${OUTPUT_DIR}/rollout_jsonl/val"
mkdir -p "$ROLLOUT_DUMP_DIR" "$VAL_DUMP_DIR"

echo "=== Stage 4: Agentic RL ==="
echo "Model: $MODEL_PATH"
echo "Train data: $TRAIN_DATA"
echo "Dev data: $DEV_DATA"
echo "Tool config: $TOOL_CONFIG"
echo "Agent loop: $AGENT_LOOP_CONFIG"
echo "Database dir: $SQL_REWARD_DB_DIR"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE"
echo ""

# ==========================================
# Pre-create database pool copies
# ==========================================
POOL_SIZE=${POOL_SIZE:-160}
echo "Pre-creating database pool copies (pool_size=$POOL_SIZE)..."

# Clean up stale pool files from previous runs
find "$SQL_REWARD_DB_DIR" -name "*_pool_*.sqlite*" -type f -delete 2>/dev/null
find "$SQL_REWARD_DB_DIR" \( -name "*_template.sqlite-wal" -o -name "*_template.sqlite-shm" -o -name "*_template.sqlite-journal" \) -type f -delete 2>/dev/null

CREATED_COUNT=0
for db_dir in "$SQL_REWARD_DB_DIR"/*/; do
    db_id=$(basename "$db_dir")
    TEMPLATE_PATH="$db_dir/${db_id}_template.sqlite"

    if [ ! -f "$TEMPLATE_PATH" ]; then
        continue
    fi

    for i in $(seq 0 $((POOL_SIZE - 1))); do
        POOL_FILE="$db_dir/${db_id}_pool_${i}.sqlite"
        cp "$TEMPLATE_PATH" "$POOL_FILE"
        chmod 644 "$POOL_FILE"
        rm -f "${POOL_FILE}-wal" "${POOL_FILE}-shm" "${POOL_FILE}-journal"
        CREATED_COUNT=$((CREATED_COUNT + 1))
    done
done
echo "Pre-created $CREATED_COUNT pool files."
echo ""

# ==========================================
# Training
# ==========================================
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent_with_db_cleanup \
    actor_rollout_ref.rollout.agent.num_workers=$NPROC_PER_NODE \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$DEV_DATA" \
    data.train_batch_size=16 \
    data.max_prompt_length=15000 \
    data.max_response_length=5000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.trace.token2text=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.001 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name="compute_score_batch" \
    reward_model.use_reward_loop=False \
    reward_model.reward_manager=batch \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=bird-rl-stage4-agentic-rl \
    trainer.experiment_name=stage4-${MODEL_NAME} \
    trainer.n_gpus_per_node=$NPROC_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.total_epochs=1 \
    trainer.rollout_data_dir="$ROLLOUT_DUMP_DIR" \
    trainer.validation_data_dir="$VAL_DUMP_DIR" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    "$@"

echo ""
echo "Stage 4 complete. Checkpoints saved to: $OUTPUT_DIR"
echo "This is the final model. Evaluate with scripts/eval/."
