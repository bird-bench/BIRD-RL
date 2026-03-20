#!/bin/bash
#
# Stage 3: Single-Turn Reasoning RL
# Optimizes SQL reasoning quality via GRPO with execution-based rewards.
#
# Input:  Stage 2 checkpoint
# Output: Checkpoint with RL-refined reasoning
#
# Usage:
#   bash scripts/train/stage3_reasoning_rl.sh
#
# Required environment variables:
#   MODEL_PATH      - Path to Stage 2 checkpoint
#   TRAIN_DATA      - Path to training prompts (.parquet)
#   DEV_DATA        - Path to validation prompts (.parquet)
#   SQL_REWARD_DB_DIR - Path to SQLite database directory
#
# Optional:
#   NPROC_PER_NODE        - Number of GPUs (default: 8)
#   OUTPUT_DIR            - Output checkpoint directory
#   SQL_EXECUTION_TIMEOUT - Per-query timeout in seconds (default: 30)

set -x

# ==========================================
# Configuration
# ==========================================
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MODEL_PATH=${MODEL_PATH:?"Set MODEL_PATH to Stage 2 checkpoint"}
TRAIN_DATA=${TRAIN_DATA:?"Set TRAIN_DATA to training parquet file"}
DEV_DATA=${DEV_DATA:?"Set DEV_DATA to validation parquet file"}
SQL_REWARD_DB_DIR=${SQL_REWARD_DB_DIR:?"Set SQL_REWARD_DB_DIR to SQLite database directory"}

export SQL_REWARD_DB_DIR
export SQL_EXECUTION_TIMEOUT=${SQL_EXECUTION_TIMEOUT:-30}
export SQL_REWARD_NUM_WORKERS=${SQL_REWARD_NUM_WORKERS:-4}
export SQL_REWARD_BATCH_SIZE=${SQL_REWARD_BATCH_SIZE:-8}

RUN_DATE=$(date +%Y%m%d_%H%M%S)
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/stage3_reasoning_rl/${MODEL_NAME}_${RUN_DATE}"}
REWARD_FN_PATH="bird_rl/rewards/critic_reward.py"

export REWARD_OUTPUT_DIR="$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
ROLLOUT_DUMP_DIR="${OUTPUT_DIR}/rollout_jsonl/train"
VAL_DUMP_DIR="${OUTPUT_DIR}/rollout_jsonl/val"
mkdir -p "$ROLLOUT_DUMP_DIR" "$VAL_DUMP_DIR"

echo "=== Stage 3: Reasoning RL (GRPO) ==="
echo "Model: $MODEL_PATH"
echo "Train data: $TRAIN_DATA"
echo "Dev data: $DEV_DATA"
echo "Database dir: $SQL_REWARD_DB_DIR"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE"
echo ""

# ==========================================
# Training
# ==========================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$DEV_DATA" \
    data.train_batch_size=16 \
    data.max_prompt_length=15000 \
    data.max_response_length=5000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
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
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.001 \
    actor_rollout_ref.rollout.n=16 \
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
    trainer.project_name=bird-rl-stage3-reasoning-rl \
    trainer.experiment_name=stage3-${MODEL_NAME} \
    trainer.n_gpus_per_node=$NPROC_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.total_epochs=1 \
    trainer.rollout_data_dir="$ROLLOUT_DUMP_DIR" \
    trainer.validation_data_dir="$VAL_DUMP_DIR" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    "$@"

echo ""
echo "Stage 3 complete. Checkpoints saved to: $OUTPUT_DIR"
echo "Next: Use the best checkpoint as input to Stage 4."
