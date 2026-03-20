#!/bin/bash
#
# Stage 1: Reasoning SFT
# Teaches structured SQL debugging with <thought> + <solution> format.
#
# Input:  Pre-trained code LLM (e.g., Qwen2.5-Coder-7B-Instruct)
# Output: Checkpoint with SQL reasoning capability
#
# Usage:
#   bash scripts/train/stage1_reasoning_sft.sh
#
# Required environment variables:
#   MODEL_PATH    - Path to base model (e.g., Qwen2.5-Coder-7B-Instruct)
#   TRAIN_DATA    - Path to reasoning SFT training data (.parquet)
#
# Optional:
#   NPROC_PER_NODE - Number of GPUs (default: 4)
#   OUTPUT_DIR     - Output checkpoint directory

set -x

# ==========================================
# Configuration
# ==========================================
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MODEL_PATH=${MODEL_PATH:?"Set MODEL_PATH to base model (e.g., Qwen2.5-Coder-7B-Instruct)"}
TRAIN_DATA=${TRAIN_DATA:?"Set TRAIN_DATA to reasoning SFT parquet file"}

RUN_DATE=$(date +%Y%m%d_%H%M%S)
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/stage1_reasoning_sft/${MODEL_NAME}_${RUN_DATE}"}

mkdir -p "$OUTPUT_DIR"

echo "=== Stage 1: Reasoning SFT ==="
echo "Model: $MODEL_PATH"
echo "Train data: $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE"
echo ""

# ==========================================
# Training
# ==========================================
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$TRAIN_DATA \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=8192 \
    data.truncation=right \
    model.partial_pretrain=$MODEL_PATH \
    model.enable_gradient_checkpointing=true \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.01 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=bird-rl-stage1-reasoning-sft \
    trainer.experiment_name=stage1-${MODEL_NAME} \
    trainer.total_epochs=3 \
    trainer.logger=console \
    trainer.save_freq=100 \
    trainer.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    trainer.n_gpus_per_node=$NPROC_PER_NODE \
    "$@"

echo ""
echo "Stage 1 complete. Checkpoints saved to: $OUTPUT_DIR"
echo "Next: Use the best checkpoint as input to Stage 2."
