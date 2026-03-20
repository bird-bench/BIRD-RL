#!/bin/bash
#
# Stage 2: Multi-Turn SFT
# Teaches tool interaction patterns (think + execute_sql + observe + submit_solution).
#
# Input:  Stage 1 checkpoint
# Output: Checkpoint with multi-turn tool-calling capability
#
# Usage:
#   bash scripts/train/stage2_multiturn_sft.sh
#
# Required environment variables:
#   MODEL_PATH    - Path to Stage 1 checkpoint
#   TRAIN_DATA    - Path to multi-turn SFT training data (.parquet)
#
# Optional:
#   NPROC_PER_NODE - Number of GPUs (default: 4)
#   OUTPUT_DIR     - Output checkpoint directory

set -x

# ==========================================
# Configuration
# ==========================================
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MODEL_PATH=${MODEL_PATH:?"Set MODEL_PATH to Stage 1 checkpoint"}
TRAIN_DATA=${TRAIN_DATA:?"Set TRAIN_DATA to multi-turn SFT parquet file"}

RUN_DATE=$(date +%Y%m%d_%H%M%S)
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/stage2_multiturn_sft/${MODEL_NAME}_${RUN_DATE}"}

mkdir -p "$OUTPUT_DIR"

echo "=== Stage 2: Multi-Turn SFT ==="
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
    data.micro_batch_size_per_gpu=2 \
    data.max_length=16384 \
    data.truncation=right \
    model.partial_pretrain=$MODEL_PATH \
    model.enable_gradient_checkpointing=true \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.01 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=bird-rl-stage2-multiturn-sft \
    trainer.experiment_name=stage2-${MODEL_NAME} \
    trainer.total_epochs=3 \
    trainer.logger=console \
    trainer.save_freq=500 \
    trainer.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    trainer.n_gpus_per_node=$NPROC_PER_NODE \
    "$@"

echo ""
echo "Stage 2 complete. Checkpoints saved to: $OUTPUT_DIR"
echo "Next: Use the best checkpoint as input to Stage 3."
