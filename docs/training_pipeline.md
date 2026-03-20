# Four-Stage Progressive Training Pipeline

## Overview

BIRD-RL uses a four-stage progressive training pipeline that decomposes SQL debugging agent training into a curriculum of increasingly complex skills.

## Stage 1: Reasoning SFT

**Script:** `scripts/train/stage1_reasoning_sft.sh`

- **Input model:** Pre-trained code LLM (e.g., Qwen2.5-Coder-7B-Instruct)
- **Training data:** Single-turn SQL debugging examples with `<thought>` + `<solution>` format
- **Method:** Full fine-tuning with VERL's FSDP SFT trainer
- **Hyperparameters:** lr=1e-5, 3 epochs, max_length=8192
- **Purpose:** Establish structured reasoning — identify SQL bugs through explicit analysis

## Stage 2: Multi-Turn SFT

**Script:** `scripts/train/stage2_multiturn_sft.sh`

- **Input model:** Stage 1 checkpoint
- **Training data:** Multi-turn tool-call trajectories (think + execute_sql + observe + submit_solution)
- **Method:** Full fine-tuning with multi-turn message format
- **Hyperparameters:** lr=1e-5, 3 epochs, max_length=16384
- **Purpose:** Teach tool interaction patterns on top of existing SQL skills

## Stage 3: Single-Turn Reasoning RL

**Script:** `scripts/train/stage3_reasoning_rl.sh`

- **Input model:** Stage 2 checkpoint
- **Training data:** SQL debugging prompts (2913 train + 504 dev)
- **Method:** GRPO with n=16 rollouts, execution-based reward (test case pass rate)
- **Hyperparameters:** lr=1e-6, KL coeff=0.01 (low_var_kl)
- **Purpose:** Optimize reasoning quality through RL before adding tool complexity

## Stage 4: Agentic RL

**Script:** `scripts/train/stage4_agentic_rl.sh`

- **Input model:** Stage 3 checkpoint
- **Training data:** SQL debugging prompts (agentic format with tool access)
- **Method:** GRPO with n=4 rollouts, multi-turn with max 5 assistant turns
- **Hyperparameters:** lr=1e-6, KL coeff=0.01, trajectory timeout=500s
- **Purpose:** Optimize multi-turn exploration strategy with persistent database sessions

## Checkpoint Chain

```
Qwen2.5-Coder-7B-Instruct
  → Stage 1 SFT checkpoint
    → Stage 2 Multi-turn SFT checkpoint
      → Stage 3 Reasoning RL checkpoint
        → Stage 4 Agentic RL checkpoint (final model)
```

Each stage uses progressively lower learning rates and builds directly on the previous checkpoint, preventing catastrophic forgetting while acquiring new capabilities.
