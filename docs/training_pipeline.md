# Four-Stage Progressive Training Pipeline

## Overview

BIRD-RL uses a four-stage progressive training pipeline that decomposes SQL debugging agent training into a curriculum of increasingly complex skills.

## Stage 1: Reasoning SFT

**Script:** `scripts/train/stage1_reasoning_sft.sh`

- **Input model:** Pre-trained code LLM (e.g., Qwen2.5-Coder-7B-Instruct)
- **Training data:** Single-turn SQL debugging examples with `<thought>` + `<solution>` format
- **Method:** Full fine-tuning with VERL's FSDP SFT trainer
- **Purpose:** Establish structured reasoning — identify SQL bugs through explicit analysis

## Stage 2: Multi-Turn SFT

**Script:** `scripts/train/stage2_multiturn_sft.sh`

- **Input model:** Stage 1 checkpoint
- **Training data:** Multi-turn tool-call trajectories (think → execute_sql → observe → submit_solution)
- **Method:** Full fine-tuning with multi-turn message format
- **Purpose:** Teach tool interaction patterns on top of existing SQL skills

## Stage 3: Single-Turn Reasoning RL

**Script:** `scripts/train/stage3_reasoning_rl.sh`

- **Input model:** Stage 2 checkpoint
- **Training data:** SQL debugging prompts
- **Method:** GRPO with execution-based reward (test case pass rate)
- **Purpose:** Optimize reasoning quality through RL before adding tool complexity

## Stage 4: Agentic RL

**Script:** `scripts/train/stage4_agentic_rl.sh`

- **Input model:** Stage 3 checkpoint
- **Training data:** SQL debugging prompts with tool access
- **Method:** GRPO with multi-turn rollouts and persistent database sessions
- **Purpose:** Optimize multi-turn exploration strategy with iterative SQL execution

## Checkpoint Chain

```
Pre-trained Code LLM
  → Stage 1 SFT checkpoint
    → Stage 2 Multi-turn SFT checkpoint
      → Stage 3 Reasoning RL checkpoint
        → Stage 4 Agentic RL checkpoint (final model)
```

Each stage builds directly on the previous checkpoint, preventing catastrophic forgetting while acquiring new capabilities.
