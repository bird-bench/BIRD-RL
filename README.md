# BIRD-RL: Open-source Agentic RL Project for SQL Tasks 

This is the official github for **"From Imitation to Interactive Exploration: A Multi-Stage Reinforcement Learning Framework for Tool-Augmented SQL Agents"**

## Project Structure

```
BIRD-RL/
├── bird_rl/                        # Core library
│   ├── data/                       # Data preprocessing for all 4 stages
│   │   ├── generate_thought_prompts.py   # Stage 1: generate prompts for thought generation
│   │   ├── call_api.py                   # Stage 1/2: call LLM API (Bedrock) with threading
│   │   ├── prepare_reasoning_sft_data.py # Stage 1: extract thoughts → SFT parquet
│   │   ├── generate_turn_prompts.py      # Stage 2: per-turn prompts with GT + history
│   │   ├── parse_turn_responses.py       # Stage 2: extract <thought> and <tool_call>
│   │   ├── execute_sql_observations.py   # Stage 2: execute SQL, collect observations
│   │   ├── build_trajectory.py           # Stage 2: accumulate multi-turn trajectories
│   │   ├── postprocess_trajectories.py   # Stage 2: extract SQL for validation
│   │   ├── prepare_multi_turn_sft_data.py# Stage 2: validated trajectories → SFT parquet
│   │   ├── prepare_reasoning_rl_data.py  # Stage 3: single-turn RL parquet (VERL format)
│   │   └── prepare_agentic_rl_data.py    # Stage 4: agentic RL parquet with tools_kwargs
│   │
│   ├── prompts/                    # Prompt templates
│   │   ├── critic_reasoning.py           # Single-turn: <thought> + <solution>
│   │   ├── critic_thought_generation.py  # Thought generation (with GT for data gen)
│   │   ├── sft_generation.py             # Multi-turn: tool-calling format (gen/train/validation)
│   │   ├── bird_generation.py            # BIRD benchmark prompts
│   │   ├── bird_sft_training.py          # BIRD SFT training prompts
│   │   └── hybrid_prompts.py             # Hybrid model prompts
│   │
│   ├── rewards/                    # Reward functions for VERL RL training
│   │   ├── critic_reward.py              # Single-turn critic reward (Stage 3)
│   │   ├── critic_reward_agentic.py      # Agentic critic reward (Stage 4)
│   │   ├── bird_reward.py                # BIRD single-turn reward
│   │   ├── bird_reward_agentic.py        # BIRD agentic reward
│   │   ├── hybrid_reward.py              # Hybrid single-turn reward
│   │   └── hybrid_reward_agentic.py      # Hybrid agentic reward
│   │
│   ├── tools/                      # Tool implementations for agentic RL
│   │   ├── critic_sql_executor.py        # execute_sql tool (critic)
│   │   ├── critic_submit_solution.py     # submit_solution tool (critic)
│   │   ├── bird_sql_executor.py          # execute_sql tool (BIRD)
│   │   ├── bird_submit_solution.py       # submit_solution tool (BIRD)
│   │   ├── hybrid_sql_executor.py        # execute_sql tool (hybrid)
│   │   ├── hybrid_submit_solution.py     # submit_solution tool (hybrid)
│   │   ├── pool_manager.py               # Database pool manager
│   │   └── sql_utils.py                  # Shared SQL utilities
│   │
│   ├── database/                   # Database connection and pooling
│   │   ├── config.py
│   │   ├── connection.py
│   │   └── pool.py
│   │
│   └── inference/                  # Inference pipelines
│       ├── vllm_infer.py                 # vLLM batch inference engine
│       ├── parse_responses.py            # Parse model responses
│       ├── execute_sql_observations.py   # Execute SQL during inference
│       ├── build_trajectory.py           # Build trajectories during inference
│       ├── critic/                       # Critic model inference
│       │   ├── generate_prompts.py
│       │   └── evaluate.py
│       └── bird/                         # BIRD benchmark inference
│           ├── generate_prompts.py
│           ├── generate_prompts_hybrid.py
│           └── evaluate.py
│
├── evaluation/                     # Evaluation code
│   ├── critic/                     # Critic evaluation (test-case based)
│   │   ├── run/run_eval.sh
│   │   └── src/
│   └── bird/                       # BIRD evaluation (official benchmark)
│       └── README.md               # → https://github.com/bird-bench/mini_dev
│
├── scripts/
│   ├── data/                       # Data preprocessing scripts
│   │   ├── stage1_generate_thought_prompts.sh
│   │   ├── stage1_call_api.sh
│   │   ├── stage1_prepare_sft_data.sh
│   │   ├── stage2_generate_trajectories.sh   # Multi-turn trajectory loop
│   │   ├── stage2_prepare_sft_data.sh        # Postprocess + create SFT parquet
│   │   ├── stage3_prepare_rl_data.sh
│   │   └── stage4_prepare_rl_data.sh
│   ├── train/                      # VERL training launch scripts
│   │   ├── stage1_reasoning_sft.sh
│   │   ├── stage2_multiturn_sft.sh
│   │   ├── stage3_reasoning_rl.sh
│   │   └── stage4_agentic_rl.sh
│   └── infer/                      # Inference scripts
│       ├── run_critic_inference.sh
│       ├── run_bird_inference.sh
│       └── run_hybrid_bird_inference.sh
│
└── verl_patch/                     # VERL framework patches
    └── stateful_tool_agent_loop.py
```

