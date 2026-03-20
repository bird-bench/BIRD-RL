# System Architecture

## Overview

BIRD-RL supports three task types through a unified framework:

- **Critic**: SQL debugging with test case evaluation
- **BIRD**: SQL generation with execution accuracy (EX metric)
- **Hybrid**: Joint training on both tasks

## Component Map

```
┌─────────────────────────────────────────────────────────┐
│                    Training Scripts                       │
│  stage1_reasoning_sft → stage2_multiturn_sft            │
│  stage3_reasoning_rl  → stage4_agentic_rl               │
└───────────────────────┬─────────────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
   ┌──────────┐  ┌───────────┐  ┌──────────┐
   │  Reward   │  │  Agent    │  │  Tools   │
   │ Functions │  │  Loop     │  │          │
   ├──────────┤  ├───────────┤  ├──────────┤
   │ critic   │  │ stateful  │  │ executor │
   │ bird     │  │ tool loop │  │ submit   │
   │ hybrid   │  │ (DB       │  │ pool_mgr │
   │          │  │  cleanup) │  │ sql_utils│
   └────┬─────┘  └─────┬─────┘  └────┬─────┘
        │               │             │
        └───────────────┼─────────────┘
                        ▼
              ┌──────────────────┐
              │  Database Pool   │
              ├──────────────────┤
              │ pool.py          │
              │ connection.py    │
              │ config.py        │
              └──────────────────┘
```

## Reward Function Dispatch (Hybrid Mode)

```
Input batch
    │
    ├── data_source="bird_critic/*" ──► critic_reward (test case pass rate)
    │
    └── data_source="bird/*" ────────► bird_reward (EX metric)

Results normalized to union key set before returning to VERL.
```

## Tool Configuration

Each task type uses different tool implementations:

| Task | SQL Executor | Submit Solution | Config |
|------|-------------|-----------------|--------|
| Critic | `critic_sql_executor.py` | `critic_submit_solution.py` | `critic_tools.yaml` |
| BIRD | `bird_sql_executor.py` | `bird_submit_solution.py` | `bird_tools.yaml` |
| Hybrid | `hybrid_sql_executor.py` | `hybrid_submit_solution.py` | `hybrid_tools.yaml` |

The critic tools use the connection pool for persistent sessions. The BIRD tools use direct SQLite connections (read-only SELECT queries). The hybrid tools dispatch to either based on the data source.
