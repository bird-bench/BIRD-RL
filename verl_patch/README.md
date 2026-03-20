# VERL Patch: Stateful Tool Agent Loop

This directory contains the custom agent loop that enables **trajectory-scoped persistent database sessions** for multi-turn RL training.

## Installation

Copy the patch file into your VERL installation:

```bash
VERL_PATH=$(python -c "import verl; print(verl.__path__[0])")
cp stateful_tool_agent_loop.py $VERL_PATH/experimental/agent_loop/
```

## What It Does

The default VERL `ToolAgentLoop` creates and destroys tool instances per call (stateless). Our `ToolAgentLoopWithDBCleanup` makes three key changes:

1. **Stable instance_id**: Passes `agent_data.request_id` to `tool.create()` so all tool calls in a trajectory share the same database instance.

2. **Deferred cleanup**: Removes the per-call `tool.release()` in the `finally` block of `_call_tool()`. DB connections are released only when the trajectory ends via `run()`'s `finally` block.

3. **Guaranteed resource release**: Wraps the entire trajectory in `try/finally` with `asyncio.wait_for()` timeout, ensuring DB connections are always returned to the pool.

## Configuration

Reference this agent loop in your training script:

```bash
actor_rollout_ref.rollout.agent.agent_loop_config_path=/path/to/configs/agent_loop.yaml
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent_with_db_cleanup
```
