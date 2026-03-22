# Trajectory-Scoped Persistent Database Sessions

## Problem

The default VERL `ToolAgentLoop` treats each tool call as stateless — a fresh tool instance is created and destroyed per call. For SQL debugging, this means database state (temporary tables, inserted rows, schema modifications) is lost between tool calls, preventing iterative exploration.

## Solution

BIRD-RL introduces a three-layer architecture for persistent database sessions:

### Layer 1: StatefulToolAgentLoop (`verl_patch/`)

Extends VERL's `ToolAgentLoop` with three key changes:

1. **Stable instance_id**: All tool calls in a trajectory share the same `request_id`
2. **No per-call release**: `tool.release()` removed from `_call_tool()`'s finally block
3. **Trajectory-level cleanup**: `run()` wrapped in try/finally with timeout

### Layer 2: DatabasePool (`bird_rl/database/`)

A thread-safe SQLite connection pool:

- Pre-creates working copies from template databases at startup
- Worker partitioning: each Ray actor gets exclusive pool indices to avoid file-level contention
- Lazy reset: DB reset happens on next acquire, not during release
- Two modes: `persistent` (tools) and `ephemeral` (reward evaluation)

### Layer 3: Shared Instance Registry (`bird_rl/tools/pool_manager.py`)

A cross-tool registry (`instance_db_map`) that:

- Maps trajectory `instance_id` → acquired `PooledDatabase`
- Enables `execute_sql` and `submit_solution` tools to share the same DB
- Uses reference counting to prevent premature release

## Configuration

### Environment Variables

```bash
# Database directory containing template databases
export SQL_REWARD_DB_DIR=/path/to/sqlite_databases

# SQL execution timeout per query (default: 30s)
export SQL_EXECUTION_TIMEOUT=30

# Trajectory timeout (default: 500s)
export TRAJECTORY_TIMEOUT=500
```
