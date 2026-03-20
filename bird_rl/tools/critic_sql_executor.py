"""
SQL Executor Tool for trajectory-based SQL debugging training.

This tool allows the model to:
1. Execute SQL queries against a persistent database session
2. Get execution feedback (success/error messages)
3. Iterate and refine based on feedback

Design Pattern: Persistent DB Session with Connection Pool
- Tool executes SQL on a PERSISTENT database (state maintained across tool calls)
- Uses SQLite connection pool - DB acquired ONCE in create(), reused for all execute() calls
- Returns 0.0 for step rewards (no evaluation during execution)
- Final scoring happens in reward_function which evaluates <solution> tags

Database Management:
- create(): Acquires DB from pool in PERSISTENT mode, runs preprocess_sql
- execute(): Uses the same DB connection (state persists - TEMP tables preserved)
- release(): Returns DB to pool for reuse
- Zero disk I/O overhead (pool pre-creates all DBs at training startup)

Why Persistent State:
- Allows model to explore incrementally (CREATE TEMP TABLE → INSERT → SELECT)
- State builds up across tool calls within ONE trajectory
- Each trajectory gets isolated DB from pool
- No copying/deleting DBs during training (fast!)

System Prompt Style:
- Model uses <think> tags for reasoning
- Model calls tool with: <tool_call>execute_sql("SELECT ...")</tool_call>
- Tool returns execution feedback (no scoring)
- Model outputs final answer in: <solution>["SQL1", "SQL2", ...]</solution>
- reward_function evaluates <solution> SQL with test cases (on fresh ephemeral DB)
"""

import logging
import os
import shutil
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from .sql_utils import is_write_operation, get_db_config, execute_sql_with_timeout
from .pool_manager import pool, instance_db_map

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SqlExecutorTool(BaseTool):
    """
    SQL Executor Tool for SQL debugging with trajectory training.

    Workflow per trajectory:
    1. create() - Acquire persistent DB from pool, run preprocess_sql
    2. execute() - Execute SQL on persistent DB (state maintained), called multiple times
    3. calc_reward() - Return 0.0 (not used, final reward from reward_function)
    4. release() - Return DB to pool

    Step Reward Strategy:
    - Tool ONLY executes SQL, does NOT evaluate correctness
    - Returns 0.0 for step rewards (no evaluation during execution)
    - Model learns from execution feedback (errors, results)
    - Final scoring happens in reward_function (batch processing with test cases)

    Connection Pool Optimization:
    - DB acquired from pool in create() - Fast! (0.1ms vs 200ms copy)
    - Same DB reused for all execute() calls in trajectory (persistent state)
    - State maintained: TEMP tables, INSERTs persist across tool calls
    - DB returned to pool in release() for next trajectory to use
    - Zero disk I/O during trajectory execution
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initialize the tool.

        Args:
            config: Tool configuration
            tool_schema: OpenAI function tool schema
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}  # Store state per trajectory

        # Configuration from environment variables
        self.db_dir, self.timeout = get_db_config()

        logger.info(f"SqlExecutorTool initialized with db_dir={self.db_dir}, timeout={self.timeout}s")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the tool schema (defines what model sees)."""
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> tuple[str, ToolResponse]:
        """
        Initialize tool state for a trajectory - acquires DB from pool.

        This is called ONCE per trajectory with data from extra_info.tools_kwargs.create_kwargs.

        Database Management:
        - Acquires persistent DB from connection pool (fast! ~0.1ms)
        - Runs preprocess_sql to set up database state
        - DB state maintained across all execute() calls in this trajectory
        - DB returned to pool in release()

        Args:
            instance_id: Unique ID for this trajectory
            ground_truth: Ground truth SQL (optional, for logging)
            **kwargs: Additional kwargs including create_kwargs from data

        Returns:
            (instance_id, ToolResponse): ID and initial tool message
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Extract from create_kwargs (passed from data)
        create_kwargs = kwargs.get("create_kwargs", {})
        db_id = create_kwargs.get("db_id")
        preprocess_sql = create_kwargs.get("preprocess_sql", [])

        if ground_truth is None:
            ground_truth = create_kwargs.get("ground_truth")

        # Validate
        if not db_id:
            raise ValueError("db_id is required in create_kwargs")

        # Check if DB already acquired for this instance (shared with submit_solution_tool)
        if instance_id in instance_db_map:
            pooled_db = instance_db_map[instance_id]["pooled_db"]
            instance_db_map[instance_id]["ref_count"] += 1
            logger.warning(f"[POOL DEBUG] Reusing {db_id}[{pooled_db.pool_index}] for instance {instance_id[:8]}... (ref_count={instance_db_map[instance_id]['ref_count']})")
        else:
            # Acquire database from pool in persistent mode (state maintained across tool calls)
            pooled_db = pool.acquire(
                db_id=db_id,
                mode="persistent",
                preprocess_sql=preprocess_sql
            )
            instance_db_map[instance_id] = {"pooled_db": pooled_db, "ref_count": 1}
            logger.warning(f"[POOL DEBUG] Acquired {db_id}[{pooled_db.pool_index}] for instance {instance_id[:8]}... (ref_count=1)")

        # Initialize state for this trajectory
        self._instance_dict[instance_id] = {
            "db_id": db_id,
            "pooled_db": pooled_db,
            "preprocess_sql": preprocess_sql,
            "ground_truth": ground_truth
        }

        return instance_id, ToolResponse(
            text=f"SQL Executor ready for database '{db_id}'. You can execute SQL queries to test them."
        )

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Execute single SQL query on persistent DB and return feedback (no scoring).

        This is called MULTIPLE times per trajectory (once per tool call).

        Persistent State:
        - Executes SQL on the same DB acquired in create()
        - State persists: TEMP tables, INSERTs, etc. remain across calls
        - Allows model to build up solution incrementally
        - Returns 0.0 for step rewards (no evaluation)

        Args:
            instance_id: Trajectory ID
            parameters: Tool parameters from model (contains "sql" - single string)
            **kwargs: Additional kwargs

        Returns:
            (ToolResponse, float, extra_info):
                - ToolResponse: Feedback text shown to model
                - float: Step reward (0.0, no evaluation)
                - extra_info: Empty dict
        """
        import asyncio

        sql = parameters.get("sql", "")

        if not sql:
            return ToolResponse(text="Error: No SQL provided"), 0.0, {}

        state = self._instance_dict[instance_id]
        pooled_db = state["pooled_db"]

        logger.debug(f"Instance {instance_id}: Executing SQL with {self.timeout}s timeout")

        # Use multiprocessing-based timeout enforcement, wrapped in run_in_executor
        # to make it properly async and cancellable by trajectory timeout
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # Use default ThreadPoolExecutor
            lambda: execute_sql_with_timeout(
                sql_list=[sql],
                db_path=pooled_db.working_path,
                preprocess_sql=[],  # Already preprocessed when DB was acquired
                timeout=self.timeout
            )
        )

        if result["success"]:
            # SQL executed successfully
            if sql.strip().upper().startswith(('SELECT', 'WITH', 'PRAGMA')):
                query_result = result.get("result")
                feedback = f"SQL executed successfully.\n\nResults: {query_result}"
            else:
                feedback = "SQL executed successfully."
            logger.debug(f"Instance {instance_id}: Execution succeeded")
        else:
            # SQL failed (syntax error or timeout)
            error_msg = result.get("error", "Unknown error")
            feedback = f"Execution failed: {error_msg}"
            logger.debug(f"Instance {instance_id}: Execution failed - {error_msg}")

        # No step reward, just execution feedback
        return ToolResponse(text=feedback), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """
        Calculate final trajectory-level reward.

        Called ONCE at the end of trajectory.

        Note: This is not actually used for final reward calculation.
        The final reward comes from reward_model (v4_hybrid) which
        extracts SQL from <solution> tags and evaluates it.

        This method just returns 0.0 for consistency.

        Args:
            instance_id: Trajectory ID
            **kwargs: Additional kwargs

        Returns:
            0.0 (not used)
        """
        logger.info(f"Instance {instance_id}: calc_reward called")
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """
        Cleanup tool state - returns DB to pool.

        Called ONCE at the end of trajectory.

        Database Management:
        - Returns persistent DB to connection pool
        - DB becomes available for next trajectory to use
        - Pool resets DB state for next user (if configured)

        Args:
            instance_id: Trajectory ID
            **kwargs: Additional kwargs
        """
        if instance_id in self._instance_dict:
            # Just cleanup tool state - DB will be released by custom agent loop
            del self._instance_dict[instance_id]
            logger.warning(f"[TOOL DEBUG] Cleaned up tool state for {instance_id} (DB cleanup in agent loop)")
