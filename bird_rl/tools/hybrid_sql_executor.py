"""
Unified SQL Executor Tool — handles both critic (pool-based) and bird (direct SQLite).

Mode detection via create_kwargs:
- If preprocess_sql is present and non-empty → critic mode (uses DB pool, persistent state)
- Otherwise → bird mode (direct read-only SQLite, no pool)

Env vars:
- SQL_REWARD_DB_DIR: Critic database directory (with pool templates)
- BIRD_DB_DIR: BIRD database directory (read-only SQLite)
"""

import asyncio
import logging
import multiprocessing
import os
import sqlite3
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Bird database directory (direct SQLite, read-only)
BIRD_DB_DIR = os.environ.get("BIRD_DB_DIR", "")

SQL_TIMEOUT = int(os.environ.get("BIRD_SQL_TIMEOUT", "30"))


def _execute_sql_readonly(sql: str, db_path: str, timeout: int = 30) -> dict:
    """Execute a single SQL query against a read-only SQLite database."""

    def _run(queue, sql, db_path):
        try:
            conn = sqlite3.connect(db_path, timeout=10.0)
            cursor = conn.execute(sql)
            result = cursor.fetchall()
            conn.close()
            queue.put({"success": True, "error": None, "result": result})
        except Exception as e:
            queue.put({"success": False, "error": str(e), "result": None})

    try:
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=_run, args=(queue, sql, db_path))
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
            return {"success": False, "error": f"Timeout after {timeout}s", "result": None}

        if not queue.empty():
            return queue.get_nowait()
        return {"success": False, "error": "No result returned", "result": None}

    except Exception as e:
        return {"success": False, "error": str(e), "result": None}


class HybridSqlExecutorTool(BaseTool):
    """
    Unified SQL Executor for both critic and bird tasks.

    - Critic: DB pool with persistent mode, preprocess_sql, state persists
    - Bird: Direct read-only SQLite, no pool, no state
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Critic pool support (lazy import to avoid errors when pool not configured)
        self._pool = None
        self._instance_db_map = None
        self._execute_sql_with_timeout = None

    def _init_pool(self):
        """Lazy init of pool resources (only needed for critic instances)."""
        if self._pool is not None:
            return
        try:
            from .pool_manager import pool, instance_db_map
            from .sql_utils import execute_sql_with_timeout, get_db_config
            self._pool = pool
            self._instance_db_map = instance_db_map
            self._execute_sql_with_timeout = execute_sql_with_timeout
            self._db_dir, self._timeout = get_db_config()
            logger.info(f"HybridSqlExecutorTool: pool initialized, db_dir={self._db_dir}")
        except Exception as e:
            logger.warning(f"HybridSqlExecutorTool: pool not available ({e}), critic mode disabled")
            self._pool = None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        db_id = create_kwargs.get("db_id")
        preprocess_sql = create_kwargs.get("preprocess_sql", [])

        if not db_id:
            raise ValueError("db_id is required in create_kwargs")

        # Detect mode: critic (has preprocess_sql) vs bird (no preprocess_sql)
        use_pool = bool(preprocess_sql)

        if use_pool:
            # Critic mode: acquire from pool
            self._init_pool()
            if self._pool is None:
                raise RuntimeError("Pool not available but preprocess_sql provided (critic mode)")

            if instance_id in self._instance_db_map:
                pooled_db = self._instance_db_map[instance_id]["pooled_db"]
                self._instance_db_map[instance_id]["ref_count"] += 1
            else:
                pooled_db = self._pool.acquire(
                    db_id=db_id,
                    mode="persistent",
                    preprocess_sql=preprocess_sql
                )
                self._instance_db_map[instance_id] = {"pooled_db": pooled_db, "ref_count": 1}

            self._instance_dict[instance_id] = {
                "mode": "pool",
                "db_id": db_id,
                "pooled_db": pooled_db,
                "db_path": pooled_db.working_path,
            }
        else:
            # Bird mode: direct SQLite
            db_path = os.path.join(BIRD_DB_DIR, db_id, f"{db_id}.sqlite")
            self._instance_dict[instance_id] = {
                "mode": "direct",
                "db_id": db_id,
                "db_path": db_path,
            }

        return instance_id, ToolResponse(
            text=f"SQL Executor ready for database '{db_id}'. You can execute SQL queries."
        )

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        sql = parameters.get("sql", "")

        if not sql:
            return ToolResponse(text="Error: No SQL provided"), 0.0, {}

        state = self._instance_dict[instance_id]
        mode = state["mode"]
        db_path = state["db_path"]

        loop = asyncio.get_event_loop()

        if mode == "pool":
            # Critic mode: use pool's execute_sql_with_timeout
            result = await loop.run_in_executor(
                None,
                lambda: self._execute_sql_with_timeout(
                    sql_list=[sql],
                    db_path=db_path,
                    preprocess_sql=[],  # Already preprocessed at acquire
                    timeout=getattr(self, '_timeout', SQL_TIMEOUT)
                )
            )
        else:
            # Bird mode: direct read-only execution
            result = await loop.run_in_executor(
                None,
                lambda: _execute_sql_readonly(sql, db_path, SQL_TIMEOUT)
            )

        if result["success"]:
            if sql.strip().upper().startswith(('SELECT', 'WITH', 'PRAGMA')):
                query_result = result.get("result")
                feedback = f"SQL executed successfully.\n\nResults: {query_result}"
            else:
                feedback = "SQL executed successfully."
        else:
            error_msg = result.get("error", "Unknown error")
            feedback = f"Execution failed: {error_msg}"

        return ToolResponse(text=feedback), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
