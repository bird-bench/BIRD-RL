"""
BIRD SQL Executor Tool — Simple read-only SQL execution.

No database pool needed. BIRD is SELECT-only, so we just open
the SQLite database directly each time.
"""

import asyncio
import logging
import os
import sqlite3
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# BIRD database directory
BIRD_DB_DIR = os.environ.get("BIRD_DB_DIR", "")

SQL_TIMEOUT = int(os.environ.get("BIRD_SQL_TIMEOUT", "30"))


def _execute_sql_readonly(sql: str, db_path: str, timeout: int = 30) -> dict:
    """Execute a single SELECT query against a SQLite database."""
    import multiprocessing

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


class BirdSqlExecutorTool(BaseTool):
    """
    SQL Executor for BIRD — read-only, no pool needed.

    Opens SQLite directly from BIRD_DB_DIR/{db_id}/{db_id}.sqlite.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

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

        if not db_id:
            raise ValueError("db_id is required in create_kwargs")

        db_path = os.path.join(BIRD_DB_DIR, db_id, f"{db_id}.sqlite")

        self._instance_dict[instance_id] = {
            "db_id": db_id,
            "db_path": db_path,
        }

        return instance_id, ToolResponse(
            text=f"SQL Executor ready for database '{db_id}'. You can execute SQL queries to explore the database."
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
        db_path = state["db_path"]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _execute_sql_readonly(sql, db_path, SQL_TIMEOUT)
        )

        if result["success"]:
            query_result = result.get("result")
            feedback = f"SQL executed successfully.\n\nResults: {query_result}"
        else:
            error_msg = result.get("error", "Unknown error")
            feedback = f"Execution failed: {error_msg}"

        return ToolResponse(text=feedback), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
