"""
Unified Submit Solution Tool — accepts sql_list for both critic and bird.

Both tasks use the same submit_solution interface:
- Accepts sql_list (array of SQL strings)
- Returns success message (no evaluation in tool)
- Actual evaluation happens in reward function

For critic instances (with pool): shares pooled DB via instance_db_map.
For bird instances (no pool): just stores db_id.
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class HybridSubmitSolutionTool(BaseTool):
    """
    Unified Submit Solution Tool for both critic and bird tasks.

    Accepts sql_list parameter (unified format).
    Does NOT evaluate correctness — reward function handles evaluation.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Lazy pool references (only for critic instances)
        self._pool = None
        self._instance_db_map = None

    def _init_pool(self):
        """Lazy init of pool resources (only needed for critic instances)."""
        if self._pool is not None:
            return
        try:
            from .pool_manager import pool, instance_db_map
            self._pool = pool
            self._instance_db_map = instance_db_map
        except Exception as e:
            logger.warning(f"HybridSubmitSolutionTool: pool not available ({e})")
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

        use_pool = bool(preprocess_sql)

        if use_pool:
            # Critic mode: share pooled DB via instance_db_map
            self._init_pool()
            if self._pool is not None and self._instance_db_map is not None:
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
            "db_id": db_id,
        }

        return instance_id, ToolResponse(
            text=f"Ready to accept solution submission for database '{db_id}'. Submit your final SQL when ready."
        )

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        sql_list = parameters.get("sql_list", [])

        if not sql_list:
            return ToolResponse(text="Error: No SQL provided in solution"), 0.0, {}

        feedback = "Solution submitted successfully. Your answer will be evaluated."
        return ToolResponse(text=feedback), 0.0, {"execution_success": True}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
