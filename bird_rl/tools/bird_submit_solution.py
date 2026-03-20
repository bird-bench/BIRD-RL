"""
BIRD Submit Solution Tool — accepts final SQL submission.

No database pool needed. Just accepts the submission and returns success.
Actual evaluation (EX metric) happens in the reward function.
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


class BirdSubmitSolutionTool(BaseTool):
    """
    Submit Solution for BIRD — accepts final SQL and terminates trajectory.

    Does NOT evaluate correctness here. The reward function handles
    EX evaluation (set equality of query results).
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
        sql = parameters.get("sql", "")

        if not sql:
            return ToolResponse(text="Error: No SQL provided in solution"), 0.0, {}

        feedback = "Solution submitted successfully. Your answer will be evaluated."
        return ToolResponse(text=feedback), 0.0, {"execution_success": True}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
