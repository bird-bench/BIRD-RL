"""
Submit Solution Tool for SQL debugging.

This tool:
1. Accepts the model's final SQL solution
2. Signals the end of the trajectory (agent loop terminates on this call)
3. Does NOT execute or validate the SQL — all evaluation is deferred to the reward function

Design Pattern: Trajectory Termination Signal
- Model calls this when confident in solution
- Tool simply records the submission and returns immediately
- Returns 0.0 step reward (actual reward computed in reward_function)
- Actual SQL execution and test case evaluation happen in reward_function
  with ephemeral DB (fresh state) and batch parallelization

Key difference from execute_sql:
- execute_sql: For exploration (schema checks, partial queries, testing ideas)
- submit_solution: For final answer submission (terminates trajectory, no execution)
"""

import logging
import os
import sqlite3
import shutil
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from .sql_utils import is_write_operation, get_db_config, execute_test_cases
from .pool_manager import pool, instance_db_map

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SubmitSolutionTool(BaseTool):
    """
    Submit Solution Tool - accepts SQL and signals trajectory completion.

    This tool should be called when the model is confident in the solution.
    It does NOT execute or validate SQL — all evaluation is deferred to reward_function.

    Workflow per trajectory:
    1. create() - Acquire same persistent DB from pool (shares with execute_sql tool)
    2. execute() - Validate submitted SQL, check execution, return 0.0 reward
    3. release() - Return DB to pool

    Important: Test Case Evaluation happens LATER
    - This tool does NOT run test cases (design change for efficiency)
    - Test cases evaluated in reward_function with batch parallelization
    - Reward function uses ephemeral DB (fresh state) for fair testing
    - This enables efficient multi-threaded reward computation

    The model learns to call this as the final step to signal "I'm done".
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

        logger.info(f"SubmitSolutionTool initialized with db_dir={self.db_dir}, timeout={self.timeout}s")

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
        Initialize tool state for a trajectory.

        Args:
            instance_id: Unique ID for this trajectory
            ground_truth: Ground truth SQL (for evaluation)
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
        test_cases = create_kwargs.get("test_cases", [])
        ground_truth_sql = create_kwargs.get("ground_truth") or ground_truth
        query = create_kwargs.get("query", "")
        schema = create_kwargs.get("schema", "")

        if ground_truth is None:
            ground_truth = ground_truth_sql

        # Validate
        if not db_id:
            raise ValueError("db_id is required in create_kwargs")

        # Check if DB already acquired for this instance (shared with sql_executor_tool)
        if instance_id in instance_db_map:
            pooled_db = instance_db_map[instance_id]["pooled_db"]
            instance_db_map[instance_id]["ref_count"] += 1
            logger.info(f"Reusing existing DB for instance {instance_id} (ref_count={instance_db_map[instance_id]['ref_count']})")
        else:
            # Acquire database from pool in persistent mode (state maintained across tool calls)
            pooled_db = pool.acquire(
                db_id=db_id,
                mode="persistent",
                preprocess_sql=preprocess_sql
            )
            instance_db_map[instance_id] = {"pooled_db": pooled_db, "ref_count": 1}
            logger.info(f"Acquired new DB for instance {instance_id} from pool (ref_count=1)")

        # Initialize state for this trajectory
        self._instance_dict[instance_id] = {
            "db_id": db_id,
            "pooled_db": pooled_db,
            "preprocess_sql": preprocess_sql,
            "ground_truth": ground_truth,
            "test_cases": test_cases,
            "ground_truth_sql": ground_truth_sql,
            "query": query,
            "schema": schema
        }

        logger.info(f"SubmitSolutionTool created for instance {instance_id} with {len(test_cases)} test cases")

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
        """
        Accept submitted solution — no execution or validation here.

        This tool simply records the submission and returns immediately.
        All SQL execution and test case evaluation happen in reward_function
        with ephemeral DB (fresh state) and batch parallelization.

        Args:
            instance_id: Trajectory ID
            parameters: Tool parameters from model (contains "sql_list")
            **kwargs: Additional kwargs

        Returns:
            (ToolResponse, reward, extra_info):
                - ToolResponse: Execution status (success/error message)
                - reward: 0.0 (actual reward computed in reward_function)
                - extra_info: Dict with execution metadata
        """
        sql_list = parameters.get("sql_list", [])

        if not sql_list:
            return ToolResponse(text="Error: No SQL provided in solution"), 0.0, {}

        state = self._instance_dict[instance_id]
        pooled_db = state["pooled_db"]

        logger.info(f"Instance {instance_id}: Submitting solution with {len(sql_list)} SQL(s)")

        # NO SQL execution here - reward function handles all validation
        # This prevents hanging on slow queries and avoids duplicate work
        # Just accept the submission and let reward_function do the evaluation

        feedback = "✅ Solution submitted successfully. Your answer will be evaluated."
        extra_info = {
            "execution_success": True
        }

        logger.info(f"Instance {instance_id}: Submission accepted (validation deferred to reward_function)")

        # Return 0.0 reward - actual reward computed in reward_function
        return ToolResponse(text=feedback), 0.0, extra_info

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """
        Return 0.0 as reward is computed in reward_function (not in tool).

        Design Change:
        - Tool does NOT run test cases (for efficiency)
        - Test cases evaluated in reward_function with batch processing
        - This method always returns 0.0

        Args:
            instance_id: Trajectory ID

        Returns:
            0.0 (actual reward computed in reward_function)
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """
        Cleanup tool state.

        DB cleanup is handled by the agent loop's run() method in a finally block,
        ensuring cleanup happens regardless of how the trajectory ends.

        Args:
            instance_id: Trajectory ID
            **kwargs: Additional kwargs
        """
        # Clean up tool state (DB cleanup handled by agent loop)
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            logger.info(f"[TOOL DEBUG] Cleaned up tool state for {instance_id} (DB cleanup in agent loop)")
