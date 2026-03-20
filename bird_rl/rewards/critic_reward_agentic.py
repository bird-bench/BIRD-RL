"""
SQL Debugging Reward Function for Agentic/Trajectory Training: Simple Execution-Based (v0)

- pure test pass rate with no partial credit.

Scoring Logic (Simplest - No Reward Hacking):
- No submit_solution: 0.0
- Syntax/execution error: 0.0
- Tests pass: test_pass_rate (0.0 to 1.0)
  - 0/5 tests pass: 0.0
  - 1/5 tests pass: 0.2
  - 3/5 tests pass: 0.6
  - 5/5 tests pass: 1.0

Formula: score = test_pass_count / test_total_count

Key Differences from v1:
- NO 0.2 base reward for "valid but wrong SQL" (removes exploitation)
- NO 0.02 reward for malformed submit_solution
- NO fallback to execute_sql
- Pure test-based scoring only

Why this is better:
- No reward hacking - can't get easy partial credit
- Forces model to actually improve correctness
- Clear optimization target: maximize test pass rate
"""

import re
import json
import sys
import os
import sqlite3
import shutil
import traceback
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

# CRITICAL: Set this BEFORE importing pool_manager
# This tells pool_manager to use reward-dedicated pool indices [16-19]
# instead of competing with tools for indices [0-15]
os.environ["USE_REWARD_POOL"] = "1"

logger = logging.getLogger(__name__)

# ============================================================
# IMPORTS FROM EVALUATION CODE
# ============================================================

try:
    from bird_rl.evaluation.critic.db_utils import (
        execute_queries,
        reset_and_restore_database,
        close_sqlite_connection,
        perform_query_on_sqlite_databases,
    )
    from bird_rl.evaluation.critic.test_utils import (
        check_sql_function_usage,
        remove_round,
        remove_distinct,
        remove_comments,
        preprocess_results,
        ex_base,
        TEST_CASE_DEFAULT,
    )
    from bird_rl.evaluation.critic.logger import NullLogger
    HAS_EVAL_UTILS = True
except ImportError as e:
    HAS_EVAL_UTILS = False

from bird_rl.rewards.critic_reward import (
    is_read_only_sql,
    _to_python,
    reset_ephemeral_database,
    execute_test_cases,
)

from bird_rl.tools.pool_manager import pool
from bird_rl.tools.sql_utils import execute_sql_with_timeout

# ============================================================
# CONFIGURATION
# ============================================================

DB_DIR = os.environ.get("SQL_REWARD_DB_DIR", "")

# ============================================================
# RESULT STRUCTURE
# ============================================================

def create_empty_result() -> dict:
    """
    Create an empty result dict with all fields initialized.
    """
    return {
        # Required score
        "score": 0.0,

        # Format validation (metrics)
        "format_valid": False,
        "format_error": "",  # String: for logging
        "has_submit_solution": False,  # Bool: whether submit_solution was called

        # Parsed components (logging)
        "solution_sql": "",         # String: extracted SQL (converted from list)

        # Execution results (metrics)
        "execution_success": False,
        "syntax_error": False,
        "test_passed": False,
        "test_pass_count": 0,
        "test_total_count": 0,
        "test_pass_rate": 0.0,
        "is_read_only": False,
        "timeout": False,

        # Error details (logging)
        "error_message": "",        # String: for logging
    }

# ============================================================
# WRAPPER FOR LIST-BASED EXECUTION
# ============================================================

def execute_single_instance_with_timeout(
    solution_sql_list: List[str],
    ground_truth_sql_list: List[str],
    test_cases: List[str],
    preprocess_sql: List[str],
    conditions: Dict,
    db_path: str,
    db_id: str,
    instance_id: str,
    is_read_only: bool = False,
    timeout: int = 30
) -> Dict:
    """
    Execute SQL lists and run test cases using DB pool.

    Key features:
    - Uses DB pool with ephemeral mode (auto-reset to clean state)
    - Multi-thread safe via pool's internal locking
    - Accepts SQL as lists, NOT strings
    - Passes lists directly to test cases (pred_sqls, sol_sqls)
    - Returns ONLY metric fields (no string/list fields)

    Args:
        solution_sql_list: List of predicted SQL statements
        ground_truth_sql_list: List of ground truth SQL statements
        test_cases: List of test case code strings
        preprocess_sql: Database setup SQL statements
        conditions: Test conditions dict
        db_path: Path to database (not used with pool, kept for compatibility)
        db_id: Database identifier
        instance_id: Instance identifier
        is_read_only: Not used with pool (pool handles mode internally)
        timeout: Execution timeout in seconds (30s for single-list execution)

    Returns:
        Execution result dict with ONLY metric fields
    """
    # Initialize result with metrics + logging fields
    result = {
        # Metrics (for VERL)
        'execution_success': False,
        'syntax_error': False,
        'test_passed': False,
        'test_pass_count': 0,
        'test_total_count': len(test_cases) if (test_cases is not None and len(test_cases) > 0) else 1,
        'test_pass_rate': 0.0,
        'is_read_only': is_read_only,
        'timeout': False,
        # Logging
        'error_message': '',
    }

    logger = NullLogger() if HAS_EVAL_UTILS else None

    # Convert preprocess_sql to list if needed
    if preprocess_sql is not None:
        if isinstance(preprocess_sql, str):
            preprocess_sql = [preprocess_sql]
        else:
            preprocess_sql = list(preprocess_sql) if preprocess_sql else []
    else:
        preprocess_sql = []

    try:
        # Acquire DB from pool in EPHEMERAL mode
        # This automatically:
        # 1. Resets DB to clean template state
        # 2. Runs preprocess_sql
        # 3. Returns pooled DB with connection
        # Note: Relies on SQLite connection timeout (30s) for lock timeouts
        with pool.acquire_context(
            db_id=db_id,
            mode="ephemeral",  # Fresh DB state for fair evaluation
            preprocess_sql=preprocess_sql
        ) as pooled_db:

            conn = pooled_db.connection
            db_path = pooled_db.working_path  # Use pooled DB path for test_cases

            # Execute predicted SQL with multiprocessing timeout
            # This ensures queries cannot hang forever
            exec_result = execute_sql_with_timeout(
                sql_list=solution_sql_list,
                db_path=db_path,
                preprocess_sql=[],  # Already preprocessed when DB was acquired
                timeout=timeout
            )

            if not exec_result["success"]:
                # SQL execution failed (syntax error or timeout)
                error_msg = exec_result.get("error", "Unknown error")
                if "Timeout" in error_msg:
                    result['timeout'] = True
                    result['syntax_error'] = False
                else:
                    result['syntax_error'] = True
                result['error_message'] = error_msg
                return result

            result['execution_success'] = True

            # Run test cases with timeout protection (prevents infinite hangs
            # on computationally explosive SQL during test case evaluation)
            test_result_holder = [None]
            test_error_holder = [None]

            def _run_tests():
                try:
                    test_result_holder[0] = execute_test_cases(
                        test_cases=test_cases,
                        pred_sqls=solution_sql_list,
                        sol_sqls=ground_truth_sql_list,
                        db_path=db_path,
                        conn=conn,
                        conditions=conditions
                    )
                except Exception as e:
                    test_error_holder[0] = e

            test_thread = threading.Thread(target=_run_tests, daemon=True)
            test_thread.start()
            test_thread.join(timeout=timeout)

            if test_thread.is_alive():
                # Test cases hung — interrupt SQLite and bail out
                try:
                    conn.interrupt()
                except Exception:
                    pass
                test_thread.join(timeout=5)
                result['timeout'] = True
                result['error_message'] = f"Test case execution timeout after {timeout}s"
                return result

            if test_error_holder[0] is not None:
                raise test_error_holder[0]

            if test_result_holder[0] is None:
                result['error_message'] = "Test execution returned no result"
                return result

            passed_count, total_count, error_msgs = test_result_holder[0]

            result['test_pass_count'] = passed_count
            result['test_total_count'] = total_count
            result['test_pass_rate'] = passed_count / total_count if total_count > 0 else 0.0
            result['test_passed'] = (passed_count == total_count)

            if error_msgs:
                result['error_message'] = error_msgs

    except Exception as e:
        # For any exception, capture error message
        result['syntax_error'] = True
        result['error_message'] = f"Execution exception: {str(e)}\n{traceback.format_exc()}"

    return result

# ============================================================
# TRAJECTORY RESPONSE PARSING
# ============================================================

def has_submit_solution_call(response: str) -> bool:
    """
    Check if trajectory contains submit_solution tool call.

    Args:
        response: Full trajectory text

    Returns:
        True if submit_solution is called, False otherwise
    """
    if not response or not isinstance(response, str):
        return False

    # Check for submit_solution in tool_call tags (JSON format)
    # Format: <tool_call>{"name": "submit_solution", "arguments": {...}}</tool_call>
    # Uses .*? with re.DOTALL to handle nested JSON objects (e.g. "arguments" before "name")
    pattern = r'<tool_call>\s*\{.*?"name"\s*:\s*"submit_solution"'
    return bool(re.search(pattern, response, re.IGNORECASE | re.DOTALL))


def _repair_json(json_str: str) -> Optional[dict]:
    """
    Attempt to repair common JSON issues from model output.

    Handles:
    - Trailing commas before ] or }
    - Truncated JSON strings in sql_list (attempts to close them)

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Parsed dict, or None if repair failed
    """
    # Fix trailing commas before ] or }
    repaired = re.sub(r',\s*([}\]])', r'\1', json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Try to recover truncated JSON: find sql_list content directly
    # Pattern: look for "sql_list" key and extract the array contents
    sql_list_match = re.search(
        r'"sql_list"\s*:\s*\[\s*(.*)',
        repaired,
        re.DOTALL
    )
    if sql_list_match:
        array_content = sql_list_match.group(1)
        # Extract all quoted strings from the partial array
        strings = re.findall(r'"((?:[^"\\]|\\.)*)"', array_content)
        if strings:
            # Reconstruct a valid tool call dict
            return {
                "name": "submit_solution",
                "arguments": {
                    "sql_list": strings
                }
            }

    return None


def _strip_markdown_fences(text: str) -> str:
    """
    Remove markdown code fences from tool call JSON content.

    Matches the stripping logic in sft_infer/parse_turn_responses.py:87-89.

    Args:
        text: Raw tool call content that may have ```json...``` wrapping

    Returns:
        Cleaned text with fences removed
    """
    text = re.sub(r'^\s*```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?\s*```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def extract_sql_from_solution(response: str) -> Optional[List[str]]:
    """
    Extract SQL from trajectory output.

    Extracts from submit_solution tool call ONLY (agentic v1 with tools):
    <tool_call>{"name": "submit_solution", "arguments": {"sql_list": ["SELECT ..."]}}</tool_call>

    Handles:
    - Markdown code fences inside <tool_call> tags
    - Malformed JSON (trailing commas, truncation)

    Args:
        response: Full trajectory text (may contain <think>, <tool_call>, etc.)

    Returns:
        List of SQL strings, or None if not found
    """
    if not response or not isinstance(response, str):
        return None

    # Extract from submit_solution tool call
    # Format: <tool_call>{"name": "submit_solution", "arguments": {"sql_list": [...]}}</tool_call>
    tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    all_tool_calls = re.findall(tool_call_pattern, response, re.DOTALL | re.IGNORECASE)

    # Find submit_solution calls (check from last to first)
    for tool_call_content in reversed(all_tool_calls):
        # Strip markdown code fences (model sometimes wraps JSON in ```json...```)
        tool_call_json = _strip_markdown_fences(tool_call_content)

        # Try direct JSON parse first
        tool_data = None
        try:
            tool_data = json.loads(tool_call_json)
        except json.JSONDecodeError:
            # Try JSON repair for common issues
            tool_data = _repair_json(tool_call_json)

        if tool_data is None:
            logger.debug(f"Failed to decode tool call (even after repair): {tool_call_json[:200]}")
            continue

        if isinstance(tool_data, dict) and tool_data.get('name') == 'submit_solution':
            arguments = tool_data.get('arguments', {})
            sql_list = arguments.get('sql_list', [])
            if isinstance(sql_list, list) and sql_list:
                return [str(sql).strip() for sql in sql_list if sql]

    # No valid submit_solution found
    return None


def extract_sql_from_last_execute(response: str) -> Optional[List[str]]:
    """
    Fallback: extract SQL from the last execute_sql tool call.

    Used for truncated trajectories where the model called execute_sql
    but never reached submit_solution (e.g., due to max token limit).

    Args:
        response: Full trajectory text

    Returns:
        List of SQL strings from the last execute_sql call, or None if not found
    """
    if not response or not isinstance(response, str):
        return None

    tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    all_tool_calls = re.findall(tool_call_pattern, response, re.DOTALL | re.IGNORECASE)

    # Search from last to first for execute_sql calls
    for tool_call_content in reversed(all_tool_calls):
        tool_call_json = _strip_markdown_fences(tool_call_content)

        tool_data = None
        try:
            tool_data = json.loads(tool_call_json)
        except json.JSONDecodeError:
            tool_data = _repair_json(tool_call_json)

        if tool_data is None:
            continue

        if isinstance(tool_data, dict) and tool_data.get('name') == 'execute_sql':
            arguments = tool_data.get('arguments', {})

            # Handle case where arguments is a string instead of dict (malformed tool call)
            # This happens when model generates invalid JSON format
            # Skip this malformed tool call - don't reward incorrect format
            if not isinstance(arguments, dict):
                continue

            # execute_sql uses 'sql' (single string) not 'sql_list'
            sql = arguments.get('sql', '')
            if isinstance(sql, str) and sql.strip():
                return [sql.strip()]
            # Also check sql_list in case model used that key
            sql_list = arguments.get('sql_list', [])
            if isinstance(sql_list, list) and sql_list:
                return [str(s).strip() for s in sql_list if s]

    return None


def count_tool_calls(response: str) -> Tuple[int, bool]:
    """
    Count total tool calls and detect execute_sql usage in a trajectory.

    Args:
        response: Full trajectory text

    Returns:
        Tuple of (num_tool_calls, has_execute_sql)
    """
    if not response or not isinstance(response, str):
        return 0, False

    tool_call_pattern = r'<tool_call>'
    all_calls = re.findall(tool_call_pattern, response, re.IGNORECASE)
    num_calls = len(all_calls)

    has_execute_sql = bool(re.search(
        r'<tool_call>\s*\{.*?"name"\s*:\s*"execute_sql"',
        response,
        re.IGNORECASE | re.DOTALL
    ))

    return num_calls, has_execute_sql


def validate_trajectory_format(response: str) -> Tuple[bool, str, bool]:
    """
    Validate that trajectory response has correct format.

    Checks for:
    - <think> tags (model's chain-of-thought reasoning)
    - submit_solution tool calls (primary solution format)
    - execute_sql tool calls (secondary indicator of tool usage)

    Args:
        response: Full trajectory text

    Returns:
        Tuple of (is_valid, error_message, has_submit_solution)
        - is_valid: Whether response has extractable SQL from submit_solution
        - error_message: Error description if invalid
        - has_submit_solution: Whether response contains submit_solution tool call
    """
    if not response or not response.strip():
        return False, "empty_response", False

    # Check for submit_solution tool call
    has_submit_call = has_submit_solution_call(response)

    # Try extracting SQL from submit_solution
    sql_list = extract_sql_from_solution(response)

    if not sql_list or len(sql_list) == 0:
        # No SQL found from submit_solution - categorize the error
        _, has_exec_sql = count_tool_calls(response)

        if not has_submit_call:
            if has_exec_sql:
                return False, "no_submit_but_has_execute_sql", False
            elif '<tool_call>' in response.lower():
                return False, "has_tool_calls_but_no_submit", False
            else:
                return False, "no_tool_calls", False
        else:
            # Has submit_solution but extraction failed
            return False, "tool_call_found_but_sql_extraction_failed", has_submit_call

    # Check if SQL is meaningful (not too short)
    sql_text = ' '.join(sql_list)
    if len(sql_text.strip()) < 5:
        return False, "solution_sql_too_short", has_submit_call

    return True, "", has_submit_call


# ============================================================
# BATCH REWARD FUNCTION
# ============================================================

def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[dict],
    **kwargs
) -> List[dict]:
    """
    Compute rewards for trajectory-based training using SQL execution.

    Processing:
    1. Count tool calls and detect execute_sql/submit_solution usage
    2. Extract SQL from submit_solution (with fence stripping + JSON repair)
    3. For truncated trajectories, fall back to last execute_sql call
    4. Execute SQL and run test cases
    5. Score purely based on test pass rate (v0 - simplest)

    Scoring (v0):
    - No submit_solution: 0.0
    - Malformed submit_solution: 0.0
    - Syntax/execution error: 0.0
    - Test pass rate: (tests_passed / tests_total)
      Examples:
      - 0/1 tests pass: 0.0
      - 1/2 tests pass: 0.5
      - 5/5 tests pass: 1.0

    Args:
        data_sources: List of data source identifiers
        solution_strs: List of trajectory texts (full model output)
        ground_truths: List of ground truth solutions
        extra_infos: List of extra info dicts containing:
            - db_id: Database identifier
            - test_cases: Test case code
            - preprocess_sql: Database setup statements
            - conditions: Test conditions

    Returns:
        List of result dicts with scores
    """
    batch_size = len(solution_strs)

    # Sanity checks
    assert len(data_sources) == batch_size
    assert len(ground_truths) == batch_size
    assert len(extra_infos) == batch_size

    results = [create_empty_result() for _ in range(batch_size)]

    print(f"[Agentic Reward v0] Processing batch of {batch_size} trajectories")
    print(f"[Agentic Reward v0] Using DB pool (ephemeral mode: auto-reset + multi-thread safe, timeout=30s)")

    # ==========================================
    # STAGE 1: Extract SQL from trajectories
    # ==========================================
    valid_sql_indices = []
    sql_list_map = {}  # Map index to SQL list (for execution, not output)

    for i in range(batch_size):
        # Count tool calls for scoring logic
        num_calls, has_exec_sql = count_tool_calls(solution_strs[i])

        is_valid, error, has_submit_call = validate_trajectory_format(solution_strs[i])
        results[i]["format_valid"] = is_valid
        results[i]["format_error"] = error  # For logging
        results[i]["has_submit_solution"] = has_submit_call

        if is_valid:
            # Successfully extracted SQL from submit_solution
            sql_list = extract_sql_from_solution(solution_strs[i])
            results[i]["solution_sql"] = ';\n'.join(sql_list) if sql_list else ""
            if sql_list:
                valid_sql_indices.append(i)
                sql_list_map[i] = sql_list
            else:
                results[i]["score"] = 0.0
        elif has_submit_call:
            # Has submit_solution but JSON extraction failed
            # v0: No partial credit - must have valid SQL
            results[i]["solution_sql"] = ""
            results[i]["score"] = 0.0
        else:
            # No submit_solution → 0.0 reward
            results[i]["solution_sql"] = ""
            results[i]["score"] = 0.0

    print(f"[Agentic Reward v0] Stage 1: {len(valid_sql_indices)}/{batch_size} have valid SQL")

    if not valid_sql_indices:
        return [_to_python(r) for r in results]

    # ==========================================
    # STAGE 2: Execute SQL and run test cases
    # ==========================================

    for i in valid_sql_indices:
        solution_sql_list = sql_list_map[i]  # Get list from map (not from results dict)

        # Extract ground_truth and test_cases
        if isinstance(ground_truths[i], dict):
            ground_truth = ground_truths[i].get('ground_truth', '')
            test_cases = ground_truths[i].get('test_cases', [])
        else:
            ground_truth = ground_truths[i]
            test_cases = []

        # Convert ground_truth to list (handle numpy arrays from parquet)
        if hasattr(ground_truth, 'tolist'):
            ground_truth = ground_truth.tolist()

        # Ensure ground_truth is a list
        if not isinstance(ground_truth, list):
            ground_truth = [str(ground_truth)] if ground_truth else []

        ground_truth_sql_list = [str(sql).strip() for sql in ground_truth if sql]

        # Fallback: check extra_infos for test_cases
        if not test_cases:
            test_cases = extra_infos[i].get('test_cases', [])

        # Convert to list if needed
        if hasattr(test_cases, 'tolist'):
            test_cases = test_cases.tolist()
        elif not isinstance(test_cases, list):
            test_cases = list(test_cases) if (test_cases is not None and len(test_cases) > 0) else []

        # Get other fields
        db_id = extra_infos[i].get('db_id', '')
        preprocess_sql = extra_infos[i].get('preprocess_sql', [])

        # Convert preprocess_sql to list
        if hasattr(preprocess_sql, 'tolist'):
            preprocess_sql = preprocess_sql.tolist()
        elif not isinstance(preprocess_sql, list):
            preprocess_sql = list(preprocess_sql) if preprocess_sql is not None else []

        conditions = extra_infos[i].get('conditions', {})
        if isinstance(conditions, str):
            try:
                conditions = json.loads(conditions)
            except:
                conditions = {}

        instance_id = extra_infos[i].get('instance_id', f'instance_{i}')

        # Validate db_id exists in pool
        # Pool automatically handles ephemeral mode (reset + preprocess)
        template_path = os.path.join(DB_DIR, db_id, f"{db_id}_template.sqlite")
        if not os.path.exists(template_path):
            results[i]["score"] = 0.0
            results[i]["error_message"] = f"Template database not found for {db_id}"
            results[i]["syntax_error"] = True
            continue

        # Execute with pool (ephemeral mode: auto-reset + preprocess)
        # Wrap in try-except to handle any crashes and skip problematic instances
        try:
            exec_result = execute_single_instance_with_timeout(
                solution_sql_list=solution_sql_list,
                ground_truth_sql_list=ground_truth_sql_list,
                test_cases=test_cases,
                preprocess_sql=preprocess_sql,
                conditions=conditions,
                db_path=template_path,  # Not used, pool determines path
                db_id=db_id,
                instance_id=instance_id,
                is_read_only=False,  # Not used, pool uses ephemeral mode
                timeout=30  # 30s for single-list execution (not full trajectory)
            )
        except Exception as e:
            # Skip problematic instance and continue with rest of batch
            print(f"[Agentic Reward v0]   Instance {i}: CRASHED - {type(e).__name__}: {str(e)}")
            exec_result = {
                'execution_success': False,
                'syntax_error': True,
                'test_passed': False,
                'test_pass_count': 0,
                'test_total_count': 1,
                'test_pass_rate': 0.0,
                'timeout': False,
                'error_message': f"Instance crashed: {type(e).__name__}"
            }

        # Update results (exec_result contains ONLY metric fields now)
        results[i].update(exec_result)

        # Calculate score (v0 - simplest):
        # - Syntax error / timeout: 0.0
        # - Otherwise: pure test pass rate (0.0 to 1.0)
        if exec_result.get('timeout') or exec_result.get('syntax_error'):
            score = 0.0
        else:
            # v0: Pure test pass rate, no base reward
            score = exec_result['test_pass_rate']

        results[i]["score"] = score

        # Per-instance logging
        test_info = f"{exec_result.get('test_pass_count', 0)}/{exec_result.get('test_total_count', 1)} tests"
        if exec_result.get('timeout'):
            test_info = "timeout"
        elif exec_result.get('syntax_error'):
            test_info = "syntax_error"

        print(f"[Agentic Reward v0]   Instance {i}: Score={score:.2f} ({test_info})")

    # ==========================================
    # Summary
    # ==========================================
    avg_score = sum(r["score"] for r in results) / batch_size
    has_submit_count = sum(1 for r in results if r.get("has_submit_solution", False))
    executed_count = sum(1 for r in results if r["execution_success"])
    test_passed_count = sum(1 for r in results if r["test_passed"])

    print(f"[Agentic Reward v0] Summary:")
    print(f"  - Total trajectories: {batch_size}")
    print(f"  - With submit_solution call: {has_submit_count} ({has_submit_count/batch_size:.1%})")
    print(f"  - Valid SQL: {len(valid_sql_indices)} ({len(valid_sql_indices)/batch_size:.1%})")
    print(f"  - Executed successfully (no syntax error): {executed_count} ({executed_count/batch_size:.1%})")
    print(f"  - Tests passed: {test_passed_count} ({test_passed_count/batch_size:.1%})")
    print(f"  - Average score: {avg_score:.3f}")

    return [_to_python(r) for r in results]


# ============================================================
# VERL COMPATIBILITY
# ============================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
    **kwargs
) -> dict:
    """
    Single instance entry point (for compatibility).

    Wraps compute_score_batch for single trajectory processing.
    """
    results = compute_score_batch(
        data_sources=[data_source],
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        extra_infos=[extra_info],
        **kwargs
    )
    return results[0]
