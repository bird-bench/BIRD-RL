"""
BIRD SQL Generation Reward Function for Agentic/Trajectory Training: EX-Based

This reward function evaluates BIRD SQL generation trajectories with tool-based
interactions (execute_sql + submit_solution).

Key Features:
1. Extracts SQL from submit_solution tool calls (single "sql" string, not "sql_list")
2. Falls back to last execute_sql call for truncated trajectories
3. Uses EX metric: set(predicted_results) == set(ground_truth_results)
4. All BIRD SQL is SELECT-only, so no ephemeral databases or pools needed
5. Strips markdown code fences from tool call JSON
6. Repairs common JSON issues (trailing commas, truncation)
7. Reward shaping gives gradient signal for partial trajectories

Model Output Format (matching bird_for_sft/sft_generation_prompt.py):
- Model uses <think>...</think> tags for reasoning
- Tool calls wrapped in <tool_call>...</tool_call> tags
- submit_solution uses "sql" (single string): {"name": "submit_solution", "arguments": {"sql": "SELECT ..."}}
- execute_sql uses "sql" (single string): {"name": "execute_sql", "arguments": {"sql": "SELECT ..."}}

Scoring Logic (reward shaping for gradient signal):
- No tool calls at all: 0.0
- Has execute_sql but no submit_solution (truncated): 0.05
- Has submit_solution but JSON extraction failed: 0.02
- SQL execution error / timeout: 0.0
- SQL executes but wrong result (EX=0): 0.1
- Correct result (EX=1): 1.0

The formula ensures gradient signal at every stage:
0.0 (nothing) < 0.02 (broken JSON) < 0.05 (tool usage) < 0.1 (valid SQL) < 1.0 (correct)
"""

import re
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from func_timeout import func_timeout, FunctionTimedOut

# ============================================================
# CONFIGURATION
# ============================================================

DB_DIR = os.environ.get("BIRD_DB_DIR", "")

NUM_WORKERS = int(os.environ.get("BIRD_REWARD_NUM_WORKERS", "8"))

SQL_TIMEOUT = int(os.environ.get("BIRD_SQL_TIMEOUT", "30"))

# ============================================================
# NUMPY COMPATIBILITY
# ============================================================

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _to_python(x):
    """Convert numpy types to Python types."""
    if HAS_NUMPY and isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _to_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_python(v) for v in x]
    return x


# ============================================================
# RESULT STRUCTURE
# ============================================================

def create_empty_result() -> dict:
    """Create an empty result dict with all fields initialized."""
    return {
        "score": 0.0,
        "format_valid": False,
        "format_error": "",
        "has_submit_solution": False,
        "solution_sql": "",
        "execution_success": False,
        "ex_score": 0,
        "error_message": "",
        "timeout": False,
    }


# ============================================================
# EX METRIC (same as official BIRD evaluation)
# ============================================================

def round_results(results: List[tuple], precision: int = 10) -> List[tuple]:
    """Round floating-point values for consistent comparison."""
    rounded = []
    for row in results:
        rounded_row = []
        for val in row:
            if isinstance(val, float):
                rounded_row.append(round(val, precision))
            else:
                rounded_row.append(val)
        rounded.append(tuple(rounded_row))
    return rounded


def calculate_ex(predicted_res: List[tuple], ground_truth_res: List[tuple]) -> int:
    """EX metric: 1 if set(predicted) == set(ground_truth), else 0."""
    return 1 if set(predicted_res) == set(ground_truth_res) else 0


# ============================================================
# SQL EXECUTION WITH TIMEOUT
# ============================================================

def _execute_sql_pair(predicted_sql: str, ground_truth_sql: str, db_path: str) -> Tuple[int, str]:
    """
    Execute both predicted and ground truth SQL, compare results.
    Called within func_timeout.

    Returns:
        Tuple of (ex_score, error_message)
    """
    predicted_res = []
    ground_truth_res = []
    error_message = ""

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        conn.close()
    except Exception as e:
        error_message = f"Predicted SQL Error: {str(e)}"
        predicted_res = []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(ground_truth_sql)
        ground_truth_res = cursor.fetchall()
        conn.close()
    except Exception as e:
        if error_message:
            error_message += f" | Ground Truth SQL Error: {str(e)}"
        else:
            error_message = f"Ground Truth SQL Error: {str(e)}"
        ground_truth_res = []

    predicted_res = round_results(predicted_res)
    ground_truth_res = round_results(ground_truth_res)

    res = calculate_ex(predicted_res, ground_truth_res)
    return res, error_message


def execute_single_instance(
    solution_sql: str,
    ground_truth_sql: str,
    db_id: str,
) -> Dict:
    """
    Execute predicted and ground truth SQL, compare results using EX metric.

    Returns:
        Result dict with ex_score and execution details.
    """
    result = {
        "execution_success": False,
        "ex_score": 0,
        "error_message": "",
        "timeout": False,
    }

    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        result["error_message"] = f"Database not found: {db_path}"
        return result

    try:
        ex_score, error_message = func_timeout(
            SQL_TIMEOUT, _execute_sql_pair,
            args=(solution_sql, ground_truth_sql, db_path),
        )
        result["execution_success"] = True
        result["ex_score"] = ex_score
        result["error_message"] = error_message
    except FunctionTimedOut:
        result["error_message"] = "Execution timeout"
        result["timeout"] = True
    except Exception as e:
        result["error_message"] = f"Execution error: {str(e)}"

    return result


# ============================================================
# TRAJECTORY RESPONSE PARSING
# ============================================================

def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from tool call JSON content."""
    text = re.sub(r'^\s*```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?\s*```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def _repair_json(json_str: str) -> Optional[dict]:
    """
    Attempt to repair common JSON issues from model output.

    Handles:
    - Trailing commas before ] or }
    - Truncated JSON strings
    """
    repaired = re.sub(r',\s*([}\]])', r'\1', json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Try to recover truncated JSON: find "sql" key content directly
    sql_match = re.search(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', repaired)
    if sql_match:
        return {
            "name": "submit_solution",
            "arguments": {
                "sql": sql_match.group(1)
            }
        }

    return None


def has_submit_solution_call(response: str) -> bool:
    """Check if trajectory contains submit_solution tool call."""
    if not response or not isinstance(response, str):
        return False
    pattern = r'<tool_call>\s*\{.*?"name"\s*:\s*"submit_solution"'
    return bool(re.search(pattern, response, re.IGNORECASE | re.DOTALL))


def extract_sql_from_solution(response: str) -> Optional[str]:
    """
    Extract SQL from the last submit_solution tool call.

    BIRD generation uses single "sql" string (not "sql_list"):
    <tool_call>{"name": "submit_solution", "arguments": {"sql": "SELECT ..."}}</tool_call>

    Returns:
        SQL string, or None if not found
    """
    if not response or not isinstance(response, str):
        return None

    tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    all_tool_calls = re.findall(tool_call_pattern, response, re.DOTALL | re.IGNORECASE)

    # Search from last to first for submit_solution
    for tool_call_content in reversed(all_tool_calls):
        tool_call_json = _strip_markdown_fences(tool_call_content)

        tool_data = None
        try:
            tool_data = json.loads(tool_call_json)
        except json.JSONDecodeError:
            tool_data = _repair_json(tool_call_json)

        if tool_data is None:
            continue

        if isinstance(tool_data, dict) and tool_data.get('name') == 'submit_solution':
            arguments = tool_data.get('arguments', {})
            if not isinstance(arguments, dict):
                continue

            # Primary: "sql" key (single string, matching BIRD prompt)
            sql = arguments.get('sql', '')
            if isinstance(sql, str) and sql.strip():
                return sql.strip()

            # Fallback: "sql_list" key (in case model uses array format)
            sql_list = arguments.get('sql_list', [])
            if isinstance(sql_list, list) and sql_list:
                return str(sql_list[0]).strip()

    return None


def extract_sql_from_last_execute(response: str) -> Optional[str]:
    """
    Fallback: extract SQL from the last execute_sql tool call.

    Used for truncated trajectories where submit_solution was never called.

    Returns:
        SQL string, or None if not found
    """
    if not response or not isinstance(response, str):
        return None

    tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    all_tool_calls = re.findall(tool_call_pattern, response, re.DOTALL | re.IGNORECASE)

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
            if not isinstance(arguments, dict):
                continue
            sql = arguments.get('sql', '')
            if isinstance(sql, str) and sql.strip():
                return sql.strip()

    return None


def count_tool_calls(response: str) -> Tuple[int, bool]:
    """
    Count total tool calls and detect execute_sql usage.

    Returns:
        Tuple of (num_tool_calls, has_execute_sql)
    """
    if not response or not isinstance(response, str):
        return 0, False

    all_calls = re.findall(r'<tool_call>', response, re.IGNORECASE)
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

    Returns:
        Tuple of (is_valid, error_message, has_submit_solution)
    """
    if not response or not response.strip():
        return False, "empty_response", False

    has_submit_call = has_submit_solution_call(response)
    sql = extract_sql_from_solution(response)

    if not sql:
        _, has_exec_sql = count_tool_calls(response)

        if not has_submit_call:
            if has_exec_sql:
                return False, "no_submit_but_has_execute_sql", False
            elif '<tool_call>' in response.lower():
                return False, "has_tool_calls_but_no_submit", False
            else:
                return False, "no_tool_calls", False
        else:
            return False, "tool_call_found_but_sql_extraction_failed", has_submit_call

    if len(sql.strip()) < 5:
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
    Compute rewards for agentic BIRD SQL generation trajectories.

    Scoring (reward shaping):
    - No tool calls at all: 0.0
    - Has execute_sql but no submit_solution (truncated): 0.05
    - Has submit_solution but JSON extraction failed: 0.02
    - SQL execution error / timeout: 0.0
    - SQL executes but wrong result (EX=0): 0.1
    - Correct result (EX=1): 1.0

    Args:
        data_sources: List of data source identifiers
        solution_strs: List of trajectory texts (full model output)
        ground_truths: List of ground truth dicts with 'ground_truth' key
        extra_infos: List of extra info dicts with 'db_id' key

    Returns:
        List of result dicts with scores
    """
    batch_size = len(solution_strs)

    assert len(data_sources) == batch_size
    assert len(ground_truths) == batch_size
    assert len(extra_infos) == batch_size

    results = [create_empty_result() for _ in range(batch_size)]

    print(f"[BIRD Agentic Reward] Processing batch of {batch_size} trajectories")

    # ==========================================
    # STAGE 1: Extract SQL from trajectories
    # ==========================================
    valid_sql_indices = []
    sql_map = {}  # index -> extracted SQL string

    for i in range(batch_size):
        num_calls, has_exec_sql = count_tool_calls(solution_strs[i])
        is_valid, error, has_submit_call = validate_trajectory_format(solution_strs[i])

        results[i]["format_valid"] = is_valid
        results[i]["format_error"] = error
        results[i]["has_submit_solution"] = has_submit_call

        if is_valid:
            sql = extract_sql_from_solution(solution_strs[i])
            results[i]["solution_sql"] = sql or ""
            if sql:
                valid_sql_indices.append(i)
                sql_map[i] = sql
            else:
                results[i]["score"] = 0.0
        elif has_submit_call:
            # Has submit_solution but JSON extraction failed
            results[i]["solution_sql"] = ""
            results[i]["score"] = 0.02
        elif has_exec_sql:
            # Has execute_sql but no submit_solution (truncated trajectory)
            # Try fallback to last execute_sql
            fallback_sql = extract_sql_from_last_execute(solution_strs[i])
            if fallback_sql:
                results[i]["solution_sql"] = fallback_sql
                valid_sql_indices.append(i)
                sql_map[i] = fallback_sql
                results[i]["score"] = 0.05  # Base score for truncated, may increase if EX=1
            else:
                results[i]["score"] = 0.05
        else:
            results[i]["score"] = 0.0

    print(f"[BIRD Agentic Reward] Stage 1: {len(valid_sql_indices)}/{batch_size} have extractable SQL")

    if not valid_sql_indices:
        return [_to_python(r) for r in results]

    # ==========================================
    # STAGE 2: Execute SQL and compute EX
    # ==========================================

    def process_instance(i: int) -> Tuple[int, Dict]:
        solution_sql = sql_map[i]

        # Get ground truth SQL
        if isinstance(ground_truths[i], dict):
            gt_sql = ground_truths[i].get('ground_truth', '')
        else:
            gt_sql = str(ground_truths[i])

        # Handle numpy arrays from parquet
        if hasattr(gt_sql, 'tolist'):
            gt_sql = gt_sql.tolist()
        if isinstance(gt_sql, list):
            gt_sql = str(gt_sql[0]) if gt_sql else ''

        db_id = extra_infos[i].get('db_id', '')

        exec_result = execute_single_instance(
            solution_sql=solution_sql,
            ground_truth_sql=gt_sql,
            db_id=db_id,
        )
        return i, exec_result

    print(f"[BIRD Agentic Reward] Stage 2: Executing {len(valid_sql_indices)} instances with {NUM_WORKERS} workers")

    completed = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_instance, i): i for i in valid_sql_indices}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                idx, exec_result = future.result(timeout=120)
                results[idx].update(exec_result)

                was_truncated = not results[idx]["has_submit_solution"]

                if exec_result.get("timeout"):
                    results[idx]["score"] = 0.0
                elif not exec_result.get("execution_success"):
                    results[idx]["score"] = 0.0
                elif exec_result["ex_score"] == 1:
                    results[idx]["score"] = 1.0
                else:
                    # SQL executed but wrong result
                    if was_truncated:
                        results[idx]["score"] = 0.05  # Keep truncated base
                    else:
                        results[idx]["score"] = 0.1
            except Exception as e:
                results[idx]["error_message"] = f"Worker error: {str(e)}"

            completed += 1
            if completed % 100 == 0 or completed == len(valid_sql_indices):
                print(f"[BIRD Agentic Reward]   Progress: {completed}/{len(valid_sql_indices)}")

    # ==========================================
    # SUMMARY
    # ==========================================
    avg_score = sum(r["score"] for r in results) / batch_size
    has_submit_count = sum(1 for r in results if r.get("has_submit_solution", False))
    executed_count = sum(1 for r in results if r["execution_success"])
    correct_count = sum(1 for r in results if r["ex_score"] == 1)
    timeout_count = sum(1 for r in results if r["timeout"])

    print(f"[BIRD Agentic Reward] Summary:")
    print(f"  - Total trajectories: {batch_size}")
    print(f"  - With submit_solution: {has_submit_count} ({has_submit_count/batch_size:.1%})")
    print(f"  - Extractable SQL: {len(valid_sql_indices)} ({len(valid_sql_indices)/batch_size:.1%})")
    print(f"  - Executed successfully: {executed_count} ({executed_count/batch_size:.1%})")
    print(f"  - Correct (EX=1): {correct_count} ({correct_count/batch_size:.1%})")
    print(f"  - Timeout: {timeout_count}")
    print(f"  - Avg score: {avg_score:.3f}")

    return [_to_python(r) for r in results]


# ============================================================
# VERL COMPATIBILITY - Entry point
# ============================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
    **kwargs
) -> dict:
    """Single instance entry point."""
    results = compute_score_batch(
        data_sources=[data_source],
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        extra_infos=[extra_info],
        **kwargs
    )
    return results[0]
