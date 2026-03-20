"""
BIRD SQL Generation Reward Function: Execution Accuracy (EX)

Scoring:
- Parse model response for <thought> and <solution> tags
- Execute predicted SQL and ground truth SQL on the database
- Compare execution results: set(predicted) == set(ground_truth)
- Score: 1.0 if match, 0.0 otherwise

All BIRD SQL is SELECT-only, so no ephemeral databases needed.
Uses func_timeout for SQL execution timeout (same as official BIRD evaluation).
"""

import re
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from func_timeout import func_timeout, FunctionTimedOut

# ============================================================
# CONFIGURATION
# ============================================================

# Base directory for BIRD databases
DB_DIR = os.environ.get("BIRD_DB_DIR", "")

# Number of parallel workers for batch processing
NUM_WORKERS = int(os.environ.get("BIRD_REWARD_NUM_WORKERS", "8"))

# Timeout for SQL execution (seconds)
SQL_TIMEOUT = int(os.environ.get("BIRD_SQL_TIMEOUT", "30"))

# ============================================================
# RESPONSE PARSING
# ============================================================

def parse_model_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse model response to extract thought and solution.

    Returns:
        Tuple of (thought, solution_sql)
    """
    if not response or not isinstance(response, str):
        return None, None

    # Extract thought
    thought_match = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL | re.IGNORECASE)
    thought = thought_match.group(1).strip() if thought_match else None

    # Extract solution
    solution_match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL | re.IGNORECASE)
    solution = solution_match.group(1).strip() if solution_match else None

    # Remove markdown code fences if present
    if solution:
        solution = re.sub(r'^```(?:sql)?\s*\n', '', solution, flags=re.MULTILINE)
        solution = re.sub(r'\n```\s*$', '', solution, flags=re.MULTILINE)
        solution = solution.strip()

    return thought, solution


def validate_response_format(response: str) -> Tuple[bool, str]:
    """Validate that response has correct format."""
    if not response or not response.strip():
        return False, "empty_response"

    if '<thought>' not in response.lower():
        return False, "missing_thought_tag"
    if '</thought>' not in response.lower():
        return False, "missing_thought_closing_tag"
    if '<solution>' not in response.lower():
        return False, "missing_solution_tag"
    if '</solution>' not in response.lower():
        return False, "missing_solution_closing_tag"

    thought, solution = parse_model_response(response)

    if not thought or len(thought.strip()) < 10:
        return False, "thought_too_short"
    if not solution or len(solution.strip()) < 5:
        return False, "solution_too_short"

    return True, ""


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
# SQL EXECUTION WITH TIMEOUT (func_timeout, same as official BIRD eval)
# ============================================================

def _execute_sql_pair(predicted_sql: str, ground_truth_sql: str, db_path: str) -> Tuple[int, str]:
    """
    Execute both predicted and ground truth SQL, compare results.
    Called within func_timeout (same as official BIRD evaluation).

    Returns:
        Tuple of (ex_score, error_message)
    """
    predicted_res = []
    ground_truth_res = []
    error_message = ""

    # Execute predicted SQL
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        conn.close()
    except Exception as e:
        error_message = f"Predicted SQL Error: {str(e)}"
        predicted_res = []

    # Execute ground truth SQL
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

    # Round results for consistent comparison (precision=10, same as official)
    predicted_res = round_results(predicted_res)
    ground_truth_res = round_results(ground_truth_res)

    res = calculate_ex(predicted_res, ground_truth_res)
    return res, error_message


# ============================================================
# SINGLE INSTANCE EXECUTION
# ============================================================

def execute_single_instance(
    solution_sql: str,
    ground_truth_sql: str,
    db_id: str,
) -> Dict:
    """
    Execute predicted and ground truth SQL, compare results.
    Single func_timeout wraps both executions (same as official BIRD eval).

    Returns:
        Result dict with score and execution details.
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
# BATCH REWARD FUNCTION
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


def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[dict],
    **kwargs
) -> List[dict]:
    """
    Compute rewards for a batch of BIRD SQL generation instances.

    Scoring: EX = 1.0 if set(pred_results) == set(gt_results), else 0.0

    Args:
        data_sources: List of data source identifiers
        solution_strs: List of model responses
        ground_truths: List of ground truth dicts with 'ground_truth' key
        extra_infos: List of extra info dicts with 'db_id' key

    Returns:
        List of result dicts with 'score' field
    """
    batch_size = len(solution_strs)

    assert len(data_sources) == batch_size
    assert len(ground_truths) == batch_size
    assert len(extra_infos) == batch_size

    results = [{
        "score": 0.0,
        "format_valid": False,
        "format_error": "",
        "thought": "",
        "solution_sql": "",
        "execution_success": False,
        "ex_score": 0,
        "error_message": "",
        "timeout": False,
    } for _ in range(batch_size)]

    print(f"[BIRD Reward] Processing batch of {batch_size} instances")

    # ==========================================
    # STAGE 1: Parse and validate format
    # ==========================================
    valid_indices = []

    for i in range(batch_size):
        is_valid, error = validate_response_format(solution_strs[i])
        results[i]["format_valid"] = is_valid
        results[i]["format_error"] = error

        if is_valid:
            thought, solution = parse_model_response(solution_strs[i])
            results[i]["thought"] = thought or ""
            results[i]["solution_sql"] = solution or ""
            valid_indices.append(i)

    print(f"[BIRD Reward] Stage 1: {len(valid_indices)}/{batch_size} valid format")

    if not valid_indices:
        return [_to_python(r) for r in results]

    # ==========================================
    # STAGE 2: Execute SQL and compute EX
    # ==========================================

    def process_instance(i: int) -> Tuple[int, Dict]:
        solution_sql = results[i]["solution_sql"]

        # Get ground truth
        if isinstance(ground_truths[i], dict):
            gt_sql = ground_truths[i].get('ground_truth', '')
        else:
            gt_sql = str(ground_truths[i])

        db_id = extra_infos[i].get('db_id', '')

        exec_result = execute_single_instance(
            solution_sql=solution_sql,
            ground_truth_sql=gt_sql,
            db_id=db_id,
        )
        return i, exec_result

    print(f"[BIRD Reward] Stage 2: Executing {len(valid_indices)} instances with {NUM_WORKERS} workers")

    completed = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_instance, i): i for i in valid_indices}

        for future in as_completed(futures):
            try:
                idx, exec_result = future.result(timeout=120)
                results[idx].update(exec_result)
                results[idx]["score"] = float(exec_result["ex_score"])
            except Exception as e:
                idx = futures[future]
                results[idx]["error_message"] = f"Worker error: {str(e)}"

            completed += 1
            if completed % 100 == 0 or completed == len(valid_indices):
                print(f"[BIRD Reward]   Progress: {completed}/{len(valid_indices)}")

    # ==========================================
    # SUMMARY
    # ==========================================
    avg_score = sum(r["score"] for r in results) / batch_size
    executed = sum(1 for r in results if r["execution_success"])
    correct = sum(1 for r in results if r["ex_score"] == 1)
    timeout_count = sum(1 for r in results if r["timeout"])

    print(f"[BIRD Reward] Summary:")
    print(f"  - Total: {batch_size}")
    print(f"  - Valid format: {len(valid_indices)} ({len(valid_indices)/batch_size:.1%})")
    print(f"  - Executed: {executed} ({executed/batch_size:.1%})")
    print(f"  - Correct (EX=1): {correct} ({correct/batch_size:.1%})")
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
