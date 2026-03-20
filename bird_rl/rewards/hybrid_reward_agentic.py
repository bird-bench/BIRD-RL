"""
Hybrid Agentic Reward Function: Dispatches between critic and bird reward evaluation.

Routes instances by data_source:
- bird_critic/*  -> critic reward (test case execution via DB pool)
- bird/* (other) -> bird reward (EX metric: set equality of query results)

Both reward functions extract SQL from submit_solution tool calls with sql_list format.

Important: The critic reward uses sql_list natively. The bird reward needs to handle
sql_list extraction (unified format) instead of the old single "sql" string format.

Key: VERL validation asserts every key appears in ALL result dicts or NONE.
Since critic and bird return different keys, we normalize all results to
a union key set with sensible defaults for missing fields.
"""

import os
import re
import json
import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ============================================================
# Import both reward modules
# ============================================================

# Critic reward (test case execution via DB pool)
from bird_rl.rewards import critic_reward_agentic as critic_reward

# Bird reward needs adaptation for sql_list format
from bird_rl.rewards import bird_reward_agentic as bird_reward_base

# ============================================================
# Result normalization (union of all keys from both rewards)
# ============================================================
# Critic-only: syntax_error, test_passed, test_pass_count, test_total_count, test_pass_rate, is_read_only
# Bird-only: ex_score
_ALL_KEYS_DEFAULTS = {
    "score": 0.0,
    "format_valid": False,
    "format_error": "",
    "has_submit_solution": False,
    "solution_sql": "",
    "execution_success": False,
    "syntax_error": False,
    "test_passed": False,
    "test_pass_count": 0,
    "test_total_count": 0,
    "test_pass_rate": 0.0,
    "is_read_only": False,
    "timeout": False,
    "error_message": "",
    "ex_score": 0,
}


def _normalize_result(result: dict) -> dict:
    """Ensure result dict has all keys from the union set."""
    for key, default in _ALL_KEYS_DEFAULTS.items():
        if key not in result:
            result[key] = default
    return result


# ============================================================
# Bird reward wrapper: adapt sql_list extraction
# ============================================================

def _extract_sql_list_from_solution(response: str) -> Optional[List[str]]:
    """
    Extract SQL from submit_solution tool call using sql_list format.

    Handles both:
    - sql_list: ["SELECT ..."] (unified format)
    - sql: "SELECT ..." (legacy format, for backwards compatibility)
    """
    if not response or not isinstance(response, str):
        return None

    tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    all_tool_calls = re.findall(tool_call_pattern, response, re.DOTALL | re.IGNORECASE)

    for tool_call_content in reversed(all_tool_calls):
        tool_call_json = bird_reward_base._strip_markdown_fences(tool_call_content)

        tool_data = None
        try:
            tool_data = json.loads(tool_call_json)
        except json.JSONDecodeError:
            tool_data = bird_reward_base._repair_json(tool_call_json)

        if tool_data is None:
            continue

        if isinstance(tool_data, dict) and tool_data.get('name') == 'submit_solution':
            arguments = tool_data.get('arguments', {})
            if not isinstance(arguments, dict):
                continue

            # Primary: sql_list (unified format)
            sql_list = arguments.get('sql_list', [])
            if isinstance(sql_list, list) and sql_list:
                return [str(s).strip() for s in sql_list if s]

            # Fallback: sql (legacy single string)
            sql = arguments.get('sql', '')
            if isinstance(sql, str) and sql.strip():
                return [sql.strip()]

    return None


def compute_bird_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[dict],
    **kwargs
) -> List[dict]:
    """
    Bird agentic reward with sql_list extraction support.

    Uses the same EX metric and reward shaping as bird_reward_fn_agentic,
    but extracts SQL from sql_list format instead of single sql string.
    """
    batch_size = len(solution_strs)
    results = [bird_reward_base.create_empty_result() for _ in range(batch_size)]

    print(f"[Hybrid Bird Agentic Reward] Processing batch of {batch_size} trajectories")

    # Stage 1: Extract SQL from trajectories (using sql_list format)
    valid_sql_indices = []
    sql_map = {}

    for i in range(batch_size):
        num_calls, has_exec_sql = bird_reward_base.count_tool_calls(solution_strs[i])
        has_submit_call = bird_reward_base.has_submit_solution_call(solution_strs[i])

        # Try sql_list extraction first
        sql_list = _extract_sql_list_from_solution(solution_strs[i])
        sql = sql_list[0] if sql_list else None

        # Fallback to original bird extraction (handles legacy "sql" key)
        if not sql:
            sql = bird_reward_base.extract_sql_from_solution(solution_strs[i])

        is_valid = sql is not None and len(sql.strip()) >= 5
        results[i]["format_valid"] = is_valid
        results[i]["has_submit_solution"] = has_submit_call

        if is_valid:
            results[i]["solution_sql"] = sql
            valid_sql_indices.append(i)
            sql_map[i] = sql
        elif has_submit_call:
            results[i]["format_error"] = "tool_call_found_but_sql_extraction_failed"
            results[i]["score"] = 0.02
        elif has_exec_sql:
            fallback_sql = bird_reward_base.extract_sql_from_last_execute(solution_strs[i])
            if fallback_sql:
                results[i]["solution_sql"] = fallback_sql
                valid_sql_indices.append(i)
                sql_map[i] = fallback_sql
                results[i]["score"] = 0.05
            else:
                results[i]["format_error"] = "no_submit_but_has_execute_sql"
                results[i]["score"] = 0.05
        else:
            results[i]["format_error"] = "no_tool_calls"
            results[i]["score"] = 0.0

    print(f"[Hybrid Bird Agentic Reward] Stage 1: {len(valid_sql_indices)}/{batch_size} have extractable SQL")

    if not valid_sql_indices:
        return [bird_reward_base._to_python(r) for r in results]

    # Stage 2: Execute SQL and compute EX (reuse bird's execution logic)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_instance(i):
        solution_sql = sql_map[i]

        if isinstance(ground_truths[i], dict):
            gt_sql = ground_truths[i].get('ground_truth', '')
        else:
            gt_sql = ground_truths[i]

        # Handle numpy arrays from parquet (e.g. array(["SELECT ..."]))
        if hasattr(gt_sql, 'tolist'):
            gt_sql = gt_sql.tolist()
        if isinstance(gt_sql, (list, tuple)):
            gt_sql = str(gt_sql[0]) if gt_sql else ''
        elif not isinstance(gt_sql, str):
            gt_sql = str(gt_sql)

        db_id = extra_infos[i].get('db_id', '')

        exec_result = bird_reward_base.execute_single_instance(
            solution_sql=solution_sql,
            ground_truth_sql=gt_sql,
            db_id=db_id,
        )
        return i, exec_result

    num_workers = bird_reward_base.NUM_WORKERS
    print(f"[Hybrid Bird Agentic Reward] Stage 2: Executing {len(valid_sql_indices)} instances with {num_workers} workers")

    completed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
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
                    if was_truncated:
                        results[idx]["score"] = 0.05
                    else:
                        results[idx]["score"] = 0.1
            except Exception as e:
                results[idx]["error_message"] = f"Worker error: {str(e)}"

            completed += 1
            if completed % 100 == 0 or completed == len(valid_sql_indices):
                print(f"[Hybrid Bird Agentic Reward]   Progress: {completed}/{len(valid_sql_indices)}")

    # Summary
    avg_score = sum(r["score"] for r in results) / batch_size
    has_submit_count = sum(1 for r in results if r.get("has_submit_solution", False))
    executed_count = sum(1 for r in results if r["execution_success"])
    correct_count = sum(1 for r in results if r["ex_score"] == 1)

    print(f"[Hybrid Bird Agentic Reward] Summary:")
    print(f"  - Total: {batch_size}")
    print(f"  - With submit_solution: {has_submit_count} ({has_submit_count/batch_size:.1%})")
    print(f"  - Extractable SQL: {len(valid_sql_indices)} ({len(valid_sql_indices)/batch_size:.1%})")
    print(f"  - Executed: {executed_count} ({executed_count/batch_size:.1%})")
    print(f"  - Correct (EX=1): {correct_count} ({correct_count/batch_size:.1%})")
    print(f"  - Avg score: {avg_score:.3f}")

    return [bird_reward_base._to_python(r) for r in results]


# ============================================================
# Dispatcher
# ============================================================

def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[dict],
    **kwargs
) -> List[dict]:
    """
    Dispatch reward computation based on data_source.

    Splits the batch into critic and bird subsets, evaluates each with
    the appropriate reward function, then merges results back in order.
    """
    batch_size = len(solution_strs)
    results = [None] * batch_size

    critic_indices = []
    bird_indices = []

    for i in range(batch_size):
        ds = data_sources[i]
        if ds.startswith("bird_critic"):
            critic_indices.append(i)
        else:
            bird_indices.append(i)

    print(f"[Hybrid Agentic Reward] Batch {batch_size}: critic={len(critic_indices)}, bird={len(bird_indices)}")

    # Process critic instances
    if critic_indices:
        c_results = critic_reward.compute_score_batch(
            data_sources=[data_sources[i] for i in critic_indices],
            solution_strs=[solution_strs[i] for i in critic_indices],
            ground_truths=[ground_truths[i] for i in critic_indices],
            extra_infos=[extra_infos[i] for i in critic_indices],
            **kwargs,
        )
        for idx, i in enumerate(critic_indices):
            results[i] = _normalize_result(c_results[idx])

    # Process bird instances
    if bird_indices:
        b_results = compute_bird_score_batch(
            data_sources=[data_sources[i] for i in bird_indices],
            solution_strs=[solution_strs[i] for i in bird_indices],
            ground_truths=[ground_truths[i] for i in bird_indices],
            extra_infos=[extra_infos[i] for i in bird_indices],
            **kwargs,
        )
        for idx, i in enumerate(bird_indices):
            results[i] = _normalize_result(b_results[idx])

    return results


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
    **kwargs
) -> dict:
    """Single instance entry point."""
    return compute_score_batch(
        data_sources=[data_source],
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        extra_infos=[extra_info],
        **kwargs,
    )[0]
