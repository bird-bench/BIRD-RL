"""
Hybrid Reward Function: Dispatches between critic and bird reward evaluation.

Routes instances by data_source:
- bird_critic/*  -> critic reward (test case execution)
- bird/* (other) -> bird reward (EX metric: set equality of query results)

Important: VERL validation asserts that every key appears in ALL result dicts
or NONE of them. Since critic and bird return different keys, we normalize
all results to a union key set with sensible defaults for missing fields.
"""

import os
from typing import Any, List

from bird_rl.rewards import critic_reward
from bird_rl.rewards import bird_reward

# Union of all keys from both reward functions, with default values.
# Critic-only: syntax_error, test_passed, test_pass_count, test_total_count, test_pass_rate, is_read_only
# Bird-only: ex_score
_ALL_KEYS_DEFAULTS = {
    "score": 0.0,
    "format_valid": False,
    "format_error": "",
    "thought": "",
    "solution_sql": "",
    "execution_success": False,
    "syntax_error": False,
    "test_passed": False,
    "test_pass_count": 0,
    "test_total_count": 0,
    "test_pass_rate": 0.0,
    "error_message": "",
    "is_read_only": False,
    "timeout": False,
    "ex_score": 0,
}


def _normalize_result(result: dict) -> dict:
    """Ensure result dict has all keys from the union set."""
    for key, default in _ALL_KEYS_DEFAULTS.items():
        if key not in result:
            result[key] = default
    return result


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

    # Split indices by task type
    critic_indices = []
    bird_indices = []

    for i in range(batch_size):
        ds = data_sources[i]
        if ds.startswith("bird_critic"):
            critic_indices.append(i)
        else:
            bird_indices.append(i)

    print(f"[Hybrid Reward] Batch {batch_size}: critic={len(critic_indices)}, bird={len(bird_indices)}")

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
        b_results = bird_reward.compute_score_batch(
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
