#!/usr/bin/env python3
"""
Evaluate BIRD inference results using Execution Accuracy (EX) metric.

Extracts the final SQL from trajectories and compares execution results
against ground truth SQL on the dev databases.

Handles both formats:
- sql (string): from BIRD-specialized model
- sql_list (array): from hybrid model
"""

import json
import sqlite3
import os
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def _execute_sql_worker(queue, sql, db_path):
    """Worker function for multiprocessing-based timeout."""
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        cursor.execute(sql)
        col_names = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()
        queue.put(("success", col_names, rows))
    except Exception as e:
        queue.put(("error", str(e), None))


def execute_sql_with_timeout(sql: str, db_path: str, timeout: int = 30):
    """Execute SQL with timeout, returning (success, col_names, rows)."""
    if not sql or not sql.strip():
        return False, [], []

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_execute_sql_worker, args=(queue, sql, db_path))
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        return False, [], []

    try:
        if not queue.empty():
            status, *rest = queue.get_nowait()
            if status == "success":
                return True, rest[0], rest[1]
    except Exception:
        pass

    return False, [], []


def normalize_result(rows):
    """Normalize query results for comparison."""
    if not rows:
        return set()
    normalized = set()
    for row in rows:
        normalized_row = tuple(
            round(v, 10) if isinstance(v, float) else str(v).strip().lower() if isinstance(v, str) else v
            for v in row
        )
        normalized.add(normalized_row)
    return normalized


def extract_final_sql(traj_item: dict) -> str:
    """Extract the final submitted SQL from a trajectory item."""
    trajectory = traj_item.get("trajectory", [])
    if not trajectory:
        return ""

    last_turn = trajectory[-1]
    if not last_turn.get("end_flag", False):
        return ""

    # Parse the action to extract SQL
    action = last_turn.get("action", "")
    if not action:
        return ""

    import re
    match = re.search(r"<tool_call>(.*?)</tool_call>", action, re.DOTALL)
    if not match:
        return ""

    try:
        tool_data = json.loads(match.group(1))
    except json.JSONDecodeError:
        # Try double-brace fix
        fixed = match.group(1).replace("{{", "{").replace("}}", "}")
        try:
            tool_data = json.loads(fixed)
        except json.JSONDecodeError:
            return ""

    if not isinstance(tool_data, dict):
        return ""

    args = tool_data.get("arguments", {})
    if not isinstance(args, dict):
        return ""

    # Handle sql_list (hybrid) or sql (bird)
    sql_list = args.get("sql_list", [])
    if isinstance(sql_list, list) and sql_list:
        return str(sql_list[0]).strip()

    sql = args.get("sql", "")
    if isinstance(sql, str) and sql.strip():
        return sql.strip()

    return ""


def evaluate_single(pred_sql: str, gold_sql: str, db_path: str, timeout: int = 30) -> bool:
    """Compare predicted SQL against ground truth using execution accuracy."""
    if not pred_sql:
        return False

    pred_ok, _, pred_rows = execute_sql_with_timeout(pred_sql, db_path, timeout)
    if not pred_ok:
        return False

    gold_ok, _, gold_rows = execute_sql_with_timeout(gold_sql, db_path, timeout)
    if not gold_ok:
        return False

    return normalize_result(pred_rows) == normalize_result(gold_rows)


def evaluate_trajectories(
    traj_path: str,
    gold_path: str,
    db_dir: str,
    output_path: str,
    num_threads: int = 8,
    timeout: int = 30,
):
    """
    Evaluate trajectories against ground truth.

    Args:
        traj_path: Path to trajectory JSONL (from build_trajectory)
        gold_path: Path to gold data (JSON array with question_id, SQL, db_id)
        db_dir: Path to database directory
        output_path: Path to write evaluation results
        num_threads: Number of parallel threads
        timeout: SQL execution timeout in seconds
    """
    # Load trajectories
    print(f"Loading trajectories from: {traj_path}")
    traj_dict = {}
    with open(traj_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                traj_dict[item.get("instance_idx")] = item

    # Load gold data
    print(f"Loading gold data from: {gold_path}")
    with open(gold_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    print(f"  {len(traj_dict)} trajectories, {len(gold_data)} gold instances")

    results = []
    tasks = []

    for instance_idx, item in enumerate(gold_data):
        db_id = item.get("db_id", "")
        gold_sql = item.get("SQL", item.get("sql", ""))
        question_id = item.get("question_id", instance_idx)

        traj_item = traj_dict.get(instance_idx, {})
        pred_sql = extract_final_sql(traj_item)

        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

        tasks.append({
            "instance_idx": instance_idx,
            "question_id": question_id,
            "db_id": db_id,
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "db_path": db_path,
        })

    correct = 0
    total = len(tasks)
    eval_results = [None] * total

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for i, task in enumerate(tasks):
            future = executor.submit(
                evaluate_single,
                task["pred_sql"], task["gold_sql"], task["db_path"], timeout
            )
            futures[future] = i

        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            task = tasks[idx]
            try:
                is_correct = future.result(timeout=120)
            except Exception:
                is_correct = False

            if is_correct:
                correct += 1

            eval_results[idx] = {
                "instance_idx": task["instance_idx"],
                "question_id": task["question_id"],
                "db_id": task["db_id"],
                "pred_sql": task["pred_sql"],
                "gold_sql": task["gold_sql"],
                "correct": is_correct,
            }

            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  Progress: {completed}/{total} (correct so far: {correct})")

    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"BIRD Evaluation Results")
    print(f"{'=' * 60}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Execution Accuracy (EX): {accuracy:.2f}%")
    print(f"{'=' * 60}")

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": eval_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_path}")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate BIRD inference with EX metric')
    parser.add_argument('--trajectory', type=str, required=True, help='Trajectory JSONL file')
    parser.add_argument('--gold', type=str, required=True, help='Gold data JSON file (mini_dev.json)')
    parser.add_argument('--db-dir', type=str, required=True, help='Database directory')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--threads', type=int, default=8, help='Number of threads')
    parser.add_argument('--timeout', type=int, default=30, help='SQL timeout in seconds')
    args = parser.parse_args()

    evaluate_trajectories(
        traj_path=args.trajectory,
        gold_path=args.gold,
        db_dir=args.db_dir,
        output_path=args.output,
        num_threads=args.threads,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
