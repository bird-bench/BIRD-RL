#!/usr/bin/env python3
"""
Execute SQL queries and collect observations for multi-turn inference.

Reads parsed responses with pred_sqls, executes them against SQLite databases,
and returns execution results as observations for the next turn.
"""

import json
import sqlite3
import os
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def _format_rows_as_json(col_names, rows, max_rows=50):
    """Format query results as readable JSON-like string."""
    if not rows:
        return "(No results)"
    display_rows = rows[:max_rows]
    lines = []
    for row in display_rows:
        row_dict = {col: (str(v) if v is not None else "NULL") for col, v in zip(col_names, row)}
        lines.append(json.dumps(row_dict, ensure_ascii=False))
    result = "\n".join(lines)
    if len(rows) > max_rows:
        result += f"\n... ({len(rows)} total rows, showing first {max_rows})"
    return result


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


def execute_sql_safe(sql: str, db_path: str, timeout: int = 30) -> dict:
    """Execute SQL with timeout, returning result dict."""
    if not sql or not sql.strip():
        return {"exec_flag": False, "exec_results": "Error: Empty SQL"}

    if not os.path.exists(db_path):
        return {"exec_flag": False, "exec_results": f"Error: Database not found: {db_path}"}

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
        return {"exec_flag": False, "exec_results": f"Error: Query timed out after {timeout}s"}

    try:
        if not queue.empty():
            status, *rest = queue.get_nowait()
            if status == "success":
                col_names, rows = rest
                formatted = _format_rows_as_json(col_names, rows)
                return {"exec_flag": True, "exec_results": formatted}
            else:
                return {"exec_flag": False, "exec_results": f"Error: {rest[0]}"}
    except Exception as e:
        return {"exec_flag": False, "exec_results": f"Error: {str(e)}"}

    return {"exec_flag": False, "exec_results": "Error: No result returned"}


def process_single_instance(item: dict, db_dir: str, timeout: int = 30) -> dict:
    """Process a single parsed instance: execute its SQL and collect observation."""
    db_id = item.get("db_id", "")
    pred_sqls = item.get("pred_sqls", [])
    end_flag = item.get("end_flag", False)

    # If submit_solution, no need to execute (reward function handles it)
    if end_flag:
        item["exec_flag"] = True
        item["exec_results"] = ""
        return item

    # Execute first SQL from pred_sqls
    sql = pred_sqls[0] if pred_sqls else ""
    if not sql:
        item["exec_flag"] = False
        item["exec_results"] = "Error: No SQL to execute"
        return item

    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    result = execute_sql_safe(sql, db_path, timeout=timeout)
    item.update(result)
    return item


def process_observations(input_path: str, output_path: str, db_dir: str, num_threads: int = 8, timeout: int = 30):
    """Process all parsed responses: execute SQL and collect observations."""
    print(f"Loading parsed responses from: {input_path}")
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"  Loaded {len(data)} instances")

    results = [None] * len(data)
    completed = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_single_instance, item, db_dir, timeout): i for i, item in enumerate(data)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result(timeout=120)
            except Exception as e:
                data[idx]["exec_flag"] = False
                data[idx]["exec_results"] = f"Error: {str(e)}"
                results[idx] = data[idx]
            completed += 1
            if completed % 100 == 0 or completed == len(data):
                print(f"  Progress: {completed}/{len(data)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    exec_count = sum(1 for r in results if r.get("exec_flag"))
    print(f"  Executed {exec_count}/{len(results)} successfully")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Execute SQL and collect observations")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL (parsed responses)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL (with observations)")
    parser.add_argument("--db-dir", type=str, required=True, help="Database directory")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--timeout", type=int, default=30, help="SQL timeout in seconds")
    args = parser.parse_args()

    process_observations(args.input, args.output, args.db_dir, args.threads, args.timeout)


if __name__ == "__main__":
    main()
