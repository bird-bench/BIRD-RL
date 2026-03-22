#!/usr/bin/env python3
"""
Execute SQL queries and collect observations for SFT data generation.

This script:
1. Reads parsed turn data with pred_sqls and tool_name
2. Loads preprocess_sql from train.jsonl for each instance
3. Detects if SQL is read-only or modifying
4. For read-only SQL without preprocess: uses template database directly
5. For modifying SQL or with preprocess_sql: creates ephemeral copy
6. Executes SQL queries with 60-second timeout using multiprocessing
7. Returns execution results as JSON (only final statement's result)

Two-Tool Design:
- execute_sql: Single SQL query (pred_sqls = ["SELECT ..."])
- submit_solution: Multiple SQL statements (pred_sqls = ["CREATE ...", "INSERT ...", "SELECT ..."])

Usage:
    python -m bird_rl.data.execute_sql_observations \
        --turn 0 \
        --input <parsed.jsonl> \
        --output <observations.jsonl> \
        --train-data <train.jsonl> \
        --db-dir <database_directory>
"""

import argparse
import json
import multiprocessing
import os
import re
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


# SQL keywords that modify the database
_MODIFYING_SQL_KEYWORDS = {
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
    'TRUNCATE', 'REPLACE', 'MERGE', 'UPSERT', 'GRANT', 'REVOKE'
}


def is_read_only_sql(*sql_parts) -> bool:
    """Check if all SQL parts are read-only (no modifications to DB)."""
    for part in sql_parts:
        if part is None:
            continue
        if isinstance(part, (list, tuple)):
            for sql in part:
                if sql and not is_read_only_sql(sql):
                    return False
            continue
        if not isinstance(part, str) or not part.strip():
            continue

        sql_upper = part.upper()
        sql_upper = re.sub(r'--[^\n]*', '', sql_upper)
        sql_upper = re.sub(r'/\*.*?\*/', '', sql_upper, flags=re.DOTALL)

        for stmt in sql_upper.split(';'):
            stmt = stmt.strip()
            if not stmt:
                continue
            words = stmt.split()
            if not words:
                continue
            first_word = words[0]
            if first_word in _MODIFYING_SQL_KEYWORDS:
                return False
            if first_word == 'WITH':
                for kw in _MODIFYING_SQL_KEYWORDS:
                    if kw in stmt:
                        return False
    return True


def _format_rows_as_json(col_names: List[str], rows: List[tuple]) -> str:
    """Format query results as JSON (list of dictionaries)."""
    if not col_names:
        return json.dumps([list(row) for row in rows], ensure_ascii=False, indent=2)

    result_list = []
    for row in rows:
        row_dict = {}
        for i, col_name in enumerate(col_names):
            value = row[i] if i < len(row) else None
            row_dict[col_name] = value
        result_list.append(row_dict)

    return json.dumps(result_list, ensure_ascii=False, indent=2)


def _execute_sql_in_process(queue, sql_list, db_path, is_read_only, preprocess_sql):
    """Worker function that runs in a separate process."""
    try:
        result = _execute_sql_worker(sql_list, db_path, is_read_only, preprocess_sql)
        queue.put(('success', result))
    except Exception as e:
        queue.put(('error', str(e)))


def _execute_sql_worker(sql_list, db_path, is_read_only, preprocess_sql=None):
    """Execute SQL queries and return results or error."""
    try:
        if is_read_only:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
            conn.execute("PRAGMA busy_timeout = 30000")
        else:
            conn = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True, timeout=30.0)
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = OFF")

        conn.execute('PRAGMA foreign_keys = ON')
        cursor = conn.cursor()

        if preprocess_sql:
            try:
                for sql in preprocess_sql:
                    if sql and sql.strip():
                        cursor.execute(sql)
                conn.commit()
            except sqlite3.Error:
                conn.rollback()

        final_result = None
        for sql in sql_list:
            if sql and sql.strip() and sql.strip() != "[MISS]":
                try:
                    cursor.execute(sql)
                    try:
                        rows = cursor.fetchall()
                        if rows:
                            col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                            final_result = _format_rows_as_json(col_names, rows)
                        else:
                            final_result = "(No results)"
                    except sqlite3.Error:
                        if not sql.strip().upper().startswith(('SELECT', 'PRAGMA', 'WITH')):
                            affected = cursor.rowcount if cursor.rowcount >= 0 else 0
                            final_result = f"(Executed successfully: {affected} rows affected)"
                        else:
                            final_result = "(No results)"
                except sqlite3.Error as e:
                    conn.rollback()
                    conn.close()
                    return False, f"SQL Error: {str(e)}"

        conn.commit()
        conn.close()

        if final_result is not None:
            if len(final_result) > 1000:
                final_result = final_result[:1000] + "\n... (truncated)"
            return True, final_result
        else:
            return True, "(No output)"

    except sqlite3.Error as e:
        return False, f"Connection Error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected Error: {type(e).__name__}: {str(e)}"


def execute_sql_safe(sql_list, db_path, db_id, db_dir, is_read_only, preprocess_sql=None, timeout=60):
    """Execute SQL queries with timeout using multiprocessing."""
    needs_reset = not is_read_only or bool(preprocess_sql)

    if needs_reset:
        template_path = f"{db_dir}/{db_id}/{db_id}_template.sqlite"
        if not os.path.exists(template_path):
            return False, f"Template database not found: {template_path}"

        try:
            for suffix in ['-wal', '-shm', '-journal']:
                path_to_remove = db_path + suffix
                if os.path.exists(path_to_remove):
                    try:
                        os.remove(path_to_remove)
                    except OSError:
                        pass

            if os.path.exists(db_path):
                os.remove(db_path)
            shutil.copy2(template_path, db_path)
        except Exception as e:
            return False, f"Failed to reset database: {str(e)}"

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_execute_sql_in_process,
        args=(queue, sql_list, db_path, is_read_only, preprocess_sql)
    )

    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        return False, f"Execution timeout after {timeout} seconds"

    try:
        if not queue.empty():
            status, result = queue.get_nowait()
            if status == 'success':
                return result
            else:
                return False, f"Execution exception: {result}"
        else:
            return False, "Process finished but no result returned"
    except Exception as e:
        return False, f"Error getting result: {str(e)}"


def process_single_instance(item, db_dir, preprocess_sql_dict, lock, worker_id):
    """Process a single instance: execute SQL and collect observation."""
    instance_idx = item.get('instance_idx')
    pred_sqls = item.get('pred_sqls', [])

    if not pred_sqls or pred_sqls == ["[MISS]"]:
        item['exec_flag'] = False
        item['exec_results'] = "Error: No SQL to execute ([MISS])"
        return item

    db_id = item.get('db_id', '')
    if not db_id:
        item['exec_flag'] = False
        item['exec_results'] = f"Error: Database not found for instance {item.get('instance_id', '')}"
        return item

    preprocess_sql = preprocess_sql_dict.get(instance_idx, [])
    read_only = is_read_only_sql(preprocess_sql, pred_sqls)

    template_path = f"{db_dir}/{db_id}/{db_id}_template.sqlite"
    if not Path(template_path).exists():
        item['exec_flag'] = False
        item['exec_results'] = f"Error: Template database not found: {template_path}"
        return item

    if preprocess_sql or not read_only:
        db_path = f"{db_dir}/{db_id}/{db_id}_sft_worker_{worker_id}.sqlite"
    else:
        db_path = template_path

    success, results = execute_sql_safe(
        sql_list=pred_sqls,
        db_path=db_path,
        db_id=db_id,
        db_dir=db_dir,
        is_read_only=read_only,
        preprocess_sql=preprocess_sql,
        timeout=60
    )

    item['exec_flag'] = success
    item['exec_results'] = results
    return item


def process_observations(input_path, train_data_path, output_path, db_dir, num_threads=8):
    """Process all instances: execute SQL and collect observations."""
    print(f"Loading parsed data from: {input_path}")
    parsed_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parsed_data.append(json.loads(line))
    print(f"  Loaded {len(parsed_data)} instances")

    print(f"Loading train data from: {train_data_path}")
    preprocess_sql_dict = {}
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for instance_idx, line in enumerate(f):
            train_item = json.loads(line)
            preprocess_sql = train_item.get('preprocess_sql', [])
            if preprocess_sql:
                preprocess_sql_dict[instance_idx] = preprocess_sql
    print(f"  Loaded preprocess_sql for {len(preprocess_sql_dict)} instances")

    print(f"Executing SQL queries with {num_threads} threads...")
    lock = threading.Lock()
    results = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for worker_id, item in enumerate(parsed_data):
            future = executor.submit(
                process_single_instance, item, db_dir, preprocess_sql_dict,
                lock, worker_id % num_threads
            )
            futures[future] = item

        for future in tqdm(as_completed(futures), total=len(futures), desc="Executing SQL"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                item = futures[future]
                item['exec_flag'] = False
                item['exec_results'] = f"Error: {type(e).__name__}: {str(e)}"
                results.append(item)

    results.sort(key=lambda x: x.get('idx', 0))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    success_count = sum(1 for r in results if r.get('exec_flag', False))
    print(f"\n  Total: {len(results)}, Success: {success_count}, Failed: {len(results) - success_count}")

    return len(results)


def main():
    parser = argparse.ArgumentParser(description='Execute SQL and collect observations')
    parser.add_argument('--turn', type=int, default=0, help='Current turn number')
    parser.add_argument('--threads', type=int, default=8, help='Number of parallel threads')
    parser.add_argument('--train-data', type=str, required=True, help='Path to train.jsonl')
    parser.add_argument('--db-dir', type=str, required=True, help='Database directory')
    parser.add_argument('--input', type=str, required=True, help='Input file (parsed data)')
    parser.add_argument('--output', type=str, required=True, help='Output file (observations)')
    args = parser.parse_args()

    print("=" * 60)
    print(f"Execute SQL - Turn {args.turn}")
    print("=" * 60)

    count = process_observations(
        input_path=args.input,
        train_data_path=args.train_data,
        output_path=args.output,
        db_dir=args.db_dir,
        num_threads=args.threads
    )

    print(f"\nObservations collected: {count} -> {args.output}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
