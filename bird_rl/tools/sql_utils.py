
"""
Shared utilities for SQL debugging tools.
"""

import os
import sys
import sqlite3
import multiprocessing
from typing import List, Dict, Tuple
from datetime import date

try:
    from bird_rl.evaluation.critic.db_utils import execute_queries
    HAS_EVAL_UTILS = True
except ImportError:
    HAS_EVAL_UTILS = False


# SQL keywords that modify database
WRITE_KEYWORDS = [
    'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER',
    'REPLACE', 'TRUNCATE', 'MERGE', 'UPSERT'
]


def is_write_operation(sql: str) -> bool:
    """
    Check if SQL modifies database.

    Args:
        sql: SQL query string

    Returns:
        True if SQL is a write operation, False for read-only
    """
    sql_upper = sql.strip().upper()
    return any(sql_upper.startswith(kw) for kw in WRITE_KEYWORDS)


def get_db_config() -> tuple[str, int]:
    """
    Get database configuration from environment variables.

    Returns:
        (db_dir, timeout): Database directory and execution timeout
    """
    db_dir = os.environ.get("SQL_REWARD_DB_DIR", "")
    timeout = int(os.environ.get("SQL_EXECUTION_TIMEOUT", "30"))
    return db_dir, timeout


def execute_sql_in_process(
    queue: multiprocessing.Queue,
    sql_list: List[str],
    preprocess_sql: List[str],
    db_path: str
):
    """
    Execute SQL list in separate process (for timeout enforcement).

    This runs in a subprocess, so it can be killed if it hangs.

    Args:
        queue: Queue to return result
        sql_list: List of SQL queries to execute
        preprocess_sql: Preprocessing SQL
        db_path: Database path
    """
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 30000")

        # Run preprocessing SQL
        for prep_sql in preprocess_sql:
            conn.execute(prep_sql)
        conn.commit()

        # Execute target SQL list in sequence
        last_result = None
        for sql in sql_list:
            cursor = conn.execute(sql)
            # Fetch results from last query (typically the final SELECT)
            try:
                last_result = cursor.fetchall()
            except:
                # Some queries don't return results (INSERT, CREATE, etc.)
                pass

        conn.commit()
        conn.close()

        queue.put({"success": True, "error": None, "result": last_result})

    except Exception as e:
        queue.put({"success": False, "error": str(e), "result": None})


def execute_sql_with_timeout(
    sql_list: List[str],
    db_path: str,
    preprocess_sql: List[str],
    timeout: int
) -> Dict:
    """
    Execute SQL list with timeout protection using multiprocessing.

    Args:
        sql_list: List of SQL queries to execute in sequence
        db_path: Path to database
        preprocess_sql: Preprocessing SQL statements
        timeout: Timeout in seconds

    Returns:
        {"success": bool, "error": str or None, "result": any}
    """
    if not sql_list:
        return {"success": True, "error": None, "result": None}

    try:
        # Execute with timeout using multiprocessing
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=execute_sql_in_process,
            args=(queue, sql_list, preprocess_sql, db_path)
        )

        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            # Timeout - kill process
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)

            return {"success": False, "error": f"Timeout after {timeout}s", "result": None}

        # Get result from queue
        if not queue.empty():
            result = queue.get_nowait()
            return result
        else:
            return {"success": False, "error": "Process finished but no result returned", "result": None}

    except Exception as e:
        return {"success": False, "error": str(e), "result": None}


def run_single_test_case(
    test_code: str,
    pred_sqls: List[str],
    sol_sqls: List[str],
    db_path: str,
    conn: sqlite3.Connection,
    conditions: Dict = None
) -> Tuple[bool, str]:
    """
    Execute a single test case.

    Args:
        test_code: Python test code to execute
        pred_sqls: Predicted SQL statements
        sol_sqls: Solution SQL statements
        db_path: Database path
        conn: Database connection
        conditions: Optional conditions dict

    Returns:
        Tuple of (test_passed, error_message)
    """
    if conditions is None:
        conditions = {}

    # Setup execution environment
    global_env = {
        'execute_queries': execute_queries if HAS_EVAL_UTILS else lambda *args, **kwargs: (None, True, False),
        'date': date,
    }

    local_env = {
        'conn': conn,
        'pred_sqls': pred_sqls,
        'sol_sqls': sol_sqls,
        'db_path': db_path,
        'conditions': conditions,
    }

    try:
        # Prepare test code
        test_case_code = "import datetime\nfrom datetime import date\n" + test_code
        test_case_code += "\n__test_case_result__ = test_case(pred_sqls, sol_sqls, db_path, conn, conditions)"

        # Execute test
        exec(test_case_code, global_env, local_env)
        return True, ""

    except AssertionError as e:
        return False, f"Assertion failed: {str(e)}"

    except Exception as e:
        return False, f"Test error: {str(e)}"


def execute_test_cases(
    test_cases: List[str],
    pred_sqls: List[str],
    sol_sqls: List[str],
    db_path: str,
    conn: sqlite3.Connection,
    conditions: Dict = None
) -> Tuple[int, int, str]:
    """
    Execute multiple test cases sequentially.

    Args:
        test_cases: List of test case code strings
        pred_sqls: Predicted SQL statements
        sol_sqls: Solution SQL statements
        db_path: Database path
        conn: Database connection
        conditions: Optional conditions dict

    Returns:
        Tuple of (passed_count, total_count, error_messages)
    """
    if not test_cases:
        test_cases = []

    if not test_cases:
        return 0, 0, "No test cases provided"

    passed_count = 0
    error_messages = []

    for i, test_code in enumerate(test_cases, start=1):
        try:
            test_passed, error_msg = run_single_test_case(
                test_code, pred_sqls, sol_sqls, db_path, conn, conditions
            )

            if test_passed:
                passed_count += 1
            elif error_msg:
                error_messages.append(f"Test {i}: {error_msg}")

        except Exception as e:
            error_messages.append(f"Test {i}: Unexpected error: {str(e)}")

    return passed_count, len(test_cases), "\n".join(error_messages)
