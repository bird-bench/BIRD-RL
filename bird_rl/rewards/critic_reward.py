"""
SQL Debugging Reward Function: Execution-Based (Batch Processing)

Scoring (v0 simple - pure test pass rate):
- format_invalid: 0.0
- syntax/execution_error/timeout: 0.0
- test pass rate: passed/total (0.0 to 1.0)
"""

import re
import json
import sys
import os
import sqlite3
import shutil
import io
import traceback
import threading
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

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

# ============================================================
# CONFIGURATION
# ============================================================

# Base directory for template databases
DB_DIR = os.environ.get("SQL_REWARD_DB_DIR", "")

# Number of parallel workers for batch processing
# CRITICAL: Each worker needs its own ephemeral database to avoid race conditions
NUM_WORKERS = int(os.environ.get("SQL_REWARD_NUM_WORKERS", "4"))

# Batch size for processing
# NOTE: Set equal to NUM_WORKERS to avoid race conditions on ephemeral databases
BATCH_SIZE = int(os.environ.get("SQL_REWARD_BATCH_SIZE", str(NUM_WORKERS)))

# ============================================================
# RESPONSE PARSING
# ============================================================

def parse_model_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse model response to extract thought and solution.

    Args:
        response: Raw model response

    Returns:
        Tuple of (thought, solution_sql)
        Returns (None, None) if parsing fails
    """
    if not response or not isinstance(response, str):
        return None, None

    # Extract thought
    thought_pattern = r'<thought>(.*?)</thought>'
    thought_match = re.search(thought_pattern, response, re.DOTALL | re.IGNORECASE)
    thought = thought_match.group(1).strip() if thought_match else None

    # Extract solution
    solution_pattern = r'<solution>(.*?)</solution>'
    solution_match = re.search(solution_pattern, response, re.DOTALL | re.IGNORECASE)
    solution = solution_match.group(1).strip() if solution_match else None

    # Remove markdown code fences if present
    if solution:
        solution = re.sub(r'^```(?:sql)?\s*\n', '', solution, flags=re.MULTILINE)
        solution = re.sub(r'\n```\s*$', '', solution, flags=re.MULTILINE)
        solution = solution.strip()

    return thought, solution


def validate_response_format(response: str) -> Tuple[bool, str]:
    """
    Validate that response has correct format.

    Args:
        response: Raw model response

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not response or not response.strip():
        return False, "empty_response"

    # Check for required tags
    if '<thought>' not in response.lower():
        return False, "missing_thought_tag"

    if '<solution>' not in response.lower():
        return False, "missing_solution_tag"

    if '</thought>' not in response.lower():
        return False, "missing_thought_closing_tag"

    if '</solution>' not in response.lower():
        return False, "missing_solution_closing_tag"

    # Try parsing
    thought, solution = parse_model_response(response)

    if not thought or len(thought.strip()) < 10:
        return False, "thought_too_short"

    if not solution or len(solution.strip()) < 10:
        return False, "solution_too_short"

    return True, ""


# ============================================================
# SQL TYPE DETECTION (Read-only vs Modifying)
# ============================================================

# SQL keywords that modify the database
_MODIFYING_SQL_KEYWORDS = {
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
    'TRUNCATE', 'REPLACE', 'MERGE', 'UPSERT', 'GRANT', 'REVOKE'
}


def is_read_only_sql(*sql_parts) -> bool:
    """
    Check if all SQL parts are read-only (no modifications to DB).

    Args:
        *sql_parts: SQL strings or lists of SQL strings

    Returns:
        True if all SQL is read-only (SELECT, WITH...SELECT, etc.)
        False if any SQL modifies the database
    """
    for part in sql_parts:
        if part is None:
            continue

        # Handle list of SQLs
        if isinstance(part, (list, tuple)):
            for sql in part:
                if sql and not is_read_only_sql(sql):
                    return False
            continue

        # Handle single SQL string
        if not isinstance(part, str) or not part.strip():
            continue

        sql_upper = part.upper()

        # Remove comments
        sql_upper = re.sub(r'--[^\n]*', '', sql_upper)
        sql_upper = re.sub(r'/\*.*?\*/', '', sql_upper, flags=re.DOTALL)

        # Check each statement
        for stmt in sql_upper.split(';'):
            stmt = stmt.strip()
            if not stmt:
                continue

            words = stmt.split()
            if not words:
                continue

            first_word = words[0]

            # Check if modifying
            if first_word in _MODIFYING_SQL_KEYWORDS:
                return False

            # WITH ... INSERT/UPDATE/DELETE
            if first_word == 'WITH':
                for kw in _MODIFYING_SQL_KEYWORDS:
                    if kw in stmt:
                        return False

    return True


# ============================================================
# EPHEMERAL DATABASE MANAGEMENT (With Global Cache)
# ============================================================

# Global cache to avoid recreating databases for every batch
_EPHEMERAL_DB_CACHE: Dict[str, List[str]] = {}
_EPHEMERAL_DB_CACHE_LOCK = threading.Lock()
_EPHEMERAL_DB_INITIALIZED = False


def _init_ephemeral_db_for_db_id(db_id: str, num_copies: int = NUM_WORKERS) -> List[str]:
    """
    Initialize ephemeral databases for a single db_id.
    Creates databases only if they don't exist in cache.
    """
    template_path = os.path.join(DB_DIR, db_id, f"{db_id}_template.sqlite")

    if not os.path.exists(template_path):
        print(f"[SQL Reward v3] WARNING: Template not found: {template_path}")
        return []

    ephemeral_paths = []

    for i in range(1, num_copies + 1):
        ephemeral_path = os.path.join(DB_DIR, db_id, f"{db_id}_reward_worker_{i}.sqlite")

        try:
            # Only create if doesn't exist
            if not os.path.exists(ephemeral_path):
                shutil.copy2(template_path, ephemeral_path)

            ephemeral_paths.append(ephemeral_path)

        except Exception as e:
            print(f"[SQL Reward v3] Failed to create ephemeral db {ephemeral_path}: {e}")

    return ephemeral_paths


def get_ephemeral_databases(db_ids: List[str], num_copies: int = NUM_WORKERS) -> Dict[str, List[str]]:
    """
    Get ephemeral database copies, using global cache.
    Only creates new databases for db_ids not already in cache.

    Args:
        db_ids: List of unique database IDs needed
        num_copies: Number of ephemeral copies per database

    Returns:
        Dict mapping db_id to list of ephemeral database paths
    """
    global _EPHEMERAL_DB_CACHE, _EPHEMERAL_DB_INITIALIZED

    with _EPHEMERAL_DB_CACHE_LOCK:
        # Check which db_ids need initialization
        new_db_ids = [db_id for db_id in db_ids if db_id not in _EPHEMERAL_DB_CACHE]

        if new_db_ids:
            if not _EPHEMERAL_DB_INITIALIZED:
                print(f"[SQL Reward v3] Initializing ephemeral database cache...")
                _EPHEMERAL_DB_INITIALIZED = True

            for db_id in new_db_ids:
                paths = _init_ephemeral_db_for_db_id(db_id, num_copies)
                if paths:
                    _EPHEMERAL_DB_CACHE[db_id] = paths
                    print(f"[SQL Reward v3] Cached {len(paths)} ephemeral DBs for {db_id}")

        # Return only requested db_ids from cache
        return {db_id: _EPHEMERAL_DB_CACHE.get(db_id, []) for db_id in db_ids}


# Backward compatibility alias
def create_ephemeral_databases(db_ids: List[str], num_copies: int = NUM_WORKERS) -> Dict[str, List[str]]:
    """Alias for get_ephemeral_databases (backward compatibility)."""
    return get_ephemeral_databases(db_ids, num_copies)


def reset_ephemeral_database(ephemeral_path: str, db_id: str) -> bool:
    """
    Reset an ephemeral database by copying from template.

    Args:
        ephemeral_path: Path to ephemeral database
        db_id: Database identifier

    Returns:
        True if successful, False otherwise
    """
    template_path = os.path.join(DB_DIR, db_id, f"{db_id}_template.sqlite")

    if not os.path.exists(template_path):
        return False

    try:
        # Clean up WAL files
        for suffix in ['-wal', '-shm', '-journal']:
            path_to_remove = ephemeral_path + suffix
            if os.path.exists(path_to_remove):
                try:
                    os.remove(path_to_remove)
                except:
                    pass

        # Remove and recreate
        if os.path.exists(ephemeral_path):
            os.remove(ephemeral_path)

        shutil.copy2(template_path, ephemeral_path)
        return True

    except Exception as e:
        print(f"[SQL Reward v3] Failed to reset {ephemeral_path}: {e}")
        return False


# ============================================================
# TEST CASE EXECUTION (Using evaluation code)
# ============================================================

def run_single_test_case(
    test_code: str,
    pred_sqls: List[str],
    sol_sqls: List[str],
    db_path: str,
    conn: sqlite3.Connection,
    conditions: Dict = None
) -> Tuple[bool, str]:
    """
    Execute a single test case using the evaluation pattern.

    This follows the pattern from single_instance_eval_sqlite.run_test_case

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

    # Setup execution environment (matching evaluation code)
    global_env = {
        'execute_queries': execute_queries if HAS_EVAL_UTILS else lambda *args, **kwargs: (None, True, False),
        'perform_query_on_sqlite_databases': perform_query_on_sqlite_databases if HAS_EVAL_UTILS else lambda *args, **kwargs: (None, None),
        'ex_base': ex_base if HAS_EVAL_UTILS else lambda *args, **kwargs: 0,
        'check_sql_function_usage': check_sql_function_usage if HAS_EVAL_UTILS else lambda *args: True,
        'remove_distinct': remove_distinct if HAS_EVAL_UTILS else lambda x: x,
        'remove_comments': remove_comments if HAS_EVAL_UTILS else lambda x: x,
        'remove_round': remove_round if HAS_EVAL_UTILS else lambda x: x,
        'preprocess_results': preprocess_results if HAS_EVAL_UTILS else lambda x: x,
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
        # Prepare test code (matching evaluation code pattern)
        test_case_code = "import datetime\nfrom datetime import date\n" + test_code
        test_case_code += "\n__test_case_result__ = test_case(pred_sqls, sol_sqls, db_path, conn, conditions)"

        # Execute test (no stdout capture - not thread-safe)
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
        # Use default test case if none provided
        test_cases = [TEST_CASE_DEFAULT] if HAS_EVAL_UTILS else []

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


# ============================================================
# SINGLE INSTANCE EXECUTION
# ============================================================

# Timeout for SQL execution (in seconds)
SQL_EXECUTION_TIMEOUT = int(os.environ.get("SQL_EXECUTION_TIMEOUT", "30"))


def _execute_in_process(queue, solution_sql, ground_truth_sql, test_cases,
                        preprocess_sql, conditions, db_path, db_id,
                        instance_id, is_read_only):
    """
    Worker function that runs in a separate process.
    Puts result into queue when done.
    """
    try:
        result = execute_single_instance(
            solution_sql=solution_sql,
            ground_truth_sql=ground_truth_sql,
            test_cases=test_cases,
            preprocess_sql=preprocess_sql,
            conditions=conditions,
            db_path=db_path,
            db_id=db_id,
            instance_id=instance_id,
            is_read_only=is_read_only
        )
        queue.put(('success', result))
    except Exception as e:
        queue.put(('error', str(e)))


def execute_single_instance_with_timeout(
    solution_sql: str,
    ground_truth_sql: str,
    test_cases: List[str],
    preprocess_sql: List[str],
    conditions: Dict,
    db_path: str,
    db_id: str,
    instance_id: str,
    is_read_only: bool = False,
    timeout: int = SQL_EXECUTION_TIMEOUT
) -> Dict:
    """
    Execute SQL with a timeout wrapper using multiprocessing.

    Uses a separate process that can be killed after timeout.
    This actually stops the SQL query (unlike threading).
    """
    # Create queue for result
    queue = multiprocessing.Queue()

    # Create process
    process = multiprocessing.Process(
        target=_execute_in_process,
        args=(queue, solution_sql, ground_truth_sql, test_cases,
              preprocess_sql, conditions, db_path, db_id,
              instance_id, is_read_only)
    )

    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        # Timeout - kill the process
        process.terminate()
        process.join(timeout=5)  # Wait for termination

        if process.is_alive():
            # Force kill if still alive
            process.kill()
            process.join(timeout=2)

        return {
            'execution_success': False,
            'syntax_error': False,
            'test_passed': False,
            'test_pass_count': 0,
            'test_total_count': len(test_cases) if test_cases else 1,
            'test_pass_rate': 0.0,
            'error_message': f'Execution timeout after {timeout} seconds',
            'is_read_only': is_read_only,
            'timeout': True,
        }

    # Process finished - get result from queue
    try:
        if not queue.empty():
            status, result = queue.get_nowait()
            if status == 'success':
                return result
            else:
                return {
                    'execution_success': False,
                    'syntax_error': False,
                    'test_passed': False,
                    'test_pass_count': 0,
                    'test_total_count': len(test_cases) if test_cases else 1,
                    'test_pass_rate': 0.0,
                    'error_message': f'Execution exception: {result}',
                    'is_read_only': is_read_only,
                    'timeout': False,
                }
        else:
            return {
                'execution_success': False,
                'syntax_error': False,
                'test_passed': False,
                'test_pass_count': 0,
                'test_total_count': len(test_cases) if test_cases else 1,
                'test_pass_rate': 0.0,
                'error_message': 'Process finished but no result returned',
                'is_read_only': is_read_only,
                'timeout': False,
            }
    except Exception as e:
        return {
            'execution_success': False,
            'syntax_error': False,
            'test_passed': False,
            'test_pass_count': 0,
            'test_total_count': len(test_cases) if test_cases else 1,
            'test_pass_rate': 0.0,
            'error_message': f'Error getting result: {str(e)}',
            'is_read_only': is_read_only,
            'timeout': False,
        }


def execute_single_instance(
    solution_sql: str,
    ground_truth_sql: str,
    test_cases: List[str],
    preprocess_sql: List[str],
    conditions: Dict,
    db_path: str,
    db_id: str,
    instance_id: str,
    is_read_only: bool = False
) -> Dict:
    """
    Execute SQL and run test cases for a single instance.

    Following the evaluation pattern from single_instance_eval_sqlite.evaluate_instance

    Args:
        solution_sql: Predicted SQL solution
        ground_truth_sql: Reference solution
        test_cases: List of test case code strings
        preprocess_sql: Database setup SQL statements
        conditions: Test conditions dict
        db_path: Path to database (template for read-only, ephemeral for modifying)
        db_id: Database identifier
        instance_id: Instance identifier
        is_read_only: If True, use template DB directly (no copy/reset needed)

    Returns:
        Execution result dict
    """
    result = {
        'execution_success': False,
        'syntax_error': False,
        'test_passed': False,
        'test_pass_count': 0,
        'test_total_count': len(test_cases) if (test_cases is not None and len(test_cases) > 0) else 1,
        'test_pass_rate': 0.0,
        'error_message': '',
        'is_read_only': is_read_only,
    }

    conn = None
    logger = NullLogger() if HAS_EVAL_UTILS else None

    try:
        if is_read_only:
            # Read-only: Use template DB directly (no copy needed)
            # Open in read-only mode for safety
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro",
                uri=True,
                timeout=30.0
            )
            conn.execute("PRAGMA busy_timeout = 30000")
        else:
            # Modifying: Reset ephemeral database from template
            if not reset_ephemeral_database(db_path, db_id):
                result['error_message'] = f"Failed to reset database for {db_id}"
                result['syntax_error'] = True
                return result

            # Connect in read-write mode
            conn = sqlite3.connect(
                f"file:{db_path}?mode=rw",
                uri=True,
                timeout=30.0
            )
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = OFF")

            # Run preprocessing SQL (only for modifying instances)
            if preprocess_sql is not None and len(preprocess_sql) > 0:
                if isinstance(preprocess_sql, str):
                    preprocess_sql = [preprocess_sql]
                else:
                    preprocess_sql = list(preprocess_sql)

                for sql in preprocess_sql:
                    if sql and sql.strip():
                        try:
                            cursor = conn.cursor()
                            cursor.execute(sql)
                            conn.commit()
                            cursor.close()
                        except Exception as e:
                            result['error_message'] = f"Preprocessing error: {str(e)}"
                            result['syntax_error'] = True
                            return result

        # Parse SQL statements
        pred_sqls = [s.strip() for s in solution_sql.split(';') if s.strip()]
        sol_sqls = [s.strip() for s in ground_truth_sql.split(';') if s.strip()]

        # Step 5: Execute predicted SQL
        for sql in pred_sqls:
            try:
                cursor = conn.cursor()
                cursor.execute(sql)

                # Commit for non-SELECT statements
                if not sql.strip().upper().startswith(('SELECT', 'WITH')):
                    conn.commit()

                cursor.close()
            except sqlite3.Error as e:
                result['syntax_error'] = True
                result['error_message'] = f"SQL Error: {str(e)}"
                return result

        result['execution_success'] = True

        # Step 6: Run test cases
        passed_count, total_count, error_msgs = execute_test_cases(
            test_cases=test_cases,
            pred_sqls=pred_sqls,
            sol_sqls=sol_sqls,
            db_path=db_path,
            conn=conn,
            conditions=conditions
        )

        result['test_pass_count'] = passed_count
        result['test_total_count'] = total_count
        result['test_pass_rate'] = passed_count / total_count if total_count > 0 else 0.0
        result['test_passed'] = (passed_count == total_count)

        if error_msgs:
            result['error_message'] = error_msgs

    except Exception as e:
        result['error_message'] = f"Execution exception: {str(e)}\n{traceback.format_exc()}"

    finally:
        # Close connection
        if conn:
            try:
                conn.commit()
            except:
                try:
                    conn.rollback()
                except:
                    pass
            try:
                conn.close()
            except:
                pass

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
    """Convert numpy types to Python types for JSON serialization."""
    if HAS_NUMPY and isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _to_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_python(v) for v in x]
    return x


def create_empty_result() -> dict:
    """Create an empty result dict with all fields initialized."""
    return {
        "score": 0.0,

        # Format validation
        "format_valid": False,
        "format_error": "",

        # Parsed components
        "thought": "",
        "solution_sql": "",

        # Execution results
        "execution_success": False,
        "syntax_error": False,
        "test_passed": False,
        "test_pass_count": 0,
        "test_total_count": 0,
        "test_pass_rate": 0.0,
        "error_message": "",
        "is_read_only": False,
        "timeout": False,
    }


def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[dict],
    **kwargs
) -> List[dict]:
    """
    V3: Compute rewards using SQL execution and test cases.

    Processing stages:
    1. Format validation: Parse <thought> and <solution> tags
    2. Create ephemeral databases for unique db_ids
    3. Execute SQL and run test cases in parallel batches

    CRITICAL: Maintains strict index alignment across all processing stages.

    Args:
        data_sources: List of data source identifiers (length=N)
        solution_strs: List of model responses (length=N)
        ground_truths: List of ground truth solutions (length=N)
            - Can be dict with 'ground_truth' and 'test_cases' keys
            - Or just string (ground truth SQL only)
        extra_infos: List of extra info dicts containing:
            - query: Problem description
            - schema: Database schema
            - issue_sql: Original problematic SQL
            - db_id: Database identifier
            - preprocess_sql: Database setup statements
            - clean_up_sql: Database cleanup statements
            - test_cases: Test case code (may be here or in ground_truths)
            - conditions: Test conditions
        **kwargs: Additional keyword arguments

    Returns:
        List of result dicts (length=N), one per input instance, in same order
    """
    batch_size = len(solution_strs)

    # Sanity checks
    assert len(data_sources) == batch_size, f"data_sources length mismatch: {len(data_sources)} != {batch_size}"
    assert len(ground_truths) == batch_size, f"ground_truths length mismatch: {len(ground_truths)} != {batch_size}"
    assert len(extra_infos) == batch_size, f"extra_infos length mismatch: {len(extra_infos)} != {batch_size}"

    # Initialize results for all instances
    results = [create_empty_result() for _ in range(batch_size)]

    print(f"[SQL Reward v3] Processing batch of {batch_size} instances")

    # ==========================================
    # STAGE 1: Parse and validate format
    # ==========================================
    valid_format_indices = []

    for i in range(batch_size):
        is_valid, error = validate_response_format(solution_strs[i])
        results[i]["format_valid"] = is_valid
        results[i]["format_error"] = error

        if is_valid:
            thought, solution = parse_model_response(solution_strs[i])
            results[i]["thought"] = thought or ""
            results[i]["solution_sql"] = solution or ""
            valid_format_indices.append(i)
        else:
            results[i]["score"] = 0.0

    print(f"[SQL Reward v3] Stage 1: {len(valid_format_indices)}/{batch_size} have valid format")

    if not valid_format_indices:
        return [_to_python(r) for r in results]

    # ==========================================
    # STAGE 2: Create ephemeral databases
    # ==========================================
    unique_db_ids = set()
    for i in valid_format_indices:
        db_id = extra_infos[i].get('db_id', '')
        if db_id:
            unique_db_ids.add(db_id)

    # Use cached ephemeral databases (only creates new ones if not in cache)
    ephemeral_pool = get_ephemeral_databases(list(unique_db_ids), num_copies=NUM_WORKERS)
    total_ephemeral = sum(len(paths) for paths in ephemeral_pool.values())
    print(f"[SQL Reward v3] Stage 2: {len(unique_db_ids)} db_ids, {total_ephemeral} ephemeral DBs (cached)")

    # ==========================================
    # STAGE 3: Execute SQL and run test cases
    # ==========================================

    def process_instance(i: int, worker_id: int) -> Tuple[int, float, Dict]:
        """Process a single instance with assigned worker."""
        solution_sql = results[i]["solution_sql"]

        # Extract ground_truth and test_cases
        if isinstance(ground_truths[i], dict):
            ground_truth_sql = ground_truths[i].get('ground_truth', '')
            test_cases = ground_truths[i].get('test_cases', [])
        else:
            ground_truth_sql = str(ground_truths[i])
            test_cases = []

        # Fallback: check extra_infos for test_cases
        if not test_cases:
            test_cases = extra_infos[i].get('test_cases', [])

        # Convert to list if needed (handle numpy arrays safely)
        if hasattr(test_cases, 'tolist'):
            test_cases = test_cases.tolist()
        elif not isinstance(test_cases, list):
            test_cases = list(test_cases) if (test_cases is not None and len(test_cases) > 0) else []

        # Get other fields
        db_id = extra_infos[i].get('db_id', '')
        preprocess_sql = extra_infos[i].get('preprocess_sql', [])

        # Convert preprocess_sql to list (handle numpy arrays)
        if hasattr(preprocess_sql, 'tolist'):
            preprocess_sql = preprocess_sql.tolist()
        elif not isinstance(preprocess_sql, list):
            preprocess_sql = list(preprocess_sql) if preprocess_sql is not None else []

        # conditions is optional - used for order-sensitive comparisons (rarely used)
        conditions = extra_infos[i].get('conditions', {})
        if isinstance(conditions, str):
            try:
                conditions = json.loads(conditions)
            except:
                conditions = {}
        instance_id = extra_infos[i].get('instance_id', f'instance_{i}')

        # Check if this instance is read-only (can use template DB directly)
        read_only = is_read_only_sql(solution_sql, preprocess_sql)

        if read_only:
            # Read-only: Use template DB directly (no copy needed, unlimited parallelism)
            template_path = os.path.join(DB_DIR, db_id, f"{db_id}_template.sqlite")
            if not os.path.exists(template_path):
                results[i]["score"] = 0.0
                results[i]["error_message"] = f"Template database not found for {db_id}"
                results[i]["syntax_error"] = True
                return i, 0.0, None
            db_path = template_path
        else:
            # Modifying: Need ephemeral database (limited parallelism)
            if db_id not in ephemeral_pool or not ephemeral_pool[db_id]:
                results[i]["score"] = 0.0
                results[i]["error_message"] = f"No ephemeral database for {db_id}"
                results[i]["syntax_error"] = True
                return i, 0.0, None
            # Round-robin assignment based on worker_id
            db_path = ephemeral_pool[db_id][worker_id % len(ephemeral_pool[db_id])]

        # Execute instance with timeout
        exec_result = execute_single_instance_with_timeout(
            solution_sql=solution_sql,
            ground_truth_sql=ground_truth_sql,
            test_cases=test_cases,
            preprocess_sql=preprocess_sql,
            conditions=conditions,
            db_path=db_path,
            db_id=db_id,
            instance_id=instance_id,
            is_read_only=read_only
        )

        # Calculate score (matching agentic v0 simple reward logic)
        # Pure test pass rate - no partial credit for valid-but-wrong SQL
        if exec_result.get('timeout') or exec_result.get('syntax_error'):
            score = 0.0  # Syntax error or timeout
        else:
            score = exec_result['test_pass_rate']

        return i, score, exec_result

    # Process instances
    # CRITICAL: Ensure batch size <= NUM_WORKERS to prevent race conditions
    # Each worker_id maps to a specific ephemeral database, and we must ensure
    # no two concurrent tasks use the same ephemeral database for the same db_id
    effective_batch_size = min(BATCH_SIZE, NUM_WORKERS)

    print(f"[SQL Reward v3] Stage 3: Executing {len(valid_format_indices)} instances")
    print(f"[SQL Reward v3] Stage 3: Using {NUM_WORKERS} workers, batch size {effective_batch_size}")

    completed_count = 0

    # Check if we should use sequential processing (safer for Ray actors)
    use_sequential = NUM_WORKERS == 1 or os.environ.get("SQL_REWARD_SEQUENTIAL", "0") == "1"

    if use_sequential:
        # Sequential processing - safer for Ray actors
        print(f"[SQL Reward v3] Stage 3: Using SEQUENTIAL processing")
        for idx_in_list, i in enumerate(valid_format_indices):
            try:
                idx, score, exec_result = process_instance(i, 0)  # Always worker_id=0

                if exec_result:
                    results[idx].update(exec_result)
                results[idx]["score"] = score

                completed_count += 1

                # Per-instance output (like v1)
                if exec_result:
                    test_info = f"{exec_result.get('test_pass_count', 0)}/{exec_result.get('test_total_count', 1)} tests"
                    if exec_result.get('timeout'):
                        test_info = "timeout"
                    elif exec_result.get('syntax_error'):
                        test_info = "syntax_error"
                else:
                    test_info = "error"
                print(f"[SQL Reward v3]   [{completed_count}/{len(valid_format_indices)}] Instance {i}: Score={score:.2f} ({test_info})")

            except Exception as e:
                print(f"[SQL Reward v3]   [{completed_count+1}/{len(valid_format_indices)}] Instance {i}: Score=0.00 (ERROR: {str(e)})")
                results[i]["score"] = 0.0
                results[i]["error_message"] = f"Processing error: {str(e)}"
                completed_count += 1
    else:
        # Parallel processing with ThreadPoolExecutor
        # Create batches with size limited to NUM_WORKERS
        batches = []
        for start_idx in range(0, len(valid_format_indices), effective_batch_size):
            batch = valid_format_indices[start_idx:start_idx + effective_batch_size]
            batches.append(batch)

        print(f"[SQL Reward v3] Stage 3: Processing {len(batches)} batches (parallel)")

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for batch_idx, batch in enumerate(batches):
                # Submit all instances in this batch
                # CRITICAL: worker_id must be < NUM_WORKERS to ensure unique ephemeral DB per concurrent task
                futures = {}
                for worker_id, i in enumerate(batch):
                    # worker_id is guaranteed to be < effective_batch_size <= NUM_WORKERS
                    future = executor.submit(process_instance, i, worker_id)
                    futures[future] = i

                # Wait for ENTIRE batch to complete before starting next batch
                # This ensures no race conditions on ephemeral databases
                for future in as_completed(futures.keys()):
                    global_idx = futures[future]
                    try:
                        idx, score, exec_result = future.result(timeout=120)

                        if exec_result:
                            results[idx].update(exec_result)
                        results[idx]["score"] = score

                        completed_count += 1

                        # Per-instance output (like v1)
                        if exec_result:
                            test_info = f"{exec_result.get('test_pass_count', 0)}/{exec_result.get('test_total_count', 1)} tests"
                            if exec_result.get('timeout'):
                                test_info = "timeout"
                            elif exec_result.get('syntax_error'):
                                test_info = "syntax_error"
                        else:
                            test_info = "error"
                        print(f"[SQL Reward v3]   [{completed_count}/{len(valid_format_indices)}] Instance {idx}: Score={score:.2f} ({test_info})")

                    except Exception as e:
                        results[global_idx]["score"] = 0.0
                        results[global_idx]["error_message"] = f"Worker error: {str(e)}"
                        completed_count += 1
                        print(f"[SQL Reward v3]   [{completed_count}/{len(valid_format_indices)}] Instance {global_idx}: Score=0.00 (ERROR: {str(e)})")

    # ==========================================
    # FINAL: Summary and return
    # ==========================================
    avg_score = sum(r["score"] for r in results) / batch_size
    executed_count = sum(1 for r in results if r["execution_success"])
    test_passed_count = sum(1 for r in results if r["test_passed"])

    print(f"[SQL Reward v3] Summary:")
    print(f"  - Total instances: {batch_size}")
    print(f"  - Valid format: {len(valid_format_indices)} ({len(valid_format_indices)/batch_size:.1%})")
    print(f"  - Executed successfully: {executed_count} ({executed_count/batch_size:.1%})")
    print(f"  - Tests passed: {test_passed_count} ({test_passed_count/batch_size:.1%})")
    print(f"  - Average score: {avg_score:.3f}")

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
    """
    Single instance entry point (for compatibility).

    Wraps compute_score_batch for single instance processing.
    """
    results = compute_score_batch(
        data_sources=[data_source],
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        extra_infos=[extra_info],
        **kwargs
    )
    return results[0]
