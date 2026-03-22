import re
from datetime import date, datetime
try:
    from .db_utils import perform_query_on_sqlite_databases, execute_queries
except ImportError:
    from db_utils import perform_query_on_sqlite_databases, execute_queries
import json
from decimal import Decimal, ROUND_HALF_UP
import logging


def remove_round_functions(sql_string):
    """
    Remove all ROUND() function calls from SQL string, including nested ones.
    Correctly handles nested functions with commas.
    """

    def find_matching_paren(text, start_pos):
        """Find the position of the matching closing parenthesis."""
        paren_count = 0
        for i in range(start_pos, len(text)):
            if text[i] == "(":
                paren_count += 1
            elif text[i] == ")":
                paren_count -= 1
                if paren_count == 0:
                    return i
        return -1

    def find_first_arg_end(text, start_pos):
        """Find the end of the first argument, considering nested parentheses."""
        paren_count = 0
        for i in range(start_pos, len(text)):
            if text[i] == "(":
                paren_count += 1
            elif text[i] == ")":
                if paren_count == 0:
                    return i  # End of ROUND function
                paren_count -= 1
            elif text[i] == "," and paren_count == 0:
                return i  # End of first argument
        return len(text)

    result = sql_string

    while True:
        pattern = re.compile(r"ROUND\s*\(", re.IGNORECASE)
        match = pattern.search(result)

        if not match:
            break

        start_pos = match.start()
        open_paren_pos = match.end() - 1

        first_arg_end = find_first_arg_end(result, open_paren_pos + 1)
        close_paren_pos = find_matching_paren(result, open_paren_pos)

        if close_paren_pos == -1:
            break  # Malformed SQL

        first_arg = result[open_paren_pos + 1 : first_arg_end].strip()
        result = result[:start_pos] + first_arg + result[close_paren_pos + 1 :]

    return result


def remove_round(sql_list):
    """
    Remove ROUND function calls while preserving the inner expression.
    Examples:
    - ROUND(column, 2) -> column
    - ROUND(ROUND(price, 2), 1) -> price (handles nested ROUND)
    """
    cleaned = []
    for sql in sql_list:
        result = sql
        result = remove_round_functions(result)
        cleaned.append(result)
        if "ROUND" in result:
            logging.warning(f"ROUND found in {result}")
    return cleaned


def process_decimals_recursive(item, decimal_places):
    """
    Recursively process decimals in any data structure (list, dict, tuple).
    Returns a new structure with all decimals rounded to the specified places.
    """
    quantizer = Decimal(1).scaleb(-decimal_places)

    if isinstance(item, Decimal):
        return item.quantize(quantizer, rounding=ROUND_HALF_UP)
    elif isinstance(item, float):
        return round(item, decimal_places)
    elif isinstance(item, (list, tuple)):
        return type(item)(process_decimals_recursive(x, decimal_places) for x in item)
    elif isinstance(item, dict):
        return {
            k: process_decimals_recursive(v, decimal_places) for k, v in item.items()
        }
    else:
        return item


def preprocess_results(results, decimal_places=2):
    """
    Process result set:
    - Replace dates with normalized string: YYYY-MM-DD
    - Convert tuples to lists for JSON serialization
    - Convert unhashable types (dicts, lists) to string representation for comparison
    - Recursively process all decimals in nested structures
    """
    processed = []
    for result in results:
        processed_result = []
        for item in result:
            if isinstance(item, (date, datetime)):
                processed_result.append(item.strftime("%Y-%m-%d"))
            else:
                processed_item = process_decimals_recursive(item, decimal_places)
                if isinstance(processed_item, (dict, list)):
                    processed_result.append(json.dumps(processed_item, sort_keys=True))
                else:
                    processed_result.append(processed_item)
        processed.append(tuple(processed_result))
    return processed


def remove_distinct(sql_list):
    """
    Remove all DISTINCT keywords (case-insensitive) from a list of SQL query strings.

    Args:
        sql_list: List of SQL query strings

    Returns:
        List of SQL query strings with all DISTINCT keywords removed
    """
    cleaned_queries = []
    for query in sql_list:
        tokens = query.split(" ")
        filtered_tokens = []
        for token in tokens:
            if token.lower() != "distinct":
                filtered_tokens.append(token)
        cleaned_query = " ".join(filtered_tokens)
        cleaned_queries.append(cleaned_query)

    return cleaned_queries


def check_sql_function_usage(sqls, required_keywords):
    """
    Check if predicted SQL queries use all specified keywords/functions.
    Returns 1 if all required keywords are present, 0 otherwise.

    Args:
        sqls: List of predicted SQL query strings
        required_keywords: List of required keywords or function names

    Returns:
        1 if all required keywords are found, 0 if at least one is missing
    """
    if not sqls:
        return 0

    combined_sql = " ".join(sql.lower() for sql in sqls)

    for kw in required_keywords:
        if kw.lower() not in combined_sql:
            return 0

    return 1


def ex_base(pred_sqls, sol_sqls, db_path, conn, conditions=None):
    """
    Compare result sets of two SQL query lists:
    - Strip comments, DISTINCT, and ORDER BY
    - Execute both
    - Normalize dates and optionally round decimals
    - Check equality (ordered or unordered comparison based on conditions)
    Returns 1 on match, 0 otherwise.
    """
    if not pred_sqls or not sol_sqls:
        return 0

    predicted_res, pred_err, pred_to = execute_queries(
        pred_sqls, db_path, conn, None, ""
    )
    ground_res, gt_err, gt_to = execute_queries(sol_sqls, db_path, conn, None, "")
    if any([pred_err, pred_to, gt_err, gt_to]):
        return 0

    predicted_res = preprocess_results(predicted_res)
    ground_res = preprocess_results(ground_res)
    if not predicted_res or not ground_res:
        return 0

    if conditions is not None and conditions.get("order", False):
        return 1 if predicted_res == ground_res else 0
    else:
        return 1 if set(predicted_res) == set(ground_res) else 0


def remove_comments(sql_list):
    """
    Remove all SQL comments from each query string in the list.
    - Block comments: /* ... */
    - Line comments: -- ... (to end of line)
    Also collapses multiple blank lines and strips leading/trailing whitespace.
    """
    cleaned = []
    for sql in sql_list:
        no_block = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        no_line = re.sub(r"--.*?(\r\n|\r|\n)", r"\1", no_block)
        no_blank = re.sub(r"\n\s*\n+", "\n", no_line)
        cleaned.append(no_blank.strip())
    return cleaned


def test_case_default(pred_sqls, sol_sqls, db_path, conn, conditions):
    """Default test case: pytest-style assertion."""
    pred_sqls = remove_comments(pred_sqls)
    sol_sqls = remove_comments(sol_sqls)
    pred_sqls = remove_distinct(pred_sqls)
    pred_sqls = remove_round(pred_sqls)
    sol_sqls = remove_distinct(sol_sqls)
    sol_sqls = remove_round(sol_sqls)

    result = ex_base(pred_sqls, sol_sqls, db_path, conn, conditions)
    assert result == 1, f"ex_base returned {result} but expected 1."
    return result


TEST_CASE_DEFAULT = """
def test_case(pred_sqls, sol_sqls, db_path, conn, conditions):
   pred_sqls = remove_comments(pred_sqls)
   sol_sqls  = remove_comments(sol_sqls)
   pred_sqls = remove_distinct(pred_sqls)
   pred_sqls = remove_round(pred_sqls)
   sol_sqls  = remove_distinct(sol_sqls)
   sol_sqls  = remove_round(sol_sqls)
   result = ex_base(pred_sqls, sol_sqls, db_path, conn, conditions)
   assert result == 1, f"ex_base returned {result} but expected 1."
   return result
"""
