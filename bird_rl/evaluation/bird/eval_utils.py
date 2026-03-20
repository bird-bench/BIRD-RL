import json
import sqlite3
import psycopg2
import pymysql
import os
from tqdm import tqdm


def load_jsonl(file_path):
    """Load data from a JSONL file or JSON array file."""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

        # Check if it's a JSON array (starts with '[')
        if content.startswith('['):
            # It's a JSON array, load it directly
            data_list = json.loads(content)
        else:
            # It's JSONL format, parse line by line
            for line in content.split('\n'):
                line = line.strip()
                if line:  # Skip empty lines
                    data_list.append(json.loads(line))
    return data_list


def safe_sort_key(value):
    # If the value is None, replace it with the empty string (or the smallest value you see fit)
    return str(value) if value is not None else ""


# psycopg2   2.9.9
def connect_postgresql(db_id):
    # Open database connection
    # Connect to the database
    db = psycopg2.connect(
        f"dbname={db_id} user=root host=localhost password={os.environ.get('PG_PASSWORD', '')} port=5432"
    )
    return db


# PyMySQL  1.1.1
def connect_mysql(db_id):
    # Open database connection
    # Connect to the database"
    db = pymysql.connect(
        host="localhost",
        user="root",
        password=os.environ.get("PG_PASSWORD", ""),
        database=db_id,
        # unix_socket="/tmp/mysql.sock",
        unix_socket="/var/run/mysqld/mysqld.sock"
        # port=3306,
    )
    return db



def connect_db(sql_dialect, db_path):
    if sql_dialect == "sqlite":
        conn = sqlite3.connect(db_path)
    elif sql_dialect.lower() == "mysql":
        conn = connect_mysql(db_path)
    elif sql_dialect.lower() == "postgresql":
        conn = connect_postgresql(db_path)
    else:
        raise ValueError("Unsupported SQL dialect")
    return conn


def execute_sql(predicted_sql, ground_truth, db_path, sql_dialect, calculate_func):
    # Round floating point results to a consistent precision for comparison
    def round_results(results, precision=10):
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
    
    predicted_res = []
    ground_truth_res = []
    error_message = ""
    
    # Execute predicted SQL
    try:
        conn = connect_db(sql_dialect, db_path)
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        conn.close()
    except Exception as e:
        error_message = f"Predicted SQL Error: {str(e)}"
        predicted_res = []
    
    # Execute ground truth SQL
    try:
        conn = connect_db(sql_dialect, db_path)
        cursor = conn.cursor()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        conn.close()
    except Exception as e:
        if error_message:
            error_message += f" | Ground Truth SQL Error: {str(e)}"
        else:
            error_message = f"Ground Truth SQL Error: {str(e)}"
        ground_truth_res = []
    
    # Apply consistent rounding to both results
    predicted_res_rounded = round_results(predicted_res)
    ground_truth_res_rounded = round_results(ground_truth_res)
    
    res = calculate_func(predicted_res_rounded, ground_truth_res_rounded)
    
    if res == 0 and not error_message:
        # Both queries executed but results don't match
        error_message = f"Result mismatch - GT: {ground_truth_res_rounded} vs Pred: {predicted_res_rounded}"
    return res, predicted_res, ground_truth_res, error_message




def package_sqls(sql_path, gold_path, mode="gpt",dialect="sqlite",db_dir=None):
    clean_sqls = []
    db_path_list = []
    if mode == "gpt":
        # use chain of thought
        gt_data_list = load_jsonl(sql_path)
        for gt_item in gt_data_list:
            sql = gt_item.get('final_sql', gt_item.get('pred_sql', ''))
            clean_sqls.append(sql)

    elif mode == "gt":
        gt_data_list = load_jsonl(gold_path)
        for gt_item in gt_data_list:
            # print(gt_item)
            sql = gt_item['SQL']
            db_name = gt_item['db_id']
            clean_sqls.append(sql)
            if dialect == "sqlite":
                db_root_path = db_dir
                # Support both {db_id}.sqlite and {db_id}_template.sqlite
                template_path = os.path.join(db_root_path, db_name, db_name + "_template.sqlite")
                regular_path = os.path.join(db_root_path, db_name, db_name + ".sqlite")

                # Check which file exists
                if os.path.exists(template_path):
                    db_path_list.append(template_path)
                elif os.path.exists(regular_path):
                    db_path_list.append(regular_path)
                else:
                    # Default to template format
                    db_path_list.append(template_path)
            else:
                db_path_list.append(db_name)

    return clean_sqls, db_path_list


def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x["sql_idx"])


def save_results_to_jsonl(exec_results, pred_file, gold_file, output_jsonl_path):
    """
    Save results to a jsonl file with specified fields and execution results
    """
    # Load original data
    pred_data = load_jsonl(pred_file)
    gold_data = load_jsonl(gold_file)
    
    # Create a mapping of sql_idx to result
    result_map = {}
    for res in exec_results:
        result_map[res["sql_idx"]] = {
            "passed": bool(res["res"]),
            "predicted_res": res.get("predicted_res", []),
            "ground_truth_res": res.get("ground_truth_res", []),
            "error_message": res.get("error_message", "")
        }
    
    # Create output data with specified fields
    output_data = []
    for i, (pred_record, gold_record) in enumerate(zip(pred_data, gold_data)):
        # Get pred_sql from pred_file
        pred_sql = pred_record.get('final_sql', pred_record.get('pred_sql', ''))
        
        # Get execution results (first 10 rows)
        result_info = result_map.get(i, {"passed": False, "predicted_res": [], "ground_truth_res": []})
        pred_results_first_10 = result_info["predicted_res"][:10] if result_info["predicted_res"] else []
        gt_results_first_10 = result_info["ground_truth_res"][:10] if result_info["ground_truth_res"] else []
        error_message = result_info.get("error_message", "")
        
        new_record = {
            "question_id": gold_record.get("question_id", ""),
            "db_id": gold_record.get("db_id", ""),
            "question": gold_record.get("question", ""),
            "evidence": gold_record.get("evidence", ""),
            "SQL": gold_record.get("SQL", ""),  # Ground truth SQL
            "pred_sql": pred_sql,
            "gt_sql_results": gt_results_first_10,
            "pred_sql_results": pred_results_first_10,
            "passed": result_info["passed"],
            "error_message": error_message,
        }
        output_data.append(new_record)
    
    # Write to output jsonl file
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for record in output_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    tqdm.write(f"Results saved to {output_jsonl_path}")


def save_results_to_txt(exec_results, gold_file, output_txt_path, score_lists, count_lists, metric="EX"):
    """
    Save results to txt file with print_data functionality and instance-level results
    """
    # Load gold data to get question_id
    gold_data = load_jsonl(gold_file)
    
    levels = ["simple", "moderate", "challenging", "total"]
    
    # Write to file
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        # Write summary statistics (previous print_data functionality)
        f.write(f"start calculate {metric}\n")
        f.write("{:20} {:20} {:20} {:20} {:20}\n".format("", *levels))
        f.write("{:20} {:<20} {:<20} {:<20} {:<20}\n".format("count", *count_lists))
        f.write(f"======================================    {metric}   =====================================\n")
        f.write("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}\n".format(metric, *score_lists))
        f.write("===========================================================================================\n")
        f.write(f"Finished {metric} evaluation for test set\n")
        f.write("\n")
        
        # Write instance-level results
        f.write("Instance-level results:\n")
        f.write("=" * 50 + "\n")
        for res in exec_results:
            idx = res["sql_idx"]
            status = "correct" if res["res"] == 1 else "failed"
            error_msg = res.get("error_message", "")
            # Get question_id from gold data
            question_id = gold_data[idx].get("question_id", f"unknown_{idx}") if idx < len(gold_data) else f"unknown_{idx}"
            
            if error_msg:
                f.write(f"instance : {question_id} {status} | Error: {error_msg}\n")
            else:
                f.write(f"instance : {question_id} {status}\n")
    
    # Also print to terminal (previous print_data functionality)
    tqdm.write(f"start calculate {metric}")
    tqdm.write("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    tqdm.write("{:20} {:<20} {:<20} {:<20} {:<20}".format("count", *count_lists))
    tqdm.write(f"======================================    {metric}   =====================================")
    tqdm.write("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format(metric, *score_lists))
    tqdm.write("===========================================================================================")
    tqdm.write(f"Finished {metric} evaluation for test set")

    tqdm.write(f"Results and summary saved to {output_txt_path}")