import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut

from eval_utils import (
    load_jsonl,
    execute_sql,
    package_sqls,
    sort_results,
    save_results_to_jsonl,
    save_results_to_txt,
)
from tqdm import tqdm  # Import tqdm for progress tracking


def result_callback(result):
    exec_result.append(result)


def calculate_ex(predicted_res, ground_truth_res):
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out,dialect):
    try:
        # res, predicted_res, ground_truth_res, error_message
        res, predicted_res, ground_truth_res, error_message = func_timeout(
            meta_time_out,
            execute_sql,
            args=(predicted_sql, ground_truth, db_place, dialect, calculate_ex),
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = 0
        predicted_res, ground_truth_res = [], []
        error_message = "Execution timeout"
    except Exception as e:
        res = 0
        predicted_res, ground_truth_res = [], []
        error_message = f"Execution error: {str(e)}"
    
    result = {
        "sql_idx": idx,
        "res": res,
        "predicted_res": predicted_res,
        "ground_truth_res": ground_truth_res,
        "error_message": error_message,
    }
    return result


def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0,dialect=None):
    pool = mp.Pool(processes=num_cpus)
    results = []
    futures = []
    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        future = pool.apply_async(
            execute_model,
            args=(
                predicted_sql,
                ground_truth,
                db_places[i],
                i,
                meta_time_out,
                dialect
            ),
            callback=result_callback,
        )
        futures.append(future)

    # Initialize tqdm progress bar
    for future in tqdm(futures, total=len(futures), desc="Evaluating EX"):
        result = (
            future.get()
        )  # Get result from future, tqdm will handle the progress update
        results.append(result)

    pool.close()
    pool.join()
    return results


def compute_acc_by_diff(exec_results, diff_json_path):
    num_queries = len(exec_results)
    tqdm.write(f"Num of queires {num_queries}")
    results = [res["res"] for res in exec_results]
    contents = load_jsonl(diff_json_path)
    limit = len(exec_results)
    contents = contents[:limit]
    simple_results, moderate_results, challenging_results = [], [], []

    for i, content in enumerate(contents):
        if content["difficulty"] == "simple":
            simple_results.append(exec_results[i])

        if content["difficulty"] == "moderate":
            moderate_results.append(exec_results[i])

        if content["difficulty"] == "challenging":
            try:
                challenging_results.append(exec_results[i])
            except:
                tqdm.write(str(i))
        # simple_results.append(exec_results[i])
    # print(f"Simple Correct: {sum([res['res'] for res in simple_results])}")
    # print(f"Moderate Correct: {sum([res['res'] for res in moderate_results])}")
    # print(f"Challenging Correct: {sum([res['res'] for res in challenging_results])}")
    simple_acc = sum([res["res"] for res in simple_results]) / len(simple_results) if len(simple_results) else 0
    moderate_acc = sum([res["res"] for res in moderate_results]) / len(moderate_results) if len(moderate_results) else 0
    challenging_acc = sum([res["res"] for res in challenging_results]) / len(
        challenging_results
    ) if len(challenging_results) else 0
    all_acc = sum(results) / num_queries
    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        num_queries,
    ]
    return (
        simple_acc * 100,
        moderate_acc * 100,
        challenging_acc * 100,
        all_acc * 100,
        count_lists,
    )


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--pred_file", type=str, required=True, default=""
    )
    args_parser.add_argument("--gold_file", type=str, required=True, default="")
    args_parser.add_argument("--output_file", type=str, default="")
    args_parser.add_argument("--dialect", type=str)
    args_parser.add_argument("--db_dir", type=str)
    args_parser.add_argument("--num_cpus", type=int, default=8)
    args_parser.add_argument("--meta_time_out", type=float, default=30.0)
    args = args_parser.parse_args()
    exec_result = []

    pred_queries, db_paths = package_sqls(
        args.pred_file,
        args.gold_file,
        mode="gpt",
        dialect=args.dialect,
        db_dir=args.db_dir
    )
    limit = len(pred_queries)
    # generate ground truth sqls:
    gt_queries, db_paths_gt, = package_sqls(
        args.pred_file,
        args.gold_file,
        mode="gt",
        dialect=args.dialect,
        db_dir=args.db_dir
    )
    gt_queries = gt_queries[:limit]
    tqdm.write(f"Num of queries {len(pred_queries)}")
    assert len(pred_queries) == len(gt_queries)
    query_pairs = list(zip(pred_queries, gt_queries))
    tqdm.write(str(query_pairs[0]))
    run_sqls_parallel(
        query_pairs,
        db_places=db_paths_gt,
        num_cpus=args.num_cpus,
        meta_time_out=args.meta_time_out,
        dialect=args.dialect
    )
    exec_result = sort_results(exec_result)
    
    # Compute accuracy by difficulty
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = compute_acc_by_diff(
        exec_result, args.gold_file
    )
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    
    # Save results
    if args.output_file:
        # Save to txt file (includes print_data functionality and instance results)
        save_results_to_txt(exec_result, args.gold_file, args.output_file, score_lists, count_lists, "EX")
        
        # Generate jsonl output filename and save
        if args.output_file.endswith('.txt'):
            output_jsonl_path = args.output_file.replace('.txt', '.jsonl')
        else:
            output_jsonl_path = args.output_file + '.jsonl'
        
        save_results_to_jsonl(exec_result, args.pred_file, args.gold_file, output_jsonl_path)