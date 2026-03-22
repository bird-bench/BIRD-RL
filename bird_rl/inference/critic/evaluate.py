#!/usr/bin/env python3
"""
Extract final SQL from critic trajectories and prepare for evaluation.

The critic evaluation uses the existing evaluation pipeline in
evaluation/critic/src/ which runs test cases against ephemeral databases.
This script extracts the submitted SQL from trajectories and formats it
for the evaluation pipeline.
"""

import json
import re
import argparse
from pathlib import Path


def extract_final_sql(traj_item: dict) -> list:
    """Extract the final submitted SQL list from a trajectory item."""
    trajectory = traj_item.get("trajectory", [])
    if not trajectory:
        return []

    last_turn = trajectory[-1]
    if not last_turn.get("end_flag", False):
        return []

    action = last_turn.get("action", "")
    if not action:
        return []

    match = re.search(r"<tool_call>(.*?)</tool_call>", action, re.DOTALL)
    if not match:
        return []

    try:
        tool_data = json.loads(match.group(1))
    except json.JSONDecodeError:
        fixed = match.group(1).replace("{{", "{").replace("}}", "}")
        try:
            tool_data = json.loads(fixed)
        except json.JSONDecodeError:
            return []

    if not isinstance(tool_data, dict):
        return []

    args = tool_data.get("arguments", {})
    if not isinstance(args, dict):
        return []

    sql_list = args.get("sql_list", [])
    if isinstance(sql_list, list) and sql_list:
        return [str(s).strip() for s in sql_list if s]

    sql = args.get("sql", "")
    if isinstance(sql, str) and sql.strip():
        return [sql.strip()]

    return []


def prepare_for_evaluation(
    traj_path: str,
    original_data_path: str,
    output_path: str,
):
    """
    Extract SQL from trajectories and merge with original data for evaluation.

    The output JSONL can be fed to evaluation/critic/src/evaluate.py.

    Args:
        traj_path: Path to trajectory JSONL
        original_data_path: Path to original input JSONL (with test_cases, db_id, etc.)
        output_path: Path to write evaluation-ready JSONL
    """
    # Load trajectories
    print(f"Loading trajectories from: {traj_path}")
    traj_dict = {}
    with open(traj_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                traj_dict[item.get("instance_idx")] = item

    # Load original data
    print(f"Loading original data from: {original_data_path}")
    original_data = []
    with open(original_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                original_data.append(json.loads(line))

    print(f"  {len(traj_dict)} trajectories, {len(original_data)} original instances")

    results = []
    extracted = 0

    for instance_idx, item in enumerate(original_data):
        traj_item = traj_dict.get(instance_idx, {})
        pred_sqls = extract_final_sql(traj_item)

        eval_item = dict(item)
        eval_item["instance_idx"] = instance_idx
        eval_item["pred_sqls"] = pred_sqls

        if pred_sqls:
            extracted += 1

        results.append(eval_item)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"  Extracted SQL from {extracted}/{len(results)} instances")
    print(f"  Saved to: {output_path}")
    print(f"\nTo evaluate, run:")
    print(f"  bash evaluation/critic/run/run_eval.sh --jsonl_file {output_path} --db_dir <DB_DIR>")


def main():
    parser = argparse.ArgumentParser(description='Extract SQL from critic trajectories for evaluation')
    parser.add_argument('--trajectory', type=str, required=True, help='Trajectory JSONL file')
    parser.add_argument('--original-data', type=str, required=True, help='Original input JSONL (with test_cases)')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL for evaluation pipeline')
    args = parser.parse_args()

    prepare_for_evaluation(
        traj_path=args.trajectory,
        original_data_path=args.original_data,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
