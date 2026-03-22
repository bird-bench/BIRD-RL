#!/usr/bin/env python3
"""
Postprocess trajectories: extract SQL from submit_solution for evaluation.

This script:
1. Extracts SQL from last turn's <tool_call> in trajectory
   - ONLY from submit_solution: extracts sql_list (array)
   - execute_sql is IGNORED (exploration only, not final answer)
2. Matches with original data file
3. Adds 'pred_sqls' field for evaluation

Usage:
    python -m bird_rl.data.postprocess_trajectories \
        --trajectory-file <traj_4.jsonl> \
        --data-file <train.jsonl> \
        --output-file <output_with_pred_sqls.jsonl>
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional


def extract_sql_from_tool_call(action: str) -> Optional[List[str]]:
    """
    Extract SQL from <tool_call> tags. Only extracts from submit_solution.

    Returns:
        List of SQL strings if submit_solution found, None otherwise
    """
    if not action or not isinstance(action, str):
        return None

    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(tool_call_pattern, action, re.DOTALL | re.IGNORECASE)

    if not matches:
        return None

    tool_call_content = matches[-1].strip()

    try:
        tool_call_data = json.loads(tool_call_content)
        tool_name = tool_call_data.get('name', '')
        arguments = tool_call_data.get('arguments', {})

        if tool_name == 'submit_solution':
            sql_list = arguments.get('sql_list', [])
            if isinstance(sql_list, list) and sql_list:
                return [str(sql).strip() for sql in sql_list if sql]
        return None

    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser(description='Postprocess trajectories for evaluation')
    parser.add_argument('--trajectory-file', type=str, required=True,
                        help='Input trajectory JSONL file')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Original data JSONL file (train.jsonl)')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output JSONL file with pred_sqls added')
    args = parser.parse_args()

    print("=" * 60)
    print("Postprocess Trajectories")
    print("=" * 60)

    # Extract SQL from trajectories
    predictions = {}
    stats = {'total': 0, 'submit_solution': 0, 'no_sql_found': 0}

    with open(args.trajectory_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            stats['total'] += 1
            instance_idx = data.get('instance_idx')
            trajectory = data.get('trajectory', [])

            if not trajectory:
                predictions[instance_idx] = []
                stats['no_sql_found'] += 1
                continue

            last_turn = trajectory[-1]
            pred_sqls = extract_sql_from_tool_call(last_turn.get('action', ''))

            if pred_sqls:
                predictions[instance_idx] = pred_sqls
                stats['submit_solution'] += 1
            else:
                predictions[instance_idx] = []
                stats['no_sql_found'] += 1

    print(f"  Extracted from {stats['total']} trajectories")
    print(f"  submit_solution: {stats['submit_solution']}, no SQL: {stats['no_sql_found']}")

    # Match with original data
    processed_data = []
    with open(args.data_file, 'r', encoding='utf-8') as f:
        for instance_idx, line in enumerate(f):
            if not line.strip():
                continue
            data_entry = json.loads(line)
            data_entry['pred_sqls'] = predictions.get(instance_idx, [])
            processed_data.append(data_entry)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"  Saved {len(processed_data)} entries -> {args.output_file}")


if __name__ == "__main__":
    main()
