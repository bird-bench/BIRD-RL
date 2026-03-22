#!/usr/bin/env python3
"""
Prepare Stage 4 RL training data: agentic RL (critic with tool use).

Creates parquet files in VERL RL format for agentic/trajectory training with
execute_sql and submit_solution tools.

Uses SFT_TRAINING prompts (multi-turn tool-calling format, without GT solution).

Output parquet schema:
{
    "data_source": "bird_critic/sqlite_train",
    "prompt": [{"role": "system", ...}, {"role": "user", ...}],
    "ability": "sql_debugging",
    "reward_model": {"ground_truth": [...], "test_cases": [...]},
    "extra_info": {
        "instance_id": "...",
        "db_id": "...",
        "test_cases": [...],
        "tools_kwargs": {
            "execute_sql": {"create_kwargs": {...}},
            "submit_solution": {"create_kwargs": {...}}
        },
        ...
    },
    "return_raw_chat": true,
    "agent_name": "tool_agent_with_db_cleanup"
}

Usage:
    python -m bird_rl.data.prepare_agentic_rl_data \
        --data <train.jsonl> \
        --schema <train_schema.jsonl> \
        --output <train_rl.parquet> \
        --data-source bird_critic/sqlite_train \
        --split train \
        [--max-samples 50] [--max-turns 5]
"""

import argparse
import json
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq

from bird_rl.prompts.sft_generation import (
    SFT_TRAINING_SYSTEM_PROMPT,
    SFT_TRAINING_USER_TEMPLATE,
)


def load_jsonl(file_path: str) -> List[dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def create_prompt_message(query: str, schema: str, issue_sql: str, max_turns: int = 5) -> List[dict]:
    """Create prompt in chat format for agentic training."""
    system_content = SFT_TRAINING_SYSTEM_PROMPT.format(
        max_turns=max_turns,
        prev_turns=max_turns - 1
    )
    user_content = SFT_TRAINING_USER_TEMPLATE.format(
        query=query.strip(),
        schema=schema.strip() if schema else "(No schema provided)",
        issue_sql=issue_sql.strip(),
        max_turns=max_turns
    )
    return [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content}
    ]


def process_split(
    data_jsonl_path: str,
    schema_jsonl_path: str,
    data_source_name: str,
    split_name: str,
    max_samples: int = None,
    max_turns: int = 5,
) -> List[dict]:
    """
    Process data split and return list of rows for VERL agentic RL training.

    Data and schema are matched by index (line-by-line), not by instance_id,
    because instance_id may have duplicates.

    Args:
        data_jsonl_path: Path to data JSONL (train.jsonl or prompts.jsonl)
        schema_jsonl_path: Path to schema JSONL
        data_source_name: Data source identifier for VERL
        split_name: Split name (train/dev)
        max_samples: Max samples to include (None = all)
        max_turns: Maximum turns for trajectory
    """
    print(f"Loading data from: {data_jsonl_path}")
    data = load_jsonl(data_jsonl_path)
    print(f"  Loaded {len(data)} instances")

    print(f"Loading schema from: {schema_jsonl_path}")
    schema_data = load_jsonl(schema_jsonl_path)
    print(f"  Loaded {len(schema_data)} schema instances")

    if len(data) != len(schema_data):
        print(f"  WARNING: data ({len(data)}) and schema ({len(schema_data)}) have different lengths!")

    # Match by index, filter valid instances
    valid_tuples = []
    skipped_count = 0

    for original_idx, (item, schema_item) in enumerate(zip(data, schema_data)):
        if not all([item.get('query'), item.get('issue_sql'), item.get('sol_sql'),
                    item.get('test_cases'), item.get('db_id')]):
            skipped_count += 1
            continue
        valid_tuples.append((item, schema_item, original_idx))

    print(f"  Valid instances: {len(valid_tuples)} / {len(data)}")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} (missing required fields)")

    if max_samples is not None and len(valid_tuples) > max_samples:
        valid_tuples = valid_tuples[:max_samples]
        print(f"  Limited to {max_samples} instances")

    processed_rows = []
    for idx, (item, schema_item, original_idx) in enumerate(valid_tuples):
        instance_id = item.get('instance_id')
        query = item.get('query', '')
        issue_sql = item.get('issue_sql', '')
        sol_sql = item.get('sol_sql', '')
        test_cases = item.get('test_cases', [])
        preprocess_sql = item.get('preprocess_sql', [])
        clean_up_sql = item.get('clean_up_sql', [])
        db_id = item.get('db_id', '')
        selected_database = item.get('selected_database', db_id)

        schema = schema_item.get('after_preprocess_schema', '')

        if isinstance(issue_sql, list):
            issue_sql = '\n'.join(issue_sql)

        # Keep sol_sql as list for agentic format
        if not isinstance(sol_sql, list):
            sol_sql = [sol_sql] if sol_sql else []

        prompt_messages = create_prompt_message(query, schema, issue_sql, max_turns)

        if hasattr(test_cases, 'tolist'):
            test_cases = test_cases.tolist()
        if hasattr(preprocess_sql, 'tolist'):
            preprocess_sql = preprocess_sql.tolist()
        if hasattr(clean_up_sql, 'tolist'):
            clean_up_sql = clean_up_sql.tolist()

        shared_create_kwargs = {
            'db_id': db_id,
            'preprocess_sql': list(preprocess_sql) if preprocess_sql else [],
            'test_cases': list(test_cases) if test_cases else [],
            'ground_truth': sol_sql,
            'query': query,
            'schema': schema,
        }

        tools_kwargs = {
            'execute_sql': {'create_kwargs': shared_create_kwargs},
            'submit_solution': {'create_kwargs': shared_create_kwargs},
        }

        row = {
            'data_source': data_source_name,
            'prompt': prompt_messages,
            'ability': 'sql_debugging',
            'reward_model': {
                'ground_truth': sol_sql,
                'test_cases': list(test_cases) if test_cases else [],
            },
            'extra_info': {
                'instance_id': instance_id,
                'query': query,
                'schema': schema,
                'issue_sql': issue_sql,
                'index': idx,
                'split': split_name,
                'db_id': db_id,
                'selected_database': selected_database,
                'preprocess_sql': list(preprocess_sql) if preprocess_sql else [],
                'clean_up_sql': list(clean_up_sql) if clean_up_sql else [],
                'test_cases': list(test_cases) if test_cases else [],
                'need_tools_kwargs': True,
                'tools_kwargs': tools_kwargs,
            },
            'return_raw_chat': True,
            'agent_name': 'tool_agent_with_db_cleanup',
        }
        processed_rows.append(row)

    print(f"  Processed {len(processed_rows)} instances")
    return processed_rows


def save_parquet(rows: List[dict], output_path: str):
    """Save rows to parquet file."""
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(output_path, index=False)
    except ImportError:
        columns = {}
        for key in rows[0].keys():
            columns[key] = [row[key] for row in rows]
        table = pa.table(columns)
        pq.write_table(table, output_path)

    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Stage 4 RL data: agentic RL (critic with tool use)"
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data JSONL (train.jsonl or prompts.jsonl)')
    parser.add_argument('--schema', type=str, required=True,
                        help='Path to schema JSONL')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output parquet file')
    parser.add_argument('--data-source', type=str, default='bird_critic/sqlite_train',
                        help='Data source name for VERL')
    parser.add_argument('--split', type=str, default='train',
                        help='Split name: train or dev')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to include (default: all)')
    parser.add_argument('--max-turns', type=int, default=5,
                        help='Maximum turns for trajectory (default: 5)')
    args = parser.parse_args()

    print("=" * 60)
    print("Prepare Stage 4 RL Data (Agentic RL)")
    print("=" * 60)

    rows = process_split(
        data_jsonl_path=args.data,
        schema_jsonl_path=args.schema,
        data_source_name=args.data_source,
        split_name=args.split,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_parquet(rows, args.output)

    print(f"\nSaved {len(rows)} instances to: {args.output}")


if __name__ == "__main__":
    main()
