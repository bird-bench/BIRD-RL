#!/usr/bin/env python3
"""
Prepare Stage 3 RL training data: single-turn reasoning RL (critic).

Creates parquet files in VERL RL format with prompt, reward_model, and extra_info
fields needed for GRPO/PPO training with the critic reward function.

The prompt uses critic_reasoning templates (thought + solution format).

Output parquet schema:
{
    "data_source": "bird_critic/sqlite_train",
    "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    "ability": "sql_debugging",
    "reward_model": {"ground_truth": "..."},
    "extra_info": {
        "instance_id": "...",
        "db_id": "...",
        "test_cases": [...],
        "preprocess_sql": [...],
        ...
    }
}

Usage:
    python -m bird_rl.data.prepare_reasoning_rl_data \
        --data <train.jsonl> \
        --schema <train_schema.jsonl> \
        --output <train_rl.parquet> \
        --data-source bird_critic/sqlite_train \
        --split train \
        [--max-samples 200]
"""

import argparse
import json
import random
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq

from bird_rl.prompts.critic_reasoning import SYSTEM_PROMPT, USER_PROMPT


def load_jsonl(file_path: str) -> List[dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def create_prompt_message(query: str, schema: str, issue_sql: str) -> List[dict]:
    """Create prompt in chat format using critic_reasoning templates."""
    user_content = USER_PROMPT.format(
        query=query.strip(),
        schema=schema.strip() if schema else "(No schema provided)",
        issue_sql=issue_sql.strip()
    )
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_content}
    ]


def process_split(
    data_jsonl_path: str,
    schema_jsonl_path: str,
    data_source_name: str,
    split_name: str,
    max_samples: int = None
) -> List[dict]:
    """
    Process data split and return list of rows for VERL RL training.

    Args:
        data_jsonl_path: Path to data JSONL (train.jsonl or prompts.jsonl)
        schema_jsonl_path: Path to schema JSONL
        data_source_name: Data source identifier for VERL
        split_name: Split name (train/dev)
        max_samples: Max samples to include (None = all)

    Returns:
        List of row dicts for parquet
    """
    print(f"Loading data from: {data_jsonl_path}")
    data = load_jsonl(data_jsonl_path)
    print(f"  Loaded {len(data)} instances")

    print(f"Loading schema from: {schema_jsonl_path}")
    schema_data = load_jsonl(schema_jsonl_path)
    print(f"  Loaded {len(schema_data)} schema instances")

    schema_lookup = {
        item['instance_id']: item['after_preprocess_schema']
        for item in schema_data
    }

    # Filter valid instances
    valid_data = []
    for item in data:
        if (item.get('query') and item.get('issue_sql') and
                item.get('sol_sql') and item.get('test_cases') and
                item.get('db_id')):
            valid_data.append(item)

    print(f"  Valid instances: {len(valid_data)} / {len(data)}")

    if max_samples is not None and len(valid_data) > max_samples:
        random.seed(42)
        valid_data = random.sample(valid_data, max_samples)
        print(f"  Sampled {max_samples} instances")

    processed_rows = []
    for idx, item in enumerate(valid_data):
        instance_id = item.get('instance_id')
        query = item.get('query', '')
        issue_sql = item.get('issue_sql', '')
        sol_sql = item.get('sol_sql', '')
        test_cases = item.get('test_cases', [])
        preprocess_sql = item.get('preprocess_sql', [])
        clean_up_sql = item.get('clean_up_sql', [])
        db_id = item.get('db_id', '')
        selected_database = item.get('selected_database', db_id)
        schema = schema_lookup.get(instance_id, '')

        if isinstance(issue_sql, list):
            issue_sql = '\n'.join(issue_sql)
        if isinstance(sol_sql, list):
            sol_sql = '\n'.join(sol_sql)

        prompt_messages = create_prompt_message(query, schema, issue_sql)

        if hasattr(test_cases, 'tolist'):
            test_cases = test_cases.tolist()
        if hasattr(preprocess_sql, 'tolist'):
            preprocess_sql = preprocess_sql.tolist()
        if hasattr(clean_up_sql, 'tolist'):
            clean_up_sql = clean_up_sql.tolist()

        row = {
            'data_source': data_source_name,
            'prompt': prompt_messages,
            'ability': 'sql_debugging',
            'reward_model': {
                'ground_truth': sol_sql,
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
            }
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
        description="Prepare Stage 3 RL data: single-turn reasoning (critic)"
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data JSONL (train.jsonl or prompts.jsonl)')
    parser.add_argument('--schema', type=str, required=True,
                        help='Path to schema JSONL')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output parquet file')
    parser.add_argument('--data-source', type=str, default='bird_critic/sqlite_train',
                        help='Data source name for VERL (default: bird_critic/sqlite_train)')
    parser.add_argument('--split', type=str, default='train',
                        help='Split name: train or dev (default: train)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to include (default: all)')
    args = parser.parse_args()

    print("=" * 60)
    print("Prepare Stage 3 RL Data (Single-Turn Reasoning)")
    print("=" * 60)

    rows = process_split(
        data_jsonl_path=args.data,
        schema_jsonl_path=args.schema,
        data_source_name=args.data_source,
        split_name=args.split,
        max_samples=args.max_samples
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_parquet(rows, args.output)

    print(f"\nSaved {len(rows)} instances to: {args.output}")


if __name__ == "__main__":
    main()
