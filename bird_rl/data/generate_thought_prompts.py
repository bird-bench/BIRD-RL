#!/usr/bin/env python3
"""
Generate prompts for creating Stage 1 SFT training data (critic).

This script:
1. Loads training data (query, schema, issue_sql, solution_sql)
2. Formats prompts using critic_thought_generation templates
3. Outputs JSONL file for LLM API calls to generate thoughts

The output JSONL has fields: instance_id, system_prompt, prompt
and can be fed to call_api.py to collect LLM responses.

Usage:
    python -m bird_rl.data.generate_thought_prompts \
        --train_data <train.jsonl> \
        --schema_data <train_schema.jsonl> \
        --output_path <output.jsonl>
"""

import argparse
import json
import os
from pathlib import Path

from bird_rl.prompts.critic_thought_generation import (
    SYSTEM_PROMPT,
    USER_PROMPT,
)


def load_jsonl(file_path: str) -> list:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def process_data(data_jsonl_path: str, schema_jsonl_path: str, limit: int = None) -> list:
    """
    Load training data and schema, format prompts for thought generation.

    Args:
        data_jsonl_path: Path to training JSONL (fields: instance_id, query, issue_sql, sol_sql, db_id)
        schema_jsonl_path: Path to schema JSONL (fields: instance_id, after_preprocess_schema)
        limit: Limit number of instances

    Returns:
        List of prompt dicts with: instance_id, system_prompt, prompt
    """
    print(f"Loading data from: {data_jsonl_path}")
    data = load_jsonl(data_jsonl_path)
    print(f"  Loaded {len(data)} instances")

    print(f"Loading schema from: {schema_jsonl_path}")
    schema_data = load_jsonl(schema_jsonl_path)
    print(f"  Loaded {len(schema_data)} schema instances")

    # Create schema lookup
    schema_lookup = {
        item['instance_id']: item['after_preprocess_schema']
        for item in schema_data
    }

    # Filter valid instances
    valid_data = [
        item for item in data
        if item.get('query') and item.get('issue_sql') and item.get('sol_sql') and item.get('db_id')
    ]
    print(f"  Found {len(valid_data)} valid instances")

    if limit and limit < len(valid_data):
        valid_data = valid_data[:limit]
        print(f"  Limited to {limit} instances")

    # Generate prompts
    results = []
    for item in valid_data:
        instance_id = item.get('instance_id')
        query = item.get('query', '')
        issue_sql = item.get('issue_sql', '')
        sol_sql = item.get('sol_sql', '')

        if isinstance(issue_sql, list):
            issue_sql = '\n'.join(issue_sql)
        if isinstance(sol_sql, list):
            sol_sql = '\n'.join(sol_sql)

        schema = schema_lookup.get(instance_id, '')

        user_prompt = USER_PROMPT.format(
            query=query.strip(),
            schema=schema.strip() if schema else "(No schema provided)",
            issue_sql=issue_sql.strip(),
            solution_sql=sol_sql.strip()
        )

        results.append({
            'instance_id': instance_id,
            'system_prompt': SYSTEM_PROMPT,
            'prompt': user_prompt,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate prompts for SFT thought generation (critic)"
    )
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data JSONL')
    parser.add_argument('--schema_data', type=str, required=True, help='Path to schema JSONL')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output JSONL file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of instances')
    args = parser.parse_args()

    print("=" * 60)
    print("Generate Thought Prompts for Stage 1 SFT")
    print("=" * 60)

    results = process_data(
        data_jsonl_path=args.train_data,
        schema_jsonl_path=args.schema_data,
        limit=args.limit
    )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(results)} prompts to: {args.output_path}")

    if results:
        ex = results[0]
        print(f"\nExample: instance_id={ex['instance_id']}, "
              f"system_prompt={len(ex['system_prompt'])} chars, "
              f"user_prompt={len(ex['prompt'])} chars")


if __name__ == "__main__":
    main()
