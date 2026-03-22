#!/usr/bin/env python3
"""
Prepare Stage 1 SFT training data: single-turn reasoning (critic).

This script:
1. Reads LLM responses containing generated <thought> blocks
2. Reads original training data (with sol_sql as ground truth)
3. Extracts thoughts and pairs them with ground-truth SQL
4. Creates SFT parquet in VERL multi-turn format:
   [
       {"role": "system", "content": system_prompt},
       {"role": "user", "content": user_prompt},
       {"role": "assistant", "content": "<thought>...</thought>\\n\\n<solution>...</solution>"}
   ]

The system/user prompts use critic_reasoning templates (WITHOUT ground truth),
so the model learns to reason and produce the solution from the problem alone.

Usage:
    python -m bird_rl.data.prepare_reasoning_sft_data \
        --response_data <responses.jsonl> \
        --train_data <train.jsonl> \
        --schema_data <train_schema.jsonl> \
        --output_path <sft_train.parquet>
"""

import argparse
import json
import re
import os
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from bird_rl.prompts.critic_reasoning import SYSTEM_PROMPT, USER_PROMPT


def load_jsonl(file_path: str) -> list:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_thought(response: str) -> Optional[str]:
    """Extract content from <thought> tags in LLM response."""
    match = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def format_assistant_response(thought: str, solution_sql: str) -> str:
    """Format the assistant response with thought and solution tags."""
    return f"""<thought>
{thought}
</thought>

<solution>
{solution_sql.strip()}
</solution>"""


def prepare_sft_data(
    response_jsonl_path: str,
    train_jsonl_path: str,
    schema_jsonl_path: str,
) -> list:
    """
    Prepare SFT training data in VERL multi-turn format.

    Args:
        response_jsonl_path: LLM responses with generated thoughts
        train_jsonl_path: Original training data with sol_sql
        schema_jsonl_path: Schema data with after_preprocess_schema

    Returns:
        List of dicts with 'messages' field for VERL SFT
    """
    print(f"Loading responses from: {response_jsonl_path}")
    responses = load_jsonl(response_jsonl_path)
    print(f"  Loaded {len(responses)} responses")

    print(f"Loading train data from: {train_jsonl_path}")
    train_data = load_jsonl(train_jsonl_path)
    print(f"  Loaded {len(train_data)} train instances")

    print(f"Loading schema from: {schema_jsonl_path}")
    schema_data = load_jsonl(schema_jsonl_path)
    print(f"  Loaded {len(schema_data)} schema instances")

    # Create lookups
    train_lookup = {item['instance_id']: item for item in train_data}
    schema_lookup = {
        item['instance_id']: item['after_preprocess_schema']
        for item in schema_data
    }

    # Process responses
    results = []
    skipped_no_thought = 0
    skipped_no_train = 0

    for resp in responses:
        instance_id = resp.get('instance_id')
        raw_response = resp.get('raw_response', '')

        thought = extract_thought(raw_response)
        if not thought:
            skipped_no_thought += 1
            continue

        train_item = train_lookup.get(instance_id)
        if not train_item:
            skipped_no_train += 1
            continue

        query = train_item.get('query', '')
        issue_sql = train_item.get('issue_sql', '')
        sol_sql = train_item.get('sol_sql', '')

        if isinstance(issue_sql, list):
            issue_sql = '\n'.join(issue_sql)
        if isinstance(sol_sql, list):
            sol_sql = '\n'.join(sol_sql)

        schema = schema_lookup.get(instance_id, '')

        # Format using critic_reasoning prompts (no ground truth exposed)
        user_content = USER_PROMPT.format(
            query=query.strip(),
            schema=schema.strip() if schema else "(No schema provided)",
            issue_sql=issue_sql.strip()
        )
        assistant_content = format_assistant_response(thought, sol_sql)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]

        results.append({"messages": messages})

    print(f"\nProcessed {len(results)} valid instances")
    print(f"Skipped {skipped_no_thought} (no thought extracted)")
    print(f"Skipped {skipped_no_train} (no matching train data)")
    return results


def save_parquet(data: list, output_path: str):
    """Save data to parquet file for VERL SFT."""
    messages_list = [item["messages"] for item in data]
    table = pa.table({"messages": messages_list})
    pq.write_table(table, output_path)
    print(f"Saved parquet to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Stage 1 SFT data: single-turn reasoning (critic)"
    )
    parser.add_argument('--response_data', type=str, required=True,
                        help='Path to LLM responses JSONL (with generated thoughts)')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data JSONL (with sol_sql)')
    parser.add_argument('--schema_data', type=str, required=True,
                        help='Path to schema JSONL')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output parquet file')
    args = parser.parse_args()

    print("=" * 60)
    print("Prepare Stage 1 SFT Data (Single-Turn Reasoning)")
    print("=" * 60)

    results = prepare_sft_data(
        response_jsonl_path=args.response_data,
        train_jsonl_path=args.train_data,
        schema_jsonl_path=args.schema_data,
    )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    save_parquet(results, args.output_path)
    print(f"\nSaved {len(results)} SFT instances to: {args.output_path}")

    if results:
        msgs = results[0]['messages']
        print(f"\nExample: {len(msgs)} messages, "
              f"assistant response: {len(msgs[2]['content'])} chars")


if __name__ == "__main__":
    main()
