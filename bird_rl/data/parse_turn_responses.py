#!/usr/bin/env python3
"""
Parse model responses from turn-based SFT data generation (Two-Tool Design).

Extracts thought and action from each response. For responses with multiple
turns, only the FIRST turn is extracted.

Two-Tool Design:
- execute_sql: Single SQL query for testing → normalized to [sql], end_flag=False
- submit_solution: Final solution with multiple statements → kept as list, end_flag=True

Usage:
    python -m bird_rl.data.parse_turn_responses \
        --turn 0 \
        --input <responses.jsonl> \
        --output <parsed.jsonl>
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple


def extract_first_thought(response: str) -> Optional[str]:
    """Extract the FIRST <thought> block from response."""
    pattern = r'<thought>(.*?)</thought>'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    return None


def extract_first_tool_call(response: str) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Extract tool name and SQL from the FIRST <tool_call> in response.

    Returns:
        Tuple of (tool_name, sql_list) or (None, None)
    """
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

    if not matches:
        return None, None

    tool_call_json = matches[0].strip()

    # Remove markdown code fences if present
    tool_call_json = re.sub(r'^\s*```(?:json)?\s*\n?', '', tool_call_json, flags=re.MULTILINE)
    tool_call_json = re.sub(r'\n?\s*```\s*$', '', tool_call_json, flags=re.MULTILINE)
    tool_call_json = tool_call_json.strip()

    try:
        tool_call = json.loads(tool_call_json)
        if not isinstance(tool_call, dict):
            return None, None

        tool_name = tool_call.get('name', '')
        arguments = tool_call.get('arguments', {})

        if not isinstance(arguments, dict):
            return None, None

        if tool_name == 'execute_sql':
            sql = arguments.get('sql', '')
            if sql and isinstance(sql, str):
                return 'execute_sql', [sql]
            return 'execute_sql', None

        elif tool_name == 'submit_solution':
            sql_list = arguments.get('sql_list', [])
            if isinstance(sql_list, list) and len(sql_list) > 0:
                return 'submit_solution', sql_list
            return 'submit_solution', None

        else:
            # Unknown tool - try fallbacks
            sql = arguments.get('sql', '')
            if sql:
                return 'execute_sql', [sql]
            sql_list = arguments.get('sql_list', [])
            if sql_list:
                return 'execute_sql', sql_list
            return None, None

    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse tool_call JSON: {e}")
        return None, None


def parse_response(response: str) -> Tuple[str, str, List[str], bool]:
    """
    Parse response to extract the FIRST thought and action.

    Returns:
        Tuple of (thought, tool_name, pred_sqls, end_flag)
    """
    thought = extract_first_thought(response)
    if thought is None:
        thought = "[MISS]"

    tool_name, pred_sqls = extract_first_tool_call(response)

    if tool_name is None or pred_sqls is None:
        return thought, "[MISS]", ["[MISS]"], False

    end_flag = (tool_name == 'submit_solution')
    return thought, tool_name, pred_sqls, end_flag


def process_responses(input_path: str, output_path: str):
    """Process API responses and extract parsed turn data."""
    print(f"Loading responses from: {input_path}")
    responses = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    print(f"  Loaded {len(responses)} responses")

    parsed_data = []
    miss_count = 0
    execute_sql_count = 0
    submit_solution_count = 0

    for item in responses:
        idx = item.get('idx', len(parsed_data))
        instance_idx = item.get('instance_idx')
        instance_id = item.get('instance_id', '')
        db_id = item.get('db_id', '')
        raw_response = item.get('raw_response', '')

        thought, tool_name, pred_sqls, end_flag = parse_response(raw_response)

        if thought == "[MISS]" or tool_name == "[MISS]" or pred_sqls == ["[MISS]"]:
            miss_count += 1
        elif tool_name == 'execute_sql':
            execute_sql_count += 1
        elif tool_name == 'submit_solution':
            submit_solution_count += 1

        parsed_data.append({
            'idx': idx,
            'instance_idx': instance_idx,
            'instance_id': instance_id,
            'db_id': db_id,
            'thought': thought,
            'tool_name': tool_name,
            'pred_sqls': pred_sqls,
            'end_flag': end_flag
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in parsed_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"  Parsed {len(parsed_data)} turns ({miss_count} with missing data)")
    print(f"  execute_sql: {execute_sql_count}, submit_solution: {submit_solution_count}")

    return len(parsed_data)


def main():
    parser = argparse.ArgumentParser(description='Parse turn responses for SFT data generation')
    parser.add_argument('--turn', type=int, default=0, help='Current turn number')
    parser.add_argument('--input', type=str, required=True, help='Input file (API responses)')
    parser.add_argument('--output', type=str, required=True, help='Output file (parsed data)')
    args = parser.parse_args()

    print("=" * 60)
    print(f"Parse Turn {args.turn} Responses")
    print("=" * 60)

    count = process_responses(input_path=args.input, output_path=args.output)
    print(f"\nParsed {count} turns -> {args.output}")


if __name__ == "__main__":
    main()
