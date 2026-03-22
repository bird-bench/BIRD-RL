#!/usr/bin/env python3
"""
Generate turn prompts for Stage 2 multi-turn SFT data generation.

This script reads training data and existing trajectory (if any) to generate
prompts for the next turn. Only generates prompts for instances where end_flag=False.

Uses SFT_GENERATION_SYSTEM_PROMPT/USER_TEMPLATE which include the ground-truth
solution so the LLM can generate realistic debugging trajectories.

Output JSONL format:
{
    "idx": 0,
    "instance_idx": 1234,
    "instance_id": "...",
    "db_id": "...",
    "current_turn": 0,
    "max_turns": 5,
    "system_prompt": "...",
    "prompt": "..."
}

Usage:
    python -m bird_rl.data.generate_turn_prompts \
        --turn 0 \
        --max-turns 5 \
        --train-data <train.jsonl> \
        --schema-data <train_schema.jsonl> \
        --traj-dir <trajectories_dir> \
        --output <output.jsonl>
"""

import argparse
import json
from pathlib import Path

from bird_rl.prompts.sft_generation import (
    SFT_GENERATION_SYSTEM_PROMPT,
    SFT_GENERATION_USER_TEMPLATE,
)


def build_history_from_trajectory(trajectory: list) -> str:
    """Build history string from trajectory list."""
    if not trajectory:
        return ""

    history_parts = ["## Previous Turns:\n"]

    for turn_idx, turn in enumerate(trajectory):
        history_parts.append(f"### Turn {turn_idx}:")

        if turn.get('thought'):
            history_parts.append(f"{turn['thought']}\n")

        if turn.get('action'):
            history_parts.append(f"{turn['action']}\n")

            if not turn.get('end_flag', False):
                observation = turn.get('observation', '')
                if observation:
                    history_parts.append(f"Observation:\n{observation}\n")

        history_parts.append("")

    return '\n'.join(history_parts)


def process_dataset(
    input_path: str,
    schema_path: str,
    output_path: str,
    traj_dir: str,
    current_turn: int,
    max_samples: int = None,
    max_turns: int = 5
):
    """
    Process dataset and generate turn prompts.

    Args:
        input_path: Path to training data JSONL
        schema_path: Path to schema JSONL
        output_path: Path to output JSONL
        traj_dir: Directory containing trajectory files
        current_turn: Current turn number (0 for first turn)
        max_samples: Maximum number of samples (None = all)
        max_turns: Maximum turns per trajectory
    """
    print(f"Loading data from: {input_path}")
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for instance_idx, line in enumerate(f):
            item = json.loads(line)
            item['instance_idx'] = instance_idx
            data.append(item)
    print(f"  Loaded {len(data)} instances")

    if max_samples is not None and max_samples < len(data):
        print(f"  Limiting to {max_samples} samples")
        data = data[:max_samples]

    print(f"Loading schema from: {schema_path}")
    schema_dict = {}
    with open(schema_path, 'r', encoding='utf-8') as f:
        for instance_idx, line in enumerate(f):
            schema_item = json.loads(line)
            schema_dict[instance_idx] = schema_item.get('after_preprocess_schema', '')
    print(f"  Loaded schema for {len(schema_dict)} instances")

    trajectory_dict = {}
    if current_turn > 0:
        traj_path = Path(traj_dir) / f"traj_{current_turn - 1}.jsonl"
        print(f"Loading trajectory from: {traj_path}")
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
        with open(traj_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    traj_item = json.loads(line)
                    trajectory_dict[traj_item['instance_idx']] = traj_item.get('trajectory', [])
        print(f"  Loaded trajectory for {len(trajectory_dict)} instances")
    else:
        print("  Turn 0 - generating prompts for all instances")

    prompts = []
    skipped = 0
    already_finished = 0

    for item in data:
        instance_idx = item['instance_idx']
        instance_id = item.get('instance_id', '')
        db_id = item.get('db_id', '')
        query = item.get('query', '')
        issue_sql = item.get('issue_sql', [])
        sol_sql = item.get('sol_sql', [])
        schema = schema_dict.get(instance_idx, '')

        if not query or not issue_sql or not sol_sql:
            skipped += 1
            continue

        trajectory = trajectory_dict.get(instance_idx, [])

        if current_turn > 0:
            if not trajectory:
                already_finished += 1
                continue
            if trajectory[-1].get('end_flag', False):
                already_finished += 1
                continue
            if len(trajectory) != current_turn:
                raise AssertionError(
                    f"Instance_idx {instance_idx}: trajectory length {len(trajectory)} != current_turn {current_turn}"
                )

        if current_turn >= max_turns:
            already_finished += 1
            continue

        history = build_history_from_trajectory(trajectory)

        issue_sql_str = '\n'.join(issue_sql) if isinstance(issue_sql, list) else issue_sql
        solution_sql_str = '\n'.join(sol_sql) if isinstance(sol_sql, list) else sol_sql

        system_prompt = SFT_GENERATION_SYSTEM_PROMPT.format(
            max_turns=max_turns,
            prev_turns=max_turns - 1
        )

        user_prompt = SFT_GENERATION_USER_TEMPLATE.format(
            query=query.strip(),
            schema=schema.strip() if schema else "(No schema provided)",
            issue_sql=issue_sql_str.strip(),
            solution_sql=solution_sql_str.strip(),
            max_turns=max_turns
        )

        if history:
            user_prompt = user_prompt + "\n\n" + history

        prompt_record = {
            'idx': len(prompts),
            'instance_idx': instance_idx,
            'instance_id': instance_id,
            'db_id': db_id,
            'current_turn': current_turn,
            'max_turns': max_turns,
            'system_prompt': system_prompt,
            'prompt': user_prompt,  # 'prompt' for call_api.py compatibility
        }

        prompts.append(prompt_record)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt, ensure_ascii=False) + '\n')

    print(f"  Generated {len(prompts)} prompts")
    print(f"  Skipped (missing fields): {skipped}")
    print(f"  Skipped (already finished): {already_finished}")

    return len(prompts)


def main():
    parser = argparse.ArgumentParser(description='Generate turn prompts for SFT data generation')
    parser.add_argument('--turn', type=int, default=0, help='Current turn number')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples')
    parser.add_argument('--max-turns', type=int, default=5, help='Maximum turns per trajectory')
    parser.add_argument('--train-data', type=str, required=True, help='Path to train.jsonl')
    parser.add_argument('--schema-data', type=str, required=True, help='Path to train_schema.jsonl')
    parser.add_argument('--traj-dir', type=str, default='', help='Trajectory directory')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    args = parser.parse_args()

    print("=" * 60)
    print(f"Generate Turn {args.turn} Prompts")
    print("=" * 60)

    prompt_count = process_dataset(
        input_path=args.train_data,
        schema_path=args.schema_data,
        output_path=args.output,
        traj_dir=args.traj_dir,
        current_turn=args.turn,
        max_samples=args.limit,
        max_turns=args.max_turns
    )

    print(f"\nGenerated {prompt_count} prompts -> {args.output}")


if __name__ == "__main__":
    main()
