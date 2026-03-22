#!/usr/bin/env python3
"""
Prepare Stage 2 multi-turn SFT training data in VERL format.

This script creates multi-turn format with alternating assistant/user messages:
- assistant messages: thought + action (TRAINED)
- user messages: observations/tool responses (NOT trained)

Only includes trajectories from instances that passed evaluation (correct SQL).

Input:
- Status file: evaluation results with 'status' field
- Trajectory file: completed trajectories
- Train/Schema data: for creating prompts

Output: Parquet file with VERL multi-turn format:
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "problem description"},
        {"role": "assistant", "content": "thought + action"},
        {"role": "user", "content": "observation"},
        {"role": "assistant", "content": "thought + action"},
        ...
    ]
}

Usage:
    python -m bird_rl.data.prepare_multi_turn_sft_data \
        --status-file <eval_status.jsonl> \
        --trajectory-file <traj_4.jsonl> \
        --train-data <train.jsonl> \
        --schema-data <train_schema.jsonl> \
        --output-path <sft_data.parquet> \
        [--max-turns 5] [--use-think-tags]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

from bird_rl.prompts.sft_generation import create_sft_training_prompt


def load_evaluation_results(status_file: str) -> Dict[int, dict]:
    """Load evaluation results and return dict of correct instances."""
    print(f"Loading evaluation status from: {status_file}")
    correct_instances = {}
    with open(status_file, 'r', encoding='utf-8') as f:
        for instance_idx, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                if item.get('status') == 'success':
                    correct_instances[instance_idx] = {'instance_id': item.get('instance_id', '')}
    print(f"  Found {len(correct_instances)} successful instances")
    return correct_instances


def load_train_data(train_path: str) -> Dict[int, dict]:
    """Load train data indexed by instance_idx."""
    print(f"Loading train data from: {train_path}")
    train_dict = {}
    with open(train_path, 'r', encoding='utf-8') as f:
        for instance_idx, line in enumerate(f):
            if line.strip():
                train_dict[instance_idx] = json.loads(line)
    print(f"  Loaded {len(train_dict)} instances")
    return train_dict


def load_schema_data(schema_path: str) -> Dict[int, str]:
    """Load schema data indexed by instance_idx."""
    print(f"Loading schema data from: {schema_path}")
    schema_dict = {}
    with open(schema_path, 'r', encoding='utf-8') as f:
        for instance_idx, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                schema_dict[instance_idx] = item.get('after_preprocess_schema', '')
    print(f"  Loaded schema for {len(schema_dict)} instances")
    return schema_dict


def load_trajectories(traj_path: str) -> Dict[int, dict]:
    """Load trajectories indexed by instance_idx."""
    print(f"Loading trajectories from: {traj_path}")
    traj_dict = {}
    with open(traj_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                traj_dict[item['instance_idx']] = item
    print(f"  Loaded {len(traj_dict)} trajectories")
    return traj_dict


def convert_trajectory_to_multiturn(
    trajectory: List[dict],
    train_item: dict,
    schema: str,
    max_turns: int = 5,
    use_think_tags: bool = True
) -> List[dict]:
    """
    Convert a trajectory to VERL multi-turn message format.

    Args:
        trajectory: List of turn dicts with thought, action, observation
        train_item: Training item with query, issue_sql
        schema: Database schema string
        max_turns: Maximum turns for prompt
        use_think_tags: If True, use <think> tags (Qwen3), else <thought> tags
    """
    messages = []

    system_prompt, user_prompt = create_sft_training_prompt(
        query=train_item['query'],
        schema=schema,
        issue_sql=train_item['issue_sql'],
        max_turns=max_turns,
        conversation_history=""
    )

    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    for i, turn in enumerate(trajectory):
        assistant_parts = [f"### Turn {i + 1}"]

        if turn.get('thought'):
            thought = turn['thought']
            if use_think_tags:
                thought = thought.replace('<thought>', '<think>').replace('</thought>', '</think>')
            assistant_parts.append(thought)

        if turn.get('action'):
            assistant_parts.append(turn['action'])

        messages.append({"role": "assistant", "content": "\n\n".join(assistant_parts)})

        if turn.get('observation') and i < len(trajectory) - 1:
            messages.append({"role": "user", "content": f"Observation:\n{turn['observation']}"})

    return messages


def main():
    parser = argparse.ArgumentParser(description='Prepare Stage 2 multi-turn SFT data')
    parser.add_argument('--status-file', type=str, required=True,
                        help='Evaluation status JSONL (with status field)')
    parser.add_argument('--trajectory-file', type=str, required=True,
                        help='Trajectory JSONL file')
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to train.jsonl')
    parser.add_argument('--schema-data', type=str, required=True,
                        help='Path to train_schema.jsonl')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Output parquet file')
    parser.add_argument('--max-turns', type=int, default=5,
                        help='Maximum turns (default: 5)')
    parser.add_argument('--use-think-tags', action='store_true', default=True,
                        help='Use <think> tags instead of <thought> (default: True)')
    parser.add_argument('--no-think-tags', action='store_false', dest='use_think_tags',
                        help='Use <thought> tags instead of <think>')
    args = parser.parse_args()

    print("=" * 60)
    print("Prepare Stage 2 Multi-turn SFT Data")
    print("=" * 60)

    correct_instances = load_evaluation_results(args.status_file)
    trajectories = load_trajectories(args.trajectory_file)
    train_dict = load_train_data(args.train_data)
    schema_dict = load_schema_data(args.schema_data)

    all_examples = []
    skipped_no_traj = 0
    skipped_no_train = 0
    skipped_no_schema = 0

    for instance_idx in sorted(correct_instances.keys()):
        if instance_idx not in trajectories:
            skipped_no_traj += 1
            continue
        if instance_idx not in train_dict:
            skipped_no_train += 1
            continue
        if instance_idx not in schema_dict:
            skipped_no_schema += 1
            continue

        messages = convert_trajectory_to_multiturn(
            trajectory=trajectories[instance_idx]['trajectory'],
            train_item=train_dict[instance_idx],
            schema=schema_dict[instance_idx],
            max_turns=args.max_turns,
            use_think_tags=args.use_think_tags
        )

        all_examples.append({"messages": messages})

    print(f"\n  Correct instances: {len(correct_instances)}")
    print(f"  Skipped - no trajectory: {skipped_no_traj}")
    print(f"  Skipped - no train data: {skipped_no_train}")
    print(f"  Skipped - no schema: {skipped_no_schema}")
    print(f"  Generated {len(all_examples)} training examples")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    messages_list = [item["messages"] for item in all_examples]
    table = pa.table({"messages": messages_list})
    pq.write_table(table, args.output_path)

    print(f"\nSaved {len(all_examples)} SFT instances to: {args.output_path}")

    if all_examples:
        sample = all_examples[0]
        n_assistant = sum(1 for m in sample['messages'] if m['role'] == 'assistant')
        print(f"Example: {len(sample['messages'])} messages, {n_assistant} assistant turns")


if __name__ == '__main__':
    main()
