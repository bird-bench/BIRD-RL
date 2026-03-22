#!/usr/bin/env python3
"""
Build trajectory files for SFT data generation (Two-Tool Design).

This script:
1. Reads previous trajectory (if turn > 0)
2. Reads current parsed responses and observations
3. Appends new turn data to trajectory
4. Validates trajectory length
5. Saves updated trajectory

Trajectory format:
{
    "idx": 0,
    "instance_idx": 1234,
    "instance_id": "...",
    "db_id": "...",
    "trajectory": [
        {
            "thought": "<thought>...</thought>",
            "action": "<tool_call>{...}</tool_call>",
            "observation": "<tool_response>...</tool_response>",
            "end_flag": false
        }
    ]
}

Usage:
    python -m bird_rl.data.build_trajectory \
        --turn 0 \
        --traj-dir <trajectories_dir> \
        --observations <observations.jsonl> \
        --output <traj_0.jsonl>
"""

import argparse
import json
from pathlib import Path
from typing import List


def format_thought(thought: str) -> str:
    """Format thought with tags."""
    if thought == "[MISS]" or not thought:
        return ""
    return f"<thought>\n{thought}\n</thought>"


def format_action(pred_sqls: List[str], tool_name: str) -> str:
    """Format action with tool_call tags (Two-Tool Design)."""
    if pred_sqls == ["[MISS]"] or not pred_sqls:
        return ""

    if tool_name == "execute_sql":
        sql = pred_sqls[0] if len(pred_sqls) > 0 else ""
        tool_call = {"name": "execute_sql", "arguments": {"sql": sql}}
    elif tool_name == "submit_solution":
        tool_call = {"name": "submit_solution", "arguments": {"sql_list": pred_sqls}}
    else:
        sql = pred_sqls[0] if len(pred_sqls) > 0 else ""
        tool_call = {"name": "execute_sql", "arguments": {"sql": sql}}

    tool_call_json = json.dumps(tool_call, ensure_ascii=False)
    return f"<tool_call>{tool_call_json}</tool_call>"


def format_observation(exec_flag: bool, exec_results: str) -> str:
    """Format observation with tool_response tags."""
    if exec_results is None or exec_results == "":
        return ""
    status = "Success" if exec_flag else "Error"
    return f"<tool_response>\nExecution Status: {status}\n\n{exec_results}\n</tool_response>"


def build_trajectory(current_turn, traj_dir, observations_path, output_path):
    """Build trajectory file for current turn."""
    print(f"Building trajectory for turn {current_turn}")

    prev_traj_dict = {}
    if current_turn > 0:
        prev_traj_path = Path(traj_dir) / f"traj_{current_turn - 1}.jsonl"
        print(f"  Loading previous trajectory from: {prev_traj_path}")
        if not prev_traj_path.exists():
            raise FileNotFoundError(f"Previous trajectory not found: {prev_traj_path}")
        with open(prev_traj_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    prev_traj_dict[item['instance_idx']] = item
        print(f"    Loaded {len(prev_traj_dict)} instances")
    else:
        print("  Turn 0 - starting fresh trajectories")

    print(f"  Loading observations from: {observations_path}")
    observations = []
    with open(observations_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                observations.append(json.loads(line))
    print(f"    Loaded {len(observations)} observations")

    obs_dict = {obs['instance_idx']: obs for obs in observations}
    new_trajectories = []

    if current_turn == 0:
        for obs_item in observations:
            turn_entry = {
                'thought': format_thought(obs_item.get('thought', '')),
                'action': format_action(obs_item.get('pred_sqls', []), obs_item.get('tool_name', 'execute_sql')),
                'observation': format_observation(obs_item.get('exec_flag', False), obs_item.get('exec_results', '')),
                'end_flag': obs_item.get('end_flag', False)
            }
            new_trajectories.append({
                'idx': obs_item.get('idx'),
                'instance_idx': obs_item.get('instance_idx'),
                'instance_id': obs_item.get('instance_id', ''),
                'db_id': obs_item.get('db_id', ''),
                'trajectory': [turn_entry]
            })
    else:
        for instance_idx, prev_item in prev_traj_dict.items():
            if instance_idx in obs_dict:
                obs_item = obs_dict[instance_idx]
                trajectory = prev_item['trajectory'].copy()
                turn_entry = {
                    'thought': format_thought(obs_item.get('thought', '')),
                    'action': format_action(obs_item.get('pred_sqls', []), obs_item.get('tool_name', 'execute_sql')),
                    'observation': format_observation(obs_item.get('exec_flag', False), obs_item.get('exec_results', '')),
                    'end_flag': obs_item.get('end_flag', False)
                }
                trajectory.append(turn_entry)
                new_trajectories.append({
                    'idx': obs_item.get('idx'),
                    'instance_idx': instance_idx,
                    'instance_id': obs_item.get('instance_id', ''),
                    'db_id': obs_item.get('db_id', ''),
                    'trajectory': trajectory
                })
            else:
                new_trajectories.append(prev_item)

    new_trajectories.sort(key=lambda x: x['instance_idx'])

    # Validate
    validation_errors = 0
    finished_count = 0
    continuing_count = 0

    for traj in new_trajectories:
        trajectory = traj['trajectory']
        last_turn = trajectory[-1]
        actual_length = len(trajectory)

        if last_turn['end_flag']:
            finished_count += 1
            if actual_length > current_turn + 1:
                validation_errors += 1
        else:
            continuing_count += 1
            if actual_length != current_turn + 1:
                validation_errors += 1

    if validation_errors > 0:
        raise AssertionError(f"Validation failed: {validation_errors} instances have incorrect trajectory length")

    print(f"  Finished: {finished_count}, Continuing: {continuing_count}")

    for i, traj in enumerate(new_trajectories):
        traj['idx'] = i

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for traj in new_trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')

    print(f"  Saved {len(new_trajectories)} trajectories -> {output_path}")
    return len(new_trajectories)


def main():
    parser = argparse.ArgumentParser(description='Build trajectory for SFT data generation')
    parser.add_argument('--turn', type=int, default=0, help='Current turn number')
    parser.add_argument('--traj-dir', type=str, required=True, help='Trajectory directory')
    parser.add_argument('--observations', type=str, required=True, help='Observations file')
    parser.add_argument('--output', type=str, required=True, help='Output trajectory file')
    args = parser.parse_args()

    print("=" * 60)
    print(f"Build Trajectory - Turn {args.turn}")
    print("=" * 60)

    build_trajectory(
        current_turn=args.turn,
        traj_dir=args.traj_dir,
        observations_path=args.observations,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
