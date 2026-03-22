#!/usr/bin/env python3
"""
Build trajectory files by accumulating turn data from observations.

Each turn adds: thought, action (tool_call), observation (execution result).
Reads previous trajectory (if exists) and appends current turn.
"""

import json
import argparse
from pathlib import Path


def format_thought(thought: str) -> str:
    """Wrap thought in <think> tags if not already wrapped."""
    if not thought:
        return ""
    thought = thought.strip()
    if not thought.startswith("<think>") and not thought.startswith("<thought>"):
        thought = f"<think>\n{thought}\n</think>"
    return thought


def format_action(pred_sqls: list, tool_name: str, submit_format: str = "sql_list") -> str:
    """
    Format action as a tool_call string.

    Args:
        pred_sqls: List of SQL strings
        tool_name: "execute_sql" or "submit_solution"
        submit_format: "sql_list" (array) or "sql" (string) for submit_solution
    """
    if not pred_sqls:
        return ""

    if tool_name == "submit_solution":
        if submit_format == "sql_list":
            args = {"sql_list": pred_sqls}
        else:
            args = {"sql": pred_sqls[0]}
    else:
        args = {"sql": pred_sqls[0]}

    tool_call = {"name": tool_name, "arguments": args}
    return f'<tool_call>{json.dumps(tool_call, ensure_ascii=False)}</tool_call>'


def format_observation(exec_flag: bool, exec_results: str) -> str:
    """Format observation from execution results."""
    if exec_flag:
        return f"<tool_response>\n{exec_results}\n</tool_response>"
    else:
        return f"<tool_response>\n{exec_results}\n</tool_response>"


def build_trajectory(
    current_turn: int,
    traj_dir: str,
    observations_path: str,
    output_path: str,
    submit_format: str = "sql_list",
):
    """
    Build trajectory by appending current turn observations to previous trajectory.

    Args:
        current_turn: Current turn number (0-based)
        traj_dir: Directory containing previous trajectory files
        observations_path: Path to current turn's observations JSONL
        output_path: Path to write updated trajectory JSONL
        submit_format: "sql_list" or "sql" for submit_solution format
    """
    # Load previous trajectory (if not first turn)
    prev_trajectories = {}
    if current_turn > 0:
        prev_path = Path(traj_dir) / f"traj_{current_turn - 1}.jsonl"
        if prev_path.exists():
            with open(prev_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        prev_trajectories[item["instance_idx"]] = item

    # Load observations for current turn
    observations = []
    with open(observations_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                observations.append(json.loads(line))

    print(f"Building trajectory for turn {current_turn}: {len(observations)} observations")

    results = []
    for obs in observations:
        instance_idx = obs.get("instance_idx")
        instance_id = obs.get("instance_id", "")
        db_id = obs.get("db_id", "")

        # Get previous trajectory or start new
        if instance_idx in prev_trajectories:
            traj_item = prev_trajectories[instance_idx]
            trajectory = traj_item.get("trajectory", [])
        else:
            traj_item = {
                "instance_idx": instance_idx,
                "instance_id": instance_id,
                "db_id": db_id,
            }
            trajectory = []

        # Build current turn
        thought = obs.get("thought", "")
        tool_name = obs.get("tool_name", "execute_sql")
        pred_sqls = obs.get("pred_sqls", [])
        end_flag = obs.get("end_flag", False)
        exec_flag = obs.get("exec_flag", False)
        exec_results = obs.get("exec_results", "")

        turn_data = {
            "thought": format_thought(thought),
            "action": format_action(pred_sqls, tool_name or "execute_sql", submit_format),
            "end_flag": end_flag,
        }

        # Add observation only if not end (submit_solution doesn't need observation)
        if not end_flag:
            turn_data["observation"] = format_observation(exec_flag, exec_results)

        trajectory.append(turn_data)

        traj_item["trajectory"] = trajectory
        results.append(traj_item)

    # Also carry forward finished instances from previous trajectory
    if current_turn > 0:
        seen = {r["instance_idx"] for r in results}
        for idx, prev_item in prev_trajectories.items():
            if idx not in seen:
                results.append(prev_item)

    # Sort by instance_idx for consistency
    results.sort(key=lambda x: x.get("instance_idx", 0))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    finished = sum(1 for r in results if r.get("trajectory", []) and r["trajectory"][-1].get("end_flag"))
    print(f"  Total instances: {len(results)}, Finished: {finished}")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build trajectory from observations")
    parser.add_argument("--turn", type=int, required=True, help="Current turn number")
    parser.add_argument("--traj-dir", type=str, required=True, help="Trajectory directory")
    parser.add_argument("--observations", type=str, required=True, help="Observations JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output trajectory JSONL")
    parser.add_argument("--submit-format", type=str, default="sql_list",
                        choices=["sql_list", "sql"], help="Format for submit_solution")
    args = parser.parse_args()

    build_trajectory(args.turn, args.traj_dir, args.observations, args.output, args.submit_format)


if __name__ == "__main__":
    main()
