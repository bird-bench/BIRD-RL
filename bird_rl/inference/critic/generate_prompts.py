#!/usr/bin/env python3
"""
Generate turn prompts for Critic (SQL debugging) inference.

Uses SFT_TRAINING prompts from bird_rl.prompts.sft_generation (without ground truth).
submit_solution takes sql_list (array format).

Input: JSONL with fields: instance_id, db_id, query, issue_sql, schema (or SQLite DB)
Output: JSONL with fields: idx, instance_idx, instance_id, db_id, system_prompt, prompt
"""

import json
import sqlite3
import os
import argparse
from pathlib import Path

from bird_rl.prompts.sft_generation import SFT_TRAINING_SYSTEM_PROMPT, SFT_TRAINING_USER_TEMPLATE


def get_schema_from_db(db_path: str, num_sample_rows: int = 3) -> str:
    """Extract schema from SQLite database as CREATE DDL + sample rows."""
    schema_parts = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()

        for table_name, create_sql in tables:
            if not create_sql:
                continue
            schema_parts.append(create_sql + ";")
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            columns = [row[1] for row in cursor.fetchall()]
            try:
                cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {num_sample_rows}')
                rows = cursor.fetchall()
            except Exception:
                rows = []
            if rows:
                header = "  " + "\t".join(columns)
                schema_parts.append(f"First {len(rows)} rows:")
                schema_parts.append(header)
                for row in rows:
                    formatted = "  " + "\t".join(
                        str(v) if v is not None else "NULL" for v in row
                    )
                    schema_parts.append(formatted)
            schema_parts.append("")
        conn.close()
    except Exception as e:
        schema_parts.append(f"(Error reading database: {e})")
    return "\n".join(schema_parts)


def build_history_from_trajectory(trajectory: list) -> str:
    """Build conversation history string from trajectory list."""
    if not trajectory:
        return ""

    history_parts = ["## Previous Turns:\n"]
    for turn_idx, turn in enumerate(trajectory):
        history_parts.append(f"### Turn {turn_idx}:")
        thought = turn.get('thought', '')
        if thought:
            history_parts.append(f"{thought}\n")
        action = turn.get('action', '')
        if action:
            history_parts.append(f"{action}\n")
            if not turn.get('end_flag', False):
                observation = turn.get('observation', '')
                if observation:
                    history_parts.append(f"Observation:\n{observation}\n")
        history_parts.append("")
    return '\n'.join(history_parts)


def process_dataset(
    input_path: str,
    db_dir: str,
    output_path: str,
    traj_dir: str,
    current_turn: int,
    max_samples: int = None,
    max_turns: int = 5,
):
    """
    Process critic dataset and generate turn prompts.

    Args:
        input_path: Path to input JSONL (each line: instance_id, db_id, query, issue_sql, ...)
        db_dir: Path to database directory (each DB at db_dir/<db_id>/<db_id>.sqlite)
        output_path: Path to output JSONL
        traj_dir: Directory containing trajectory files (traj_0.jsonl, traj_1.jsonl, ...)
        current_turn: Current turn number (0 for first turn)
        max_samples: Limit number of samples
        max_turns: Maximum turns per trajectory
    """
    print(f"Loading data from: {input_path}")

    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for instance_idx, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                item['instance_idx'] = instance_idx
                data.append(item)

    print(f"  Loaded {len(data)} instances")

    if max_samples is not None and max_samples < len(data):
        print(f"  Limiting to {max_samples} samples")
        data = data[:max_samples]

    schema_cache = {}

    # Load trajectory if not first turn
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
                    trajectory_dict[traj_item.get('instance_idx')] = traj_item.get('trajectory', [])
        print(f"  Loaded trajectory for {len(trajectory_dict)} instances")

    prompts = []
    skipped = 0
    already_finished = 0

    for item in data:
        instance_idx = item['instance_idx']
        db_id = item.get('db_id', '')
        query = item.get('query', '')
        issue_sql = item.get('issue_sql', '')
        instance_id = item.get('instance_id', str(instance_idx))

        if not query or not db_id:
            skipped += 1
            continue

        # Check trajectory state
        trajectory = trajectory_dict.get(instance_idx, [])
        if current_turn > 0:
            if not trajectory:
                already_finished += 1
                continue
            if trajectory[-1].get('end_flag', False):
                already_finished += 1
                continue

        if current_turn >= max_turns:
            already_finished += 1
            continue

        # Get schema
        if db_id not in schema_cache:
            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
            if os.path.exists(db_path):
                schema_cache[db_id] = get_schema_from_db(db_path)
            else:
                print(f"  WARNING: Database not found: {db_path}")
                schema_cache[db_id] = "(Database not found)"

        schema = schema_cache[db_id]

        # Use schema from data if available and DB schema is empty
        if not schema or schema == "(Database not found)":
            schema = item.get('schema', schema)

        # Build history
        history = build_history_from_trajectory(trajectory)

        # Format prompts
        if isinstance(issue_sql, list):
            issue_sql_str = '\n'.join(issue_sql)
        else:
            issue_sql_str = str(issue_sql)

        system_prompt = SFT_TRAINING_SYSTEM_PROMPT.format(
            max_turns=max_turns,
            prev_turns=max_turns - 1
        )

        user_prompt = SFT_TRAINING_USER_TEMPLATE.format(
            query=query.strip(),
            schema=schema.strip() if schema else "(No schema provided)",
            issue_sql=issue_sql_str.strip(),
            max_turns=max_turns
        )

        if history:
            user_prompt += "\n\n" + history

        prompt_record = {
            'idx': len(prompts),
            'instance_idx': instance_idx,
            'instance_id': instance_id,
            'db_id': db_id,
            'current_turn': current_turn,
            'system_prompt': system_prompt,
            'prompt': user_prompt,
        }
        prompts.append(prompt_record)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"  Generated {len(prompts)} prompts (skipped: {skipped}, finished: {already_finished})")
    print(f"  Saved to: {output_path}")
    return len(prompts)


def main():
    parser = argparse.ArgumentParser(description='Generate turn prompts for Critic inference')
    parser.add_argument('--turn', type=int, default=0, help='Current turn number')
    parser.add_argument('--max-turns', type=int, default=5, help='Maximum turns per trajectory')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--db-dir', type=str, required=True, help='Database directory')
    parser.add_argument('--traj-dir', type=str, required=True, help='Trajectory directory')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples')
    args = parser.parse_args()

    process_dataset(
        input_path=args.input,
        db_dir=args.db_dir,
        output_path=args.output,
        traj_dir=args.traj_dir,
        current_turn=args.turn,
        max_samples=args.limit,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    main()
