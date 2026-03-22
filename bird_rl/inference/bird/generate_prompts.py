#!/usr/bin/env python3
"""
Generate turn prompts for BIRD (SQL generation) inference.

Uses SFT_TRAINING prompts from bird_rl.prompts.bird_sft_training (without ground truth).
submit_solution takes a single `sql` string.

Input: JSON array (mini_dev.json) with fields: question_id, db_id, question, evidence
Output: JSONL with fields: idx, instance_idx, question_id, db_id, system_prompt, prompt
"""

import json
import sqlite3
import os
import argparse
from pathlib import Path

from bird_rl.prompts.bird_sft_training import SFT_TRAINING_SYSTEM_PROMPT, SFT_TRAINING_USER_TEMPLATE


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


def get_column_descriptions(db_id: str, column_meanings: dict) -> str:
    """Get column descriptions for a database from column_meaning.json."""
    table_columns = {}
    prefix = f"{db_id}|"
    for key, description in column_meanings.items():
        if not key.startswith(prefix):
            continue
        parts = key.split("|")
        if len(parts) != 3:
            continue
        _, table_name, col_name = parts
        if table_name not in table_columns:
            table_columns[table_name] = []
        table_columns[table_name].append((col_name, description))

    if not table_columns:
        return "(No column descriptions available)"

    lines = []
    for table_name in sorted(table_columns.keys()):
        lines.append(f"Table: {table_name}")
        for col_name, desc in table_columns[table_name]:
            lines.append(f"  - {col_name}: {desc}")
    return "\n".join(lines)


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
    column_meaning_path: str,
    output_path: str,
    traj_dir: str,
    current_turn: int,
    max_samples: int = None,
    max_turns: int = 5,
):
    """
    Process BIRD dev dataset and generate turn prompts.

    Args:
        input_path: Path to mini_dev.json (JSON array with question_id)
        db_dir: Path to dev_databases directory
        column_meaning_path: Path to column_meaning.json
        output_path: Path to output JSONL
        traj_dir: Directory containing trajectory files
        current_turn: Current turn number (0 for first turn)
        max_samples: Limit number of samples
        max_turns: Maximum turns per trajectory
    """
    print(f"Loading data from: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    data = []
    for instance_idx, item in enumerate(raw_data):
        item['instance_idx'] = instance_idx
        data.append(item)

    print(f"  Loaded {len(data)} instances")

    if max_samples is not None and max_samples < len(data):
        print(f"  Limiting to {max_samples} samples")
        data = data[:max_samples]

    # Load column meanings
    print(f"Loading column meanings from: {column_meaning_path}")
    with open(column_meaning_path, 'r', encoding='utf-8') as f:
        column_meanings = json.load(f)
    print(f"  Loaded {len(column_meanings)} column descriptions")

    schema_cache = {}
    col_desc_cache = {}

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
        question = item.get('question', '')
        evidence = item.get('evidence', '')
        question_id = item.get('question_id', instance_idx)

        if not question or not db_id:
            skipped += 1
            continue

        trajectory = trajectory_dict.get(instance_idx, [])
        if current_turn > 0:
            if not trajectory or trajectory[-1].get('end_flag', False):
                already_finished += 1
                continue

        if current_turn >= max_turns:
            already_finished += 1
            continue

        # Build schema from SQLite (cached per db_id)
        if db_id not in schema_cache:
            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
            if os.path.exists(db_path):
                schema_cache[db_id] = get_schema_from_db(db_path)
            else:
                print(f"  WARNING: Database not found: {db_path}")
                schema_cache[db_id] = "(Database not found)"

        if db_id not in col_desc_cache:
            col_desc_cache[db_id] = get_column_descriptions(db_id, column_meanings)

        schema = schema_cache[db_id]
        column_descriptions = col_desc_cache[db_id]

        history = build_history_from_trajectory(trajectory)

        system_prompt = SFT_TRAINING_SYSTEM_PROMPT.format(
            max_turns=max_turns,
            prev_turns=max_turns - 1
        )

        user_prompt = SFT_TRAINING_USER_TEMPLATE.format(
            question=question.strip(),
            evidence=evidence.strip() if evidence else "(No evidence provided)",
            schema=schema.strip() if schema else "(No schema provided)",
            column_descriptions=column_descriptions.strip() if column_descriptions else "(No column descriptions available)",
            max_turns=max_turns
        )

        if history:
            user_prompt += "\n\n" + history

        prompt_record = {
            'idx': len(prompts),
            'instance_idx': instance_idx,
            'question_id': question_id,
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
    parser = argparse.ArgumentParser(description='Generate turn prompts for BIRD inference')
    parser.add_argument('--turn', type=int, default=0, help='Current turn number')
    parser.add_argument('--max-turns', type=int, default=5, help='Maximum turns per trajectory')
    parser.add_argument('--dev-data', type=str, required=True, help='Path to mini_dev.json')
    parser.add_argument('--db-dir', type=str, required=True, help='Path to dev databases')
    parser.add_argument('--column-meaning', type=str, required=True, help='Path to column_meaning.json')
    parser.add_argument('--traj-dir', type=str, required=True, help='Trajectory directory')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples')
    args = parser.parse_args()

    process_dataset(
        input_path=args.dev_data,
        db_dir=args.db_dir,
        column_meaning_path=args.column_meaning,
        output_path=args.output,
        traj_dir=args.traj_dir,
        current_turn=args.turn,
        max_samples=args.limit,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    main()
