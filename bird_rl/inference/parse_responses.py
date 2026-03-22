#!/usr/bin/env python3
"""
Parse model responses: extract <think>/<thought> blocks and <tool_call> JSON.

Handles both BIRD (single sql string) and hybrid (sql_list array) formats.
Includes double-brace fix for RL-trained models that output {{ }} instead of { }.
"""

import json
import re
import argparse
from pathlib import Path


def _fix_doubled_braces(text: str) -> str:
    """Fix {{ -> { and }} -> } from RL-trained models, preserving JSON strings."""
    if "{{" not in text and "}}" not in text:
        return text
    result = []
    in_string = False
    escape_next = False
    i = 0
    while i < len(text):
        c = text[i]
        if escape_next:
            result.append(c)
            escape_next = False
            i += 1
            continue
        if c == "\\":
            escape_next = True
            result.append(c)
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
            i += 1
            continue
        if not in_string:
            if c == "{" and i + 1 < len(text) and text[i + 1] == "{":
                result.append("{")
                i += 2
                continue
            if c == "}" and i + 1 < len(text) and text[i + 1] == "}":
                result.append("}")
                i += 2
                continue
        result.append(c)
        i += 1
    return "".join(result)


def _try_parse_json(json_str: str) -> dict:
    """Try parsing JSON, with doubled-brace fallback."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        fixed = _fix_doubled_braces(json_str)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = re.sub(r"^```(?:json)?\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def extract_first_think(response: str) -> str:
    """Extract first <think> or <thought> block."""
    for tag in ["think", "thought"]:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def extract_first_tool_call(response: str):
    """
    Extract the first tool_call from the response.

    Returns:
        (tool_name, pred_sqls, end_flag) or (None, None, False) if no tool call found.
        - tool_name: "execute_sql" or "submit_solution"
        - pred_sqls: list of SQL strings
        - end_flag: True if submit_solution
    """
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL | re.IGNORECASE)
    if not matches:
        return None, None, False

    for content in matches:
        content = _strip_markdown_fences(content)
        tool_data = _try_parse_json(content)
        if tool_data is None or not isinstance(tool_data, dict):
            continue

        tool_name = tool_data.get("name")
        arguments = tool_data.get("arguments", {})
        if not isinstance(arguments, dict):
            continue

        if tool_name == "submit_solution":
            # Prefer sql_list, fallback to sql
            sql_list = arguments.get("sql_list", [])
            if isinstance(sql_list, list) and sql_list:
                return tool_name, [str(s).strip() for s in sql_list if s], True
            sql = arguments.get("sql", "")
            if isinstance(sql, str) and sql.strip():
                return tool_name, [sql.strip()], True

        elif tool_name == "execute_sql":
            # Prefer sql (string), also handle sql_list
            sql = arguments.get("sql", "")
            if isinstance(sql, str) and sql.strip():
                return tool_name, [sql.strip()], False
            sql_list = arguments.get("sql_list", [])
            if isinstance(sql_list, list) and sql_list:
                return tool_name, [str(sql_list[0]).strip()], False

    return None, None, False


def parse_response(response: str):
    """
    Parse a model response into thought, tool_name, pred_sqls, end_flag.

    Returns:
        (thought, tool_name, pred_sqls, end_flag)
    """
    thought = extract_first_think(response)
    tool_name, pred_sqls, end_flag = extract_first_tool_call(response)
    return thought, tool_name, pred_sqls, end_flag


def process_responses(input_path: str, output_path: str):
    """Process a JSONL file of vLLM responses, extracting thought and tool calls."""
    results = []
    stats = {"total": 0, "parsed": 0, "no_tool_call": 0, "submit": 0, "execute": 0}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            stats["total"] += 1

            response = item.get("raw_response", "")
            if item.get("skipped"):
                response = ""

            thought, tool_name, pred_sqls, end_flag = parse_response(response)

            result = {
                "idx": item.get("idx"),
                "instance_idx": item.get("instance_idx"),
                "instance_id": item.get("instance_id", item.get("question_id", "")),
                "db_id": item.get("db_id", ""),
                "thought": thought,
                "tool_name": tool_name,
                "pred_sqls": pred_sqls or [],
                "end_flag": end_flag,
            }
            results.append(result)

            if tool_name:
                stats["parsed"] += 1
                if end_flag:
                    stats["submit"] += 1
                else:
                    stats["execute"] += 1
            else:
                stats["no_tool_call"] += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Parsed {stats['total']} responses: {stats['parsed']} with tool calls "
          f"({stats['execute']} execute, {stats['submit']} submit), "
          f"{stats['no_tool_call']} without")
    return results


def main():
    parser = argparse.ArgumentParser(description="Parse model responses")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL (vLLM output)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL (parsed)")
    args = parser.parse_args()
    process_responses(args.input, args.output)


if __name__ == "__main__":
    main()
