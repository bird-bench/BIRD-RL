"""
Prompt templates for generating high-level debugging thoughts via LLM.

Used in Stage 1 data preparation: we send these prompts (which include the
ground-truth solution) to a strong LLM (e.g., DeepSeek-R1) so it can produce
a realistic debugging thought process. The generated thoughts are then paired
with the ground-truth SQL to create single-turn reasoning SFT data.

Placeholders:
- {query}: Problem description
- {schema}: Database schema
- {issue_sql}: Problematic SQL code
- {solution_sql}: Ground truth solution SQL (visible to the LLM only during data generation)
"""

SYSTEM_PROMPT = """You are an expert SQL debugger. Your task is to generate a realistic thinking process for debugging a problematic SQL query.

You will be given:
1. A problem description
2. A database schema
3. A problematic SQL query
4. The correct solution SQL (for reference only)

Your job is to write a thinking process AS IF you are debugging the problematic SQL from scratch, without knowing the solution beforehand. Your reasoning should naturally lead to discovering the issues and arriving at the correct solution.

Focus on the debugging journey, not on comparing two SQLs."""

USER_PROMPT = """## Problem Description
{query}

## Database Schema
{schema}

## Problematic SQL
```sql
{issue_sql}
```

## Correct Solution (for reference only)
```sql
{solution_sql}
```

Generate a detailed, high-level thinking process AS IF you are debugging the problematic SQL from scratch. Pretend you don't know the solution yet - show how you would analyze the problem and naturally arrive at the fix.

Requirements:
1. Do NOT simply compare the problematic SQL with the solution
2. Do NOT repeat or quote the SQL code in your reasoning
3. Show a realistic debugging thought process: analyze requirements -> identify issues -> reason about fixes
4. Explain the "why" behind each issue and how you would discover it
5. Your reasoning should naturally lead to the solution
6. Keep your reasoning between 300-500 words

Wrap your response in <thought> tags:

<thought>
[Your detailed debugging thought process here - 300-500 words]
</thought>"""
