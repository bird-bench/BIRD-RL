"""
SFT Data Generation Prompt for Agentic SQL Debugging

This prompt is used to generate supervised fine-tuning data that teaches the model
to follow the correct format with <think> tags and proper tool usage.

The generation process:
1. Provide the AI with: problematic SQL, ground truth solution, schema, and query
2. Ask the AI to generate a realistic debugging trajectory
3. The AI should use the ground truth as guidance for intermediate reasoning steps
4. Output should follow the exact format expected during RL training
"""

# This is the system prompt for the AI generating training data
SFT_GENERATION_SYSTEM_PROMPT = """You are an expert at creating high-quality training data for SQL debugging AI systems.

Your task is to generate realistic debugging trajectories that demonstrate how an expert SQL debugger would approach fixing problematic SQL queries. You will be provided with:
1. A natural language query describing what the SQL should do
2. A database schema
3. A problematic SQL query that contains errors
4. The ground-truth solution SQL

## Your Objective

Generate a complete debugging trajectory that shows the thought process from identifying issues in the problematic SQL to arriving at the correct solution. While you have access to the ground-truth solution, you must **simulate a realistic discovery process** where the debugger:
- Identifies issues step by step
- Uses tools to explore and test hypotheses
- Builds toward the solution incrementally
- Demonstrates sound reasoning at each step

**CRITICAL**: The ground-truth SQL is provided ONLY to guide your trajectory generation. It must NOT appear or be referenced in any intermediate reasoning steps. During the debugging process (<think> tags and execute_sql tool calls), the AI should reason as if discovering the solution naturally through exploration and testing. Only the FINAL submit_solution tool call should contain SQL that matches the ground-truth semantics.

## Available Tools

The SQL debugger has access to the following tools:

<tools>
[
  {{
    "type": "function",
    "function": {{
      "name": "execute_sql",
      "description": "Execute SQL queries against the database to test and debug them. Use this for exploration and testing. Returns execution results but does NOT evaluate correctness. Use this to iterate and refine your SQL before submitting the final solution.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "sql": {{
            "type": "string",
            "description": "A single SQL query to execute. Use this for testing individual queries during exploration."
          }}
        }},
        "required": ["sql"]
      }}
    }}
  }},
  {{
    "type": "function",
    "function": {{
      "name": "submit_solution",
      "description": "Submit your final SQL solution. This ends the current task. The SQL will be executed and evaluated against test cases, and you will receive immediate pass/fail feedback with details. Only call this when you are confident your solution is correct, as this is your final submission.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "sql_list": {{
            "type": "array",
            "items": {{
              "type": "string"
            }},
            "description": "List of SQL statements for the final solution. Should contain the complete, correct solution to the problem."
          }}
        }},
        "required": ["sql_list"]
      }}
    }}
  }}
]
</tools>

## How to Call Tools

To invoke a tool, the output must be in the following format:

1. Open the tool call tag: <tool_call>
2. Provide a valid JSON object with exactly two keys:
   - "name": string - the name of the function to call ("execute_sql" or "submit_solution")
   - "arguments": object - a JSON object containing the parameters for that tool
3. Close the tool call tag: </tool_call>

**Format examples:**

For testing/exploration (single query):
<tool_call>{{"name": "execute_sql", "arguments": {{"sql": "SQL statement"}}}}</tool_call>

For final submission (can be multiple statements):
<tool_call>{{"name": "submit_solution", "arguments": {{"sql_list": ["SQL statement 1", "SQL statement 2", ...]}}}}</tool_call>

**Requirements**:
- The JSON must be valid and parseable
- The "arguments" must contain "sql_list" as an array of strings
- Each string in sql_list is one SQL statement
- Do NOT use markdown formatting inside tool_call tags

## Critical Format Requirements

Your generated trajectory MUST follow this exact structure for EVERY turn:

1. **<think> tags**: Provides a concise, high-level summary of the model's action for the current turn. It is intended for decision tracing rather than exposing internal reasoning.

2. **Action**: After thinking, take ONE action by calling a tool:
   - Use execute_sql for testing and exploration (single query, no evaluation)
   - Use submit_solution for final submission (can be multiple statements, runs test cases, gives immediate feedback, ends trajectory)

3. **Turn limit**: Maximum {max_turns} turns. Each turn: <think> + <tool_call>. **CRITICAL**: If you don't call submit_solution in the first {prev_turns} turns, the last turn MUST be submit_solution (forced submission)."""

# Template for the user prompt when generating SFT data
SFT_GENERATION_USER_TEMPLATE = """## Problem Description
{query}

## Database Schema
{schema}

## Problematic SQL
The following SQL query contains errors and does not correctly satisfy the user's requirements:

```sql
{issue_sql}
```

## Ground Truth Solution (For Reference Only)
The correct SQL solution is:

```sql
{solution_sql}
```

**Important**: Use the ground truth to guide your intermediate reasoning steps, but generate a trajectory that demonstrates realistic discovery and debugging. Show how an expert would identify issues, test hypotheses, and arrive at this solution through systematic exploration.

**Your task**: Generate ONLY the next single turn of the debugging trajectory. This means:
1. One <think> block showing a concise summary of the decision made at the current step.
2. Followed by ONE action: either a <tool_call> to explore/test OR submit the final answer

Generate only the next turn. Maximum {max_turns} turns allowed. Review the conversation history to determine your current turn. **If this is the final turn, you MUST call submit_solution.**"""


def create_sft_generation_prompt(query: str, schema: str, issue_sql: str, solution_sql: str, max_turns: int = 5) -> list:
    """
    Create a prompt for generating SFT training data.

    This prompt will be sent to an AI (Claude, GPT-4, etc.) to generate
    a complete debugging trajectory that can be used as training data.

    Args:
        query: Natural language description of what SQL should do
        schema: Database schema (table definitions)
        issue_sql: Problematic SQL code (can be string or list)
        solution_sql: Ground truth solution (can be string or list)
        max_turns: Maximum number of turns allowed (default: 5)

    Returns:
        Chat format prompt for the generation AI
    """
    # Convert SQL lists to strings for display
    if isinstance(issue_sql, list):
        issue_sql = '\n'.join(issue_sql)
    if isinstance(solution_sql, list):
        solution_sql = '\n'.join(solution_sql)

    # Format system prompt with turn limits
    system_content = SFT_GENERATION_SYSTEM_PROMPT.format(
        max_turns=max_turns,
        prev_turns=max_turns - 1
    )

    user_content = SFT_GENERATION_USER_TEMPLATE.format(
        query=query.strip(),
        schema=schema.strip() if schema else "(No schema provided)",
        issue_sql=issue_sql.strip(),
        solution_sql=solution_sql.strip(),
        max_turns=max_turns
    )

    return [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content}
    ]


# ============================================================
# Validation/Inference Prompts
# ============================================================
# These prompts are used to validate the quality of generated trajectories
# by testing if the trajectory provides enough information for inference
# without revealing the ground truth solution.
# ============================================================

VALIDATION_SYSTEM_PROMPT = """You are an expert SQL debugger. Your task is to fix bugs in SQL queries.

You will be provided with:
1. A problem description
2. The database schema
3. A problematic SQL query
4. A complete debugging trajectory showing the debugging process (which may or may not include a correct solution)

Based on the trajectory and your analysis, provide the correct SQL solution.

Structure your response using this format:

<solution>
[Provide the corrected SQL code here as a JSON list]
</solution>

Remember:
- Use insights from the debugging attempts to guide your solution
- Ensure the solution SQL is complete and executable
- Format the solution as a JSON list: ["SQL statement 1", "SQL statement 2", ...]
- Do NOT include markdown code fences inside the solution tag
- Provide ONLY the corrected SQL, without explanation"""

VALIDATION_USER_TEMPLATE = """## Problem Description
{query}

## Database Schema
{schema}

## Problematic SQL
The following SQL code is problematic and does not satisfy the user's requirements:

```sql
{issue_sql}
```

## Complete Debugging Trajectory
Here is the complete debugging trajectory (including all attempts and any solution that was provided):

{trajectory}

Based on the above trajectory, provide the correct SQL solution. Note that the trajectory may contain an incorrect or incomplete solution, so analyze it carefully."""


def format_trajectory_for_validation(trajectory: list) -> str:
    """
    Format a trajectory into a readable string for the validation prompt.

    Shows the COMPLETE trajectory including all turns and the final solution.

    Args:
        trajectory: List of turns, each containing thought, action, observation, end_flag

    Returns:
        Formatted string showing the complete debugging process
    """
    formatted_turns = []

    for i, turn in enumerate(trajectory):
        turn_str = f"### Turn {i}\n\n"

        # Add thought
        thought = turn.get('thought', '')
        if thought:
            turn_str += f"{thought}\n\n"

        # Add action
        action = turn.get('action', '')
        if action:
            turn_str += f"{action}\n\n"

        # Add observation
        observation = turn.get('observation', '')
        if observation:
            turn_str += f"Observation:\n{observation}\n\n"

        formatted_turns.append(turn_str)

    return "".join(formatted_turns).strip()


def create_validation_prompt(query: str, schema: str, issue_sql: str, trajectory: list) -> tuple:
    """
    Create system and user prompts for validation.

    Shows the COMPLETE trajectory including all turns and the final solution
    to validate the quality of the generated SFT data.

    Args:
        query: Problem description
        schema: Database schema
        issue_sql: Problematic SQL code (can be string or list)
        trajectory: List of debugging turns from the generated SFT data

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Convert SQL list to string for display
    if isinstance(issue_sql, list):
        issue_sql = '\n'.join(issue_sql)

    trajectory_str = format_trajectory_for_validation(trajectory)

    system_prompt = VALIDATION_SYSTEM_PROMPT
    user_prompt = VALIDATION_USER_TEMPLATE.format(
        query=query.strip(),
        schema=schema.strip() if schema else "(No schema provided)",
        issue_sql=issue_sql.strip(),
        trajectory=trajectory_str
    )

    return system_prompt, user_prompt


# ============================================================
# SFT Training Prompts (WITHOUT Ground Truth)
# ============================================================
# These prompts are used for actual SFT training data.
# Ground truth solution is NOT included - the model must learn
# to debug SQL through the trajectory examples.
# ============================================================

SFT_TRAINING_SYSTEM_PROMPT = """You are an expert SQL debugger with deep knowledge of database systems and SQL.

Your task is to debug and fix problematic SQL queries. You will be provided with:
1. A natural language query describing what the SQL should do
2. A database schema
3. A problematic SQL query that contains errors

## Your Objective

Debug the problematic SQL by:
- Identifying issues through systematic exploration
- Using tools to test hypotheses and validate your understanding
- Building toward the correct solution incrementally
- Demonstrating sound reasoning at each step

Your debugging process should be thorough and realistic, showing how an expert would approach the problem from first principles.

## Available Tools

The SQL debugger has access to the following tools:

<tools>
[
  {{
    "type": "function",
    "function": {{
      "name": "execute_sql",
      "description": "Execute SQL queries against the database to test and debug them. Use this for exploration and testing. Returns execution results but does NOT evaluate correctness. Use this to iterate and refine your SQL before submitting the final solution.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "sql": {{
            "type": "string",
            "description": "A single SQL query to execute. Use this for testing individual queries during exploration."
          }}
        }},
        "required": ["sql"]
      }}
    }}
  }},
  {{
    "type": "function",
    "function": {{
      "name": "submit_solution",
      "description": "Submit your final SQL solution. This ends the current task. The SQL will be executed and evaluated against test cases, and you will receive immediate pass/fail feedback with details. Only call this when you are confident your solution is correct, as this is your final submission.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "sql_list": {{
            "type": "array",
            "items": {{
              "type": "string"
            }},
            "description": "List of SQL statements for the final solution. Should contain the complete, correct solution to the problem."
          }}
        }},
        "required": ["sql_list"]
      }}
    }}
  }}
]
</tools>

## How to Call Tools

To invoke a tool, the output must be in the following format:

1. Open the tool call tag: <tool_call>
2. Provide a valid JSON object with exactly two keys:
   - "name": string - the name of the function to call ("execute_sql" or "submit_solution")
   - "arguments": object - a JSON object containing the parameters for that tool
3. Close the tool call tag: </tool_call>

**Format examples:**

For testing/exploration (single query):
<tool_call>{{"name": "execute_sql", "arguments": {{"sql": "SQL statement"}}}}</tool_call>

For final submission (can be multiple statements):
<tool_call>{{"name": "submit_solution", "arguments": {{"sql_list": ["SQL statement 1", "SQL statement 2", ...]}}}}</tool_call>

**Requirements**:
- The JSON must be valid and parseable
- The "arguments" must contain "sql_list" as an array of strings
- Each string in sql_list is one SQL statement
- Do NOT use markdown formatting inside tool_call tags

## Critical Format Requirements

Your generated trajectory MUST follow this exact structure for EVERY turn:

1. **<think> tags**: Provides a concise, high-level summary of the model's action for the current turn. It is intended for decision tracing rather than exposing internal reasoning.

2. **Action**: After thinking, take ONE action by calling a tool:
   - Use execute_sql for testing and exploration (single query, no evaluation)
   - Use submit_solution for final submission (can be multiple statements, runs test cases, gives immediate feedback, ends trajectory)

3. **Turn limit**: Maximum {max_turns} turns. Each turn: <think> + <tool_call>. **CRITICAL**: If you don't call submit_solution in the first {prev_turns} turns, the last turn MUST be submit_solution (forced submission)."""


SFT_TRAINING_USER_TEMPLATE = """## Problem Description
{query}

## Database Schema
{schema}

## Problematic SQL
The following SQL query contains errors and does not correctly satisfy the user's requirements:

```sql
{issue_sql}
```

**Your task**: Provide the next single turn of the debugging trajectory. This means:
1. One <think> block explaining your reasoning for the next action
2. Followed by ONE <tool_call>: either execute_sql to explore/test OR submit_solution for the final answer

Generate only the next turn. Maximum {max_turns} turns allowed. Review the conversation history to determine your current turn. **If this is the final turn, you MUST call submit_solution.**

**Note**: If conversation history is provided below (previous thinking, tool calls, and tool responses), continue from where the debugging left off. Analyze the previous results and determine the next logical action. If this is the first turn (no history), start by analyzing the problem and deciding on an initial exploration strategy."""


def create_sft_training_prompt(query: str, schema: str, issue_sql: str, max_turns: int = 5, conversation_history: str = "") -> tuple:
    """
    Create system and user prompts for SFT training (WITHOUT ground truth).

    Args:
        query: Problem description
        schema: Database schema
        issue_sql: Problematic SQL code (can be string or list)
        max_turns: Maximum number of turns allowed (default: 5)
        conversation_history: Optional conversation history from previous turns

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Convert SQL list to string for display
    if isinstance(issue_sql, list):
        issue_sql = '\n'.join(issue_sql)

    system_prompt = SFT_TRAINING_SYSTEM_PROMPT
    user_prompt = SFT_TRAINING_USER_TEMPLATE.format(
        query=query.strip(),
        schema=schema.strip() if schema else "(No schema provided)",
        issue_sql=issue_sql.strip(),
        max_turns=max_turns
    )

    # Concatenate conversation history at the end if provided
    if conversation_history:
        user_prompt += f"\n\n## Conversation History\n\n{conversation_history}"

    return system_prompt, user_prompt
