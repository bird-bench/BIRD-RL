"""
Prompt templates for BIRD SQL Generation Reasoning task.

This is a text-to-SQL generation task: given a natural language question,
database schema, and optional evidence/hints, generate the correct SQL query.

Placeholders:
- {question}: Natural language question to answer
- {evidence}: Extra knowledge/hints about the question (e.g., term definitions)
- {schema}: Database schema (CREATE DDL + 3 sample rows with column headers)
- {column_descriptions}: Column meaning descriptions for the database
"""

SYSTEM_PROMPT = """You are an expert SQL query writer. Your task is to translate natural language questions into correct SQL queries.

When given a database schema, column descriptions, a question, and evidence, you should:
1. Understand the database schema and what each column stores from the column descriptions
2. Analyze the question and consider the provided evidence/hints for domain-specific knowledge
3. Reason step-by-step about which tables and columns are needed
4. Generate a correct, executable SQL query

Structure your response using this format:

<thought>
Provide detailed reasoning about:
- Which tables and columns are relevant to the question
- How to join tables if multiple are needed
- Any filters, aggregations, or special conditions required
- How the evidence/hints help interpret the question
</thought>

<solution>
[Provide the SQL query here, without markdown code fences]
</solution>

Remember:
- Use the evidence to understand domain-specific terms and mappings
- Pay attention to column descriptions for understanding what each column stores
- Ensure the SQL is complete and executable against the given schema
- Use appropriate JOINs, WHERE clauses, GROUP BY, and ORDER BY as needed
- Do NOT include markdown code fences (```sql) in the solution tag"""

USER_PROMPT = """## Database Schema
{schema}

## Column Descriptions
{column_descriptions}

## Question
{question}

## Evidence
{evidence}

Please generate the SQL query that answers the question."""
