"""
core/sql_handler.py
-------------------
Modular SQL query handler with token optimization.
"""

import json
from typing import Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import text as sa_text

from .token_manager import token_manager
from .retry_handler import retry_handler
from db.database import engine, get_table_schema
from guardrails.guardrails import validate_sql, sanitize_output


class SQLHandler:
    """Optimized SQL query generation and execution."""

    def __init__(self, max_explanation_tokens: int = 300):
        self.max_explanation_tokens = max_explanation_tokens

    def _get_schema_context(self) -> str:
        """Get optimized database schema."""
        schema = get_table_schema()
        # Truncate schema if too long
        return token_manager.optimize_context(schema, max_tokens=1000)

    def _generate_sql(self, question: str) -> str:
        """Generate SQL from natural language question."""
        schema = self._get_schema_context()

        system_prompt = f"""Convert the question to a single SELECT SQL query.

Schema:
{schema}

Rules:
- Only SELECT queries
- Use JOINs when needed
- Limit to 50 rows unless specified
- Return ONLY the SQL query, no explanation"""

        def _make_sql_call():
            # Check token limits
            prompt_tokens = token_manager.count_tokens(system_prompt) + token_manager.count_tokens(question)
            if not token_manager.rate_limiter.can_make_request(prompt_tokens + 200):
                # Wait if needed instead of throwing exception
                token_manager.rate_limiter.wait_if_needed(prompt_tokens + 200)
                # Check again after waiting
                if not token_manager.rate_limiter.can_make_request(prompt_tokens + 200):
                    raise Exception("Rate limit would be exceeded even after waiting")

            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                max_tokens=300,  # Limit SQL generation tokens
            )

            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ])

            sql = response.content.strip()
            sql = sql.replace("```sql", "").replace("```", "").strip()

            # Record usage
            tokens_used = token_manager.count_tokens(sql)
            token_manager.rate_limiter.record_request(prompt_tokens + tokens_used)

            return sql

        return retry_handler.execute_with_retry(_make_sql_call)

    def _execute_sql(self, sql: str) -> Tuple[list, str]:
        """Execute SQL and return results."""
        try:
            with engine.connect() as conn:
                rows = conn.execute(sa_text(sql)).fetchall()

            if not rows:
                return [], "No records found"

            # Convert to dict format
            cols = list(rows[0]._fields)
            data = [dict(zip(cols, row)) for row in rows]

            return data, f"Found {len(data)} records"

        except Exception as e:
            return [], f"SQL execution error: {str(e)}"

    def _generate_explanation(self, question: str, sql: str, data: list) -> str:
        """Generate human-readable explanation."""
        if not data:
            return "No data to explain."

        # Sample first few rows for explanation
        sample_data = data[:3]
        sample_text = json.dumps(sample_data, default=str)

        explanation_prompt = f"""Explain these SQL results in 2-3 sentences:
Question: {question}
SQL: {sql}
Sample results: {sample_text}

Use ₹ for currency amounts."""

        def _make_explanation_call():
            # Check token limits
            prompt_tokens = token_manager.count_tokens(explanation_prompt)
            if not token_manager.rate_limiter.can_make_request(prompt_tokens + self.max_explanation_tokens):
                # Wait if needed instead of throwing exception
                token_manager.rate_limiter.wait_if_needed(prompt_tokens + self.max_explanation_tokens)
                # Check again after waiting
                if not token_manager.rate_limiter.can_make_request(prompt_tokens + self.max_explanation_tokens):
                    raise Exception("Rate limit would be exceeded even after waiting")

            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                max_tokens=self.max_explanation_tokens,
            )

            response = llm.invoke([HumanMessage(content=explanation_prompt)])

            # Record usage
            tokens_used = token_manager.count_tokens(response.content)
            token_manager.rate_limiter.record_request(prompt_tokens + tokens_used)

            return response.content.strip()

        try:
            return retry_handler.execute_with_retry(_make_explanation_call)
        except Exception as e:
            return f"Explanation generation failed: {str(e)}"

    def query(self, question: str) -> str:
        """Complete SQL query pipeline."""
        try:
            # Generate SQL
            sql = self._generate_sql(question)

            # Validate SQL
            safe, reason = validate_sql(sql)
            if not safe:
                return f"Query blocked: {reason}"

            # Execute SQL
            data, exec_message = self._execute_sql(sql)

            # Generate explanation
            explanation = self._generate_explanation(question, sql, data)

            # Format response
            result_parts = [
                f"SQL: {sql}",
                f"Execution: {exec_message}",
                f"Explanation: {explanation}"
            ]

            if data:
                # Limit data display to prevent token overflow
                display_data = data[:20]  # Max 20 rows
                result_parts.append(f"Data: {json.dumps(display_data, default=str, indent=2)}")

            return sanitize_output("\n\n".join(result_parts))

        except Exception as e:
            return f"Query failed: {str(e)}"