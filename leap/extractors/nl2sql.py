"""AutoPrep-derived NL2SQL answer extractor."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from vllm import SamplingParams

from leap.core.table import Table
from leap.core.types import ExtractorResult

from .base import ExtractionContext
from .nl2sql_prompt import DEMO_NL2SQLER_PREP, QUERY_NL2SQLER_PREP

MAX_ATTEMPTS = 3
MAX_PROMPT_ROWS = 100


class NL2SQLExtractor:
    name = "nl2sql"

    async def extract(self, context: ExtractionContext) -> ExtractorResult:
        last_error: str | None = None
        last_sql: str | None = None
        failed_attempts: list[dict[str, Any]] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            response: str | None = None
            attempt_sql: str | None = None
            prompt_table = rotate_columns(with_row_id(context.table), attempt - 1)
            prompt = build_nl2sql_prompt(prompt_table, context.question, last_error)
            params = SamplingParams(
                temperature=getattr(context.worker, "effective_temperature", lambda value: value)(0.0),
                max_tokens=256,
                stop_token_ids=[context.worker.tokenizer.eos_token_id],
            )
            try:
                response = await context.worker.generate_text(
                    prompt,
                    f"{context.request_id}_extract_nl2sql_{attempt}",
                    params,
                )
                attempt_sql = parse_sql(response)
                last_sql = attempt_sql
                answers = execute_sql(prompt_table, attempt_sql)
                if not answers:
                    raise ValueError("The SQL answer is empty")
                return ExtractorResult(
                    method=self.name,
                    answers=answers,
                    accuracy=0.0,
                    attempts=attempt,
                    metadata={"sql": attempt_sql, "response": response, "failed_attempts": failed_attempts},
                )
            except Exception as exc:
                last_error = str(exc)
                failure = {
                    "attempt": attempt,
                    "error": last_error,
                    "sql": attempt_sql,
                    "response": response,
                }
                failed_attempts.append(failure)
                print(
                    f"[NL2SQL FAILED ATTEMPT] request_id={context.request_id} attempt={attempt}/{MAX_ATTEMPTS} "
                    f"error={last_error!r} sql={attempt_sql!r}"
                )

        return ExtractorResult(
            method=self.name,
            answers=[],
            accuracy=0.0,
            error=last_error or "NL2SQL extraction failed",
            attempts=MAX_ATTEMPTS,
            metadata={"sql": last_sql, "failed_attempts": failed_attempts},
        )


def parse_sql(response: str) -> str:
    if not response or not response.strip():
        raise ValueError("Empty model response")
    cleaned = response.replace("SQL:", "").strip()
    match = re.search(r"```(?:sql|SQL)?\s*(.*?)```", cleaned, re.DOTALL)
    sql = (match.group(1) if match else cleaned).strip()
    if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT or WITH queries are allowed")
    return sql.rstrip("; ")


def with_row_id(table: Table) -> Table:
    if "row_id" in table.columns:
        return table
    return Table(columns=["row_id", *table.columns], rows=[[index, *row] for index, row in enumerate(table.rows)])


def rotate_columns(table: Table, offset: int) -> Table:
    if offset == 0 or len(table.columns) < 2:
        return table
    shift = offset % len(table.columns)
    indices = list(range(len(table.columns)))
    indices = indices[shift:] + indices[:shift]
    return Table(
        columns=[table.columns[index] for index in indices],
        rows=[[row[index] for index in indices] for row in table.rows],
    )


def build_nl2sql_prompt(table: Table, question: str, last_error: str | None = None) -> str:
    schema = ",\n".join(f"    {quote_identifier(column)} {infer_column_type(table, index)}" for index, column in enumerate(table.columns))
    rows = ["\t".join(table.columns)]
    rows.extend("\t".join(str(value) for value in row) for row in table.rows[:MAX_PROMPT_ROWS])
    if len(table.rows) > MAX_PROMPT_ROWS:
        rows.append("......")
    query = QUERY_NL2SQLER_PREP.format(
        create_table_text=f"CREATE TABLE w(\n{schema})",
        table="\n".join(rows),
        question=question,
    )
    if last_error:
        query = query.replace("SQL:", f"Last Error: {last_error}\nSQL:")
    return f"{DEMO_NL2SQLER_PREP}\n\n{query}"


def quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def infer_column_type(table: Table, column_index: int) -> str:
    values = [row[column_index] for row in table.rows if row[column_index] is not None and str(row[column_index]).strip()]
    if values and all(_is_int(value) for value in values):
        return "INTEGER"
    if values and all(_is_float(value) for value in values):
        return "REAL"
    return "TEXT"


def _is_int(value: Any) -> bool:
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def _is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def execute_sql(table: Table, sql: str) -> list[str]:
    connection = sqlite3.connect(":memory:")
    try:
        columns = ", ".join(f"{quote_identifier(column)} {infer_column_type(table, index)}" for index, column in enumerate(table.columns))
        connection.execute(f"CREATE TABLE w ({columns})")
        placeholders = ", ".join("?" for _ in table.columns)
        connection.executemany(f"INSERT INTO w VALUES ({placeholders})", table.rows)
        connection.execute("PRAGMA query_only = ON")
        cursor = connection.execute(sql)
        result_rows = cursor.fetchall()
        column_names = [description[0] for description in (cursor.description or [])]
        start_index = 1 if column_names and column_names[0].lower() == "row_id" else 0
        return [str(value).strip() for row in result_rows for value in row[start_index:] if value is not None and str(value).strip()]
    finally:
        connection.close()
