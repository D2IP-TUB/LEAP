"""Additional answer extractors adapted from AutoPrep's TableQA agents."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from vllm import SamplingParams

from leap.core.table import Table
from leap.core.types import ExtractorResult

from .autoprep_prompts import (
    COT_END2ENDER_DEMO,
    COT_END2ENDER_QUERY,
    END2ENDER_DEMO,
    END2ENDER_QUERY,
    NL2CODE_DEMO,
    NL2CODE_QUERY,
)
from .base import ExtractionContext
from .nl2sql import MAX_ATTEMPTS, MAX_PROMPT_ROWS, with_row_id


class NL2CodeExtractor:
    name = "nl2code"

    async def extract(self, context: ExtractionContext) -> ExtractorResult:
        failures: list[dict[str, Any]] = []
        last_error: str | None = None
        last_code: str | None = None
        table = with_row_id(context.table)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            response: str | None = None
            attempt_code: str | None = None
            prompt = build_nl2code_prompt(table, context.question, last_error)
            try:
                response = await _generate(context, prompt, self.name, attempt=attempt, max_tokens=512)
                attempt_code = parse_code(response)
                last_code = attempt_code
                answers = execute_code(table, attempt_code)
                if not answers:
                    raise ValueError("The generated code returned an empty answer")
                return ExtractorResult(
                    method=self.name,
                    answers=answers,
                    accuracy=0.0,
                    attempts=attempt,
                    metadata={"code": attempt_code, "response": response, "failed_attempts": failures},
                )
            except Exception as exc:
                last_error = str(exc)
                failures.append({"attempt": attempt, "error": last_error, "code": attempt_code, "response": response})
                print(
                    f"[NL2CODE FAILED ATTEMPT] request_id={context.request_id} attempt={attempt}/{MAX_ATTEMPTS} "
                    f"error={last_error!r} code={attempt_code!r}"
                )

        return ExtractorResult(
            method=self.name,
            answers=[],
            accuracy=0.0,
            error=last_error or "NL2Code extraction failed",
            attempts=MAX_ATTEMPTS,
            metadata={"code": last_code, "failed_attempts": failures},
        )


class End2EnderExtractor:
    name = "end2ender"

    async def extract(self, context: ExtractionContext) -> ExtractorResult:
        prompt = END2ENDER_DEMO + "\n\n" + END2ENDER_QUERY.format(table=format_cotable(context.table), question=context.question)
        response = await _generate(context, prompt, self.name, max_tokens=256)
        answers = parse_end2end_answer(response)
        return _answer_result(self.name, answers, response)


class CoTEnd2EnderExtractor:
    name = "cot_end2ender"

    async def extract(self, context: ExtractionContext) -> ExtractorResult:
        prompt = COT_END2ENDER_DEMO + "\n\n" + COT_END2ENDER_QUERY.format(table=format_cotable(context.table), question=context.question)
        response = await _generate(context, prompt, self.name, max_tokens=512)
        answers = parse_end2end_answer(response)
        return _answer_result(self.name, answers, response)


async def _generate(
    context: ExtractionContext,
    prompt: str,
    method: str,
    *,
    attempt: int = 1,
    max_tokens: int,
) -> str:
    params = SamplingParams(
        temperature=getattr(context.worker, "effective_temperature", lambda value: value)(0.0),
        max_tokens=max_tokens,
        stop_token_ids=[context.worker.tokenizer.eos_token_id],
    )
    response = await context.worker.generate_text(prompt, f"{context.request_id}_extract_{method}_{attempt}", params)
    if not response or not response.strip():
        raise ValueError("Empty model response")
    return response


def _answer_result(method: str, answers: list[str], response: str) -> ExtractorResult:
    if not answers:
        return ExtractorResult(
            method=method,
            answers=[],
            accuracy=0.0,
            error="Could not parse a non-empty answer",
            metadata={"response": response},
        )
    return ExtractorResult(method=method, answers=answers, accuracy=0.0, metadata={"response": response})


def format_cotable(table: Table) -> str:
    columns = " | ".join(table.columns)
    rows = [f"col : {columns}"]
    rows.extend(f"row {index + 1} : " + " | ".join(str(value) for value in row) for index, row in enumerate(table.rows[:MAX_PROMPT_ROWS]))
    if len(table.rows) > MAX_PROMPT_ROWS:
        rows.append("......")
    return "\n".join(rows)


def format_tabular(table: Table) -> str:
    rows = ["\t".join(table.columns)]
    rows.extend("\t".join(str(value) for value in row) for row in table.rows[:MAX_PROMPT_ROWS])
    if len(table.rows) > MAX_PROMPT_ROWS:
        rows.append("......")
    return "\n".join(rows)


def build_nl2code_prompt(table: Table, question: str, last_error: str | None = None) -> str:
    query = NL2CODE_QUERY.format(table=format_tabular(table), question=question)
    if last_error:
        query = query.replace("Code:", f"Last Error: {last_error}\nCode:")
    return f"{NL2CODE_DEMO}\n\n{query}"


def parse_code(response: str) -> str:
    cleaned = response.replace("Code:", "", 1).strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    code = (match.group(1) if match else cleaned).strip()
    if not code:
        raise ValueError("Generated code is empty")
    return code


def execute_code(table: Table, code: str) -> list[str]:
    if not code:
        raise ValueError("Generated code is empty")
    if re.search(r"(^|\n)\s*(import|from)\s+", code) or "__" in code:
        raise ValueError("Imports and dunder access are not allowed in generated code")

    df = pd.DataFrame([list(row) for row in table.rows], columns=list(table.columns))
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace: dict[str, Any] = {"df": df, "pd": pd}
    exec(code, {"__builtins__": safe_builtins, "pd": pd}, namespace)
    if "result" not in namespace:
        raise ValueError("Generated code did not assign the `result` variable")
    result = namespace["result"]
    if isinstance(result, (pd.Series, pd.Index)):
        result = result.tolist()
    if isinstance(result, (list, tuple, set)):
        return [str(value).strip() for value in result if value is not None and str(value).strip()]
    return [str(result).strip()] if result is not None and str(result).strip() else []


def parse_end2end_answer(response: str) -> list[str]:
    matches = list(re.finditer(r"(?:therefore,\s*)?the answer is\s*:?\s*", response, re.IGNORECASE))
    answer = response[matches[-1].end() :] if matches else response
    answer = answer.strip().splitlines()[0].strip().rstrip(".")
    return [part.strip() for part in answer.split("|") if part.strip()]
