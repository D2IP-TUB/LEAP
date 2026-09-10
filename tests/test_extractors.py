from __future__ import annotations

import asyncio
import json

import pytest

from leap.config.loader import _build_extractor_config
from leap.core import ExtractorResult, Table
from leap.extractors.autoprep_extractors import (
    CoTEnd2EnderExtractor,
    End2EnderExtractor,
    NL2CodeExtractor,
    execute_code,
    parse_code,
    parse_end2end_answer,
)
from leap.extractors.base import ExtractionContext, run_extractors
from leap.extractors.direct_query import parse_direct_answers
from leap.extractors.nl2sql import NL2SQLExtractor, execute_sql, parse_sql, with_row_id
from main import calculate_extractor_accuracy_report, print_final_extractor_summary, write_extractor_accuracy_report


class _Tokenizer:
    eos_token_id = 0


class _Worker:
    tokenizer = _Tokenizer()

    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def effective_temperature(self, temperature):
        return temperature

    async def generate_text(self, prompt, request_id, sampling_params):
        self.prompts.append((prompt, request_id))
        return next(self.responses)


def _context(worker):
    return ExtractionContext(
        question="How many wins did Alice have?",
        table=Table(columns=["Name", "Wins"], rows=[["Alice", "7"], ["Bob", "3"]]),
        request_id="req_1",
        action_history=("end()", "direct_query()"),
        worker=worker,
    )


def test_extractor_config_is_ordered_and_validated():
    assert _build_extractor_config(["nl2sql", "direct_query"]) == ("nl2sql", "direct_query")
    with pytest.raises(ValueError, match="Duplicate"):
        _build_extractor_config(["nl2sql", "nl2sql"])
    with pytest.raises(ValueError, match="Unknown"):
        _build_extractor_config(["python"])
    assert _build_extractor_config(["nl2code", "end2ender", "cot_end2ender"]) == (
        "nl2code",
        "end2ender",
        "cot_end2ender",
    )


def test_parse_direct_and_sql_responses():
    assert parse_direct_answers('["Alice", "Bob"]') == ["Alice", "Bob"]
    assert parse_sql("SQL: ```sql\nSELECT Wins FROM w\n```") == "SELECT Wins FROM w"
    with pytest.raises(ValueError, match="Only SELECT"):
        parse_sql("DROP TABLE w")


def test_parse_and_execute_nl2code():
    assert parse_code("Code: ```python\nresult = df['Wins'].max()\n```") == "result = df['Wins'].max()"
    table = with_row_id(Table(columns=["Name", "Wins"], rows=[["Alice", 7], ["Bob", 3]]))
    assert execute_code(table, "result = df.loc[df['Wins'].idxmax(), 'Name']") == ["Alice"]
    with pytest.raises(ValueError, match="Imports"):
        execute_code(table, "import os\nresult = 'bad'")


def test_parse_autoprep_end2end_answers():
    assert parse_end2end_answer("Explanation. Therefore, the answer is: Alice.") == ["Alice"]
    assert parse_end2end_answer("The answer is A | B.") == ["A", "B"]


def test_execute_sql_flattens_answers_and_omits_selected_row_id():
    table = with_row_id(Table(columns=["Name", "Wins"], rows=[["Alice", "7"], ["Bob", "3"]]))
    assert execute_sql(table, 'SELECT "Wins" FROM w WHERE "Name" = \'Alice\'') == ["7"]
    assert execute_sql(table, "SELECT * FROM w WHERE row_id = 0") == ["Alice", "7"]


def test_nl2sql_retries_with_error_and_scores_failure_zero_later(capsys):
    worker = _Worker(["DROP TABLE w", "not sql", ""])
    result = asyncio.run(NL2SQLExtractor().extract(_context(worker)))

    assert result.method == "nl2sql"
    assert result.answers == []
    assert result.accuracy == 0.0
    assert result.attempts == 3
    assert result.error
    assert "Last Error:" in worker.prompts[1][0]
    assert len(worker.prompts) == 3
    assert len(result.metadata["failed_attempts"]) == 3
    assert result.metadata["failed_attempts"][0]["response"] == "DROP TABLE w"
    output = capsys.readouterr().out
    assert output.count("[NL2SQL FAILED ATTEMPT]") == 3
    assert "request_id=req_1 attempt=1/3" in output


def test_nl2sql_executes_generated_query():
    worker = _Worker(['```SELECT "Wins" FROM w WHERE "Name" = \'Alice\'```'])
    result = asyncio.run(NL2SQLExtractor().extract(_context(worker)))

    assert result.answers == ["7"]
    assert result.attempts == 1
    assert result.metadata["sql"].startswith("SELECT")
    assert result.metadata["failed_attempts"] == []


def test_nl2sql_preserves_failed_attempt_before_success():
    worker = _Worker(["not sql", '```SELECT "Wins" FROM w WHERE "Name" = \'Alice\'```'])
    result = asyncio.run(NL2SQLExtractor().extract(_context(worker)))

    assert result.answers == ["7"]
    assert result.attempts == 2
    assert result.metadata["failed_attempts"] == [
        {"attempt": 1, "error": "Only SELECT or WITH queries are allowed", "sql": None, "response": "not sql"}
    ]


def test_nl2code_executes_generated_code_and_retries(capsys):
    worker = _Worker(["```result = missing_name```", "```result = df.loc[df['Name'] == 'Alice', 'Wins'].values[0]```"])
    result = asyncio.run(NL2CodeExtractor().extract(_context(worker)))

    assert result.answers == ["7"]
    assert result.attempts == 2
    assert len(result.metadata["failed_attempts"]) == 1
    assert "[NL2CODE FAILED ATTEMPT]" in capsys.readouterr().out
    assert "Last Error:" in worker.prompts[1][0]


def test_autoprep_prompt_extractors_parse_answers():
    end_worker = _Worker(["Alice."])
    end_result = asyncio.run(End2EnderExtractor().extract(_context(end_worker)))
    assert end_result.answers == ["Alice"]

    cot_worker = _Worker(["Alice has the most wins. Therefore, the answer is Alice."])
    cot_result = asyncio.run(CoTEnd2EnderExtractor().extract(_context(cot_worker)))
    assert cot_result.answers == ["Alice"]


def test_extractor_failure_is_isolated():
    class Good:
        name = "good"

        async def extract(self, context):
            return ExtractorResult(method=self.name, answers=["7"], accuracy=0.0)

    class Bad:
        name = "bad"

        async def extract(self, context):
            raise RuntimeError("broken")

    results = asyncio.run(run_extractors((Bad(), Good()), _context(_Worker([]))))
    assert [result.method for result in results] == ["bad", "good"]
    assert results[0].error == "broken"
    assert results[1].answers == ["7"]


def test_extractor_accuracy_report_includes_failures_in_denominator(tmp_path):
    class Result:
        def __init__(self, request_id, extractor_results):
            self.request_id = request_id
            self.extractor_results = extractor_results

    results = [
        Result("one", [ExtractorResult("direct_query", ["a"], 1.0), ExtractorResult("nl2sql", ["a"], 1.0)]),
        Result("two", [ExtractorResult("direct_query", ["b"], 0.0), ExtractorResult("nl2sql", [], 0.0, error="bad")]),
    ]
    report = calculate_extractor_accuracy_report(results, ("direct_query", "nl2sql"))
    assert report == {
        "method_accuracies": {"direct_query": 0.5, "nl2sql": 0.5},
        "average_accuracy": 0.5,
        "examples": 2,
    }

    output = tmp_path / "extractor_accuracy.json"
    write_extractor_accuracy_report(report, output)
    assert json.loads(output.read_text()) == report


def test_final_extractor_summary_prints_every_method_and_average(capsys):
    print_final_extractor_summary(
        {
            "method_accuracies": {"direct_query": 0.57, "nl2sql": 0.52},
            "average_accuracy": 0.545,
            "examples": 300,
        }
    )

    output = capsys.readouterr().out
    assert "FINAL EXTRACTOR ACCURACY SUMMARY" in output
    assert "direct_query: 0.570 (57.0%) | 171/300 correct" in output
    assert "nl2sql: 0.520 (52.0%) | 156/300 correct" in output
    assert "Average across 2 extractors: 0.545 (54.5%)" in output
