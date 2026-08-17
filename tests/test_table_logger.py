import json
from pathlib import Path

import pytest

from leap.config.loader import GenerationConfig as GenerationSettings
from leap.core import Action, ExecutionMetrics, InferenceResult, Table
from leap.generation.sampling import SamplingConfig, SamplingResult
from leap.utils.table_logger import TableLogger
from main import write_results_to_jsonl


def _sample_table(rows=3, cols=3):
    table = Table(
        columns=[f"c{i}" for i in range(cols)],
        rows=[[f"r{r}c{c}" for c in range(cols)] for r in range(rows)],
    )
    return table


def test_setup_logging_directory_creates_dir(tmp_path):
    log_dir = tmp_path / "table_logs"
    TableLogger(log_dir=str(log_dir))
    assert log_dir.exists() and log_dir.is_dir()


def test_log_table_state_writes_log_and_csv(tmp_path):
    log_dir = tmp_path / "logs"
    logger = TableLogger(log_dir=str(log_dir), save_readable_tables=True)

    table = _sample_table(rows=5, cols=4)
    request_id = "req123"
    step = 1
    action = "select_row([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"

    logger.log_table_state(
        request_id=request_id,
        step=step,
        action=action,
        table=table,
        success=True,
        generation_mode="constrained",
    )

    assert request_id in logger.log_entries
    assert len(logger.log_entries[request_id]) == 1
    entry = logger.log_entries[request_id][0]
    assert entry.request_id == request_id
    assert entry.step == step
    assert entry.action == action
    assert entry.success is True
    assert entry.table_summary is not None

    log_file = log_dir / f"{request_id}_log.json"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert ("-" * 80) in content

    clean_action = logger._clean_filename(action)
    csv_file = log_dir / f"{request_id}_step{step:02d}_{clean_action}.csv"
    assert csv_file.exists()
    # csv_text = csv_file.read_text(encoding="utf-8").strip()
    # assert csv_text.startswith(",".join(table.columns))


def test_create_summary_report_metrics(tmp_path):
    logger = TableLogger(log_dir=str(tmp_path / "logs_summary"))

    table_initial = _sample_table(rows=10, cols=3)
    table_mid = _sample_table(rows=8, cols=4)
    table_final = _sample_table(rows=7, cols=4)

    rid_a = "A"
    logger.log_table_state(rid_a, 0, "initial", table_initial, success=True, generation_mode="constrained")
    logger.log_table_state(
        rid_a,
        1,
        "select_row([1, 3, 4])",
        table_mid,
        success=True,
        generation_mode="constrained",
    )
    logger.log_table_state(
        rid_a,
        2,
        "select_column(['Democratic\\nParty'])",
        table_mid,
        success=False,
        failure_type="validity_failure",
        generation_mode="constrained",
    )
    logger.log_table_state(
        rid_a,
        2,
        "validity_failed:",
        table_mid,
        success=False,
        failure_type="validity_failure",
        generation_mode="constrained",
    )

    rid_b = "B"
    logger.log_table_state(rid_b, 0, "initial", table_initial, success=True, generation_mode="none")

    rid_c = "C"
    logger.log_table_state(rid_c, 0, "initial", table_initial, success=True, generation_mode="constrained")
    logger.log_table_state(
        rid_c,
        1,
        "select_row([1, 3, 4])",
        table_mid,
        success=True,
        generation_mode="constrained",
    )
    logger.log_table_state(
        rid_c,
        2,
        "select_column(['Democratic\\nParty'])",
        table_mid,
        success=True,
        generation_mode="constrained",
    )
    logger.log_table_state(rid_c, 3, "end()", table_final, success=True, generation_mode="constrained")

    summary = logger.create_summary_report(generation_mode="mix")
    assert summary["total_requests"] == 3

    assert summary["total_transformations"] == 3
    assert summary["average_steps_per_request"] == pytest.approx(1.5)
    assert rid_b in summary["incomplete_requests"]
    assert 0.0 < summary["completion_rate"] < 1.0

    assert 0.0 <= summary["validity_rate"] <= 1.0
    assert summary["generation_mode"] == "mix"

    assert summary["generation_mode_counts"].get("constrained", 0) >= 2
    assert summary["generation_mode_counts"].get("none", 0) >= 1

    assert any(change["request_id"] == rid_a for change in summary["table_size_changes"])


def test_create_summary_report_includes_error_summary_from_results(tmp_path):
    logger = TableLogger(log_dir=str(tmp_path / "logs_errors"))
    table = _sample_table()
    logger.log_table_state("R", 0, "initial", table)
    logger.log_table_state("R", 1, "validity_failed:action_generation_failed", table, success=False, failure_type="validity_failure")

    sampling_result = SamplingResult(
        action=Action("end", []),
        n_requested=4,
        n_generated=3,
        n_valid=0,
        winner_votes=0,
        total_votes=0,
        candidate_actions=["select_row([99])", "select_column(['missing'])", "select_row([-1])"],
        valid_actions=[],
        fallback_reason="no_valid_candidates",
    )
    result = InferenceResult(
        action_history=["end()", "direct_query()"],
        final_table=table,
        execution_metrics=ExecutionMetrics(
            execution_accuracy=0.0,
            answer_found_in_final=False,
            answer_found_in_original=False,
            terminated_properly=True,
            matched_answers_final=[],
            matched_answers_original=[],
            num_actions=1,
            execution_error="answer generation failed",
        ),
        request_id="R",
        question="q",
        ground_truth_answers=["a"],
        sampling_metadata=[sampling_result],
    )
    logger.record_inference_result(result)

    summary = logger.create_summary_report()
    error_summary = summary["error_summary"]
    assert error_summary["invalid_generation_end_count"] == 1
    assert error_summary["invalid_generation_end_rate"] == pytest.approx(1.0)
    assert error_summary["invalid_candidate_count"] == 3
    assert error_summary["missing_generation_count"] == 1
    assert error_summary["logged_failure_count"] == 1
    assert error_summary["execution_error_count"] == 1
    assert error_summary["total_error_count"] == 6
    assert error_summary["fallback_reason_counts"]["no_valid_candidates"] == 1


def test_write_summary_report_creates_file_and_prints(tmp_path, capsys):
    logger = TableLogger(log_dir=str(tmp_path / "logs_print"))
    table = _sample_table()
    logger.log_table_state("Z", 0, "initial", table)
    logger.write_summary_report(generation_mode="constrained")

    summary_file = Path(logger.log_dir) / "summary_report.json"
    assert summary_file.exists()
    data = json.loads(summary_file.read_text(encoding="utf-8"))
    assert data.get("generation_mode") == "constrained"

    out = capsys.readouterr().out
    assert "Table transformation summary written to" in out
    assert "Generation mode:" in out


def test_disabled_logging_skips_file_writes(tmp_path):
    log_dir = tmp_path / "disabled"
    logger = TableLogger(log_dir=str(log_dir), enable_logging=False)
    table = _sample_table()
    logger.log_table_state("X", 1, "noop", table)

    # No directory is created and no entries are recorded
    assert not log_dir.exists()
    assert logger.log_entries == {}


def test_enabled_logging_without_saving_tables_writes_log_only(tmp_path):
    log_dir = tmp_path / "nosave"
    logger = TableLogger(log_dir=str(log_dir), enable_logging=True, save_readable_tables=False)

    table = _sample_table()
    request_id = "NS"
    step = 2
    action = "select_row([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"

    logger.log_table_state(request_id, step, action, table, success=True)

    # In-memory entry present
    assert request_id in logger.log_entries
    entry = logger.log_entries[request_id][0]
    assert entry.table_preview is None

    # JSON log exists
    assert (log_dir / f"{request_id}_log.json").exists()

    # No CSV file should be created since saving readable tables is disabled
    clean_action = logger._clean_filename(action)
    assert not (log_dir / f"{request_id}_step{step:02d}_{clean_action}.csv").exists()


def test_parallel_results_jsonl_created_with_expected_entries(tmp_path, capsys):
    generation_config = GenerationSettings(
        use_constraints=True,
        use_global_constraints=True,
        strategy="cot",
        sampling=SamplingConfig(
            enabled=True,
            n_samples=8,
        ),
    )

    winner_action = Action("select_row", [0, 1])
    sampling_result = SamplingResult(
        action=winner_action,
        n_requested=8,
        n_generated=8,
        n_valid=6,
        winner_votes=4,
        total_votes=6,
        candidate_actions=[
            "select_row([0, 1])",
            "select_row([1, 2])",
            "select_row([0])",
        ],
        valid_actions=[
            "select_row([0, 1])",
            "select_row([0])",
        ],
    )

    results = InferenceResult(
        action_history=[
            "select_row([row 0, row 1])",
            "select_column(names=['A','B'])",
            "end()",
        ],
        execution_metrics=ExecutionMetrics(
            execution_accuracy=1.0,
            answer_found_in_final=True,
            answer_found_in_original=False,
            terminated_properly=True,
            matched_answers_final=["42"],
            matched_answers_original=[],
            num_actions=3,
            final_table_size=[5, 2],
            evaluation_method="wikitablequestions_logic_with_dataset_answers",
        ),
        request_id="req123",
        question="What is the answer?",
        ground_truth_answers=["42"],
        final_table=_sample_table(rows=5, cols=2),
        sampling_metadata=[sampling_result],
    )

    out_file = tmp_path / "results.jsonl"
    write_results_to_jsonl(
        [results],
        str(out_file),
        generation_config,
        config_key="config-key-123",
        config_label="model/a | cot | constrained | xgrammar | function",
    )

    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    obj = json.loads(lines[0])

    assert obj["id"] == results.request_id
    assert obj["request_id"] == results.request_id
    assert obj["example_id"] == results.request_id
    assert obj["metadata"]["example_id"] == results.request_id
    assert obj["metadata"]["config_key"] == "config-key-123"
    assert obj["metadata"]["config_label"] == "model/a | cot | constrained | xgrammar | function"
    assert obj["metadata"]["is_correct"] is True
    assert obj["is_correct"] is True
    assert obj["comparison"]["label"] == "correct"
    assert obj["generated_answers"] is None
    assert obj["question"] == results.question
    assert obj["ground_truth_answers"] == results.ground_truth_answers

    assert isinstance(obj["actions"], list) and len(obj["actions"]) == 3
    assert obj["actions"][0]["action"] == "select_row"
    assert obj["actions"][0]["args"] == [0, 1]
    assert obj["actions"][1]["action"] == "select_column"
    assert obj["actions"][2]["action"] == "end"

    metrics = obj["execution_metrics"]
    assert metrics["answer_found_in_final"] is True
    assert metrics["final_table_size"] == [5, 2]

    metadata = obj["metadata"]
    assert metadata["num_steps"] == 3
    assert isinstance(metadata.get("generation_mode"), str)
    assert metadata["evaluation_method"] == "wikitablequestions_logic_with_dataset_answers"

    assert "sampling_metadata" in obj
    sm = obj["sampling_metadata"]
    assert isinstance(sm, list) and len(sm) == 1
    m0 = sm[0]
    assert m0["candidate_actions"] == sampling_result.candidate_actions
    assert m0["valid_actions"] == sampling_result.valid_actions
    assert m0["n_requested"] == sampling_result.n_requested
    assert m0["n_generated"] == sampling_result.n_generated
    assert m0["n_valid"] == sampling_result.n_valid
    assert m0["winner_votes"] == sampling_result.winner_votes
    assert m0["total_votes"] == sampling_result.total_votes
    assert m0["fallback_reason"] is None
    assert m0["winner"]["action"] == "select_row"
    assert m0["winner"]["args"] == list(winner_action.arguments)

    # Printed confirmation
    out = capsys.readouterr().out
    assert "Results written to" in out
