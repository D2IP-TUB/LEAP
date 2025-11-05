import json
import os
import sys
import vllm
from pathlib import Path
import pytest
from table_logger import TableLogger
from main import write_results_to_jsonl


def _sample_table(rows=3, cols=3):
    columns = [f"c{i}" for i in range(cols)]
    data_rows = [[f"r{r}c{c}" for c in range(cols)] for r in range(rows)]
    return {"columns": columns, "rows": data_rows}


def test_setup_logging_directory_creates_dir(tmp_path):
    log_dir = tmp_path / "table_logs"
    logger = TableLogger(log_dir=str(log_dir))
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
        model_type="gpt"
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
    csv_text = csv_file.read_text(encoding="utf-8").strip()
    assert csv_text.startswith(",".join(table["columns"]))

def test_create_summary_report_metrics(tmp_path):
    logger = TableLogger(log_dir=str(tmp_path / "logs_summary"))

    table_initial = _sample_table(rows=10, cols=3)
    table_mid = _sample_table(rows=8, cols=4)
    table_final = _sample_table(rows=7, cols=4)
    
    rid_a = "A"
    logger.log_table_state(rid_a, 0, "initial", table_initial, success=True, generation_mode="constrained", model_type="gpt2")
    logger.log_table_state(rid_a, 1, "select_row([1, 3, 4])", table_mid, success=True, generation_mode="constrained", model_type="gpt2")
    logger.log_table_state(rid_a, 2, "select_column(['Democratic\\nParty'])", table_mid, success=False, failure_type= "validity_failure", generation_mode="constrained", model_type="gpt2")
    logger.log_table_state(rid_a, 2, "validity_failed:", table_mid, success=False, failure_type="validity_failure", generation_mode="constrained", model_type="gpt2")

    rid_b = "B"
    logger.log_table_state(rid_b, 0, "initial", table_initial, success=True, generation_mode="none", model_type="gpt2")

    rid_c = "C"
    logger.log_table_state(rid_c, 0, "initial", table_initial, success=True, generation_mode="constrained", model_type="gpt2")
    logger.log_table_state(rid_c, 1, "select_row([1, 3, 4])", table_mid, success=True, generation_mode="constrained", model_type="gpt2")
    logger.log_table_state(rid_c, 2, "select_column(['Democratic\\nParty'])", table_mid, success=True, generation_mode="constrained", model_type="gpt2")
    logger.log_table_state(rid_c, 3, "end()", table_final, success=True, generation_mode="constrained", model_type="gpt2")

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
    # Prepare synthetic results and examples
    results = [
        {
            "action_history": [
                "select_row([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])",
                "select_column(names=['A','B'])",
                "end()"
            ],
            "execution_accuracy_metrics": {
                "execution_accuracy": 1.0,
                "answer_found_in_final": True,
                "answer_found_in_original": False,
                "terminated_properly": True,
                "matched_answers_final": ["42"],
                "matched_answers_original": [],
                "num_actions": 3,
                "final_table_size": [5, 2],
                "evaluation_method": "wikitablequestions_logic_with_dataset_answers"
            }
        }
    ]

    examples = [
        {
            "question": "What is the answer?",
            "answers": ["42"]
        }
    ]

    out_file = tmp_path / "parallel_results.jsonl"
    write_results_to_jsonl(results, examples, str(out_file))

    # Validate file creation and content
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    obj = json.loads(lines[0])
    # Core fields
    assert obj["id"].startswith("nt-")
    assert obj["question"] == examples[0]["question"]
    assert obj["ground_truth_answers"] == examples[0]["answers"]
    # Actions parsing: last one should be invalid
    assert isinstance(obj["actions"], list) and len(obj["actions"]) == 3
    assert obj["actions"][0]["action"] == "select_row"
    assert obj["actions"][1]["action"] == "select_column"
    assert obj["actions"][2]["action"] == "end"
    # Metrics present
    metrics = obj["execution_metrics"]
    assert metrics["answer_found_in_final"] is True
    assert metrics["final_table_size"] == [5, 2]
    # Metadata present
    metadata = obj["metadata"]
    assert metadata["num_steps"] == 3
    assert isinstance(metadata.get("generation_mode"), str)

    # Printed confirmation
    out = capsys.readouterr().out
    assert "Results written to" in out