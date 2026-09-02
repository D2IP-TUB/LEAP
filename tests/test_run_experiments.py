import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.run_experiments import (
    DEFAULT_SPEC_PATH,
    ExperimentJob,
    ExperimentMatrix,
    ExperimentSession,
    ExperimentSpec,
    JobResult,
    _run_child_process,
    build_job_config,
    build_report,
    collect_run_metrics,
    create_jobs,
    expand_matrix,
    generate_comparison_artifacts,
    group_jobs,
    load_experiment_spec,
    main,
    render_markdown_report,
    run_job,
    run_session_group,
)


def _write_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _base_config(tmp_path):
    models_path = tmp_path / "configs" / "models.yaml"
    _write_yaml(models_path, {"models": {"model/a": {}, "model/b": {}}})
    base_config = {
        "model": {"id": "model/a", "presets_path": str(models_path), "results_file": "./logs/results.jsonl"},
        "dataset": {"loader": "huggingface", "name": "wikitablequestions", "split": "train", "trust_remote_code": True},
        "run": {"max_examples": 300},
        "generation": {
            "strategy": "cot",
            "use_constraints": True,
            "use_global_constraints": False,
            "enabled_actions": ["select_row", "end"],
            "sampling": {"enabled": True, "n_samples": 8},
        },
        "logging": {"enable_logging": True},
    }
    base_path = tmp_path / "configs" / "default.yaml"
    _write_yaml(base_path, base_config)
    return base_path, base_config


def _write_result(run_dir: Path, *, accuracy: float, rows: list[dict]):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps({"run_dir": str(run_dir)}), encoding="utf-8")
    (run_dir / "end_to_end_accuracy.json").write_text(json.dumps({"end_to_end_accuracy": accuracy}), encoding="utf-8")
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_comparison_run(run_dir: Path, *, config_key: str, config_label: str, rows: list[dict]):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps({"config_key": config_key, "config_label": config_label, "run_dir": str(run_dir)}),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _matrix():
    return ExperimentMatrix(
        strategies=["iterative", "cot", "direct_query"],
        use_constraints=[False, True],
        use_global_constraints=[False, True],
        constraint_backends=["legacy_state_machine", "xgrammar"],
        output_formats=["function", "json"],
        force_zero_temperature=[False, True],
    )


def _spec_data(base_path: Path):
    return {
        "base_config": str(base_path),
        "models": ["model/a"],
        "matrix": {
            "strategies": ["iterative", "cot", "direct_query"],
            "use_constraints": [False, True],
            "use_global_constraints": [False, True],
            "constraint_backends": ["legacy_state_machine", "xgrammar"],
            "output_formats": ["function", "json"],
            "force_zero_temperature": [False, True],
        },
        "repeats": 3,
        "extractors": ["direct_query", "nl2sql", "nl2code", "end2ender", "cot_end2ender"],
        "enabled_actions": ["select_row", "select_column", "group_by", "sort_by", "end"],
    }


def _job(tmp_path, **overrides):
    values = {
        "model": "model/a",
        "strategy": "cot",
        "use_constraints": False,
        "use_global_constraints": False,
        "constraint_backend": "xgrammar",
        "output_format": "function",
        "force_zero_temperature": False,
        "add_column": True,
        "repeat": 1,
        "config_path": tmp_path / "config.yaml",
        "results_root": tmp_path / "runs",
    }
    values.update(overrides)
    return ExperimentJob(**values)


def test_build_job_config_applies_all_matrix_and_fixed_settings(tmp_path):
    _, base_config = _base_config(tmp_path)
    extractors = ["direct_query", "nl2sql", "nl2code", "end2ender", "cot_end2ender"]
    actions = ["select_row", "select_column", "group_by", "sort_by", "end"]
    config = build_job_config(
        base_config,
        model="model/b",
        strategy="iterative",
        use_constraints=True,
        use_global_constraints=True,
        constraint_backend="xgrammar",
        output_format="json",
        force_zero_temperature=True,
        add_column=False,
        max_examples=10,
        extractors=extractors,
        enabled_actions=actions,
    )

    assert config["model"]["id"] == "model/b"
    assert config["run"]["max_examples"] == 10
    assert config["generation"]["output_format"] == "json"
    assert config["generation"]["strategy"] == "iterative"
    assert config["generation"]["use_constraints"] is True
    assert config["generation"]["use_global_constraints"] is True
    assert config["generation"]["constraint_backend"] == "xgrammar"
    assert config["generation"]["force_zero_temperature"] is True
    assert config["generation"]["enabled_actions"] == actions
    assert "add_column" not in config["generation"]["enabled_actions"]

    add_column_config = build_job_config(
        base_config,
        model="model/b",
        strategy="iterative",
        use_constraints=True,
        use_global_constraints=True,
        constraint_backend="xgrammar",
        output_format="json",
        force_zero_temperature=True,
        add_column=True,
        max_examples=10,
        extractors=extractors,
        enabled_actions=actions,
    )
    assert add_column_config["generation"]["enabled_actions"] == [*actions[:-1], "add_column", "end"]
    assert config["extractors"] == extractors
    assert base_config["generation"]["enabled_actions"] == ["select_row", "end"]


def test_expand_matrix_produces_42_unique_settings_and_canonical_direct_query():
    settings = expand_matrix(_matrix())

    assert len(settings) == 42
    assert len({tuple(setting.values()) for setting in settings}) == 42
    assert not any(setting["constraint_backend"] == "legacy_state_machine" and setting["output_format"] == "json" for setting in settings)
    assert all(setting["constraint_backend"] == "xgrammar" for setting in settings if not setting["use_constraints"])
    direct = [setting for setting in settings if setting["strategy"] == "direct_query"]
    assert direct == [
        {
            "strategy": "direct_query",
            "use_constraints": False,
            "use_global_constraints": False,
            "constraint_backend": "xgrammar",
            "output_format": "function",
            "force_zero_temperature": False,
            "add_column": True,
        },
        {
            "strategy": "direct_query",
            "use_constraints": False,
            "use_global_constraints": False,
            "constraint_backend": "xgrammar",
            "output_format": "function",
            "force_zero_temperature": True,
            "add_column": True,
        },
    ]


def test_load_experiment_spec_rejects_removed_output_format(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_path, _ = _base_config(tmp_path)
    data = _spec_data(base_path)
    data["matrix"]["output_formats"] = ["mc" + "p"]
    spec_path = tmp_path / "configs" / "experiments.yaml"
    _write_yaml(spec_path, data)

    with pytest.raises(ValueError, match="Unknown output_formats"):
        load_experiment_spec(spec_path)


def test_load_experiment_spec_rejects_legacy_modes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_path, _ = _base_config(tmp_path)
    spec_path = tmp_path / "configs" / "experiments.yaml"
    _write_yaml(spec_path, {"base_config": str(base_path), "models": ["model/a"], "modes": ["bad_mode"]})

    with pytest.raises(ValueError, match="modes.*no longer supported"):
        load_experiment_spec(spec_path)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("strategies", [], "strategies"),
        ("strategies", ["bad"], "Unknown strategies"),
        ("use_constraints", ["false"], "booleans"),
        ("use_global_constraints", [False, False], "duplicates"),
        ("constraint_backends", ["bad"], "Unknown constraint_backends"),
        ("output_formats", ["xml"], "Unknown output_formats"),
        ("force_zero_temperature", ["no"], "booleans"),
    ],
)
def test_load_experiment_spec_validates_matrix_lists(tmp_path, monkeypatch, key, value, message):
    monkeypatch.chdir(tmp_path)
    base_path, _ = _base_config(tmp_path)
    data = _spec_data(base_path)
    data["matrix"][key] = value
    spec_path = tmp_path / "configs" / "experiments.yaml"
    _write_yaml(spec_path, data)

    with pytest.raises(ValueError, match=message):
        load_experiment_spec(spec_path)


def test_load_experiment_spec_defaults_model_reuse_on_and_accepts_opt_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_path, _ = _base_config(tmp_path)
    data = _spec_data(base_path)
    spec_path = tmp_path / "experiments.yaml"
    _write_yaml(spec_path, data)
    assert load_experiment_spec(spec_path).reuse_models is True

    data["reuse_models"] = False
    _write_yaml(spec_path, data)
    assert load_experiment_spec(spec_path).reuse_models is False


def test_load_experiment_spec_defaults_python_executable_to_current_interpreter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.run_experiments.sys.executable", "/tmp/uv/bin/python")
    base_path, _ = _base_config(tmp_path)
    data = _spec_data(base_path)
    data.pop("python_executable", None)
    spec_path = tmp_path / "experiments.yaml"
    _write_yaml(spec_path, data)

    spec = load_experiment_spec(spec_path)

    assert spec.python_executable == Path("/tmp/uv/bin/python")


def test_example_matrix_creates_504_unique_jobs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_path, _ = _base_config(tmp_path)
    data = _spec_data(base_path)
    data["models"] = ["model/a", "model/b", "model/c", "model/d"]
    spec_path = tmp_path / "experiments.yaml"
    _write_yaml(spec_path, data)
    spec = load_experiment_spec(spec_path)

    jobs = create_jobs(spec, tmp_path / "experiment")

    assert len(jobs) == 504
    assert len({job.config_path.name for job in jobs}) == 504
    generated = yaml.safe_load(jobs[0].config_path.read_text(encoding="utf-8"))
    assert generated["extractors"] == data["extractors"]
    assert generated["generation"]["force_zero_temperature"] in {False, True}
    assert generated["generation"]["enabled_actions"] == data["enabled_actions"]


def test_group_jobs_creates_one_session_per_model_and_runtime(tmp_path):
    settings = expand_matrix(_matrix())
    jobs = []
    for model in ("model/a", "model/b", "model/c", "model/d"):
        for setting in settings:
            jobs.append(_job(tmp_path, model=model, **setting))

    sessions = group_jobs(jobs)

    assert len(sessions) == 8
    assert {(session.model, session.expected_runtime) for session in sessions} == {
        (model, runtime) for model in ("model/a", "model/b", "model/c", "model/d") for runtime in ("legacy", "modern")
    }
    assert sum(len(session.jobs) for session in sessions) == len(jobs)


def test_collect_run_metrics_from_results(tmp_path):
    run_dir = tmp_path / "run"
    _write_result(
        run_dir,
        accuracy=0.5,
        rows=[
            {
                "execution_accuracy": 1.0,
                "execution_metrics": {"answer_found_in_final": True, "terminated_properly": True, "num_actions": 2},
            },
            {
                "execution_accuracy": 0.0,
                "execution_metrics": {
                    "answer_found_in_final": False,
                    "terminated_properly": True,
                    "num_actions": 4,
                    "execution_error": "failed",
                },
                "sampling_metadata": [
                    {
                        "winner": {"action": "end", "args": []},
                        "n_requested": 4,
                        "n_generated": 3,
                        "n_valid": 0,
                        "fallback_reason": "no_valid_candidates",
                    }
                ],
            },
        ],
    )

    metrics = collect_run_metrics(run_dir)

    assert metrics["accuracy"] == 0.5
    assert metrics["examples"] == 2
    assert metrics["successful_examples"] == 1
    assert metrics["answer_found_rate"] == 0.5
    assert metrics["termination_rate"] == 1.0
    assert metrics["average_actions"] == 3.0
    assert metrics["error_rate"] == 0.5
    assert metrics["invalid_generation_end_count"] == 1
    assert metrics["invalid_candidate_count"] == 3
    assert metrics["missing_generation_count"] == 1
    assert metrics["total_error_count"] == 5


def test_collect_run_metrics_reads_extractor_report(tmp_path):
    run_dir = tmp_path / "run"
    _write_result(
        run_dir,
        accuracy=0.25,
        rows=[
            {
                "execution_accuracy": 0.0,
                "execution_metrics": {"answer_found_in_final": False, "terminated_properly": True, "num_actions": 2},
                "extractor_results": [
                    {"method": "direct_query", "accuracy": 0.0},
                    {"method": "nl2sql", "accuracy": 1.0},
                ],
            }
        ],
    )
    (run_dir / "extractor_accuracy.json").write_text(
        json.dumps(
            {
                "method_accuracies": {"direct_query": 0.25, "nl2sql": 0.75},
                "average_accuracy": 0.5,
                "examples": 1,
            }
        ),
        encoding="utf-8",
    )

    metrics = collect_run_metrics(run_dir)

    assert metrics["accuracy"] == 0.25
    assert metrics["method_accuracies"] == {"direct_query": 0.25, "nl2sql": 0.75}
    assert metrics["average_extractor_accuracy"] == 0.5


def test_collect_run_metrics_prefers_summary_report_errors(tmp_path):
    run_dir = tmp_path / "run"
    _write_result(
        run_dir,
        accuracy=0.0,
        rows=[
            {
                "execution_accuracy": 0.0,
                "execution_metrics": {"answer_found_in_final": False, "terminated_properly": True, "num_actions": 1},
            }
        ],
    )
    summary_dir = run_dir / "table_logs"
    summary_dir.mkdir()
    (summary_dir / "summary_report.json").write_text(
        json.dumps(
            {
                "total_requests": 4,
                "error_summary": {
                    "invalid_generation_end_count": 2,
                    "invalid_candidate_count": 7,
                    "missing_generation_count": 3,
                    "total_error_count": 12,
                },
            }
        ),
        encoding="utf-8",
    )

    metrics = collect_run_metrics(run_dir)

    assert metrics["error_rate"] == 0.5
    assert metrics["invalid_generation_end_count"] == 2
    assert metrics["invalid_candidate_count"] == 7
    assert metrics["missing_generation_count"] == 3
    assert metrics["total_error_count"] == 12


def test_experiment_report_includes_error_columns(tmp_path):
    spec = ExperimentSpec(
        base_config=tmp_path / "config.yaml",
        models=["model/a"],
        matrix=_matrix(),
        repeats=1,
        max_examples=2,
        output_root=tmp_path / "experiments",
        python_executable=Path("python"),
        extractors=["direct_query", "nl2sql", "nl2code", "end2ender", "cot_end2ender"],
        enabled_actions=["select_row", "end"],
    )
    result = JobResult(
        model="model/a",
        strategy="cot",
        use_constraints=True,
        use_global_constraints=False,
        constraint_backend="legacy_state_machine",
        output_format="function",
        expected_runtime="legacy",
        repeat=1,
        status="ok",
        return_code=0,
        runtime_seconds=12.0,
        config_path=str(tmp_path / "config.yaml"),
        run_dir=str(tmp_path / "run"),
        stdout_log=str(tmp_path / "stdout.log"),
        stderr_log=str(tmp_path / "stderr.log"),
        accuracy=0.25,
        examples=2,
        successful_examples=1,
        answer_found_rate=0.5,
        termination_rate=1.0,
        average_actions=2.0,
        error_rate=0.5,
        invalid_generation_end_count=1,
        invalid_candidate_count=3,
        missing_generation_count=1,
        total_error_count=5,
        config_key="config-key-a",
        config_label="model/a | cot | constrained | legacy_state_machine | function | standard_temp",
        average_extractor_accuracy=0.3,
        method_accuracies={
            "direct_query": 0.5,
            "nl2sql": 0.4,
            "nl2code": 0.3,
            "end2ender": 0.2,
            "cot_end2ender": 0.1,
        },
        session_id="model-a-legacy-attempt-1",
        model_load_id="model-a-legacy-attempt-1-load-1",
        model_reused=True,
    )

    report = build_report(spec, "exp", tmp_path / "experiment", [result])
    markdown = render_markdown_report(report)

    assert report["jobs"][0]["error_rate"] == 0.5
    assert report["required_vllm_runtimes"] == ["legacy"]
    assert report["model_load_count"] == 1
    assert report["comparison_inputs"][0]["config_key"] == "config-key-a"
    assert report["comparison_inputs"][0]["config_label"] == "model/a | cot | constrained | legacy_state_machine | function | standard_temp"
    assert "Required vLLM runtimes: legacy" in markdown
    assert "Model loads: 1" in markdown
    assert report["summary_by_configuration"][0]["invalid_generation_end_count"] == 1
    assert report["summary_by_configuration"][0]["mean_method_accuracies"] == {
        "cot_end2ender": 0.1,
        "direct_query": 0.5,
        "end2ender": 0.2,
        "nl2code": 0.3,
        "nl2sql": 0.4,
    }
    assert "Error Rate" in markdown
    assert "`direct_query` Accuracy" in markdown
    assert "`nl2sql` Accuracy" in markdown
    assert "`nl2code` Accuracy" in markdown
    assert "`end2ender` Accuracy" in markdown
    assert "`cot_end2ender` Accuracy" in markdown
    assert "Mean Extractor Accuracy" in markdown
    assert "| 0.500 | 0.400 | 0.300 | 0.200 | 0.100 | 0.300 |" in markdown
    assert "legacy_state_machine" in markdown
    assert "## Comparison Inputs" in markdown
    assert "Config Key" in markdown


def test_generate_comparison_artifacts_writes_pairwise_outputs(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_comparison_run(
        run_a,
        config_key="config-a",
        config_label="model/a | cot | constrained | xgrammar | function | standard_temp",
        rows=[
            {
                "example_id": "example_1",
                "question": "Q1",
                "ground_truth_answers": ["A"],
                "execution_accuracy": 1.0,
                "is_correct": True,
                "generated_answers": ["A"],
                "comparison": {"is_correct": True, "label": "correct"},
                "execution_metrics": {"answer_found_in_final": True, "terminated_properly": True},
            },
            {
                "example_id": "example_2",
                "question": "Q2",
                "ground_truth_answers": ["B"],
                "execution_accuracy": 1.0,
                "is_correct": True,
                "generated_answers": ["B"],
                "comparison": {"is_correct": True, "label": "correct"},
                "execution_metrics": {"answer_found_in_final": True, "terminated_properly": True},
            },
        ],
    )
    _write_comparison_run(
        run_b,
        config_key="config-b",
        config_label="model/a | cot | unconstrained | xgrammar | function | standard_temp",
        rows=[
            {
                "example_id": "example_1",
                "question": "Q1",
                "ground_truth_answers": ["A"],
                "execution_accuracy": 0.0,
                "is_correct": False,
                "generated_answers": ["wrong"],
                "comparison": {"is_correct": False, "label": "incorrect"},
                "execution_metrics": {"answer_found_in_final": False, "terminated_properly": True},
            },
            {
                "example_id": "example_2",
                "question": "Q2",
                "ground_truth_answers": ["B"],
                "execution_accuracy": 1.0,
                "is_correct": True,
                "generated_answers": ["B"],
                "comparison": {"is_correct": True, "label": "correct"},
                "execution_metrics": {"answer_found_in_final": True, "terminated_properly": True},
            },
        ],
    )

    report = generate_comparison_artifacts([run_a, run_b], tmp_path / "comparison", baseline_config_key="config-a")

    assert report["baseline_config_key"] == "config-a"
    assert report["pairwise_count"] == 1
    pair = report["pairwise_comparisons"][0]
    assert pair["summary"]["total_examples"] == 2
    assert pair["summary"]["shared_examples"] == 2
    assert pair["summary"]["left_wins"] == 1
    assert pair["summary"]["right_wins"] == 0
    assert pair["summary"]["both_correct"] == 1
    assert pair["rows"][0]["outcome"] == "left_only_correct"

    output_dir = tmp_path / "comparison"
    assert (output_dir / "comparison_report.json").exists()
    assert (output_dir / "comparison_report.md").exists()
    assert (output_dir / "comparison_rows.csv").exists()
    markdown = (output_dir / "comparison_report.md").read_text(encoding="utf-8")
    assert "Pair 1" in markdown
    assert "Left wins" in markdown
    csv_text = (output_dir / "comparison_rows.csv").read_text(encoding="utf-8")
    assert "left_only_correct" in csv_text


def test_generate_comparison_artifacts_tracks_missing_examples(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_comparison_run(
        run_a,
        config_key="config-a",
        config_label="model/a | cot | constrained | xgrammar | function | standard_temp",
        rows=[
            {
                "example_id": "example_1",
                "question": "Q1",
                "ground_truth_answers": ["A"],
                "execution_accuracy": 1.0,
                "is_correct": True,
                "generated_answers": ["A"],
                "comparison": {"is_correct": True, "label": "correct"},
                "execution_metrics": {"answer_found_in_final": True, "terminated_properly": True},
            },
            {
                "example_id": "example_2",
                "question": "Q2",
                "ground_truth_answers": ["B"],
                "execution_accuracy": 0.0,
                "is_correct": False,
                "generated_answers": ["wrong"],
                "comparison": {"is_correct": False, "label": "incorrect"},
                "execution_metrics": {"answer_found_in_final": False, "terminated_properly": True},
            },
        ],
    )
    _write_comparison_run(
        run_b,
        config_key="config-b",
        config_label="model/a | cot | unconstrained | xgrammar | function | standard_temp",
        rows=[
            {
                "example_id": "example_1",
                "question": "Q1",
                "ground_truth_answers": ["A"],
                "execution_accuracy": 1.0,
                "is_correct": True,
                "generated_answers": ["A"],
                "comparison": {"is_correct": True, "label": "correct"},
                "execution_metrics": {"answer_found_in_final": True, "terminated_properly": True},
            },
            {
                "example_id": "example_3",
                "question": "Q3",
                "ground_truth_answers": ["C"],
                "execution_accuracy": 1.0,
                "is_correct": True,
                "generated_answers": ["C"],
                "comparison": {"is_correct": True, "label": "correct"},
                "execution_metrics": {"answer_found_in_final": True, "terminated_properly": True},
            },
        ],
    )

    report = generate_comparison_artifacts([run_a, run_b], tmp_path / "comparison", baseline_config_key="config-a")
    pair = report["pairwise_comparisons"][0]

    assert pair["summary"]["total_examples"] == 3
    assert pair["summary"]["shared_examples"] == 1
    assert pair["summary"]["left_only_examples"] == 1
    assert pair["summary"]["right_only_examples"] == 1
    assert pair["summary"]["left_missing"] == 1
    assert pair["summary"]["right_missing"] == 1
    outcomes = {row["example_id"]: row["outcome"] for row in pair["rows"]}
    assert outcomes["example_2"] == "right_missing"
    assert outcomes["example_3"] == "left_missing"


def test_run_job_fails_when_subprocess_writes_empty_run(tmp_path, monkeypatch):
    seen_env = {}

    def fake_run_child_process(*args, **kwargs):
        seen_env.update(kwargs["env"])
        run_dir = tmp_path / "runs" / "empty_run"
        run_dir.mkdir(parents=True)
        (run_dir / "run_config.json").write_text(json.dumps({"run_dir": str(run_dir)}), encoding="utf-8")
        kwargs["stdout_log"].write_text("Failed to start all workers. Exiting.", encoding="utf-8")
        kwargs["stderr_log"].write_text("", encoding="utf-8")
        return 0, 0.1

    monkeypatch.setattr("scripts.run_experiments._run_child_process", fake_run_child_process)
    job = _job(tmp_path, strategy="direct_query")

    result = run_job(job, experiment_dir=tmp_path / "experiment", project_root=tmp_path, python_executable=Path("python"))

    assert result.status == "failed"
    assert result.return_code == 0
    assert result.examples == 0
    assert result.error == "Run completed without results.jsonl entries"
    assert seen_env["LEAP_TQDM_TO_TTY"] == "1"
    assert seen_env["LEAP_TQDM_POSITION"] == "1"
    assert seen_env["LEAP_TQDM_LEAVE"] == "0"


def test_run_job_converts_metric_errors_to_failed_result(tmp_path, monkeypatch):
    def fake_run_child_process(*args, **kwargs):
        run_dir = tmp_path / "runs" / "bad_run"
        run_dir.mkdir(parents=True)
        (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
        return 0, 0.1

    monkeypatch.setattr("scripts.run_experiments._run_child_process", fake_run_child_process)
    monkeypatch.setattr("scripts.run_experiments.collect_run_metrics", lambda path: (_ for _ in ()).throw(ValueError("bad metrics")))

    result = run_job(_job(tmp_path), experiment_dir=tmp_path / "experiment", project_root=tmp_path, python_executable=Path("python"))

    assert result.status == "failed"
    assert result.error == "ValueError: bad metrics"


def test_run_child_process_terminates_child_on_runner_error(tmp_path, monkeypatch):
    class FakeProcess:
        def __init__(self):
            self.poll_count = 0
            self.terminated = False
            self.waited = False

        def poll(self):
            self.poll_count += 1
            if self.poll_count == 1:
                raise RuntimeError("poll failed")
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True
            return 0

    process = FakeProcess()
    monkeypatch.setattr("scripts.run_experiments.subprocess.Popen", lambda *args, **kwargs: process)

    with pytest.raises(RuntimeError, match="poll failed"):
        _run_child_process(
            ["python", "main.py"],
            project_root=tmp_path,
            env={},
            stdout_log=tmp_path / "stdout.log",
            stderr_log=tmp_path / "stderr.log",
            job=_job(tmp_path),
        )

    assert process.terminated is True
    assert process.waited is True


def test_persistent_session_crash_marks_active_job_and_continues(tmp_path, monkeypatch):
    jobs = [_job(tmp_path, repeat=1), _job(tmp_path, repeat=2)]
    session = ExperimentSession(session_id="model-a-modern", model="model/a", expected_runtime="modern", jobs=jobs)
    attempts = []

    def fake_session_process(*args, **kwargs):
        manifest = json.loads((kwargs["status_path"].parent / "manifest.json").read_text(encoding="utf-8"))
        attempts.append([job["job_id"] for job in manifest["jobs"]])
        first = manifest["jobs"][0]
        kwargs["on_event"]({"event": "started", "job_id": first["job_id"], "model_load_id": "load"})
        if len(attempts) == 1:
            return 139, "session crashed"
        run_dir = tmp_path / "run"
        kwargs["on_event"](
            {
                "event": "completed",
                "job_id": first["job_id"],
                "run_dir": str(run_dir),
                "runtime_seconds": 1.0,
                "model_load_id": "load-2",
                "model_reused": False,
            }
        )
        return 0, None

    monkeypatch.setattr("scripts.run_experiments._run_session_process", fake_session_process)
    monkeypatch.setattr(
        "scripts.run_experiments.collect_run_metrics",
        lambda path: {"examples": 1, "accuracy": 1.0, "successful_examples": 1},
    )
    results = []

    run_session_group(
        session,
        experiment_dir=tmp_path / "experiment",
        project_root=tmp_path,
        python_executable=Path("python"),
        on_result=results.append,
    )

    assert [result.status for result in results] == ["failed", "ok"]
    assert results[0].restart_reason == "Persistent session process crashed"
    assert len(attempts) == 2
    assert len(attempts[0]) == 2
    assert len(attempts[1]) == 1


def test_main_defaults_to_example_spec(monkeypatch, tmp_path):
    seen = {}

    def fake_load_experiment_spec(path):
        seen["path"] = path
        raise SystemExit(0)

    monkeypatch.setattr("scripts.run_experiments.load_experiment_spec", fake_load_experiment_spec)

    with pytest.raises(SystemExit):
        main([])

    assert seen["path"] == DEFAULT_SPEC_PATH


def test_runner_imports_leap_when_launched_from_scripts_directory():
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-c", "import run_experiments; print(run_experiments.DEFAULT_SPEC_PATH)"],
        cwd=project_root / "scripts",
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == str(project_root / "configs/experiments.example.yaml")


def test_main_continues_after_job_and_report_errors(monkeypatch, tmp_path):
    spec = ExperimentSpec(
        base_config=tmp_path / "config.yaml",
        models=["model/a"],
        matrix=_matrix(),
        repeats=1,
        max_examples=1,
        output_root=tmp_path / "experiments",
        python_executable=Path("python"),
        extractors=["direct_query"],
        enabled_actions=["end"],
        reuse_models=False,
    )
    jobs = [_job(tmp_path, repeat=1), _job(tmp_path, repeat=2)]
    calls = []

    def fake_run_job(job, **kwargs):
        calls.append(job.repeat)
        if job.repeat == 1:
            raise RuntimeError("launch failed")
        return SimpleNamespace(
            model=job.model,
            strategy=job.strategy,
            repeat=job.repeat,
            status="ok",
            examples=1,
            accuracy=1.0,
            runtime_seconds=0.1,
            error=None,
            stderr_log="stderr.log",
        )

    build_report_calls = []

    def fake_build_report(*args):
        build_report_calls.append(1)
        if len(build_report_calls) == 1:
            raise ValueError("bad report data")
        return {}

    report_calls = []

    def fake_write_reports(*args):
        report_calls.append(1)
        if len(report_calls) == 1:
            raise OSError("report busy")

    monkeypatch.setattr("scripts.run_experiments.load_experiment_spec", lambda path: spec)
    monkeypatch.setattr("scripts.run_experiments.validate_models", lambda value: None)
    monkeypatch.setattr("scripts.run_experiments.validate_python_executable", lambda value: None)
    monkeypatch.setattr("scripts.run_experiments.create_jobs", lambda value, path: jobs)
    monkeypatch.setattr("scripts.run_experiments.run_job", fake_run_job)
    monkeypatch.setattr("scripts.run_experiments.build_report", fake_build_report)
    monkeypatch.setattr("scripts.run_experiments.write_reports", fake_write_reports)
    monkeypatch.setattr("scripts.run_experiments._write_yaml", lambda *args: None)
    monkeypatch.setattr("scripts.run_experiments._read_yaml_raw", lambda *args: {})

    assert main(["spec.yaml"]) == 1
    assert calls == [1, 2]
    assert len(build_report_calls) == 3
    assert len(report_calls) == 2


def test_main_does_not_swallow_keyboard_interrupt(monkeypatch, tmp_path):
    spec = ExperimentSpec(
        base_config=tmp_path / "config.yaml",
        models=["model/a"],
        matrix=_matrix(),
        repeats=1,
        max_examples=1,
        output_root=tmp_path / "experiments",
        python_executable=Path("python"),
        extractors=["direct_query"],
        enabled_actions=["end"],
        reuse_models=False,
    )
    monkeypatch.setattr("scripts.run_experiments.load_experiment_spec", lambda path: spec)
    monkeypatch.setattr("scripts.run_experiments.validate_models", lambda value: None)
    monkeypatch.setattr("scripts.run_experiments.validate_python_executable", lambda value: None)
    monkeypatch.setattr("scripts.run_experiments.create_jobs", lambda value, path: [_job(tmp_path)])
    monkeypatch.setattr("scripts.run_experiments.run_job", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr("scripts.run_experiments._write_yaml", lambda *args: None)
    monkeypatch.setattr("scripts.run_experiments._read_yaml_raw", lambda *args: {})

    with pytest.raises(KeyboardInterrupt):
        main(["spec.yaml"])
