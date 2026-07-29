import json
from pathlib import Path

import pytest
import yaml

from scripts.run_experiments import (
    DEFAULT_SPEC_PATH,
    ExperimentJob,
    ExperimentSpec,
    JobResult,
    build_job_config,
    build_report,
    collect_run_metrics,
    load_experiment_spec,
    main,
    render_markdown_report,
    run_job,
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


def test_build_job_config_uses_existing_generation_strategy_flag(tmp_path):
    _, base_config = _base_config(tmp_path)

    cot = build_job_config(base_config, model="model/b", mode="cot", max_examples=10)
    constrained = build_job_config(base_config, model="model/b", mode="constrained_cot", max_examples=10)
    direct = build_job_config(base_config, model="model/b", mode="direct_query", max_examples=10)

    assert cot["model"]["id"] == "model/b"
    assert cot["run"]["max_examples"] == 10
    assert cot["generation"]["strategy"] == "cot"
    assert cot["generation"]["use_constraints"] is False
    assert constrained["generation"]["strategy"] == "cot"
    assert constrained["generation"]["use_constraints"] is True
    assert direct["generation"]["strategy"] == "direct_query"
    assert direct["generation"]["use_constraints"] is False


def test_load_experiment_spec_rejects_unknown_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_path, _ = _base_config(tmp_path)
    spec_path = tmp_path / "configs" / "experiments.yaml"
    _write_yaml(spec_path, {"base_config": str(base_path), "models": ["model/a"], "modes": ["bad_mode"]})

    with pytest.raises(ValueError, match="Unknown modes"):
        load_experiment_spec(spec_path)


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
        modes=["cot"],
        repeats=1,
        max_examples=2,
        output_root=tmp_path / "experiments",
        continue_on_error=True,
        python_executable=Path("python"),
    )
    result = JobResult(
        model="model/a",
        mode="cot",
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
    )

    report = build_report(spec, "exp", tmp_path / "experiment", [result])
    markdown = render_markdown_report(report)

    assert report["jobs"][0]["error_rate"] == 0.5
    assert "vllm_runtime" in report
    assert "vLLM runtime:" in markdown
    assert report["summary_by_mode"][0]["invalid_generation_end_count"] == 1
    assert "Error Rate" in markdown
    assert "Invalid->End" in markdown


def test_run_job_fails_when_subprocess_writes_empty_run(tmp_path, monkeypatch):
    def fake_run_child_process(*args, **kwargs):
        run_dir = tmp_path / "runs" / "empty_run"
        run_dir.mkdir(parents=True)
        (run_dir / "run_config.json").write_text(json.dumps({"run_dir": str(run_dir)}), encoding="utf-8")
        kwargs["stdout_log"].write_text("Failed to start all workers. Exiting.", encoding="utf-8")
        kwargs["stderr_log"].write_text("", encoding="utf-8")
        return 0, 0.1

    monkeypatch.setattr("scripts.run_experiments._run_child_process", fake_run_child_process)
    job = ExperimentJob(
        model="model/a",
        mode="direct_query",
        repeat=1,
        config_path=tmp_path / "config.yaml",
        results_root=tmp_path / "runs",
    )

    result = run_job(job, experiment_dir=tmp_path / "experiment", project_root=tmp_path, python_executable=Path("python"))

    assert result.status == "failed"
    assert result.return_code == 0
    assert result.examples == 0
    assert result.error == "Run completed without results.jsonl entries"


def test_main_defaults_to_example_spec(monkeypatch, tmp_path):
    seen = {}

    def fake_load_experiment_spec(path):
        seen["path"] = path
        raise SystemExit(0)

    monkeypatch.setattr("scripts.run_experiments.load_experiment_spec", fake_load_experiment_spec)

    with pytest.raises(SystemExit):
        main([])

    assert seen["path"] == DEFAULT_SPEC_PATH
