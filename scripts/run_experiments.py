from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm

# Running this file directly puts scripts/ on sys.path rather than the project
# root, so make the repository package importable before importing leap.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leap.vllm_runtime import (  # noqa: E402
    BOOTSTRAPPED_ENV_VAR,
    ensure_vllm_runtime,
    runtime_for_experiment,
    runtime_for_generation,
)

VALID_STRATEGIES = {"iterative", "cot", "direct_query"}
VALID_CONSTRAINT_BACKENDS = {"legacy_state_machine", "xgrammar"}
VALID_OUTPUT_FORMATS = {"function", "json"}
VALID_EXTRACTORS = {"direct_query", "nl2sql", "nl2code", "end2ender", "cot_end2ender"}
VALID_ACTIONS = {"select_row", "select_column", "group_by", "sort_by", "add_column", "end"}
DEFAULT_SPEC_PATH = PROJECT_ROOT / "configs/experiments.example.yaml"
LOGGER = logging.getLogger("leap.experiments")


@dataclass(frozen=True)
class ExperimentMatrix:
    strategies: list[str]
    use_constraints: list[bool]
    use_global_constraints: list[bool]
    constraint_backends: list[str]
    output_formats: list[str]


@dataclass(frozen=True)
class ExperimentSpec:
    base_config: Path
    models: list[str]
    matrix: ExperimentMatrix
    repeats: int
    max_examples: int | None
    output_root: Path
    python_executable: Path
    extractors: list[str]
    enabled_actions: list[str]


@dataclass(frozen=True)
class ExperimentJob:
    model: str
    strategy: str
    use_constraints: bool
    use_global_constraints: bool
    constraint_backend: str
    output_format: str
    repeat: int
    config_path: Path
    results_root: Path

    @property
    def expected_runtime(self) -> str:
        return runtime_for_generation(
            use_constraints=self.use_constraints,
            constraint_backend=self.constraint_backend,
            output_format=self.output_format,
        ).name


@dataclass(frozen=True)
class JobResult:
    model: str
    strategy: str
    use_constraints: bool
    use_global_constraints: bool
    constraint_backend: str
    output_format: str
    expected_runtime: str
    repeat: int
    status: str
    return_code: int
    runtime_seconds: float
    config_path: str
    run_dir: str | None
    stdout_log: str
    stderr_log: str
    accuracy: float | None
    examples: int
    successful_examples: int
    answer_found_rate: float | None
    termination_rate: float | None
    average_actions: float | None
    error_rate: float | None
    invalid_generation_end_count: int
    invalid_candidate_count: int
    missing_generation_count: int
    total_error_count: int
    error: str | None = None
    average_extractor_accuracy: float | None = None
    method_accuracies: dict[str, float] = field(default_factory=dict)


def load_experiment_spec(spec_path: Path) -> ExperimentSpec:
    raw = _load_yaml(spec_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment spec {spec_path} must contain a mapping.")

    project_root = PROJECT_ROOT
    base_config = _resolve_path(raw.get("base_config", "configs/default.yaml"), spec_path.parent, project_root)
    output_root = _resolve_path(raw.get("output_root", "results/experiments"), spec_path.parent, project_root)
    python_executable = _resolve_path(
        raw.get("python_executable") or os.environ.get("LEAP_EXPERIMENT_PYTHON") or _default_python_executable(),
        spec_path.parent,
        project_root,
    )

    models = raw.get("models")
    if not isinstance(models, list) or not models or not all(isinstance(model, str) for model in models):
        raise ValueError("'models' must be a non-empty list of model IDs.")
    if len(models) != len(set(models)):
        raise ValueError("'models' must not contain duplicates.")

    if "modes" in raw:
        raise ValueError("'modes' is no longer supported; use the 'matrix' generation settings instead.")
    if "continue_on_error" in raw:
        raise ValueError("'continue_on_error' is no longer supported; experiment jobs now always continue after failures.")
    matrix_raw = raw.get("matrix")
    if not isinstance(matrix_raw, dict):
        raise ValueError("'matrix' must be a mapping of generation setting lists.")
    matrix = ExperimentMatrix(
        strategies=_string_list(matrix_raw, "strategies", VALID_STRATEGIES),
        use_constraints=_bool_list(matrix_raw, "use_constraints"),
        use_global_constraints=_bool_list(matrix_raw, "use_global_constraints"),
        constraint_backends=_string_list(matrix_raw, "constraint_backends", VALID_CONSTRAINT_BACKENDS),
        output_formats=_string_list(matrix_raw, "output_formats", VALID_OUTPUT_FORMATS),
    )
    if not expand_matrix(matrix):
        raise ValueError("The experiment matrix contains no supported generation-setting combinations.")

    repeats = raw.get("repeats", 3)
    if type(repeats) is not int or repeats <= 0:
        raise ValueError("'repeats' must be positive.")

    max_examples = raw.get("max_examples")
    if max_examples is not None:
        if type(max_examples) is not int or max_examples < 0:
            raise ValueError("'max_examples' must be non-negative when provided.")

    extractors = _string_list(raw, "extractors", VALID_EXTRACTORS)
    enabled_actions = _string_list(raw, "enabled_actions", VALID_ACTIONS)

    return ExperimentSpec(
        base_config=base_config,
        models=models,
        matrix=matrix,
        repeats=repeats,
        max_examples=max_examples,
        output_root=output_root,
        python_executable=python_executable,
        extractors=extractors,
        enabled_actions=enabled_actions,
    )


def validate_models(spec: ExperimentSpec) -> None:
    base_config = _load_yaml(spec.base_config)
    model_section = base_config.get("model", {})
    presets_path = _resolve_path(model_section.get("presets_path", "configs/models.yaml"), spec.base_config.parent, PROJECT_ROOT)
    presets = _load_yaml(presets_path).get("models", _load_yaml(presets_path))
    missing = [model for model in spec.models if model not in presets]
    if missing:
        raise ValueError(f"Models not found in {presets_path}: {missing}")


def validate_python_executable(python_executable: Path) -> None:
    if not python_executable.exists():
        raise FileNotFoundError(f"Python executable not found: {python_executable}")
    subprocess.run(
        [
            str(python_executable),
            "-c",
            "import datasets, transformers, yaml",
        ],
        check=True,
    )


def create_jobs(spec: ExperimentSpec, experiment_dir: Path) -> list[ExperimentJob]:
    base_config = _load_yaml(spec.base_config)
    config_dir = experiment_dir / "configs"
    run_root = experiment_dir / "runs"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    settings = expand_matrix(spec.matrix)
    for model in spec.models:
        model_slug = _slugify(model)
        for setting in settings:
            for repeat in range(1, spec.repeats + 1):
                job_config = build_job_config(
                    base_config,
                    model=model,
                    max_examples=spec.max_examples,
                    extractors=spec.extractors,
                    enabled_actions=spec.enabled_actions,
                    **setting,
                )
                stem = _job_stem(model_slug, repeat=repeat, **setting)
                config_path = config_dir / f"{stem}.yaml"
                _write_yaml(config_path, job_config)
                jobs.append(
                    ExperimentJob(
                        model=model,
                        repeat=repeat,
                        config_path=config_path,
                        results_root=run_root,
                        **setting,
                    )
                )
    return jobs


def expand_matrix(matrix: ExperimentMatrix) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for strategy in matrix.strategies:
        if strategy == "direct_query":
            candidates = [(False, False, "xgrammar", "function")]
        else:
            candidates = itertools.product(
                matrix.use_constraints,
                matrix.use_global_constraints,
                matrix.constraint_backends,
                matrix.output_formats,
            )
        for use_constraints, use_global_constraints, constraint_backend, output_format in candidates:
            if constraint_backend == "legacy_state_machine" and output_format == "json":
                continue
            if not use_constraints:
                constraint_backend = "xgrammar"
            key = (strategy, use_constraints, use_global_constraints, constraint_backend, output_format)
            if key in seen:
                continue
            seen.add(key)
            settings.append(
                {
                    "strategy": strategy,
                    "use_constraints": use_constraints,
                    "use_global_constraints": use_global_constraints,
                    "constraint_backend": constraint_backend,
                    "output_format": output_format,
                }
            )
    return settings


def describe_matrix_adjustments(matrix: ExperimentMatrix) -> list[str]:
    notes = []
    transformation_strategies = set(matrix.strategies) & {"iterative", "cot"}
    if transformation_strategies and "legacy_state_machine" in matrix.constraint_backends and "json" in matrix.output_formats:
        notes.append("Omitted legacy_state_machine + JSON combinations because the legacy backend only supports function output.")
    if transformation_strategies and False in matrix.use_constraints and len(matrix.constraint_backends) > 1:
        notes.append("Deduplicated unconstrained backend variants and selected xgrammar because unconstrained jobs use the modern runtime.")
    if "direct_query" in matrix.strategies:
        notes.append("Collapsed direct_query to one unconstrained function configuration because action constraints do not apply.")
    return notes


def build_job_config(
    base_config: dict[str, Any],
    *,
    model: str,
    strategy: str,
    use_constraints: bool,
    use_global_constraints: bool,
    constraint_backend: str,
    output_format: str,
    max_examples: int | None,
    extractors: list[str],
    enabled_actions: list[str],
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("model", {})["id"] = model
    if max_examples is not None:
        config.setdefault("run", {})["max_examples"] = max_examples

    config["extractors"] = list(extractors)
    generation = config.setdefault("generation", {})
    generation.update(
        {
            "strategy": strategy,
            "use_constraints": use_constraints,
            "use_global_constraints": use_global_constraints,
            "constraint_backend": constraint_backend,
            "output_format": output_format,
            "enabled_actions": list(enabled_actions),
        }
    )

    return config


def run_job(job: ExperimentJob, *, experiment_dir: Path, project_root: Path, python_executable: Path) -> JobResult:
    log_dir = experiment_dir / "job_logs"
    log_stem = _job_stem(
        _slugify(job.model),
        strategy=job.strategy,
        use_constraints=job.use_constraints,
        use_global_constraints=job.use_global_constraints,
        constraint_backend=job.constraint_backend,
        output_format=job.output_format,
        repeat=job.repeat,
    )
    stdout_log = log_dir / f"{log_stem}.stdout.log"
    stderr_log = log_dir / f"{log_stem}.stderr.log"
    started_at = time.time()
    return_code = -1
    run_dir = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        existing_manifests = {path.resolve() for path in job.results_root.glob("**/run_config.json")}
        env = os.environ.copy()
        env["LEAP_CONFIG_PATH"] = str(job.config_path)
        env["LEAP_RESULTS_ROOT"] = str(job.results_root)
        env["PYTHONUNBUFFERED"] = "1"
        # Keep each child run's per-question tqdm visible below the outer suite
        # bar without redirecting the rest of the child's stderr away from logs.
        env["LEAP_TQDM_TO_TTY"] = "1"
        env["LEAP_TQDM_POSITION"] = "1"
        env["LEAP_TQDM_LEAVE"] = "0"
        # Each generated job config selects its own runtime. Do not let the
        # experiment runner's bootstrap prevent a child from switching V0/V1.
        env.pop(BOOTSTRAPPED_ENV_VAR, None)

        return_code, runtime_seconds = _run_child_process(
            [str(python_executable), "main.py"],
            project_root=project_root,
            env=env,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            job=job,
        )

        run_dir = _find_new_run_dir(job.results_root, existing_manifests)
        metrics = collect_run_metrics(run_dir) if run_dir else {}
        examples = metrics.get("examples", 0)
        accuracy = metrics.get("average_extractor_accuracy")
        if accuracy is None:
            accuracy = metrics.get("accuracy")
        status = "ok" if return_code == 0 and run_dir and examples > 0 and accuracy is not None else "failed"
        error = None
        if return_code != 0:
            error = _format_process_error(return_code, stderr_log, stdout_log)
        elif not run_dir:
            error = "No run_config.json was produced"
        elif examples == 0:
            error = "Run completed without results.jsonl entries"
        elif accuracy is None:
            error = "Run completed without an accuracy metric"

        return _job_result(
            job,
            status=status,
            return_code=return_code,
            runtime_seconds=runtime_seconds,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            run_dir=run_dir,
            metrics=metrics,
            error=error,
        )
    except Exception as exc:
        LOGGER.exception("Job handling failed for %s", log_stem)
        return _job_result(
            job,
            status="failed",
            return_code=return_code,
            runtime_seconds=time.time() - started_at,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            run_dir=run_dir,
            metrics={},
            error=f"{type(exc).__name__}: {exc}",
        )


def _job_result(
    job: ExperimentJob,
    *,
    status: str,
    return_code: int,
    runtime_seconds: float,
    stdout_log: Path,
    stderr_log: Path,
    run_dir: Path | None,
    metrics: dict[str, Any],
    error: str | None,
) -> JobResult:
    return JobResult(
        model=job.model,
        strategy=job.strategy,
        use_constraints=job.use_constraints,
        use_global_constraints=job.use_global_constraints,
        constraint_backend=job.constraint_backend,
        output_format=job.output_format,
        expected_runtime=job.expected_runtime,
        repeat=job.repeat,
        status=status,
        return_code=return_code,
        runtime_seconds=runtime_seconds,
        config_path=str(job.config_path),
        run_dir=str(run_dir) if run_dir else None,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        accuracy=metrics.get("accuracy"),
        examples=metrics.get("examples", 0),
        successful_examples=metrics.get("successful_examples", 0),
        answer_found_rate=metrics.get("answer_found_rate"),
        termination_rate=metrics.get("termination_rate"),
        average_actions=metrics.get("average_actions"),
        error_rate=metrics.get("error_rate"),
        invalid_generation_end_count=metrics.get("invalid_generation_end_count", 0),
        invalid_candidate_count=metrics.get("invalid_candidate_count", 0),
        missing_generation_count=metrics.get("missing_generation_count", 0),
        total_error_count=metrics.get("total_error_count", 0),
        error=error,
        average_extractor_accuracy=metrics.get("average_extractor_accuracy"),
        method_accuracies=metrics.get("method_accuracies", {}),
    )


def collect_run_metrics(run_dir: Path) -> dict[str, Any]:
    results_file = run_dir / "results.jsonl"
    accuracy_file = run_dir / "end_to_end_accuracy.json"
    extractor_accuracy_file = run_dir / "extractor_accuracy.json"
    rows = _read_jsonl(results_file)
    examples = len(rows)

    accuracy = None
    if accuracy_file.exists():
        accuracy = json.loads(accuracy_file.read_text(encoding="utf-8")).get("end_to_end_accuracy")
    if accuracy is None and examples:
        accuracy = _mean(row.get("execution_accuracy", 0.0) for row in rows)

    method_accuracies: dict[str, float] = {}
    average_extractor_accuracy = None
    if extractor_accuracy_file.exists():
        try:
            extractor_report = json.loads(extractor_accuracy_file.read_text(encoding="utf-8"))
            method_accuracies = {str(method): float(value) for method, value in extractor_report.get("method_accuracies", {}).items()}
            raw_average = extractor_report.get("average_accuracy")
            average_extractor_accuracy = float(raw_average) if raw_average is not None else None
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            method_accuracies = {}
    elif examples and any(row.get("extractor_results") for row in rows):
        methods = {extractor.get("method") for row in rows for extractor in row.get("extractor_results", []) if extractor.get("method")}
        method_accuracies = {
            method: _mean(
                next(
                    (float(item.get("accuracy", 0.0)) for item in row.get("extractor_results", []) if item.get("method") == method),
                    0.0,
                )
                for row in rows
            )
            for method in sorted(methods)
        }
        average_extractor_accuracy = _mean(method_accuracies.values())

    successful_examples = sum(1 for row in rows if float(row.get("execution_accuracy", 0.0)) == 1.0)
    answer_found_rate = _mean(1.0 if row.get("execution_metrics", {}).get("answer_found_in_final") else 0.0 for row in rows)
    termination_rate = _mean(1.0 if row.get("execution_metrics", {}).get("terminated_properly") else 0.0 for row in rows)
    average_actions = _mean(row.get("execution_metrics", {}).get("num_actions", 0) for row in rows)
    error_summary = _collect_error_summary(run_dir, rows)

    return {
        "accuracy": accuracy,
        "average_extractor_accuracy": average_extractor_accuracy,
        "method_accuracies": method_accuracies,
        "examples": examples,
        "successful_examples": successful_examples,
        "answer_found_rate": answer_found_rate,
        "termination_rate": termination_rate,
        "average_actions": average_actions,
        "error_rate": error_summary["error_rate"],
        "invalid_generation_end_count": error_summary["invalid_generation_end_count"],
        "invalid_candidate_count": error_summary["invalid_candidate_count"],
        "missing_generation_count": error_summary["missing_generation_count"],
        "total_error_count": error_summary["total_error_count"],
    }


def _collect_error_summary(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_file = run_dir / "table_logs" / "summary_report.json"
    if summary_file.exists():
        try:
            summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
            error_summary = summary_data.get("error_summary", {})
            invalid_generation_end_count = int(error_summary.get("invalid_generation_end_count", 0) or 0)
            total_requests = int(summary_data.get("total_requests", len(rows)) or len(rows))
            return {
                "error_rate": (invalid_generation_end_count / total_requests) if total_requests > 0 else None,
                "invalid_generation_end_count": invalid_generation_end_count,
                "invalid_candidate_count": int(error_summary.get("invalid_candidate_count", 0) or 0),
                "missing_generation_count": int(error_summary.get("missing_generation_count", 0) or 0),
                "total_error_count": int(error_summary.get("total_error_count", 0) or 0),
            }
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    invalid_generation_end_count = 0
    invalid_candidate_count = 0
    missing_generation_count = 0
    execution_error_count = 0

    for row in rows:
        if row.get("execution_metrics", {}).get("execution_error"):
            execution_error_count += 1
        for metadata in row.get("sampling_metadata", []) or []:
            n_requested = int(metadata.get("n_requested", 0) or 0)
            n_generated = int(metadata.get("n_generated", 0) or 0)
            n_valid = int(metadata.get("n_valid", 0) or 0)
            winner = metadata.get("winner") or {}
            fallback_reason = metadata.get("fallback_reason")
            if winner.get("action") == "end" and (
                fallback_reason in {"no_valid_candidates", "action_type_generation_failed"} or n_valid == 0
            ):
                invalid_generation_end_count += 1
            invalid_candidate_count += max(n_generated - n_valid, 0)
            missing_generation_count += max(n_requested - n_generated, 0)

    total_error_count = invalid_candidate_count + missing_generation_count + execution_error_count
    return {
        "error_rate": (invalid_generation_end_count / len(rows)) if rows else None,
        "invalid_generation_end_count": invalid_generation_end_count,
        "invalid_candidate_count": invalid_candidate_count,
        "missing_generation_count": missing_generation_count,
        "total_error_count": total_error_count,
    }


def build_report(spec: ExperimentSpec, experiment_id: str, experiment_dir: Path, job_results: list[JobResult]) -> dict[str, Any]:
    serial_jobs = [asdict(result) for result in job_results]
    configuration_keys = ("strategy", "use_constraints", "use_global_constraints", "constraint_backend", "output_format")
    return {
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "base_config": str(spec.base_config),
        "models": spec.models,
        "matrix": asdict(spec.matrix),
        "matrix_adjustments": describe_matrix_adjustments(spec.matrix),
        "extractors": spec.extractors,
        "enabled_actions": spec.enabled_actions,
        "repeats": spec.repeats,
        "max_examples": spec.max_examples,
        "python_executable": str(spec.python_executable),
        "required_vllm_runtimes": sorted({result.expected_runtime for result in job_results}),
        "jobs": serial_jobs,
        "summary_by_configuration": _summarize(job_results, configuration_keys),
        "summary_by_model_and_configuration": _summarize(job_results, ("model", *configuration_keys)),
    }


def write_reports(report: dict[str, Any], experiment_dir: Path) -> None:
    json_path = experiment_dir / "experiment_report.json"
    md_path = experiment_dir / "experiment_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Experiment Report",
        "",
        f"Experiment ID: {report['experiment_id']}",
        f"Base config: {report['base_config']}",
        f"Max examples: {report['max_examples']}",
        f"Repeats: {report['repeats']}",
        f"Python: {report['python_executable']}",
        f"Required vLLM runtimes: {', '.join(report['required_vllm_runtimes']) or 'none'}",
        *(f"Matrix adjustment: {note}" for note in report["matrix_adjustments"]),
        "",
        "## Summary By Configuration",
        "",
        "| Strategy | Constraints | Global | Backend | Format | Jobs | Mean Accuracy | Std Dev | Error Rate | Failed | Avg Runtime |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary_by_configuration"]:
        lines.append(
            "| {strategy} | {constraints} | {global_constraints} | {backend} | {format} | {jobs} | "
            "{mean_accuracy} | {std_dev} | {error_rate} | {failed_jobs} | {runtime} |".format(
                strategy=row["strategy"],
                constraints=_fmt_bool(row["use_constraints"]),
                global_constraints=_fmt_bool(row["use_global_constraints"]),
                backend=row["constraint_backend"],
                format=row["output_format"],
                jobs=row["jobs"],
                mean_accuracy=_fmt_float(row["mean_accuracy"]),
                std_dev=_fmt_float(row["std_dev"]),
                error_rate=_fmt_float(row["error_rate"]),
                failed_jobs=row["failed_jobs"],
                runtime=_format_seconds(row["avg_runtime_seconds"]),
            )
        )

    lines.extend(
        [
            "",
            "## Summary By Model And Configuration",
            "",
            "| Model | Strategy | Constraints | Global | Backend | Format | Runs | Mean Accuracy | "
            "Error Rate | Successful Examples | Avg Runtime |",
            "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["summary_by_model_and_configuration"]:
        lines.append(
            "| {model} | {strategy} | {constraints} | {global_constraints} | {backend} | {format} | {runs} | "
            "{mean_accuracy} | {error_rate} | {success}/{examples} | {runtime} |".format(
                model=row["model"],
                strategy=row["strategy"],
                constraints=_fmt_bool(row["use_constraints"]),
                global_constraints=_fmt_bool(row["use_global_constraints"]),
                backend=row["constraint_backend"],
                format=row["output_format"],
                runs=row["jobs"],
                mean_accuracy=_fmt_float(row["mean_accuracy"]),
                error_rate=_fmt_float(row["error_rate"]),
                success=row["successful_examples"],
                examples=row["examples"],
                runtime=_format_seconds(row["avg_runtime_seconds"]),
            )
        )

    lines.extend(
        [
            "",
            "## Individual Runs",
            "",
            "| Model | Strategy | Constraints | Global | Backend | Format | vLLM | Repeat | Accuracy | "
            "Examples | Runtime | Status | Error | Run Dir |",
            "|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in report["jobs"]:
        lines.append(
            "| {model} | {strategy} | {constraints} | {global_constraints} | {backend} | {format} | {vllm} | {repeat} | "
            "{accuracy} | {examples} | {runtime} | {status} | {error} | {run_dir} |".format(
                model=row["model"],
                strategy=row["strategy"],
                constraints=_fmt_bool(row["use_constraints"]),
                global_constraints=_fmt_bool(row["use_global_constraints"]),
                backend=row["constraint_backend"],
                format=row["output_format"],
                vllm=row["expected_runtime"],
                repeat=row["repeat"],
                accuracy=_fmt_float(
                    row["average_extractor_accuracy"] if row["average_extractor_accuracy"] is not None else row["accuracy"]
                ),
                examples=row["examples"],
                runtime=_format_seconds(row["runtime_seconds"]),
                status=row["status"],
                error=_markdown_cell(row["error"] or ""),
                run_dir=row["run_dir"] or "",
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run configured LEAP experiments.")
    parser.add_argument(
        "spec",
        nargs="?",
        type=Path,
        default=DEFAULT_SPEC_PATH,
        help=f"Path to an experiments YAML file. Defaults to {DEFAULT_SPEC_PATH}.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Defaults to INFO.")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    spec = load_experiment_spec(args.spec)
    validate_models(spec)
    validate_python_executable(spec.python_executable)

    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = spec.output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(experiment_dir / "experiment_spec.yaml", _read_yaml_raw(args.spec))

    jobs = create_jobs(spec, experiment_dir)
    LOGGER.info("Experiment %s", experiment_id)
    LOGGER.info("Output directory: %s", experiment_dir)
    LOGGER.info("Python executable: %s", spec.python_executable)
    LOGGER.info("Prepared %s jobs", len(jobs))
    for note in describe_matrix_adjustments(spec.matrix):
        LOGGER.info("Matrix adjustment: %s", note)

    job_results = []
    project_root = PROJECT_ROOT
    progress = tqdm(jobs, desc="Experiments", unit="job")
    for job in progress:
        progress.set_postfix_str(f"{job.strategy} r{job.repeat}", refresh=False)
        LOGGER.info(
            "Starting job: model=%s strategy=%s constraints=%s global=%s backend=%s format=%s repeat=%s",
            job.model,
            job.strategy,
            job.use_constraints,
            job.use_global_constraints,
            job.constraint_backend,
            job.output_format,
            job.repeat,
        )
        try:
            result = run_job(
                job,
                experiment_dir=experiment_dir,
                project_root=project_root,
                python_executable=spec.python_executable,
            )
        except Exception as exc:
            LOGGER.exception("Unexpected runner failure; recording the job and continuing")
            log_dir = experiment_dir / "job_logs"
            result = _job_result(
                job,
                status="failed",
                return_code=-1,
                runtime_seconds=0.0,
                stdout_log=log_dir / f"{_job_identifier(job)}.stdout.log",
                stderr_log=log_dir / f"{_job_identifier(job)}.stderr.log",
                run_dir=None,
                metrics={},
                error=f"{type(exc).__name__}: {exc}",
            )
        job_results.append(result)
        try:
            write_reports(build_report(spec, experiment_id, experiment_dir, job_results), experiment_dir)
        except Exception:
            LOGGER.exception("Could not update experiment reports; continuing with the next job")
        if result.status == "ok":
            LOGGER.info(
                "Finished job: model=%s strategy=%s repeat=%s examples=%s accuracy=%s runtime=%s",
                result.model,
                result.strategy,
                result.repeat,
                result.examples,
                _fmt_float(result.accuracy),
                _format_seconds(result.runtime_seconds),
            )
        else:
            LOGGER.error(
                "Failed job: model=%s strategy=%s repeat=%s error=%s stderr=%s",
                result.model,
                result.strategy,
                result.repeat,
                result.error,
                result.stderr_log,
            )

    try:
        write_reports(build_report(spec, experiment_id, experiment_dir, job_results), experiment_dir)
    except Exception:
        LOGGER.exception("Could not write final experiment reports")
        return 1
    LOGGER.info("Experiment report written to %s", experiment_dir / "experiment_report.md")
    return 0 if job_results and all(result.status == "ok" for result in job_results) else 1


def _run_child_process(
    cmd: list[str],
    *,
    project_root: Path,
    env: dict[str, str],
    stdout_log: Path,
    stderr_log: Path,
    job: ExperimentJob,
) -> tuple[int, float]:
    start = time.time()
    process = None
    with stdout_log.open("w", encoding="utf-8") as stdout_f, stderr_log.open("w", encoding="utf-8") as stderr_f:
        try:
            process = subprocess.Popen(
                cmd,
                cwd=project_root,
                env=env,
                stdout=stdout_f,
                stderr=stderr_f,
                text=True,
            )
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                time.sleep(1)
        except BaseException:
            if process is not None:
                _terminate_process(process)
            raise
    return int(return_code), time.time() - start


def _terminate_process(process: subprocess.Popen) -> None:
    try:
        running = process.poll() is None
    except Exception:
        running = True
    if not running:
        return
    try:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    except Exception:
        LOGGER.exception("Could not terminate failed experiment child process")


def _summarize(results: list[JobResult], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[JobResult]] = {}
    for result in results:
        grouped.setdefault(tuple(getattr(result, key) for key in keys), []).append(result)

    rows = []
    for group_key, group_results in sorted(grouped.items()):
        ok_results = [result for result in group_results if result.status == "ok"]
        accuracies = [
            result.average_extractor_accuracy if result.average_extractor_accuracy is not None else result.accuracy
            for result in ok_results
            if result.average_extractor_accuracy is not None or result.accuracy is not None
        ]
        row = {key: value for key, value in zip(keys, group_key)}
        row.update(
            {
                "jobs": len(group_results),
                "mean_accuracy": _mean(accuracies),
                "std_dev": statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0 if accuracies else None,
                "answer_found_rate": _mean(result.answer_found_rate for result in ok_results if result.answer_found_rate is not None),
                "termination_rate": _mean(result.termination_rate for result in ok_results if result.termination_rate is not None),
                "error_rate": _mean(result.error_rate for result in ok_results if result.error_rate is not None),
                "failed_jobs": sum(1 for result in group_results if result.status != "ok"),
                "avg_runtime_seconds": _mean(result.runtime_seconds for result in group_results),
                "successful_examples": sum(result.successful_examples for result in ok_results),
                "examples": sum(result.examples for result in ok_results),
                "average_actions": _mean(result.average_actions for result in ok_results if result.average_actions is not None),
                "invalid_generation_end_count": sum(result.invalid_generation_end_count for result in ok_results),
                "invalid_candidate_count": sum(result.invalid_candidate_count for result in ok_results),
                "missing_generation_count": sum(result.missing_generation_count for result in ok_results),
                "total_error_count": sum(result.total_error_count for result in ok_results),
            }
        )
        rows.append(row)
    return rows


def _find_new_run_dir(results_root: Path, existing_manifests: set[Path]) -> Path | None:
    candidates = [path.resolve() for path in results_root.glob("**/run_config.json") if path.resolve() not in existing_manifests]
    if not candidates:
        return None
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return newest.parent


def _format_process_error(return_code: int, stderr_log: Path, stdout_log: Path) -> str:
    stderr_tail = _tail(stderr_log)
    stdout_tail = _tail(stdout_log)
    detail = stderr_tail or stdout_tail
    if detail:
        return f"Process exited with code {return_code}: {detail}"
    return f"Process exited with code {return_code}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:].strip()


def _mean(values) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = int(round(seconds))
    minutes, remainder = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {remainder}s"
    return f"{remainder}s"


def _fmt_bool(value: bool) -> str:
    return "on" if value else "off"


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _job_stem(
    model_slug: str,
    *,
    strategy: str,
    use_constraints: bool,
    use_global_constraints: bool,
    constraint_backend: str,
    output_format: str,
    repeat: int,
) -> str:
    return "_".join(
        [
            model_slug,
            _slugify(strategy),
            f"constraints-{_fmt_bool(use_constraints)}",
            f"global-{_fmt_bool(use_global_constraints)}",
            _slugify(constraint_backend),
            _slugify(output_format),
            f"r{repeat}",
        ]
    )


def _job_identifier(job: ExperimentJob) -> str:
    return _job_stem(
        _slugify(job.model),
        strategy=job.strategy,
        use_constraints=job.use_constraints,
        use_global_constraints=job.use_global_constraints,
        constraint_backend=job.constraint_backend,
        output_format=job.output_format,
        repeat=job.repeat,
    )


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")


def _string_list(raw: dict[str, Any], key: str, valid_values: set[str]) -> list[str]:
    values = raw.get(key)
    if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"'{key}' must be a non-empty list of strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"'{key}' must not contain duplicates.")
    unknown = sorted(set(values) - valid_values)
    if unknown:
        raise ValueError(f"Unknown {key} {unknown}. Valid values: {sorted(valid_values)}")
    return values


def _bool_list(raw: dict[str, Any], key: str) -> list[bool]:
    values = raw.get(key)
    if not isinstance(values, list) or not values or not all(type(value) is bool for value in values):
        raise ValueError(f"'{key}' must be a non-empty list of booleans.")
    if len(values) != len(set(values)):
        raise ValueError(f"'{key}' must not contain duplicates.")
    return values


def _resolve_path(value, base_dir: Path, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate.resolve()
    return (project_root / path).resolve()


def _default_python_executable() -> str:
    return sys.executable


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def _read_yaml_raw(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("spec", nargs="?", type=Path, default=DEFAULT_SPEC_PATH)
    bootstrap_parser.add_argument("--log-level")
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    ensure_vllm_runtime(runtime_for_experiment(bootstrap_args.spec, project_root=PROJECT_ROOT), argv=sys.argv, project_root=PROJECT_ROOT)
    raise SystemExit(main())
