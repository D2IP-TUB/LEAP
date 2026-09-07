from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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

from leap.utils.config_labels import build_generation_config_label  # noqa: E402
from leap.vllm_runtime import BOOTSTRAPPED_ENV_VAR, runtime_for_generation, supports_legacy_state_machine  # noqa: E402

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
    force_zero_temperature: list[bool]
    add_column: list[bool] = field(default_factory=lambda: [True])


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
    reuse_models: bool = True


@dataclass(frozen=True)
class ExperimentJob:
    model: str
    strategy: str
    use_constraints: bool
    use_global_constraints: bool
    constraint_backend: str
    output_format: str
    force_zero_temperature: bool
    add_column: bool
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
class ExperimentSession:
    session_id: str
    model: str
    expected_runtime: str
    jobs: list[ExperimentJob]


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
    add_column: bool = True
    add_column_requested_candidate_count: int = 0
    add_column_failed_candidate_count: int = 0
    add_column_affected_request_count: int = 0
    add_column_affected_sampling_step_count: int = 0
    add_column_failure_code_counts: dict[str, int] = field(default_factory=dict)
    config_key: str | None = None
    config_label: str | None = None
    error: str | None = None
    average_extractor_accuracy: float | None = None
    method_accuracies: dict[str, float] = field(default_factory=dict)
    session_id: str | None = None
    model_load_id: str | None = None
    model_reused: bool = False
    restart_reason: str | None = None


@dataclass(frozen=True)
class ComparisonRun:
    config_key: str
    config_label: str
    run_dir: Path
    rows: list[dict[str, Any]]


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
    extractors = _string_list(raw, "extractors", VALID_EXTRACTORS)
    enabled_actions = _string_list(raw, "enabled_actions", VALID_ACTIONS)
    matrix_raw = raw.get("matrix")
    if not isinstance(matrix_raw, dict):
        raise ValueError("'matrix' must be a mapping of generation setting lists.")
    matrix = ExperimentMatrix(
        strategies=_string_list(matrix_raw, "strategies", VALID_STRATEGIES),
        use_constraints=_bool_list(matrix_raw, "use_constraints"),
        use_global_constraints=_bool_list(matrix_raw, "use_global_constraints"),
        constraint_backends=_string_list(matrix_raw, "constraint_backends", VALID_CONSTRAINT_BACKENDS),
        output_formats=_string_list(matrix_raw, "output_formats", VALID_OUTPUT_FORMATS),
        force_zero_temperature=_bool_list(matrix_raw, "force_zero_temperature"),
        add_column=_bool_list(matrix_raw, "add_column", default=["add_column" in enabled_actions]),
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

    reuse_models = raw.get("reuse_models", True)
    if type(reuse_models) is not bool:
        raise ValueError("'reuse_models' must be a boolean.")

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
        reuse_models=reuse_models,
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


def create_jobs(spec: ExperimentSpec, experiment_dir: Path, *, write_configs: bool = True) -> list[ExperimentJob]:
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
            legacy_backend_not_supported = (
                setting["use_constraints"]
                and setting["constraint_backend"] == "legacy_state_machine"
                and not supports_legacy_state_machine(model)
            )
            if legacy_backend_not_supported:
                continue
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
                if write_configs:
                    _write_yaml(config_path, job_config)
                elif not config_path.exists():
                    raise FileNotFoundError(f"Frozen experiment config is missing: {config_path}")
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


def group_jobs(jobs: list[ExperimentJob]) -> list[ExperimentSession]:
    grouped: dict[tuple[str, str], list[ExperimentJob]] = {}
    for job in jobs:
        grouped.setdefault((job.model, job.expected_runtime), []).append(job)
    return [
        ExperimentSession(
            session_id=f"{_slugify(model)}-{runtime}",
            model=model,
            expected_runtime=runtime,
            jobs=session_jobs,
        )
        for (model, runtime), session_jobs in grouped.items()
    ]


def expand_matrix(matrix: ExperimentMatrix) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for strategy in matrix.strategies:
        if strategy == "direct_query":
            candidates = [
                (False, False, "xgrammar", "function", force_zero_temperature, matrix.add_column[0])
                for force_zero_temperature in matrix.force_zero_temperature
            ]
        else:
            candidates = itertools.product(
                matrix.use_constraints,
                matrix.use_global_constraints,
                matrix.constraint_backends,
                matrix.output_formats,
                matrix.force_zero_temperature,
                matrix.add_column,
            )
        for use_constraints, use_global_constraints, constraint_backend, output_format, force_zero_temperature, add_column in candidates:
            if constraint_backend == "legacy_state_machine" and output_format != "function":
                continue
            if not use_constraints:
                constraint_backend = "xgrammar"
            key = (strategy, use_constraints, use_global_constraints, constraint_backend, output_format, force_zero_temperature, add_column)
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
                    "force_zero_temperature": force_zero_temperature,
                    "add_column": add_column,
                }
            )
    return settings


def describe_matrix_adjustments(matrix: ExperimentMatrix) -> list[str]:
    notes = []
    transformation_strategies = set(matrix.strategies) & {"iterative", "cot"}
    structured_formats = {"json"} & set(matrix.output_formats)
    if transformation_strategies and "legacy_state_machine" in matrix.constraint_backends and structured_formats:
        notes.append(
            "Omitted legacy_state_machine + structured-output combinations because the legacy backend only supports function output."
        )
    if transformation_strategies and "legacy_state_machine" in matrix.constraint_backends:
        notes.append("Omitted legacy_state_machine jobs for non-Qwen2.5 models.")
    if transformation_strategies and False in matrix.use_constraints and len(matrix.constraint_backends) > 1:
        notes.append("Deduplicated unconstrained backend variants and selected xgrammar because unconstrained jobs use the modern runtime.")
    if "direct_query" in matrix.strategies:
        notes.append(
            """Collapsed direct_query to the unconstrained function path while retaining the temperature-mode sweep because action 
            constraints do not apply."""
        )
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
    force_zero_temperature: bool,
    add_column: bool,
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
            "force_zero_temperature": force_zero_temperature,
            "enabled_actions": _enabled_actions_with_add_column(enabled_actions, add_column),
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
        force_zero_temperature=job.force_zero_temperature,
        add_column=job.add_column,
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
        env = _child_environment(config_path=job.config_path, results_root=job.results_root)

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


def _config_identity_from_path(config_path: Path) -> dict[str, Any]:
    try:
        raw = _read_yaml_raw(config_path)
    except FileNotFoundError:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    return {
        "model": {
            "id": raw.get("model", {}).get("id"),
            "instruct": raw.get("model", {}).get("instruct"),
            "hardware": raw.get("model", {}).get("hardware"),
            "tokenizer_config": raw.get("model", {}).get("tokenizer_config"),
        },
        "dataset": raw.get("dataset"),
        "run": raw.get("run"),
        "generation": raw.get("generation"),
        "extractors": raw.get("extractors", []),
    }


def _config_identity_from_job(job: ExperimentJob) -> dict[str, Any]:
    identity = _config_identity_from_path(job.config_path)
    if identity:
        return identity
    return {
        "model": {"id": job.model},
        "generation": {
            "strategy": job.strategy,
            "use_constraints": job.use_constraints,
            "use_global_constraints": job.use_global_constraints,
            "constraint_backend": job.constraint_backend,
            "output_format": job.output_format,
            "enabled_actions": _enabled_actions_with_add_column([], job.add_column),
        },
    }


def _config_identity_key(job: ExperimentJob) -> str:
    identity_json = json.dumps(_config_identity_from_job(job), sort_keys=True)
    return hashlib.sha1(identity_json.encode("utf-8")).hexdigest()


def _config_identity_label(job: ExperimentJob) -> str:
    identity = _config_identity_from_job(job)
    model_id = identity.get("model", {}).get("id", job.model)
    generation = identity.get("generation", {}) or {}
    return build_generation_config_label(
        model_id=model_id,
        generation=generation,
        strategy=job.strategy,
        use_constraints=job.use_constraints,
        use_global_constraints=job.use_global_constraints,
        constraint_backend=job.constraint_backend,
        output_format=job.output_format,
        force_zero_temperature=job.force_zero_temperature,
        add_column_enabled=job.add_column,
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
    session_id: str | None = None,
    model_load_id: str | None = None,
    model_reused: bool = False,
    restart_reason: str | None = None,
) -> JobResult:
    model_load_id = model_load_id or f"standalone-{_job_identifier(job)}"
    config_key = _config_identity_key(job)
    config_label = _config_identity_label(job)
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
        add_column=job.add_column,
        add_column_requested_candidate_count=metrics.get("add_column_requested_candidate_count", 0),
        add_column_failed_candidate_count=metrics.get("add_column_failed_candidate_count", 0),
        add_column_affected_request_count=metrics.get("add_column_affected_request_count", 0),
        add_column_affected_sampling_step_count=metrics.get("add_column_affected_sampling_step_count", 0),
        add_column_failure_code_counts=metrics.get("add_column_failure_code_counts", {}),
        config_key=config_key,
        config_label=config_label,
        error=error,
        average_extractor_accuracy=metrics.get("average_extractor_accuracy"),
        method_accuracies=metrics.get("method_accuracies", {}),
        session_id=session_id,
        model_load_id=model_load_id,
        model_reused=model_reused,
        restart_reason=restart_reason,
    )


def run_session_group(
    session: ExperimentSession,
    *,
    experiment_dir: Path,
    project_root: Path,
    python_executable: Path,
    on_result,
) -> None:
    pending = list(session.jobs)
    attempt = _latest_session_attempt(experiment_dir, session.session_id)
    while pending:
        attempt += 1
        attempt_id = f"{session.session_id}-attempt-{attempt}"
        session_dir = experiment_dir / "sessions" / attempt_id
        session_dir.mkdir(parents=True, exist_ok=True)
        status_path = session_dir / "status.jsonl"
        manifest_path = session_dir / "manifest.json"
        manifest = {
            "session_id": attempt_id,
            "status_path": str(status_path),
            "jobs": [
                {
                    "job_id": _job_identifier(job),
                    "config_path": str(job.config_path),
                    "results_root": str(job.results_root),
                }
                for job in pending
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log_dir = experiment_dir / "job_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = log_dir / f"{attempt_id}.stdout.log"
        stderr_log = log_dir / f"{attempt_id}.stderr.log"
        env = _child_environment(config_path=pending[0].config_path, results_root=pending[0].results_root)
        env["LEAP_SESSION_MANIFEST"] = str(manifest_path)

        terminal_ids: set[str] = set()
        started_ids: list[str] = []
        job_by_id = {_job_identifier(job): job for job in pending}

        def handle_event(event: dict[str, Any]) -> None:
            job_id = event.get("job_id")
            if job_id not in job_by_id:
                return
            if event.get("event") == "started":
                started_ids.append(job_id)
                return
            if event.get("event") not in {"completed", "failed"} or job_id in terminal_ids:
                return
            terminal_ids.add(job_id)
            on_result(_result_from_session_event(job_by_id[job_id], event, stdout_log, stderr_log, attempt_id))

        return_code, process_error = _run_session_process(
            [str(python_executable), "main.py"],
            project_root=project_root,
            env=env,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            status_path=status_path,
            on_event=handle_event,
        )

        unfinished = [job for job in pending if _job_identifier(job) not in terminal_ids]
        if unfinished:
            active_id = next((job_id for job_id in reversed(started_ids) if job_id not in terminal_ids), _job_identifier(unfinished[0]))
            active_job = job_by_id[active_id]
            terminal_ids.add(active_id)
            on_result(
                _job_result(
                    active_job,
                    status="failed",
                    return_code=return_code,
                    runtime_seconds=0.0,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    run_dir=None,
                    metrics={},
                    error=process_error or _format_process_error(return_code, stderr_log, stdout_log),
                    session_id=attempt_id,
                    model_load_id=f"{attempt_id}-crashed",
                    restart_reason="Persistent session process crashed",
                )
            )
        pending = [job for job in pending if _job_identifier(job) not in terminal_ids]


def _result_from_session_event(
    job: ExperimentJob,
    event: dict[str, Any],
    stdout_log: Path,
    stderr_log: Path,
    session_id: str,
) -> JobResult:
    run_dir = Path(event["run_dir"]) if event.get("run_dir") else None
    metrics = {}
    error = event.get("error")
    status = "failed"
    try:
        metrics = collect_run_metrics(run_dir) if run_dir else {}
        primary_accuracy = metrics.get("average_extractor_accuracy")
        if primary_accuracy is None:
            primary_accuracy = metrics.get("accuracy")
        if event["event"] == "completed" and metrics.get("examples", 0) > 0 and primary_accuracy is not None:
            status = "ok"
        elif event["event"] == "completed":
            error = "Session completed without usable result metrics"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return _job_result(
        job,
        status=status,
        return_code=0 if status == "ok" else 1,
        runtime_seconds=float(event.get("runtime_seconds", 0.0)),
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        run_dir=run_dir,
        metrics=metrics,
        error=error,
        session_id=session_id,
        model_load_id=event.get("model_load_id"),
        model_reused=bool(event.get("model_reused", False)),
        restart_reason=event.get("restart_reason"),
    )


def _child_environment(*, config_path: Path, results_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["LEAP_CONFIG_PATH"] = str(config_path)
    env["LEAP_RESULTS_ROOT"] = str(results_root)
    env["PYTHONUNBUFFERED"] = "1"
    env["LEAP_TQDM_TO_TTY"] = "1"
    env["LEAP_TQDM_POSITION"] = "1"
    env["LEAP_TQDM_LEAVE"] = "0"
    env.pop(BOOTSTRAPPED_ENV_VAR, None)
    return env


def _run_session_process(
    cmd: list[str],
    *,
    project_root: Path,
    env: dict[str, str],
    stdout_log: Path,
    stderr_log: Path,
    status_path: Path,
    on_event,
) -> tuple[int, str | None]:
    process = None
    status_offset = 0
    process_error = None
    try:
        with stdout_log.open("w", encoding="utf-8") as stdout_f, stderr_log.open("w", encoding="utf-8") as stderr_f:
            process = subprocess.Popen(cmd, cwd=project_root, env=env, stdout=stdout_f, stderr=stderr_f, text=True)
            while process.poll() is None:
                status_offset = _consume_session_events(status_path, status_offset, on_event)
                time.sleep(1)
            status_offset = _consume_session_events(status_path, status_offset, on_event)
            return int(process.returncode), None
    except BaseException as exc:
        if process is not None:
            _terminate_process(process)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        process_error = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Persistent experiment session failed")
        _consume_session_events(status_path, status_offset, on_event)
        return int(process.returncode) if process and process.returncode is not None else -1, process_error


def _latest_session_attempt(experiment_dir: Path, session_id: str) -> int:
    sessions_dir = experiment_dir / "sessions"
    attempts = []
    for path in sessions_dir.glob(f"{session_id}-attempt-*"):
        try:
            attempts.append(int(path.name.rsplit("-", 1)[1]))
        except ValueError:
            continue
    return max(attempts, default=0)


def _consume_session_events(status_path: Path, offset: int, on_event) -> int:
    if not status_path.exists():
        return offset
    with status_path.open("r", encoding="utf-8") as status_file:
        status_file.seek(offset)
        while True:
            line_offset = status_file.tell()
            line = status_file.readline()
            if not line:
                return line_offset
            if not line.endswith("\n"):
                return line_offset
            if line.strip():
                on_event(json.loads(line))


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
        "add_column_requested_candidate_count": error_summary["add_column_requested_candidate_count"],
        "add_column_failed_candidate_count": error_summary["add_column_failed_candidate_count"],
        "add_column_affected_request_count": error_summary["add_column_affected_request_count"],
        "add_column_affected_sampling_step_count": error_summary["add_column_affected_sampling_step_count"],
        "add_column_failure_code_counts": error_summary["add_column_failure_code_counts"],
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
                "add_column_requested_candidate_count": int(error_summary.get("add_column_requested_candidate_count", 0) or 0),
                "add_column_failed_candidate_count": int(error_summary.get("add_column_failed_candidate_count", 0) or 0),
                "add_column_affected_request_count": int(error_summary.get("add_column_affected_request_count", 0) or 0),
                "add_column_affected_sampling_step_count": int(error_summary.get("add_column_affected_sampling_step_count", 0) or 0),
                "add_column_failure_code_counts": dict(error_summary.get("add_column_failure_code_counts", {}) or {}),
            }
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    invalid_generation_end_count = 0
    invalid_candidate_count = 0
    missing_generation_count = 0
    execution_error_count = 0
    add_column_requested_candidate_count = 0
    add_column_failed_candidate_count = 0
    add_column_affected_request_count = 0
    add_column_affected_sampling_step_count = 0
    add_column_failure_code_counts: dict[str, int] = {}

    for row in rows:
        request_had_add_column_failure = False
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
            diagnostics = metadata.get("add_column_diagnostics", []) or []
            selected_add_column = winner.get("action") == "add_column" or bool(diagnostics)
            if selected_add_column:
                add_column_requested_candidate_count += n_requested
            if diagnostics:
                request_had_add_column_failure = True
                add_column_affected_sampling_step_count += 1
                add_column_failed_candidate_count += len(diagnostics)
                for diagnostic in diagnostics:
                    code = diagnostic.get("failure_code", "unknown")
                    add_column_failure_code_counts[code] = add_column_failure_code_counts.get(code, 0) + 1
        if request_had_add_column_failure:
            add_column_affected_request_count += 1

    total_error_count = invalid_candidate_count + missing_generation_count + execution_error_count
    return {
        "error_rate": (invalid_generation_end_count / len(rows)) if rows else None,
        "invalid_generation_end_count": invalid_generation_end_count,
        "invalid_candidate_count": invalid_candidate_count,
        "missing_generation_count": missing_generation_count,
        "total_error_count": total_error_count,
        "add_column_requested_candidate_count": add_column_requested_candidate_count,
        "add_column_failed_candidate_count": add_column_failed_candidate_count,
        "add_column_affected_request_count": add_column_affected_request_count,
        "add_column_affected_sampling_step_count": add_column_affected_sampling_step_count,
        "add_column_failure_code_counts": add_column_failure_code_counts,
    }


def build_report(
    spec: ExperimentSpec,
    experiment_id: str,
    experiment_dir: Path,
    job_results: list[JobResult],
    *,
    resumed: bool = False,
) -> dict[str, Any]:
    serial_jobs = [asdict(result) for result in job_results]
    configuration_keys = ("strategy", "use_constraints", "use_global_constraints", "constraint_backend", "output_format", "add_column")
    comparison_inputs = [
        {
            "config_key": result.config_key,
            "config_label": result.config_label,
            "run_dir": result.run_dir,
            "examples": result.examples,
            "accuracy": result.accuracy,
            "successful_examples": result.successful_examples,
            "status": result.status,
        }
        for result in job_results
    ]
    return {
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "resumed": resumed,
        "base_config": str(spec.base_config),
        "models": spec.models,
        "matrix": asdict(spec.matrix),
        "matrix_adjustments": describe_matrix_adjustments(spec.matrix),
        "extractors": spec.extractors,
        "enabled_actions": spec.enabled_actions,
        "repeats": spec.repeats,
        "max_examples": spec.max_examples,
        "python_executable": str(spec.python_executable),
        "reuse_models": spec.reuse_models,
        "model_load_count": len({result.model_load_id for result in job_results if result.model_load_id}),
        "required_vllm_runtimes": sorted({result.expected_runtime for result in job_results}),
        "jobs": serial_jobs,
        "comparison_inputs": comparison_inputs,
        "summary_by_configuration": _summarize(job_results, configuration_keys),
        "summary_by_model_and_configuration": _summarize(job_results, ("model", *configuration_keys)),
    }


def write_reports(report: dict[str, Any], experiment_dir: Path) -> None:
    json_path = experiment_dir / "experiment_report.json"
    md_path = experiment_dir / "experiment_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    extractors = report.get("extractors", [])
    extractor_headers = [f"`{_markdown_cell(method)}` Accuracy" for method in extractors]

    def table_row(cells: list[Any]) -> str:
        return "| " + " | ".join(str(cell) for cell in cells) + " |"

    lines = [
        "# Experiment Report",
        "",
        f"Experiment ID: {report['experiment_id']}",
        f"Resumed: {report.get('resumed', False)}",
        f"Base config: {report['base_config']}",
        f"Max examples: {report['max_examples']}",
        f"Repeats: {report['repeats']}",
        f"Python: {report['python_executable']}",
        f"Reuse models: {report['reuse_models']}",
        f"Model loads: {report['model_load_count']}",
        f"Required vLLM runtimes: {', '.join(report['required_vllm_runtimes']) or 'none'}",
        *(f"Matrix adjustment: {note}" for note in report["matrix_adjustments"]),
        "",
        "## Comparison Inputs",
        "",
        table_row(["Config Key", "Config Label", "Run Dir", "Examples", "Correct", "Status"]),
        table_row(["---", "---", "---", "---:", "---:", "---"]),
        *[
            table_row(
                [
                    row.get("config_key") or "",
                    row.get("config_label") or "",
                    row.get("run_dir") or "",
                    row.get("examples", 0),
                    row.get("successful_examples", 0),
                    row.get("status") or "",
                ]
            )
            for row in report.get("comparison_inputs", [])
        ],
        "",
        "## Summary By Configuration",
        "",
        table_row(
            ["Strategy", "Constraints", "Global", "Backend", "Format", "Add Column", "Jobs"]
            + extractor_headers
            + ["Mean Extractor Accuracy", "Std Dev", "Error Rate", "Failed", "Avg Runtime"]
        ),
        table_row(["---"] * 6 + ["---:"] + ["---:"] * len(extractors) + ["---:"] * 5),
    ]
    for row in report["summary_by_configuration"]:
        lines.append(
            table_row(
                [
                    row["strategy"],
                    _fmt_bool(row["use_constraints"]),
                    _fmt_bool(row["use_global_constraints"]),
                    row["constraint_backend"],
                    row["output_format"],
                    _fmt_bool(row["add_column"]),
                    row["jobs"],
                ]
                + [_fmt_float(row.get("mean_method_accuracies", {}).get(method)) for method in extractors]
                + [
                    _fmt_float(row["mean_accuracy"]),
                    _fmt_float(row["std_dev"]),
                    _fmt_float(row["error_rate"]),
                    row["failed_jobs"],
                    _format_seconds(row["avg_runtime_seconds"]),
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Summary By Model And Configuration",
            "",
            table_row(
                ["Model", "Strategy", "Constraints", "Global", "Backend", "Format", "Add Column", "Runs"]
                + extractor_headers
                + ["Mean Extractor Accuracy", "Error Rate", "Successful Examples", "Avg Runtime"]
            ),
            table_row(["---"] * 7 + ["---:"] + ["---:"] * len(extractors) + ["---:"] * 4),
        ]
    )
    for row in report["summary_by_model_and_configuration"]:
        lines.append(
            table_row(
                [
                    row["model"],
                    row["strategy"],
                    _fmt_bool(row["use_constraints"]),
                    _fmt_bool(row["use_global_constraints"]),
                    row["constraint_backend"],
                    row["output_format"],
                    _fmt_bool(row["add_column"]),
                    row["jobs"],
                ]
                + [_fmt_float(row.get("mean_method_accuracies", {}).get(method)) for method in extractors]
                + [
                    _fmt_float(row["mean_accuracy"]),
                    _fmt_float(row["error_rate"]),
                    f"{row['successful_examples']}/{row['examples']}",
                    _format_seconds(row["avg_runtime_seconds"]),
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Individual Runs",
            "",
            table_row(
                ["Model", "Config Key", "Config Label", "Strategy", "Constraints", "Global", "Backend", "Format", "vLLM", "Repeat"]
                + extractor_headers
                + [
                    "Mean Extractor Accuracy",
                    "Examples",
                    "Runtime",
                    "Reused",
                    "Session",
                    "Restart",
                    "Status",
                    "Error",
                    "Run Dir",
                ]
            ),
            table_row(["---"] * 8 + ["---:"] + ["---:"] * len(extractors) + ["---:", "---:", "---:"] + ["---"] * 6),
        ]
    )
    for row in report["jobs"]:
        lines.append(
            table_row(
                [
                    row["model"],
                    row.get("config_key") or "",
                    _markdown_cell(row.get("config_label") or ""),
                    row["strategy"],
                    _fmt_bool(row["use_constraints"]),
                    _fmt_bool(row["use_global_constraints"]),
                    row["constraint_backend"],
                    row["output_format"],
                    row["expected_runtime"],
                    row["repeat"],
                ]
                + [_fmt_float(row.get("method_accuracies", {}).get(method)) for method in extractors]
                + [
                    _fmt_float(row["average_extractor_accuracy"] if row["average_extractor_accuracy"] is not None else row["accuracy"]),
                    row["examples"],
                    _format_seconds(row["runtime_seconds"]),
                    row["model_reused"],
                    row["session_id"] or "",
                    _markdown_cell(row["restart_reason"] or ""),
                    row["status"],
                    _markdown_cell(row["error"] or ""),
                    row["run_dir"] or "",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def load_comparison_run(run_dir: Path) -> ComparisonRun:
    manifest_path = run_dir / "run_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(run_dir / "results.jsonl")
    config_key = str(manifest.get("config_key") or manifest.get("config_slug") or run_dir.name)
    config_label = str(manifest.get("config_label") or manifest.get("config_slug") or run_dir.name)
    return ComparisonRun(config_key=config_key, config_label=config_label, run_dir=run_dir, rows=rows)


def _comparison_example_id(row: dict[str, Any]) -> str:
    return str(row.get("example_id") or row.get("request_id") or row.get("id") or "")


def _comparison_is_correct(row: dict[str, Any] | None) -> bool | None:
    if row is None:
        return None
    if "is_correct" in row:
        return bool(row.get("is_correct"))
    comparison = row.get("comparison", {}) or {}
    if "is_correct" in comparison:
        return bool(comparison.get("is_correct"))
    return float(row.get("execution_accuracy", 0.0)) == 1.0


def _comparison_state(row: dict[str, Any] | None) -> str:
    if row is None:
        return "missing"
    return "correct" if _comparison_is_correct(row) else "incorrect"


def _comparison_row_payload(run: ComparisonRun, row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "config_key": run.config_key,
            "config_label": run.config_label,
            "state": "missing",
            "is_correct": None,
            "execution_accuracy": None,
            "generated_answers": None,
            "answer_found_in_final": None,
            "terminated_properly": None,
            "execution_error": None,
        }
    comparison = row.get("comparison", {}) or {}
    metrics = row.get("execution_metrics", {}) or {}
    return {
        "config_key": run.config_key,
        "config_label": run.config_label,
        "state": _comparison_state(row),
        "is_correct": _comparison_is_correct(row),
        "execution_accuracy": row.get("execution_accuracy"),
        "generated_answers": row.get("generated_answers"),
        "answer_found_in_final": comparison.get("answer_found_in_final", metrics.get("answer_found_in_final")),
        "terminated_properly": comparison.get("terminated_properly", metrics.get("terminated_properly")),
        "execution_error": comparison.get("execution_error", metrics.get("execution_error")),
    }


def build_pairwise_comparison_report(left: ComparisonRun, right: ComparisonRun) -> dict[str, Any]:
    left_rows = {_comparison_example_id(row): row for row in left.rows}
    right_rows = {_comparison_example_id(row): row for row in right.rows}
    example_ids = sorted({example_id for example_id in left_rows if example_id} | {example_id for example_id in right_rows if example_id})

    summary = {
        "total_examples": len(example_ids),
        "shared_examples": 0,
        "left_only_examples": 0,
        "right_only_examples": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "left_only_correct": 0,
        "right_only_correct": 0,
        "left_missing": 0,
        "right_missing": 0,
        "both_missing": 0,
    }
    rows = []
    for example_id in example_ids:
        left_row = left_rows.get(example_id)
        right_row = right_rows.get(example_id)
        left_state = _comparison_state(left_row)
        right_state = _comparison_state(right_row)
        if left_state != "missing" and right_state != "missing":
            summary["shared_examples"] += 1
        elif left_state == "missing" and right_state != "missing":
            summary["left_missing"] += 1
            summary["right_only_examples"] += 1
        elif right_state == "missing" and left_state != "missing":
            summary["right_missing"] += 1
            summary["left_only_examples"] += 1
        else:
            summary["both_missing"] += 1

        if left_state == "correct" and right_state == "correct":
            summary["both_correct"] += 1
            outcome = "both_correct"
        elif left_state == "incorrect" and right_state == "incorrect":
            summary["both_wrong"] += 1
            outcome = "both_wrong"
        elif left_state == "correct" and right_state == "incorrect":
            summary["left_only_correct"] += 1
            outcome = "left_only_correct"
        elif left_state == "incorrect" and right_state == "correct":
            summary["right_only_correct"] += 1
            outcome = "right_only_correct"
        elif left_state == "missing" and right_state == "missing":
            outcome = "both_missing"
        elif left_state == "missing":
            outcome = "left_missing"
        else:
            outcome = "right_missing"

        rows.append(
            {
                "example_id": example_id,
                "question": (left_row or right_row or {}).get("question"),
                "ground_truth_answers": (left_row or right_row or {}).get("ground_truth_answers"),
                "outcome": outcome,
                "left": _comparison_row_payload(left, left_row),
                "right": _comparison_row_payload(right, right_row),
            }
        )

    summary["left_wins"] = summary["left_only_correct"]
    summary["right_wins"] = summary["right_only_correct"]

    return {
        "left": {"config_key": left.config_key, "config_label": left.config_label, "run_dir": str(left.run_dir)},
        "right": {"config_key": right.config_key, "config_label": right.config_label, "run_dir": str(right.run_dir)},
        "summary": summary,
        "rows": rows,
    }


def generate_comparison_artifacts(run_dirs: list[Path], output_dir: Path, baseline_config_key: str | None = None) -> dict[str, Any]:
    runs = [load_comparison_run(run_dir) for run_dir in run_dirs]
    if baseline_config_key:
        runs.sort(key=lambda run: 0 if run.config_key == baseline_config_key else 1)

    pairwise_reports = [build_pairwise_comparison_report(left, right) for left, right in itertools.combinations(runs, 2)]
    report = {
        "baseline_config_key": baseline_config_key,
        "runs": [{"config_key": run.config_key, "config_label": run.config_label, "run_dir": str(run.run_dir)} for run in runs],
        "pairwise_count": len(pairwise_reports),
        "pairwise_comparisons": pairwise_reports,
    }
    write_comparison_artifacts(report, output_dir)
    return report


def write_comparison_artifacts(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "comparison_report.md").write_text(render_comparison_report(report), encoding="utf-8")

    csv_path = output_dir / "comparison_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "pair_index",
                "example_id",
                "question",
                "outcome",
                "left_config_key",
                "left_config_label",
                "left_state",
                "left_is_correct",
                "left_execution_accuracy",
                "left_generated_answers",
                "right_config_key",
                "right_config_label",
                "right_state",
                "right_is_correct",
                "right_execution_accuracy",
                "right_generated_answers",
            ]
        )
        for pair_index, pair in enumerate(report.get("pairwise_comparisons", []), start=1):
            for row in pair.get("rows", []):
                left = row.get("left", {})
                right = row.get("right", {})
                writer.writerow(
                    [
                        pair_index,
                        row.get("example_id"),
                        row.get("question"),
                        row.get("outcome"),
                        left.get("config_key"),
                        left.get("config_label"),
                        left.get("state"),
                        left.get("is_correct"),
                        left.get("execution_accuracy"),
                        json.dumps(left.get("generated_answers")),
                        right.get("config_key"),
                        right.get("config_label"),
                        right.get("state"),
                        right.get("is_correct"),
                        right.get("execution_accuracy"),
                        json.dumps(right.get("generated_answers")),
                    ]
                )


def render_comparison_report(report: dict[str, Any]) -> str:
    def table_row(cells: list[Any]) -> str:
        return "| " + " | ".join(str(cell) for cell in cells) + " |"

    lines = ["# Comparison Report", ""]
    if report.get("baseline_config_key"):
        lines.extend([f"Baseline config key: `{report['baseline_config_key']}`", ""])

    lines.extend(["## Pairwise Summaries", ""])
    for index, pair in enumerate(report.get("pairwise_comparisons", []), start=1):
        left = pair["left"]
        right = pair["right"]
        summary = pair["summary"]
        lines.extend(
            [
                f"### Pair {index}: {left['config_label']} vs {right['config_label']}",
                "",
                table_row(["Metric", "Count"]),
                table_row(["---", "---:"]),
                table_row(["Total examples", summary["total_examples"]]),
                table_row(["Shared examples", summary["shared_examples"]]),
                table_row(["Left wins", summary["left_wins"]]),
                table_row(["Right wins", summary["right_wins"]]),
                table_row(["Both correct", summary["both_correct"]]),
                table_row(["Both wrong", summary["both_wrong"]]),
                table_row(["Left missing", summary["left_missing"]]),
                table_row(["Right missing", summary["right_missing"]]),
                table_row(["Both missing", summary["both_missing"]]),
                "",
            ]
        )
        disagreements = [row for row in pair.get("rows", []) if row.get("outcome") not in {"both_correct", "both_wrong"}]
        if disagreements:
            lines.extend(["Disagreements:", "", table_row(["Example", "Outcome", "Question"]), table_row(["---", "---", "---"])])
            for row in disagreements[:20]:
                lines.append(table_row([row.get("example_id"), row.get("outcome"), _markdown_cell(row.get("question") or "")]))
            if len(disagreements) > 20:
                lines.append(f"_... {len(disagreements) - 20} more disagreements omitted._")
            lines.append("")
    return "\n".join(lines) + "\n"


def _load_resume_spec(spec_path: Path) -> ExperimentSpec:
    raw = _read_yaml_raw(spec_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Frozen experiment specification {spec_path} must contain a mapping.")
    raw = copy.deepcopy(raw)
    for key in ("base_config", "output_root", "python_executable"):
        value = raw.get(key)
        if value is not None and not Path(value).is_absolute():
            candidate = PROJECT_ROOT / value
            if key == "base_config" and not candidate.exists():
                raise FileNotFoundError(f"Frozen base config is missing: {candidate}")
            raw[key] = str(candidate)
    temporary_spec = spec_path.with_name(f".{spec_path.name}.resume.yaml")
    try:
        _write_yaml(temporary_spec, raw)
        return load_experiment_spec(temporary_spec)
    finally:
        temporary_spec.unlink(missing_ok=True)


def _load_resume_results(experiment_dir: Path, jobs: list[ExperimentJob]) -> tuple[list[JobResult], set[str]]:
    job_by_id = {_job_identifier(job): job for job in jobs}
    results: list[JobResult] = []
    terminal_ids: set[str] = set()
    for status_path in sorted((experiment_dir / "sessions").glob("*/status.jsonl")):
        session_id = status_path.parent.name
        stdout_log = experiment_dir / "job_logs" / f"{session_id}.stdout.log"
        stderr_log = experiment_dir / "job_logs" / f"{session_id}.stderr.log"
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            job_id = event.get("job_id")
            if job_id not in job_by_id or event.get("event") not in {"completed", "failed"}:
                continue
            if job_id in terminal_ids:
                continue
            terminal_ids.add(job_id)
            results.append(_result_from_session_event(job_by_id[job_id], event, stdout_log, stderr_log, session_id))
    return results, terminal_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run configured LEAP experiments.")
    parser.add_argument(
        "spec",
        nargs="?",
        type=Path,
        default=DEFAULT_SPEC_PATH,
        help=f"Path to an experiments YAML file. Defaults to {DEFAULT_SPEC_PATH}.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="EXPERIMENT_DIR",
        help="Resume an existing experiment directory without regenerating its job configs.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Defaults to INFO.")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    resumed = args.resume is not None
    if resumed:
        experiment_dir = args.resume.resolve()
        spec_path = experiment_dir / "experiment_spec.yaml"
        if not spec_path.exists():
            raise FileNotFoundError(f"Frozen experiment specification not found: {spec_path}")
        spec = _load_resume_spec(spec_path)
        experiment_id = experiment_dir.name
    else:
        spec = load_experiment_spec(args.spec)
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = spec.output_root / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        _write_yaml(experiment_dir / "experiment_spec.yaml", _read_yaml_raw(args.spec))

    validate_models(spec)
    validate_python_executable(spec.python_executable)
    jobs = create_jobs(spec, experiment_dir, write_configs=False) if resumed else create_jobs(spec, experiment_dir)
    restored_results, terminal_ids = _load_resume_results(experiment_dir, jobs) if resumed else ([], set())
    pending_jobs = [job for job in jobs if _job_identifier(job) not in terminal_ids]

    LOGGER.info("Experiment %s%s", experiment_id, " (resumed)" if resumed else "")
    LOGGER.info("Output directory: %s", experiment_dir)
    LOGGER.info("Python executable: %s", spec.python_executable)
    LOGGER.info("Prepared %s jobs; %s already terminal, %s pending", len(jobs), len(restored_results), len(pending_jobs))
    for note in describe_matrix_adjustments(spec.matrix):
        LOGGER.info("Matrix adjustment: %s", note)

    job_results: list[JobResult] = list(restored_results)
    project_root = PROJECT_ROOT
    progress = tqdm(total=len(jobs), desc="Experiments", unit="job", initial=len(restored_results))

    def record_result(result: JobResult) -> None:
        job_results.append(result)
        progress.update(1)
        progress.set_postfix_str(f"{result.strategy} r{result.repeat}", refresh=False)
        try:
            report = (
                build_report(spec, experiment_id, experiment_dir, job_results, resumed=True)
                if resumed
                else build_report(spec, experiment_id, experiment_dir, job_results)
            )
            write_reports(report, experiment_dir)
        except Exception:
            LOGGER.exception("Could not update experiment reports; continuing with the next job")
        if result.status == "ok":
            LOGGER.info(
                "Finished job: model=%s strategy=%s repeat=%s examples=%s accuracy=%s runtime=%s reused=%s",
                result.model,
                result.strategy,
                result.repeat,
                result.examples,
                _fmt_float(result.accuracy),
                _format_seconds(result.runtime_seconds),
                getattr(result, "model_reused", False),
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
        if spec.reuse_models:
            sessions = group_jobs(pending_jobs)
            LOGGER.info("Prepared %s persistent model/runtime sessions", len(sessions))
            for session in sessions:
                LOGGER.info(
                    "Starting persistent session: model=%s runtime=%s jobs=%s",
                    session.model,
                    session.expected_runtime,
                    len(session.jobs),
                )
                run_session_group(
                    session,
                    experiment_dir=experiment_dir,
                    project_root=project_root,
                    python_executable=spec.python_executable,
                    on_result=record_result,
                )
        else:
            for job in pending_jobs:
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
                record_result(result)
    finally:
        progress.close()

    try:
        report = (
            build_report(spec, experiment_id, experiment_dir, job_results, resumed=True)
            if resumed
            else build_report(spec, experiment_id, experiment_dir, job_results)
        )
        write_reports(report, experiment_dir)
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
        methods = sorted({method for result in ok_results for method in result.method_accuracies})
        row.update(
            {
                "jobs": len(group_results),
                "mean_accuracy": _mean(accuracies),
                "mean_method_accuracies": {
                    method: _mean(result.method_accuracies[method] for result in ok_results if method in result.method_accuracies)
                    for method in methods
                },
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
                "add_column_requested_candidate_count": sum(result.add_column_requested_candidate_count for result in ok_results),
                "add_column_failed_candidate_count": sum(result.add_column_failed_candidate_count for result in ok_results),
                "add_column_affected_request_count": sum(result.add_column_affected_request_count for result in ok_results),
                "add_column_affected_sampling_step_count": sum(result.add_column_affected_sampling_step_count for result in ok_results),
                "add_column_failure_code_counts": _sum_failure_code_counts(result.add_column_failure_code_counts for result in ok_results),
            }
        )
        rows.append(row)
    return rows


def _sum_failure_code_counts(counts_by_result) -> dict[str, int]:
    totals: dict[str, int] = {}
    for counts in counts_by_result:
        for code, count in counts.items():
            totals[code] = totals.get(code, 0) + count
    return totals


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
    force_zero_temperature: bool,
    add_column: bool,
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
            "zero-temp" if force_zero_temperature else "standard-temp",
            "add-column-on" if add_column else "add-column-off",
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
        force_zero_temperature=job.force_zero_temperature,
        add_column=job.add_column,
        repeat=job.repeat,
    )


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")


def _enabled_actions_with_add_column(enabled_actions: list[str], add_column: bool) -> list[str]:
    actions = [action for action in enabled_actions if action != "add_column"]
    if add_column:
        try:
            actions.insert(actions.index("end"), "add_column")
        except ValueError:
            actions.append("add_column")
    return actions


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


def _bool_list(raw: dict[str, Any], key: str, default: list[bool] | None = None) -> list[bool]:
    values = raw.get(key, default)
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
    """Use the interpreter that is already running the command (for example, `uv run`)."""
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
    raise SystemExit(main())
