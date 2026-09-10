import json
from dataclasses import replace
from types import SimpleNamespace

from leap.config.loader import (
    AppConfig,
    DatasetConfig,
    GenerationConfig,
    HardwareConfig,
    LoggingConfig,
    ModelConfig,
    RunConfig,
    TokenizerConfig,
)
from main import (
    RunOutputPaths,
    RuntimeContext,
    ServerExecutionError,
    build_config_key,
    build_config_label,
    run_experiment_session,
    write_run_manifest,
)


def _app_config(tmp_path):
    tokenizer = TokenizerConfig(False, 1, 2, 3, 4, 5, 6, [6], {})
    return AppConfig(
        model=ModelConfig(
            id="model/test",
            instruct=True,
            log_dir="test",
            hardware=HardwareConfig(1, 1, [0]),
            tokenizer_config=tokenizer,
            results_file=str(tmp_path / "results.jsonl"),
        ),
        dataset=DatasetConfig(loader="json", data_files="test.json"),
        run=RunConfig(max_examples=1),
        generation=GenerationConfig(
            use_constraints=True,
            use_global_constraints=False,
            constraint_backend="xgrammar",
        ),
        logging=LoggingConfig(True, str(tmp_path / "logs"), False, False, "readable", 1000),
        extractors=("direct_query",),
    )


def test_config_label_omits_inapplicable_dimensions(tmp_path):
    app_config = _app_config(tmp_path)

    assert build_config_label(app_config) == "model/test | cot | constrained | xgrammar | function"

    unconstrained = replace(
        app_config,
        generation=replace(
            app_config.generation,
            use_constraints=False,
            use_global_constraints=False,
            constraint_backend="xgrammar",
            output_format="json",
            force_zero_temperature=True,
        ),
    )
    assert build_config_label(unconstrained) == "model/test | cot | unconstrained | json | zero_temp"

    direct_query = replace(
        app_config,
        generation=replace(
            app_config.generation,
            strategy="direct_query",
            use_constraints=False,
            constraint_backend="xgrammar",
            output_format="function",
            force_zero_temperature=True,
        ),
    )
    assert build_config_label(direct_query) == "model/test | direct_query | zero_temp"


def test_run_manifest_records_vllm_runtime(tmp_path):
    tokenizer = TokenizerConfig(False, 1, 2, 3, 4, 5, 6, [6], {})
    app_config = AppConfig(
        model=ModelConfig(
            id="model/test",
            instruct=True,
            log_dir="test",
            hardware=HardwareConfig(1, 1, [0]),
            tokenizer_config=tokenizer,
            results_file=str(tmp_path / "results.jsonl"),
        ),
        dataset=DatasetConfig(loader="json", data_files="test.json"),
        run=RunConfig(max_examples=1),
        generation=GenerationConfig(
            use_constraints=True,
            use_global_constraints=False,
            constraint_backend="xgrammar",
        ),
        logging=LoggingConfig(True, str(tmp_path / "logs"), False, False, "readable", 1000),
    )
    paths = RunOutputPaths(
        run_dir=tmp_path,
        results_file=tmp_path / "results.jsonl",
        table_log_dir=tmp_path / "logs",
        accuracy_file=tmp_path / "accuracy.json",
        extractor_accuracy_file=tmp_path / "extractor_accuracy.json",
        manifest_file=tmp_path / "run_config.json",
        config_slug="test",
        config_key=build_config_key(app_config),
        config_label=build_config_label(app_config),
        timestamp="now",
    )

    write_run_manifest(app_config, paths, tmp_path / "config.yaml")

    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    assert manifest["config_key"] == paths.config_key
    assert manifest["config_label"] == paths.config_label
    assert manifest["vllm_runtime"]["runtime"] == "modern"
    assert manifest["vllm_runtime"]["constraint_backend"] == "xgrammar"
    assert manifest["vllm_runtime"]["vllm_version"] is not None
    assert manifest["vllm_runtime"]["engine"] == "V1"


def test_unconstrained_legacy_backend_manifest_records_modern_runtime(tmp_path):
    tokenizer = TokenizerConfig(False, 1, 2, 3, 4, 5, 6, [6], {})
    app_config = AppConfig(
        model=ModelConfig(
            id="model/test",
            instruct=True,
            log_dir="test",
            hardware=HardwareConfig(1, 1, [0]),
            tokenizer_config=tokenizer,
            results_file=str(tmp_path / "results.jsonl"),
        ),
        dataset=DatasetConfig(loader="json", data_files="test.json"),
        run=RunConfig(max_examples=1),
        generation=GenerationConfig(
            use_constraints=False,
            use_global_constraints=False,
            constraint_backend="legacy_state_machine",
        ),
        logging=LoggingConfig(True, str(tmp_path / "logs"), False, False, "readable", 1000),
    )
    paths = RunOutputPaths(
        run_dir=tmp_path,
        results_file=tmp_path / "results.jsonl",
        table_log_dir=tmp_path / "logs",
        accuracy_file=tmp_path / "accuracy.json",
        extractor_accuracy_file=tmp_path / "extractor_accuracy.json",
        manifest_file=tmp_path / "run_config.json",
        config_slug="test",
        config_key=build_config_key(app_config),
        config_label=build_config_label(app_config),
        timestamp="now",
    )

    write_run_manifest(app_config, paths, tmp_path / "config.yaml")

    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    assert manifest["config_key"] == paths.config_key
    assert manifest["config_label"] == paths.config_label
    assert manifest["vllm_runtime"]["runtime"] == "modern"
    assert manifest["vllm_runtime"]["constraint_backend"] == "legacy_state_machine"
    assert manifest["vllm_runtime"]["use_constraints"] is False
    assert manifest["vllm_runtime"]["engine"] == "V1"


def test_experiment_session_reuses_one_server_for_compatible_configs(tmp_path, monkeypatch):
    app_config = _app_config(tmp_path)
    runtime = RuntimeContext(config=app_config, prompt_builder=SimpleNamespace(), tokenizer=SimpleNamespace(), dataset=[])
    status_path = tmp_path / "status.jsonl"
    config_paths = [tmp_path / "one.yaml", tmp_path / "two.yaml"]
    manifest_path = tmp_path / "session.json"
    manifest_path.write_text(
        json.dumps(
            {
                "session_id": "session",
                "status_path": str(status_path),
                "jobs": [
                    {"job_id": f"job-{index}", "config_path": str(path), "results_root": str(tmp_path / "results")}
                    for index, path in enumerate(config_paths)
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeServer:
        def __init__(self):
            self.shutdown_calls = 0

        def start_workers(self, timeout):
            return True

        def workers_healthy(self):
            return True

        def shutdown(self, **kwargs):
            self.shutdown_calls += 1

        def finish_run(self):
            pass

    server = FakeServer()
    create_calls = []
    execute_calls = []

    def fake_execute_config_run(**kwargs):
        execute_calls.append(kwargs["config_path"])
        paths = kwargs["run_paths"]
        return paths, {"examples": 1}

    monkeypatch.setattr("main.build_runtime", lambda path: runtime)
    monkeypatch.setattr("main.load_runtime_config", lambda path, tokenizer: app_config)
    monkeypatch.setattr("main.create_server", lambda config, tokenizer: create_calls.append(config) or server)
    monkeypatch.setattr("main.execute_config_run", fake_execute_config_run)

    assert run_experiment_session(manifest_path) == 0
    events = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines()]
    completed = [event for event in events if event["event"] == "completed"]

    assert len(create_calls) == 1
    assert execute_calls == config_paths
    assert [event["model_reused"] for event in completed] == [False, True]
    assert server.shutdown_calls == 1


def test_experiment_session_restarts_server_after_batch_failure(tmp_path, monkeypatch):
    app_config = _app_config(tmp_path)
    runtime = RuntimeContext(config=app_config, prompt_builder=SimpleNamespace(), tokenizer=SimpleNamespace(), dataset=[])
    status_path = tmp_path / "status.jsonl"
    manifest_path = tmp_path / "session.json"
    manifest_path.write_text(
        json.dumps(
            {
                "session_id": "session",
                "status_path": str(status_path),
                "jobs": [
                    {"job_id": f"job-{index}", "config_path": str(tmp_path / f"{index}.yaml"), "results_root": str(tmp_path)}
                    for index in range(2)
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeServer:
        def start_workers(self, timeout):
            return True

        def workers_healthy(self):
            return True

        def shutdown(self, **kwargs):
            pass

        def finish_run(self):
            pass

    servers = [FakeServer(), FakeServer()]
    execute_count = 0

    def fake_execute(**kwargs):
        nonlocal execute_count
        execute_count += 1
        if execute_count == 1:
            raise ServerExecutionError("engine failed")
        return kwargs["run_paths"], {"examples": 1}

    monkeypatch.setattr("main.build_runtime", lambda path: runtime)
    monkeypatch.setattr("main.load_runtime_config", lambda path, tokenizer: app_config)
    monkeypatch.setattr("main.create_server", lambda config, tokenizer: servers.pop(0))
    monkeypatch.setattr("main.execute_config_run", fake_execute)

    assert run_experiment_session(manifest_path) == 0
    events = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines()]
    terminal = [event for event in events if event["event"] in {"completed", "failed"}]

    assert [event["event"] for event in terminal] == ["failed", "completed"]
    assert terminal[1]["model_reused"] is False
    assert terminal[1]["restart_reason"].startswith("Server restart after job-0")
