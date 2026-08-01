import json

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
from main import RunOutputPaths, write_run_manifest


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
        timestamp="now",
    )

    write_run_manifest(app_config, paths, tmp_path / "config.yaml")

    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
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
        timestamp="now",
    )

    write_run_manifest(app_config, paths, tmp_path / "config.yaml")

    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    assert manifest["vllm_runtime"]["runtime"] == "modern"
    assert manifest["vllm_runtime"]["constraint_backend"] == "legacy_state_machine"
    assert manifest["vllm_runtime"]["use_constraints"] is False
    assert manifest["vllm_runtime"]["engine"] == "V1"
