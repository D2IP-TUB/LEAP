import os
import sys
from pathlib import Path

import pytest
import yaml

from leap.vllm_runtime import (
    BOOTSTRAPPED_ENV_VAR,
    LEGACY_BACKEND,
    MODERN_BACKEND,
    RUNTIMES,
    ensure_vllm_runtime,
    load_constraint_backend,
    resolve_experiment_base_config,
    runtime_for_config,
    runtime_for_experiment,
    runtime_for_generation,
    supports_legacy_state_machine,
    version_matches,
)


def _write_yaml(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_missing_backend_selects_xgrammar(tmp_path):
    config = tmp_path / "current.yaml"
    _write_yaml(config, {"generation": {"use_constraints": True}})

    assert load_constraint_backend(config) == MODERN_BACKEND


def test_explicit_xgrammar_selects_modern(tmp_path):
    config = tmp_path / "current.yaml"
    _write_yaml(config, {"generation": {"constraint_backend": MODERN_BACKEND}})

    assert load_constraint_backend(config) == MODERN_BACKEND


def test_unconstrained_legacy_backend_selects_modern_runtime(tmp_path):
    config = tmp_path / "unconstrained.yaml"
    _write_yaml(
        config,
        {
            "model": {"id": "meta-llama/Llama-4-Scout-17B-16E-Instruct"},
            "generation": {"use_constraints": False, "constraint_backend": LEGACY_BACKEND},
        },
    )

    assert runtime_for_config(config).name == "modern"


def test_constrained_qwen2_5_legacy_backend_selects_legacy_runtime(tmp_path):
    config = tmp_path / "constrained.yaml"
    _write_yaml(
        config,
        {
            "model": {"id": "Qwen/Qwen2.5-7B-Instruct"},
            "generation": {"use_constraints": True, "constraint_backend": LEGACY_BACKEND},
        },
    )

    assert runtime_for_config(config).name == "legacy"


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-235B-A22B", "Qwen/Qwen3.8-27B"])
def test_qwen3_models_select_the_modern_runtime(tmp_path, model_id):
    config = tmp_path / "qwen3.yaml"
    _write_yaml(
        config,
        {
            "model": {"id": model_id},
            "generation": {"use_constraints": True, "constraint_backend": MODERN_BACKEND},
        },
    )

    assert runtime_for_config(config).name == "modern"


def test_constrained_non_qwen2_5_legacy_backend_is_rejected(tmp_path):
    config = tmp_path / "constrained.yaml"
    _write_yaml(
        config,
        {
            "model": {"id": "meta-llama/Llama-4-Scout-17B-16E-Instruct"},
            "generation": {"use_constraints": True, "constraint_backend": LEGACY_BACKEND},
        },
    )

    with pytest.raises(ValueError, match="Only Qwen2.5 models"):
        runtime_for_config(config)


def test_legacy_backend_is_limited_to_qwen2_5_models():
    assert supports_legacy_state_machine("Qwen/Qwen2.5-32B-Instruct")
    assert not supports_legacy_state_machine("Qwen/Qwen3-235B-A22B")
    assert not supports_legacy_state_machine("Qwen/Qwen3.8-27B")


@pytest.mark.parametrize("backend", [LEGACY_BACKEND, MODERN_BACKEND])
def test_unconstrained_generation_selects_modern_runtime(backend):
    assert runtime_for_generation(use_constraints=False, constraint_backend=backend).name == "modern"


def test_removed_output_format_is_rejected():
    with pytest.raises(ValueError, match="Unsupported output format"):
        runtime_for_generation(
            use_constraints=True,
            constraint_backend=MODERN_BACKEND,
            output_format="mc" + "p",
        )


def test_configured_legacy_backend_forces_function_format_before_runtime_selection(tmp_path):
    config = tmp_path / "legacy-json.yaml"
    _write_yaml(
        config,
        {
            "model": {"id": "Qwen/Qwen2.5-7B-Instruct"},
            "generation": {"use_constraints": True, "constraint_backend": LEGACY_BACKEND, "output_format": "json"},
        },
    )
    assert runtime_for_config(config).name == "legacy"
    assert (
        runtime_for_generation(
            use_constraints=True,
            constraint_backend=LEGACY_BACKEND,
            output_format="json",
        ).name
        == "legacy"
    )


def test_experiment_resolves_base_config_from_project_root(tmp_path):
    config = tmp_path / "configs" / "default.yaml"
    spec = tmp_path / "specs" / "experiment.yaml"
    _write_yaml(config, {"model": {"id": "model/a"}, "generation": {"constraint_backend": MODERN_BACKEND}})
    _write_yaml(spec, {"base_config": "configs/default.yaml"})

    assert resolve_experiment_base_config(spec, tmp_path) == config
    assert runtime_for_experiment(spec, tmp_path).name == "modern"


@pytest.mark.parametrize(
    ("backend", "installed", "matches"),
    [
        (LEGACY_BACKEND, "0.10.0", True),
        (LEGACY_BACKEND, "0.10.1", False),
        (MODERN_BACKEND, "0.28.0", True),
        (MODERN_BACKEND, "0.28.4", True),
        (MODERN_BACKEND, "0.29.0", False),
    ],
)
def test_version_pairing(backend, installed, matches):
    assert version_matches(RUNTIMES[backend], installed) is matches


def test_bootstrap_constructs_isolated_uv_command(tmp_path, monkeypatch):
    captured = {}

    def fake_exec(executable, command, environment):
        captured.update(executable=executable, command=command, environment=environment)
        raise SystemExit(17)

    monkeypatch.setattr("leap.vllm_runtime.installed_vllm_version", lambda: "0.12.0")
    monkeypatch.setattr("leap.vllm_runtime.shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr("leap.vllm_runtime.os.execvpe", fake_exec)
    monkeypatch.delenv(BOOTSTRAPPED_ENV_VAR, raising=False)

    with pytest.raises(SystemExit, match="17"):
        ensure_vllm_runtime(RUNTIMES[LEGACY_BACKEND], argv=["main.py"], project_root=tmp_path)

    assert captured["command"][-2:] == ["python", "main.py"]
    assert "vllm-legacy" in captured["command"]
    assert captured["environment"]["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / ".venv-vllm-legacy")
    assert captured["environment"]["VLLM_USE_V1"] == "0"


def test_bootstrap_rejects_mismatch_after_reexecution(tmp_path, monkeypatch):
    monkeypatch.setenv(BOOTSTRAPPED_ENV_VAR, "1")
    monkeypatch.setattr("leap.vllm_runtime.installed_vllm_version", lambda: "0.12.0")

    with pytest.raises(RuntimeError, match="contains vLLM 0.12.0"):
        ensure_vllm_runtime(RUNTIMES[LEGACY_BACKEND], argv=["main.py"], project_root=tmp_path)


@pytest.mark.parametrize("backend", [LEGACY_BACKEND, MODERN_BACKEND])
def test_cuda_paths_are_not_configured_before_switching_environments(tmp_path, monkeypatch, backend):
    from unittest.mock import Mock

    configure_cuda = Mock()
    monkeypatch.setattr("leap.vllm_runtime._configure_environment_cuda", configure_cuda)
    monkeypatch.setattr("leap.vllm_runtime.installed_vllm_version", lambda: None)
    monkeypatch.setattr("leap.vllm_runtime.shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.delenv(BOOTSTRAPPED_ENV_VAR, raising=False)

    def fake_exec(*args):
        raise SystemExit(17)

    monkeypatch.setattr("leap.vllm_runtime.os.execvpe", fake_exec)
    with pytest.raises(SystemExit, match="17"):
        ensure_vllm_runtime(RUNTIMES[backend], argv=["main.py"], project_root=tmp_path)
    configure_cuda.assert_not_called()


@pytest.mark.parametrize("backend, installed", [(LEGACY_BACKEND, "0.10.0"), (MODERN_BACKEND, "0.28.0")])
def test_cuda_paths_are_configured_in_selected_environment(tmp_path, monkeypatch, backend, installed):
    from unittest.mock import Mock

    configure_cuda = Mock()
    runtime = RUNTIMES[backend]
    monkeypatch.setattr("leap.vllm_runtime._configure_environment_cuda", configure_cuda)
    monkeypatch.setattr("leap.vllm_runtime.installed_vllm_version", lambda: installed)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / runtime.environment))

    ensure_vllm_runtime(runtime, argv=["main.py"], project_root=tmp_path)
    configure_cuda.assert_called_once_with()


def test_runtime_environment_names_are_distinct():
    assert RUNTIMES[LEGACY_BACKEND].environment != RUNTIMES[MODERN_BACKEND].environment
    assert os.path.basename(RUNTIMES[LEGACY_BACKEND].environment) == ".venv-vllm-legacy"
