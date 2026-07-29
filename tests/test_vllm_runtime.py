import os
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
    version_matches,
)


def _write_yaml(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_missing_backend_selects_legacy(tmp_path):
    config = tmp_path / "old.yaml"
    _write_yaml(config, {"generation": {"use_constraints": True}})

    assert load_constraint_backend(config) == LEGACY_BACKEND


def test_explicit_xgrammar_selects_modern(tmp_path):
    config = tmp_path / "current.yaml"
    _write_yaml(config, {"generation": {"constraint_backend": MODERN_BACKEND}})

    assert load_constraint_backend(config) == MODERN_BACKEND


def test_unconstrained_legacy_backend_selects_modern_runtime(tmp_path):
    config = tmp_path / "unconstrained.yaml"
    _write_yaml(
        config,
        {"generation": {"use_constraints": False, "constraint_backend": LEGACY_BACKEND}},
    )

    assert runtime_for_config(config).name == "modern"


def test_constrained_legacy_backend_selects_legacy_runtime(tmp_path):
    config = tmp_path / "constrained.yaml"
    _write_yaml(
        config,
        {"generation": {"use_constraints": True, "constraint_backend": LEGACY_BACKEND}},
    )

    assert runtime_for_config(config).name == "legacy"


@pytest.mark.parametrize("backend", [LEGACY_BACKEND, MODERN_BACKEND])
def test_unconstrained_generation_selects_modern_runtime(backend):
    assert runtime_for_generation(use_constraints=False, constraint_backend=backend).name == "modern"


def test_experiment_resolves_base_config_from_project_root(tmp_path):
    config = tmp_path / "configs" / "default.yaml"
    spec = tmp_path / "specs" / "experiment.yaml"
    _write_yaml(config, {"generation": {"constraint_backend": MODERN_BACKEND}})
    _write_yaml(spec, {"base_config": "configs/default.yaml"})

    assert resolve_experiment_base_config(spec, tmp_path) == config
    assert runtime_for_experiment(spec, tmp_path).name == "modern"


@pytest.mark.parametrize(
    ("backend", "installed", "matches"),
    [
        (LEGACY_BACKEND, "0.10.0", True),
        (LEGACY_BACKEND, "0.10.1", False),
        (MODERN_BACKEND, "0.12.0", True),
        (MODERN_BACKEND, "0.12.4", True),
        (MODERN_BACKEND, "0.13.0", False),
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


def test_runtime_environment_names_are_distinct():
    assert RUNTIMES[LEGACY_BACKEND].environment != RUNTIMES[MODERN_BACKEND].environment
    assert os.path.basename(RUNTIMES[LEGACY_BACKEND].environment) == ".venv-vllm-legacy"
