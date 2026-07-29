"""Select and validate the isolated vLLM runtime for a LEAP run."""

from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

LEGACY_BACKEND = "legacy_state_machine"
MODERN_BACKEND = "xgrammar"
SUPPORTED_BACKENDS = {LEGACY_BACKEND, MODERN_BACKEND}
RUNTIME_ENV_VAR = "LEAP_VLLM_RUNTIME"
BOOTSTRAPPED_ENV_VAR = "LEAP_VLLM_BOOTSTRAPPED"


@dataclass(frozen=True)
class VLLMRuntime:
    name: str
    backend: str
    extra: str
    environment: str
    version_spec: str
    engine: str
    use_v1: str


RUNTIMES = {
    LEGACY_BACKEND: VLLMRuntime(
        name="legacy",
        backend=LEGACY_BACKEND,
        extra="vllm-legacy",
        environment=".venv-vllm-legacy",
        version_spec="==0.10.0",
        engine="V0",
        use_v1="0",
    ),
    MODERN_BACKEND: VLLMRuntime(
        name="modern",
        backend=MODERN_BACKEND,
        extra="vllm-modern",
        environment=".venv-vllm-modern",
        version_spec=">=0.12,<0.13",
        engine="V1",
        use_v1="1",
    ),
}


def load_constraint_backend(config_path: Path) -> str:
    """Read a backend without importing the application configuration stack."""
    config = _load_yaml(config_path)
    generation = config.get("generation") or {}
    if not isinstance(generation, dict):
        raise ValueError(f"{config_path}: 'generation' must be a mapping.")
    backend = generation.get("constraint_backend", LEGACY_BACKEND)
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"{config_path}: generation.constraint_backend must be one of {sorted(SUPPORTED_BACKENDS)}, got {backend!r}.")
    return backend


def load_generation_runtime_settings(config_path: Path) -> tuple[bool, str]:
    """Read the settings that determine whether legacy logits processing is active."""
    config = _load_yaml(config_path)
    generation = config.get("generation") or {}
    if not isinstance(generation, dict):
        raise ValueError(f"{config_path}: 'generation' must be a mapping.")
    use_constraints = generation.get("use_constraints", False)
    if not isinstance(use_constraints, bool):
        raise ValueError(f"{config_path}: generation.use_constraints must be a boolean.")
    return use_constraints, load_constraint_backend(config_path)


def resolve_experiment_base_config(spec_path: Path, project_root: Path | None = None) -> Path:
    """Resolve an experiment specification's base config like the runner does."""
    project_root = (project_root or Path.cwd()).resolve()
    spec_path = spec_path.resolve()
    spec = _load_yaml(spec_path)
    value = spec.get("base_config", "configs/default.yaml")
    path = Path(value)
    if path.is_absolute():
        return path
    relative_to_spec = spec_path.parent / path
    if relative_to_spec.exists():
        return relative_to_spec.resolve()
    return (project_root / path).resolve()


def runtime_for_generation(*, use_constraints: bool, constraint_backend: str) -> VLLMRuntime:
    """Select V0 only when the legacy logits processor will actually be used."""
    if constraint_backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported constraint backend: {constraint_backend!r}")
    backend = LEGACY_BACKEND if use_constraints and constraint_backend == LEGACY_BACKEND else MODERN_BACKEND
    return RUNTIMES[backend]


def runtime_for_config(config_path: Path) -> VLLMRuntime:
    use_constraints, constraint_backend = load_generation_runtime_settings(config_path)
    return runtime_for_generation(use_constraints=use_constraints, constraint_backend=constraint_backend)


def runtime_for_experiment(spec_path: Path, project_root: Path | None = None) -> VLLMRuntime:
    return runtime_for_config(resolve_experiment_base_config(spec_path, project_root))


def installed_vllm_version() -> str | None:
    try:
        return version("vllm")
    except PackageNotFoundError:
        return None


def version_matches(runtime: VLLMRuntime, installed: str | None) -> bool:
    parsed = _release_tuple(installed)
    if parsed is None:
        return False
    if runtime.name == "legacy":
        return parsed[:3] == (0, 10, 0)
    return parsed >= (0, 12, 0) and parsed < (0, 13, 0)


def ensure_vllm_runtime(runtime: VLLMRuntime, argv: list[str] | None = None, project_root: Path | None = None) -> None:
    """Re-execute the current script in the selected persistent uv environment."""
    project_root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    environment = (project_root / runtime.environment).resolve()
    installed = installed_vllm_version()
    in_selected_environment = Path(sys.prefix).resolve() == environment

    os.environ["VLLM_USE_V1"] = runtime.use_v1
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    os.environ[RUNTIME_ENV_VAR] = runtime.name

    if in_selected_environment and version_matches(runtime, installed):
        return

    if os.environ.get(BOOTSTRAPPED_ENV_VAR) == "1":
        raise RuntimeError(
            f"Selected the {runtime.name} runtime ({runtime.version_spec}), but environment "
            f"{environment} contains vLLM {installed or 'not installed'}. "
            "Run 'uv lock' and retry, or remove that generated environment."
        )

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to select the vLLM runtime but was not found on PATH.")

    child_env = os.environ.copy()
    child_env["UV_PROJECT_ENVIRONMENT"] = str(environment)
    child_env[BOOTSTRAPPED_ENV_VAR] = "1"
    command = [
        uv,
        "run",
        "--project",
        str(project_root),
        "--locked",
        "--no-default-groups",
        "--extra",
        runtime.extra,
        "python",
        *(argv or sys.argv),
    ]
    os.execvpe(uv, command, child_env)


def validate_installed_runtime(*, use_constraints: bool, constraint_backend: str) -> VLLMRuntime:
    """Fail early when application configuration and installed vLLM disagree."""
    runtime = runtime_for_generation(use_constraints=use_constraints, constraint_backend=constraint_backend)
    installed = installed_vllm_version()
    if not version_matches(runtime, installed) or os.environ.get("VLLM_USE_V1") != runtime.use_v1:
        raise RuntimeError(
            f"use_constraints={use_constraints!r} with constraint_backend={constraint_backend!r} "
            f"requires vLLM {runtime.version_spec} "
            f"with engine {runtime.engine}, but found vLLM {installed or 'not installed'} and "
            f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1')!r}. Start the run through the documented uv command."
        )
    return runtime


def runtime_metadata(constraint_backend: str | None = None, *, use_constraints: bool | None = None) -> dict[str, Any]:
    """Return reproducibility metadata without requiring vLLM to be importable."""
    runtime = None
    if constraint_backend is not None and use_constraints is not None:
        runtime = runtime_for_generation(use_constraints=use_constraints, constraint_backend=constraint_backend)
    if runtime is None:
        runtime_name = os.environ.get(RUNTIME_ENV_VAR)
        runtime = next((candidate for candidate in RUNTIMES.values() if candidate.name == runtime_name), None)
    return {
        "runtime": runtime.name if runtime else os.environ.get(RUNTIME_ENV_VAR, "unknown"),
        "constraint_backend": constraint_backend,
        "use_constraints": use_constraints,
        "vllm_version": installed_vllm_version(),
        "vllm_version_spec": runtime.version_spec if runtime else None,
        "engine": runtime.engine if runtime else None,
    }


def runtime_asdict(runtime: VLLMRuntime) -> dict[str, str]:
    return asdict(runtime)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
    except OSError as exc:
        raise FileNotFoundError(f"Unable to read runtime configuration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def _release_tuple(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())
