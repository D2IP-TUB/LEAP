## Installation

We strongly recommend using [uv](https://docs.astral.sh/uv/) for environment and package management.

```bash
uv sync
source .venv/bin/activate
```

This creates the common project environment. The selected vLLM runtime is installed separately on first use as described below.

## vLLM Runtime Selection

LEAP keeps the incompatible legacy and current vLLM releases in separate, automatically managed environments. Unconstrained generation automatically uses the current V1 runtime. For constrained generation, choose the constraint backend in the generation section of your config:

```yaml
generation:
  # Current constrained generation: vLLM >=0.12,<0.13 with the V1 engine.
  constraint_backend: xgrammar

  # Historical benchmark reproduction: vLLM 0.10.0 with the V0 engine.
  # constraint_backend: legacy_state_machine
```

Run LEAP normally with `uv run main.py`, or run a benchmark matrix with `uv run scripts/run_experiments.py`. Before importing vLLM, the entrypoint reads the config and re-executes itself in one of these persistent environments:

- `.venv-vllm-modern` for unconstrained generation and `xgrammar`
- `.venv-vllm-legacy` only for active `legacy_state_machine` constraints

The first run for each runtime downloads and installs its vLLM and PyTorch stack, so it can take substantially longer than later runs. Unconstrained configs use the modern runtime regardless of `constraint_backend`. Constrained configs inherit their backend from the experiment `base_config`; older constrained configs without `generation.constraint_backend` select the legacy runtime for backward compatibility.

The backend/version pairing is strict. If LEAP reports a mismatch, update the lockfile with `uv lock` and retry. If an environment was interrupted or corrupted during installation, remove only the named generated environment from the error message and rerun the command. The `xgrammar` backend does not support the `add_column` action.

## Developer Setup

If you plan to contribute to the codebase, install the development dependencies and enable the pre-commit hooks:

```bash
uv sync --group dev
pre-commit install
```

This installs the linting/formatting tools defined in pyproject.toml and ensures they run automatically before each commit.
