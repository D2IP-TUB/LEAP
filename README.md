## Installation

We strongly recommend using [uv](https://docs.astral.sh/uv/) for environment and package management.

```bash
uv sync --group dev --extra vllm-modern
```

This creates the common project environment with development dependencies. The selected vLLM runtime is installed separately on first use as described below.

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

## Experiment matrices

Run the example benchmark matrix with:

```bash
uv run scripts/run_experiments.py configs/experiments.example.yaml
```

The experiment spec uses explicit generation dimensions:

```yaml
matrix:
  strategies: [iterative, cot, direct_query]
  use_constraints: [false, true]
  use_global_constraints: [false, true]
  constraint_backends: [legacy_state_machine, xgrammar]
  output_formats: [function, json, mcp]
  force_zero_temperature: [false, true]
```

Only unique supported jobs are generated. Unconstrained runs use the modern xgrammar runtime, legacy constrained decoding supports function output only, and `direct_query` is emitted once per model, repeat, and temperature mode because action constraints do not affect it. Set `force_zero_temperature: true` to force every model request in that run to use temperature 0, including dynamic-plan, generate-args, and answer-extraction calls. The temperature-mode sweep roughly doubles the matrix size. Every job failure is recorded in the experiment report and the runner continues with the remaining jobs; the final command exits nonzero if any job failed. Invalid suite configuration and explicit user interruption still stop the runner.

Model reuse is enabled by default and can be controlled explicitly:

```yaml
reuse_models: true
```

The runner groups jobs by model and vLLM runtime. Strategy, constraint, output-format, and repeat changes within a compatible group reuse the loaded model; changing the model or switching between the modern V1 and legacy V0 runtimes starts a new model session. Set `reuse_models: false` to restore isolated one-process-per-job execution. Reports include the session, model-load count, reuse state, and restart reason for each job.

## JSON operation mode

Iterative and Chain-of-Table generation can use named JSON operation objects instead of the original `f_action(...)` protocol:

```yaml
generation:
  strategy: cot       # or iterative
  output_format: json
  use_constraints: true
```

With constraints enabled, JSON mode uses vLLM JSON Schema structured outputs on the modern V1 runtime. Selecting `constraint_backend: legacy_state_machine` automatically forces `output_format: function`; JSON operation mode therefore requires the modern `xgrammar` backend setting. With constraints disabled, the same JSON prompts and strict JSON parser are used without structured decoding when `constraint_backend` is modern. The default remains `output_format: function` for backward compatibility.

JSON operations use named fields, for example `{"action":"select_row","rows":["row 0"]}` and `{"action":"sort_by","column":"Year","order":"desc"}`. CoT remains two-phase: action selection emits only the `action` field, followed by a selected-action argument object.

## MCP operation mode

Set `generation.output_format: mcp` to have the model emit official JSON-RPC `tools/call` envelopes and execute transformations through a local MCP Python SDK server over stdio:

```yaml
generation:
  strategy: iterative  # or cot
  output_format: mcp
  constraint_backend: xgrammar
  enabled_actions: [select_row, select_column, group_by, sort_by, end]
```

The server is stateless with respect to table data. Each vLLM worker owns one persistent MCP client context whose serialized broker lazily starts the stdio server on the first MCP operation and reuses that session until worker shutdown. Function/JSON-only runs never launch the MCP subprocess. If the transport fails, the broker reconnects and retries the stateless call once.

LEAP injects the current table into every MCP tool invocation, and the server returns the complete new table state. It exposes one tool for each supported operation: `select_row`, `select_column`, `group_by`, `sort_by`, and `end`. `add_column` is intentionally unsupported in this prototype.

Iterative mode generates one complete MCP request per step. CoT remains two-phase: phase one emits a `tools/call`-shaped selection with an empty `arguments` object, and phase two emits the complete request that LEAP sends through the MCP client. The runtime, not the model, injects the table argument. Existing function and JSON modes are unchanged.

MCP mode uses the modern runtime. Selecting `legacy_state_machine` continues to force function mode for backward compatibility. Profiling reports total `mcp_table_transformation` time in the operation breakdown and reports `mcp_startup` and warm `mcp_call` diagnostics separately. See [`docs/mcp-architecture.md`](docs/mcp-architecture.md) for before/after architecture and information-flow diagrams.

Run the standalone server with:

```bash
uv run python -m leap.mcp.server
```

## Developer Setup

If you plan to contribute to the codebase, install the development dependencies and enable the pre-commit hooks:

```bash
uv sync --group dev --extra vllm-modern
uv run pre-commit install
```

This installs the linting/formatting tools defined in pyproject.toml and ensures they run automatically before each commit.
