# LEAP

LEAP extends Chain-of-Tables (COT) with constrained generation for LLM-based question answering over tables. It restricts model output during decoding to reduce malformed table operations and invalid arguments, using either a logit-masking state-machine backend or XGrammar2. The repository supports comparisons with unconstrained generation to evaluate how these restrictions affect end-to-end answer accuracy.

See the [project handover report](docs/project-handover.md) for the backend overview, experimental results, and future direction. The MCP-based implementation is maintained in a separate repository, [LEAP-MCP](https://github.com/D2IP-TUB/LEAP-MCP).

## Setup and first run

Use Python 3.11 or later and [uv](https://docs.astral.sh/uv/) for environment and package management:

```bash
uv sync --group dev --extra vllm-modern
```

This installs the modern vLLM runtime and development dependencies in `.venv`. LEAP manages the incompatible legacy runtime in a separate environment when needed.

Before running, review:

- [configs/default.yaml](configs/default.yaml) for the model, dataset, example limit, generation settings, and answer extractors.
- [configs/models.yaml](configs/models.yaml) for model-specific hardware and tokenizer settings. Choose a model and GPU allocation that fit your available hardware.

The checked-in default config selects Qwen2.5-32B with legacy constraints. To use modern constrained generation, set `generation.constraint_backend: xgrammar`.

Run the default config:

```bash
uv run main.py
```

To use another config:

```bash
LEAP_CONFIG_PATH=path/to/config.yaml uv run main.py
```

## Constrained generation

Choose the constraint backend independently of the generation strategy:

```yaml
generation:
  strategy: cot
  use_constraints: true
  constraint_backend: xgrammar
  output_format: function
  use_global_constraints: true
```

| Mode | Decoding restrictions | Runtime | Environment |
|---|---|---|---|
| Unconstrained | No token-level restrictions; output is still parsed and validated. | vLLM >=0.28,<0.29, V1 | `.venv` |
| `xgrammar` | Grammar or JSON Schema constraints on operation syntax and arguments. | vLLM >=0.28,<0.29, V1 | `.venv` |
| `legacy_state_machine` | A custom state machine masks tokens during function-call generation. Qwen2.5 models only. | vLLM 0.10.0, V0 | `.venv-vllm-legacy` |

Set `use_constraints: false` for unconstrained generation. These runs always use the modern runtime, regardless of the backend setting. If `constraint_backend` is omitted, it defaults to `xgrammar`. Selecting `legacy_state_machine` forces function output; active legacy constraints also select the legacy runtime.

The entrypoints select the environment before importing vLLM. First use may download and install the selected vLLM and PyTorch stack. The backend/version pairing is strict. If LEAP reports a mismatch, follow the error's instructions to update the lockfile with `uv lock` and retry. If installation was interrupted, remove only the generated environment named in the error before retrying.

### Output formats and validation

Both `iterative` and `cot` support function calls and JSON operation objects. Function output is the default:

```text
f_sort_by("Year", "desc")
```

For JSON, set `output_format: json` and `constraint_backend: xgrammar`:

```json
{"action":"sort_by","column":"Year","order":"desc"}
```

With constraints enabled, JSON mode uses vLLM's structured-output backend. With constraints disabled, LEAP uses the same JSON prompts and strict parser without token-level restrictions. COT remains two-phase: action selection emits the `action` field, then argument generation emits an object for that action.

LEAP validates operations against the current table after decoding in both constrained and unconstrained modes. XGrammar supports `add_column` with exactly one value per table row. Unconstrained runs enforce the same count after parsing and discard malformed candidates.

## Strategies and sampling

| Strategy | Behavior |
|---|---|
| `iterative` | Generates one complete operation per attempt, with no candidate voting or shuffle sampling. |
| `cot` | Selects an action, then generates its arguments. With sampling enabled, row and column selection each use eight argument candidates and voting. Other actions use one candidate; `end` needs no arguments. |
| `direct_query` | Skips table transformations and runs answer extraction on the original table as a baseline. |

The `extractors` setting selects answer methods applied to each strategy's final table. The `direct_query` extractor is distinct from the strategy of the same name: it can answer from either a transformed table or the original table.

Set `generation.sampling.enabled: false` to make COT argument generation single-shot. `generation.sampling.shuffle_invariant` applies only to COT row selection. Legacy `n_samples` and `per_action_samples` fields remain accepted but cannot override these counts. Startup messages and `run_config.json` report the effective policy. Earlier iterative results may use multiple candidates and are not directly comparable to the current single-shot policy.

Set `generation.force_zero_temperature: true` to use temperature 0 for every model request, including action selection, argument generation, and answer extraction. LEAP also passes `enable_thinking=False` to chat templates; models that support this option generate answers without a thinking phase.

Iterative `add_column` generates the complete operation in one response and is unavailable when the full table cannot fit the model context. COT generates the complete column in one argument-generation response.

## Experiment matrices

The [example experiment spec](configs/experiments.example.yaml) defines the models, repetitions, example limit, extractors, and generation settings to compare:

```bash
uv run scripts/run_experiments.py configs/experiments.example.yaml
```

Its matrix includes these dimensions:

```yaml
matrix:
  strategies: [iterative, cot, direct_query]
  use_constraints: [false, true]
  use_global_constraints: [false, true]
  constraint_backends: [legacy_state_machine, xgrammar]
  output_formats: [function, json]
  force_zero_temperature: [true]
  add_column: [true, false]
```

The runner generates only unique, supported jobs. Legacy constrained jobs use function output and run only for Qwen2.5 models. Unconstrained jobs use the modern runtime. The direct-query baseline runs once per model, repeat, and temperature mode because action constraints do not affect it. Add `false` to `force_zero_temperature` to compare both temperature modes, roughly doubling the matrix size.

Job failures are recorded in the experiment report, and the runner continues with the remaining jobs. It exits nonzero if any job failed. Invalid suite configuration and explicit user interruption stop the runner.

### Model reuse

Model reuse is enabled by default. Set this at the top level of the experiment spec to make it explicit:

```yaml
reuse_models: true
```

Compatible jobs reuse the loaded model across strategy, constraint, output-format, and repeat changes. A model change or switch between V1 and V0 starts a new session. Set `reuse_models: false` for isolated one-process-per-job execution. Reports record sessions, model loads, reuse status, and restart reasons.

### Resuming an interrupted matrix

```bash
uv run scripts/run_experiments.py --resume results/experiments/<experiment-id>
```

The runner uses the saved experiment specification and generated job configs, skips jobs with terminal status events, and reruns the job active at interruption. Completed run artifacts are not overwritten. Resuming uses the current working-tree code, so preserve the code revision as well as the experiment configuration for reproducibility.

## Results and diagnostics

The default single-run config writes results to `logs/results.jsonl` and table logs under the configured `logging.log_dir`. The example matrix writes experiment outputs under `results/experiments`.

Normal console output includes progress, warnings, errors, and aggregate summaries. Set `generation.sampling.debug: true` to also print full prompts, responses, action payloads, and per-question summaries. Debug output is disabled by default. Structured results, table logs, and compact failed-candidate counts remain available; raw candidate diagnostics are not persisted.

## Development

The setup command above installs development dependencies. Enable the pre-commit hooks with:

```bash
uv run pre-commit install
```

The hooks run Ruff linting and formatting before commits.
