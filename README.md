# LEAP

LEAP is an agentic loop for question answering over tables, based on Chain-of-Tables. The model
picks a table operation, LEAP executes it, and the resulting table goes back into the prompt for
the next step. It is reader-based: the full table sits in the context window at every step, so the
model reads the table rather than querying it. LEAP extends this loop with constrained generation —
restricting decoding so the model emits only valid operations over the table in front of it.

LEAP runs locally at scale on a self-hosted inference backend such as vLLM. Requests are issued
concurrently and served through continuous batching, so large experiment matrices — many models,
strategies, and repeats over hundreds of questions — run efficiently on local GPUs rather than
against a paid API.

RQ1. Does constrained generation improve accuracy over unconstrained Chain-of-Tables?
RQ2. Does MCP tool calling improve accuracy over text-based operation generation?

## Status

Active. Current work expands LEAP's tool usage — exposing table operations as callable tools and
widening the operation set — and makes the loop more robust, so that answers hold up against
malformed operations and against changes in how the table is presented.

- [LEAP](https://github.com/D2IP-TUB/LEAP) — backend, experiment runner, constrained generation.
- [LEAP-MCP](https://github.com/D2IP-TUB/LEAP-MCP) — MCP host/client/server variant; no constrained mode.
- LEAP_permutation — row-order sensitivity and marginalization experiments.

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev --extra vllm-modern
uv run main.py                                  # uses configs/default.yaml
LEAP_CONFIG_PATH=path/to.yaml uv run main.py    # another config
```

Review [configs/default.yaml](configs/default.yaml) (model, dataset, generation, extractors) and
[configs/models.yaml](configs/models.yaml) (per-model hardware) before running.

## Configuration

| Strategy | Behavior |
|---|---|
| `iterative` | One complete operation per loop step; single-shot. |
| `cot` | Action selection, then argument generation with candidate voting. |
| `direct_query` | No transformation; baseline over the original table. |

| `constraint_backend` | Restriction | Runtime |
|---|---|---|
| *(`use_constraints: false`)* | None; output still parsed and validated. | vLLM ≥0.28,<0.29 V1 (`.venv`) |
| `xgrammar` | Grammar or JSON Schema over syntax and arguments. | vLLM ≥0.28,<0.29 V1 (`.venv`) |
| `legacy_state_machine` | Python state machine masks tokens. Qwen2.5 only, function output. | vLLM 0.10.0 V0 (`.venv-vllm-legacy`) |

Runtime is selected automatically at launch. `output_format` is `function` or `json` (JSON requires
XGrammar). All modes validate operations against the current table after decoding.

## Experiments

```bash
uv run scripts/run_experiments.py configs/experiments.example.yaml
uv run scripts/run_experiments.py --resume results/experiments/<experiment-id>
```

The matrix spans strategies, constraint flags, backends, output formats, and repeats; only
supported combinations run. Compatible jobs reuse a loaded model across configurations. Results
land in `results/experiments`, single runs in `logs/`.
