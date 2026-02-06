# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LEAP (Large-scale Experiment for Automated table Processing) is a table reasoning experiment runner. It uses LLMs (via vLLM) to iteratively transform tables to answer questions, following the Chain-of-Table paper methodology. The primary dataset is WikiTableQuestions.

## Commands

### Environment Setup
```bash
uv sync                    # Install dependencies
source .venv/bin/activate  # Activate virtualenv
uv sync --group dev        # Install dev dependencies (includes pytest, pre-commit)
pre-commit install         # Enable pre-commit hooks
```

### Running
```bash
python main.py                              # Run with default config (configs/default.yaml)
LEAP_CONFIG_PATH=configs/custom.yaml python main.py  # Run with custom config
```

### Testing
```bash
.venv/bin/python -m pytest tests/                          # Run all tests
.venv/bin/python -m pytest tests/test_prompt_builder.py    # Run single test file
.venv/bin/python -m pytest tests/test_prompt_builder.py::TestClassName::test_name  # Single test
```

### Linting / Formatting
```bash
ruff check --fix .   # Lint with auto-fix
ruff format .        # Format code
```
Pre-commit hooks run ruff check and ruff format automatically. Config: `line-length = 140`, rules `E`, `F`, `I`.

## Architecture

### Execution Flow

`main.py` → `build_runtime()` loads config, tokenizer, dataset → creates `PromptBuilder` and generation strategies → starts `ProcessParallelVLLM` workers → processes dataset examples through chosen strategy → evaluates results.

### Core Pipeline (two-phase Chain-of-Table)

1. **Action Selection** (Phase 1): LLM picks an action type (e.g., `select_row`, `add_column`, `end`)
2. **Argument Generation** (Phase 2): LLM generates arguments for that action (e.g., row indices, column names)
3. **Table Transformation**: Action is applied to produce a new table
4. **Repeat** until `end` action → then `direct_query` generates the final answer

### Key Modules

- **`leap/core/actions/`** — Action registry pattern. Each action (`select_row`, `select_column`, `add_column`, `group_by`, `sort_by`, `end`) is an `ActionDefinition` subclass registered in the global `REGISTRY` singleton. `direct_query` is NOT registered — it's auto-applied after `end`.
- **`leap/core/table.py`** — Immutable `Table` dataclass (frozen, uses tuples internally). Single source of truth for table operations and CSV serialization.
- **`leap/generation/strategies.py`** — Three strategies inheriting `BaseGenerationStrategy`: `ChainOfTableGenerationStrategy` (two-phase), `IterativeGenerationStrategy` (single-call), `DirectQueryGenerationStrategy` (baseline, no transformations).
- **`leap/generation/sampling.py`** — `SamplingLayer` generates N candidates in parallel via `asyncio.gather()`, filters invalid ones, and selects winner by majority voting. `SamplingConfig` supports per-action sample counts.
- **`leap/generation/prompt_builder.py`** — `PromptBuilder` constructs all prompts. Uses `tokenizer.apply_chat_template()` for instruct models with few-shot examples as conversation history.
- **`leap/generation/action_examples.py`** — Loads few-shot examples from `configs/action_examples.yaml`. `ActionPromptBuilder` builds templates per action with instructions and examples.
- **`leap/inference/vllm_server.py`** — `ProcessParallelVLLM` manages multiple `VLLMWorkerProcess` instances (each a `mp.Process`). Workers run vLLM `AsyncLLMEngine` on assigned GPUs. Each worker process gets its own `REGISTRY` instance configured via `GenerationConfig.enabled_actions`.
- **`leap/config/loader.py`** — Loads `configs/default.yaml` + `configs/models.yaml` into frozen dataclasses (`AppConfig`, `ModelConfig`, `GenerationConfig`, etc.). Model presets define hardware allocation, tokenizer token IDs, and tensor parallel config.
- **`leap/inference/constraints.py`** — Logits processors that constrain LLM output to valid action grammar (optional, controlled by `use_constraints` config).

### Configuration

- `configs/default.yaml` — Main config: model selection, dataset, generation strategy, enabled actions, sampling params, logging.
- `configs/models.yaml` — Per-model presets: GPU allocation, tensor parallelism, tokenizer token IDs.
- `configs/action_examples.yaml` — Few-shot examples for each action type used in prompts.

### Important Patterns

- **Action Registry**: Actions are registered at import time in `leap/core/actions/__init__.py`. The `REGISTRY` global controls which actions are enabled per run. Worker processes must re-configure their own `REGISTRY` since each is a separate process.
- **Frozen Dataclasses**: `Table`, all config classes, and `InferenceResult` are frozen (immutable). `Table` accepts mutable lists in `__init__` but stores tuples.
- **`build_cot_arguments_prompt`** returns a fully formatted string (chat template applied for instruct models). It uses few-shot examples as alternating user/assistant messages.
- **Multiprocessing**: Workers communicate via `mp.Queue`. `mp.set_start_method("spawn")` is required. Each worker loads its own model copy with CUDA_VISIBLE_DEVICES set per `gpu_allocation`.
