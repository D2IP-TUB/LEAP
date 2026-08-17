# ruff: noqa: I001  # Runtime bootstrap must execute before imports that load vLLM.
if __name__ == "__main__":
    import os as _bootstrap_os
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    from leap.vllm_runtime import ensure_vllm_runtime, runtime_for_config

    _bootstrap_config = _BootstrapPath(_bootstrap_os.environ.get("LEAP_CONFIG_PATH", "configs/default.yaml"))
    ensure_vllm_runtime(runtime_for_config(_bootstrap_config), argv=_bootstrap_sys.argv)

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from datasets import load_from_disk
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer

from leap.config.loader import (
    AppConfig,
    DatasetConfig,
    get_model_id,
    load_runtime_config_tool,
    update_runtime_config_dataset_tool,
)
from leap.config.loader import (
    GenerationConfig as GenerationSettings,
)
from leap.core import Action, InferenceRequest
from leap.generation.prompt_builder import PromptBuilder
from leap.generation.sampling import SamplingConfig, SamplingLayer
from leap.generation.shuffle_invariant_sampling import ShuffleInvariantSamplingLayer
from leap.generation.strategies import (
    ChainOfTableGenerationStrategy,
    DirectQueryGenerationStrategy,
    IterativeGenerationStrategy,
)
from leap.inference.vllm_server import ProcessParallelVLLM
from leap.vllm_runtime import validate_installed_runtime
from leap.utils.apply_action import apply_single_action

# shut off llm logging in case not important
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

CONFIG_PATH = Path(os.environ.get("LEAP_CONFIG_PATH", "configs/default.yaml"))
SECRET_TOKEN = "Leap_tool_secret_token"


@dataclass(frozen=True)
class RuntimeContext:
    config: AppConfig
    prompt_builder: PromptBuilder
    tokenizer: AutoTokenizer
    dataset: Any


def load_dataset_from_config(dataset_config: DatasetConfig):
    loader = dataset_config.loader.lower()

    if loader == "disk":
        path = dataset_config.path
        if not path:
            raise ValueError("Disk dataset loader requires 'path'")
        return load_from_disk(path)

    raise ValueError(f"Unsupported dataset loader: {loader}")


def build_runtime_tool(config_path: Path = CONFIG_PATH) -> RuntimeContext:
    model_id = get_model_id(config_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    app_config: AppConfig = load_runtime_config_tool(config_path, tokenizer)
    validate_installed_runtime(
        use_constraints=app_config.generation.use_constraints,
        constraint_backend=app_config.generation.constraint_backend,
        output_format=app_config.generation.output_format,
    )

    prompt_builder = PromptBuilder(
        tokenizer=tokenizer,
        is_instruct=app_config.model.instruct,
        output_format=app_config.generation.output_format,
    )
    return RuntimeContext(
        config=app_config,
        prompt_builder=prompt_builder,
        tokenizer=tokenizer,
        dataset=None,
    )


def build_dataset_config(app_config: AppConfig, dataset_path: Path):
    return update_runtime_config_dataset_tool(app_config, dataset_path)


def get_generation_mode_string(generation_config: GenerationSettings):
    """Get a descriptive string for the current generation mode"""
    format_prefix = f"{generation_config.output_format}_" if generation_config.output_format != "function" else ""
    if generation_config.strategy == "cot":
        constraint_desc = "with_constraints" if generation_config.use_constraints else "without_constraints"
        return f"{format_prefix}chain_of_table_{constraint_desc}"
    elif generation_config.use_constraints:
        if generation_config.use_global_constraints:
            return f"{format_prefix}constrained_with_global"
        else:
            return f"{format_prefix}constrained_local_only"
    else:
        return f"{format_prefix}unconstrained_with_postprocessing"


def create_sampling_layer(generation_settings: GenerationSettings) -> SamplingLayer:
    if not generation_settings.sampling or not generation_settings.sampling.enabled:
        return SamplingLayer(
            config=SamplingConfig(
                enabled=True,
                n_samples=1,
                debug=False,
            )
        )
    elif generation_settings.sampling.shuffle_invariant:
        print(f"Shuffle-invariant sampling enabled: {generation_settings.sampling.n_samples} samples per step")
        return ShuffleInvariantSamplingLayer(config=generation_settings.sampling)
    else:
        print(f"Sampling enabled: {generation_settings.sampling.n_samples} samples per step")
        return SamplingLayer(config=generation_settings.sampling)


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = build_runtime_tool(CONFIG_PATH)

    app_config = runtime.config
    model_settings = app_config.model
    generation_settings = app_config.generation

    # Create sampling layer (always created, with n=1 when "disabled")
    sampling_layer = create_sampling_layer(generation_settings)

    # Create strategies
    iterative_strategy = IterativeGenerationStrategy(
        prompt_builder=runtime.prompt_builder,
        sampling_layer=sampling_layer,
    )
    cot_strategy = ChainOfTableGenerationStrategy(
        prompt_builder=runtime.prompt_builder,
        sampling_layer=sampling_layer,
    )
    direct_query_strategy = DirectQueryGenerationStrategy(
        prompt_builder=runtime.prompt_builder,
        sampling_layer=sampling_layer,
    )

    # Validate strategy selection
    valid_strategies = ["iterative", "cot", "direct_query"]
    if generation_settings.strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy '{generation_settings.strategy}'. Must be one of: {valid_strategies}")

    print(f"Using generation strategy: {generation_settings.strategy}_generation")

    # Initialize the server with typed configs (no more dicts!)
    num_workers = model_settings.hardware.num_workers
    server = ProcessParallelVLLM(
        model_id=model_settings.id,
        num_workers=num_workers,
        gpu_allocation=model_settings.hardware.gpu_allocation,
        generation_config=generation_settings,
        tokenizer_config=model_settings.tokenizer_config,
        logging_config=app_config.logging,
        generation_functions={
            "iterative_generation": iterative_strategy.generate_instance,
            "cot_generation": cot_strategy.generate_instance,
            "direct_query_generation": direct_query_strategy.generate_instance,
        },
        tensor_parallel_size=model_settings.hardware.tensor_parallel_size,
        max_concurrent_requests=model_settings.hardware.max_concurrent_requests,
        max_model_len=model_settings.hardware.max_model_len,
    )
    app.state.app_config = app_config
    app.state.server = server

    # Start server
    print(f"Starting server with {num_workers} workers...")
    print(f"Logging configuration: {server.get_logging_stats()}")

    if not server.start_workers():
        print("Failed to start all workers. Exiting.")
        return
    print(" Server is Ready!")

    yield

    server.shutdown()


app = FastAPI(lifespan=lifespan)


class ExperimentRequest(BaseModel):
    dataset_path: str
    question: str


class OutputResponse(BaseModel):
    question: str
    actions: list
    sampling_metadata: list
    num_steps: int


@app.post("/process")
async def process_experiment(request: Request, req_data: ExperimentRequest, x_secret_token: str = Header(None)):
    if x_secret_token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid secret token")

    try:
        app_config = request.app.state.app_config
        server = request.app.state.server

        app_config = update_runtime_config_dataset_tool(app_config, req_data.dataset_path)

        dataset = load_dataset_from_config(app_config.dataset)

        inference_request = InferenceRequest.from_example(dataset[0])
        initial_table = inference_request.table
        result = server.generate_batch([inference_request])

        steps_log = [{"index": 0, "action": "initial", "table": initial_table, "sampling_metadata": None}]
        current_table = initial_table
        total_sampling_metadata = result[0].sampling_metadata

        for i in range(len(result[0].action_history)):
            action_str = result[0].action_history[i]
            action = Action.parse(action_str)
            if not action:
                if action_str == "direct_query()":
                    continue
                step = {
                    "index": len(steps_log),
                    "action": {"action": "invalid", "args": [action_str]},
                    "table": None,
                    "sampling_metadata": None,
                }
                steps_log.append(step)
                continue
            else:
                if action.name in {"initial", "direct_query"}:
                    continue
                if action.name == "end":
                    step = {"index": len(steps_log), "action": action, "table": current_table, "sampling_metadata": None}

                    steps_log.append(step)
                else:
                    m = total_sampling_metadata[i]
                    table = apply_single_action(current_table, action_str)

                    sampling_metadata = {
                        "candidate_actions": m.candidate_actions,
                        "valid_actions": m.valid_actions,
                        "winner": m.action.to_dict(),
                        "n_requested": m.n_requested,
                        "n_generated": m.n_generated,
                        "n_valid": m.n_valid,
                        "winner_votes": m.winner_votes,
                        "total_votes": m.total_votes,
                    }

                    step = {"index": len(steps_log), "action": action, "table": table, "sampling_metadata": sampling_metadata}

                    steps_log.append(step)

                    current_table = table
        return steps_log

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
