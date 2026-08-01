# ruff: noqa: I001  # Runtime bootstrap must execute before imports that load vLLM.
if __name__ == "__main__":
    import os as _bootstrap_os
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    from leap.vllm_runtime import ensure_vllm_runtime, runtime_for_config

    _bootstrap_config = _BootstrapPath(_bootstrap_os.environ.get("LEAP_CONFIG_PATH", "configs/default.yaml"))
    ensure_vllm_runtime(runtime_for_config(_bootstrap_config), argv=_bootstrap_sys.argv)

import json
import logging
import multiprocessing as mp
import os
import re
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

from leap.config.loader import (
    AppConfig,
    DatasetConfig,
    get_model_id,
    load_runtime_config,
)
from leap.config.loader import (
    GenerationConfig as GenerationSettings,
)
from leap.core import Action, InferenceRequest, InferenceResult
from leap.extractors import build_extractors
from leap.generation.prompt_builder import PromptBuilder
from leap.generation.sampling import SamplingConfig, SamplingLayer
from leap.generation.shuffle_invariant_sampling import ShuffleInvariantSamplingLayer
from leap.generation.strategies import (
    ChainOfTableGenerationStrategy,
    DirectQueryGenerationStrategy,
    IterativeGenerationStrategy,
)
from leap.inference.vllm_server import ProcessParallelVLLM
from leap.vllm_runtime import runtime_metadata, validate_installed_runtime
from leap.utils.profiler import get_aggregate_profiler

# shut off llm logging in case not important
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

CONFIG_PATH = Path(os.environ.get("LEAP_CONFIG_PATH", "configs/default.yaml"))
RESULTS_ROOT = Path(os.environ.get("LEAP_RESULTS_ROOT", "results"))


@dataclass(frozen=True)
class RuntimeContext:
    config: AppConfig
    prompt_builder: PromptBuilder
    tokenizer: AutoTokenizer
    dataset: Any


@dataclass(frozen=True)
class RunOutputPaths:
    run_dir: Path
    results_file: Path
    table_log_dir: Path
    accuracy_file: Path
    extractor_accuracy_file: Path
    manifest_file: Path
    config_slug: str
    timestamp: str


def load_dataset_from_config(dataset_config: DatasetConfig):
    loader = dataset_config.loader.lower()

    if loader == "huggingface":
        if not dataset_config.name:
            raise ValueError("HuggingFace dataset loader requires 'name'")
        name = dataset_config.name
        split = dataset_config.split
        kwargs = {}
        if split:
            kwargs["split"] = split
        if dataset_config.trust_remote_code is not None:
            kwargs["trust_remote_code"] = dataset_config.trust_remote_code
        return load_dataset(name, **kwargs)

    if loader == "json":
        data_files = dataset_config.data_files
        if not data_files:
            raise ValueError("JSON dataset loader requires 'data_files'")
        split = dataset_config.split
        return load_dataset("json", data_files=data_files, split=split)

    if loader == "disk":
        path = dataset_config.path
        if not path:
            raise ValueError("Disk dataset loader requires 'path'")
        return load_from_disk(path)

    raise ValueError(f"Unsupported dataset loader: {loader}")


def build_runtime(config_path: Path = CONFIG_PATH) -> RuntimeContext:
    # Load tokenizer first
    model_id = get_model_id(config_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Now load the full config with tokenizer
    app_config: AppConfig = load_runtime_config(config_path, tokenizer)
    validate_installed_runtime(
        use_constraints=app_config.generation.use_constraints,
        constraint_backend=app_config.generation.constraint_backend,
    )

    prompt_builder = PromptBuilder(tokenizer=tokenizer, is_instruct=app_config.model.instruct)
    dataset = load_dataset_from_config(app_config.dataset)
    return RuntimeContext(
        config=app_config,
        prompt_builder=prompt_builder,
        tokenizer=tokenizer,
        dataset=dataset,
    )


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "run"


def build_config_slug(app_config: AppConfig) -> str:
    identity = {
        "model_id": app_config.model.id,
        "model_log_dir": app_config.model.log_dir,
        "dataset": app_config.dataset,
        "run": app_config.run,
        "generation": app_config.generation,
        "extractors": app_config.extractors,
    }
    identity_json = json.dumps(_json_safe(identity), sort_keys=True)
    config_hash = sha1(identity_json.encode("utf-8")).hexdigest()[:8]
    base = f"{app_config.model.log_dir}_{app_config.generation.strategy}_{config_hash}"
    return _sanitize_slug(base)


def create_run_output_paths(app_config: AppConfig, results_root: Path | str = "results") -> RunOutputPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    config_slug = build_config_slug(app_config)
    run_dir = Path(results_root) / config_slug / timestamp
    return RunOutputPaths(
        run_dir=run_dir,
        results_file=run_dir / "results.jsonl",
        table_log_dir=run_dir / "table_logs",
        accuracy_file=run_dir / "end_to_end_accuracy.json",
        extractor_accuracy_file=run_dir / "extractor_accuracy.json",
        manifest_file=run_dir / "run_config.json",
        config_slug=config_slug,
        timestamp=timestamp,
    )


def apply_run_output_paths(app_config: AppConfig, paths: RunOutputPaths) -> AppConfig:
    return replace(
        app_config,
        model=replace(app_config.model, results_file=str(paths.results_file)),
        logging=replace(app_config.logging, log_dir=str(paths.table_log_dir)),
    )


def write_end_to_end_accuracy(accuracy: float, output_file: Path | str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"end_to_end_accuracy": accuracy}, f, indent=2)


def write_run_manifest(app_config: AppConfig, paths: RunOutputPaths, config_path: Path) -> None:
    manifest = {
        "config_slug": paths.config_slug,
        "timestamp": paths.timestamp,
        "config_path": str(config_path),
        "run_dir": str(paths.run_dir),
        "results_file": str(paths.results_file),
        "table_log_dir": str(paths.table_log_dir),
        "end_to_end_accuracy_file": str(paths.accuracy_file),
        "extractor_accuracy_file": str(paths.extractor_accuracy_file),
        "model": _json_safe(app_config.model),
        "dataset": _json_safe(app_config.dataset),
        "run": _json_safe(app_config.run),
        "generation": _json_safe(app_config.generation),
        "logging": _json_safe(app_config.logging),
        "vllm_runtime": runtime_metadata(
            app_config.generation.constraint_backend,
            use_constraints=app_config.generation.use_constraints,
        ),
        "extractors": list(app_config.extractors),
    }
    paths.manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_results_to_jsonl(results: list[InferenceResult], output_file, generation_config: GenerationSettings):
    """Write results to JSONL file with execution accuracy metrics"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, result in enumerate(results):
            actions = []
            for action_str in result.action_history:
                action = Action.parse(action_str)
                if action:
                    actions.append(action.to_dict())
                else:
                    actions.append({"action": "invalid", "args": [action_str]})

            # Create entry with WikiTableQuestions logic-based execution accuracy metrics
            # All data comes from the self-contained result
            entry = {
                "id": f"nt-{i + 1}",
                "question": result.question,
                "ground_truth_answers": result.ground_truth_answers,
                "actions": actions,
                "execution_accuracy": result.execution_accuracy,
                "execution_metrics": result.execution_metrics.to_dict(),  # Use to_dict() method
                "extractor_results": [extractor.to_dict() for extractor in (result.extractor_results or [])],
                "sampling_metadata": [
                    {
                        "candidate_actions": m.candidate_actions,
                        "valid_actions": m.valid_actions,
                        "winner": m.action.to_dict(),
                        "n_requested": m.n_requested,
                        "n_generated": m.n_generated,
                        "n_valid": m.n_valid,
                        "winner_votes": m.winner_votes,
                        "total_votes": m.total_votes,
                        "fallback_reason": m.fallback_reason,
                    }
                    for m in (result.sampling_metadata or [])
                ],
                "metadata": {
                    "num_steps": len(actions),
                    "generation_mode": get_generation_mode_string(generation_config),
                    "evaluation_method": "wikitablequestions_logic_with_dataset_answers",
                },
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Results written to {output_file}")


def get_generation_mode_string(generation_config: GenerationSettings):
    """Get a descriptive string for the current generation mode"""
    if generation_config.strategy == "direct_query":
        return "direct_query"
    elif generation_config.strategy == "cot":
        constraint_desc = "with_constraints" if generation_config.use_constraints else "without_constraints"
        return f"chain_of_table_{constraint_desc}"
    elif generation_config.use_constraints:
        if generation_config.use_global_constraints:
            return "constrained_with_global"
        else:
            return "constrained_local_only"
    else:
        return "unconstrained_with_postprocessing"


def create_sampling_layer(generation_settings: GenerationSettings) -> SamplingLayer:
    if not generation_settings.sampling or not generation_settings.sampling.enabled:
        # When sampling is disabled, use n=1 (no voting, single generation)
        print("Sampling disabled - using single-sample generation (n=1)")
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


def build_inference_requests(runtime: RuntimeContext, max_examples: int | None) -> list[InferenceRequest]:
    requests = []
    dataset = runtime.dataset
    subset_size = len(dataset) if max_examples is None else min(max_examples, len(dataset))

    for i, example in enumerate(dataset):
        if i >= subset_size:
            break
        requests.append(InferenceRequest.from_example(example, index=i))

    return requests


def calculate_extractor_accuracy_report(results: list[InferenceResult], configured_extractors: tuple[str, ...]) -> dict[str, Any]:
    method_accuracies = {}
    for method in configured_extractors:
        scores_by_request = {
            result.request_id: extractor.accuracy
            for result in results
            for extractor in (result.extractor_results or [])
            if extractor.method == method
        }
        method_accuracies[method] = sum(scores_by_request.values()) / len(results) if results else 0.0
    average_accuracy = sum(method_accuracies.values()) / len(method_accuracies) if method_accuracies else 0.0
    return {"method_accuracies": method_accuracies, "average_accuracy": average_accuracy, "examples": len(results)}


def write_extractor_accuracy_report(report: dict[str, Any], output_file: Path | str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def print_final_extractor_summary(report: dict[str, Any]) -> None:
    examples = int(report.get("examples", 0))
    method_accuracies = report.get("method_accuracies", {})

    print(f"\n{'=' * 80}")
    print("FINAL EXTRACTOR ACCURACY SUMMARY")
    print(f"{'=' * 80}")
    for method, accuracy in method_accuracies.items():
        correct = round(float(accuracy) * examples)
        print(f"{method}: {float(accuracy):.3f} ({float(accuracy) * 100:.1f}%) | {correct}/{examples} correct")
    average = float(report.get("average_accuracy", 0.0))
    print(f"Average across {len(method_accuracies)} extractors: {average:.3f} ({average * 100:.1f}%)")


def finalize_results(
    results: list[InferenceResult],
    model_settings,
    generation_settings: GenerationSettings,
    run_paths: RunOutputPaths,
    app_config: AppConfig,
) -> dict[str, Any]:
    profiler = get_aggregate_profiler()
    for result in results:
        if result.profiling_data:
            profiler.add_request_profile(
                result.profiling_data["total_time"],
                result.profiling_data["operation_timings"],
                result.profiling_data["num_steps"],
            )

    end_to_end_accuracy = analyze_execution_accuracy(results)
    write_results_to_jsonl(results, model_settings.results_file, generation_settings)
    write_end_to_end_accuracy(end_to_end_accuracy, run_paths.accuracy_file)
    extractor_report = calculate_extractor_accuracy_report(results, app_config.extractors)
    write_extractor_accuracy_report(extractor_report, run_paths.extractor_accuracy_file)
    print_sample_results(results)
    get_aggregate_profiler().print_summary()
    return extractor_report


def main() -> int:
    """Main function using the modular vLLM server with comprehensive logging"""
    print("Setting up modular vLLM server with integrated logging...")

    runtime = build_runtime(CONFIG_PATH)
    original_config = runtime.config
    run_paths = create_run_output_paths(original_config, results_root=RESULTS_ROOT)
    app_config = apply_run_output_paths(original_config, run_paths)
    model_settings = app_config.model
    generation_settings = app_config.generation

    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run output directory: {run_paths.run_dir}")
    write_run_manifest(app_config, run_paths, CONFIG_PATH)

    # Create sampling layer (always created, with n=1 when "disabled")
    sampling_layer = create_sampling_layer(generation_settings)

    # Create strategies and the configured answer-extractor ensemble.
    extractors = build_extractors(app_config.extractors, runtime.prompt_builder)
    iterative_strategy = IterativeGenerationStrategy(
        prompt_builder=runtime.prompt_builder,
        sampling_layer=sampling_layer,
        extractors=extractors,
    )
    cot_strategy = ChainOfTableGenerationStrategy(
        prompt_builder=runtime.prompt_builder,
        sampling_layer=sampling_layer,
        extractors=extractors,
    )
    direct_query_strategy = DirectQueryGenerationStrategy(
        prompt_builder=runtime.prompt_builder,
        sampling_layer=sampling_layer,
        extractors=extractors,
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

    extractor_report = None
    try:
        # Start server
        print(f"Starting server with {num_workers} workers...")
        print(f"Logging configuration: {server.get_logging_stats()}")

        if not server.start_workers(timeout=600):
            print("Failed to start all workers. Exiting.")
            return 1

        requests = build_inference_requests(runtime, app_config.run.max_examples)
        print(f"Processing {len(requests)} questions...")
        print(f"Generation mode: {get_generation_mode_string(generation_settings)}")

        # Generate responses with comprehensive logging
        start_time = time.time()
        results = server.generate_batch(requests)
        end_time = time.time()

        print(f"Total time: {end_time - start_time:.2f} seconds")
        if requests:
            print(f"Average time per request: {(end_time - start_time) / len(requests):.2f} seconds")

        extractor_report = finalize_results(results, model_settings, generation_settings, run_paths, app_config)
        # Demonstrate logging analysis
        demonstrate_logging_analysis(server, results)

    finally:
        # Shutdown will automatically generate summary report
        server.shutdown()

    if extractor_report is not None:
        print_final_extractor_summary(extractor_report)
    return 0


def analyze_execution_accuracy(results: list[InferenceResult]):
    """Analyze and print execution accuracy metrics"""
    execution_accuracies = []
    answer_found_rates = []
    proper_termination_rates = []
    baseline_rates = []

    print(f"\n{'=' * 80}")
    print("EXECUTION ACCURACY ANALYSIS")
    print(f"{'=' * 80}")

    for result in results:
        metrics = result.execution_metrics
        execution_accuracies.append(metrics.execution_accuracy)
        answer_found_rates.append(1.0 if metrics.answer_found_in_final else 0.0)
        proper_termination_rates.append(1.0 if metrics.terminated_properly else 0.0)
        baseline_rates.append(1.0 if metrics.answer_found_in_original else 0.0)

    if execution_accuracies:
        overall_execution_accuracy = sum(execution_accuracies) / len(execution_accuracies)
        overall_answer_found_rate = sum(answer_found_rates) / len(answer_found_rates)
        overall_baseline_rate = sum(baseline_rates) / len(baseline_rates)
        overall_termination_rate = sum(proper_termination_rates) / len(proper_termination_rates)

        print(f"Overall Execution Accuracy: {overall_execution_accuracy:.3f} ({overall_execution_accuracy * 100:.1f}%)")
        print(f"Answer Found in Final Table Rate: {overall_answer_found_rate:.3f} ({overall_answer_found_rate * 100:.1f}%)")
        print(f"Answer Found in Original Table Rate (Baseline): {overall_baseline_rate:.3f} ({overall_baseline_rate * 100:.1f}%)")
        print(f"Proper Termination Rate: {overall_termination_rate:.3f} ({overall_termination_rate * 100:.1f}%)")

        if overall_baseline_rate > 0:
            improvement = overall_answer_found_rate - overall_baseline_rate
            improvement_pct = (improvement / overall_baseline_rate) * 100 if overall_baseline_rate > 0 else 0
            print(f"Improvement over baseline: {improvement:+.3f} ({improvement_pct:+.1f}%)")

        success_cases = sum(1 for acc in execution_accuracies if acc == 1.0)
        print(f"Successful Cases: {success_cases}/{len(execution_accuracies)} ({success_cases / len(execution_accuracies) * 100:.1f}%)")
        return overall_execution_accuracy

    return 0.0


def print_sample_results(results: list[InferenceResult]):
    """Print sample results for inspection"""
    print(f"\n{'=' * 80}")
    print("SAMPLE RESULTS")
    print(f"{'=' * 80}")

    for i in range(min(5, len(results))):
        result = results[i]
        metrics = result.execution_metrics

        print(f"\nExample {i + 1}:")
        print("Question:", result.question)
        print("Ground Truth Answers:", result.ground_truth_answers)
        print("Actions:")
        for j, action in enumerate(result.action_history):
            print(f"  Step {j + 1}: {action}")

        print(f"Execution Accuracy: {metrics.execution_accuracy:.1f}")
        print(f"Answer Found in Final: {metrics.answer_found_in_final}")
        print(f"Answer Found in Original: {metrics.answer_found_in_original}")
        print(f"Terminated Properly: {metrics.terminated_properly}")

        if metrics.matched_answers_final:
            print(f"Matched Answers: {metrics.matched_answers_final}")

        if metrics.final_table_size:
            rows, cols = metrics.final_table_size
            print(f"Final Table Size: {rows} rows × {cols} columns")

        print("-" * 80)


def demonstrate_logging_analysis(server, results):
    """Demonstrate logging analysis capabilities"""
    if not server.get_logging_stats().get("enabled", False):
        print("Logging not enabled - skipping analysis demonstration")
        return

    print(f"\n{'=' * 80}")
    print("LOGGING ANALYSIS DEMONSTRATION")
    print(f"{'=' * 80}")

    # Show logging statistics
    stats = server.get_logging_stats()
    print("Logging Statistics:")
    print(f"  Directory: {stats.get('log_dir', 'N/A')}")
    print(f"  Requests logged: {stats.get('requests_logged', 0)}")
    print(f"  Total log entries: {stats.get('total_entries', 0)}")
    print(f"  Save readable tables: {stats.get('save_readable_tables', False)}")

    # Try to analyze logs for first few requests (they would have been logged during processing)
    print("\nSample request analysis:")

    # Note: In a real scenario, we would have the actual request IDs from the processing
    # For demonstration, we show what the analysis would look like
    print("  (Request-specific logs would be available after processing)")
    print("  Use server.analyze_request_logs(request_id) to analyze specific requests")
    print("  Use server.write_summary_report() to generate comprehensive reports")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
