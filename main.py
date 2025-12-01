import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

from leap.inference.vllm_server import ProcessParallelVLLM, create_generation_config, create_logging_config
from leap.generation.strategies import (
    ChainOfTableGenerationStrategy,
    IterativeGenerationStrategy,
)
from leap.generation.prompt_builder import PromptBuilder
from leap.core import Table, Action, InferenceRequest, InferenceResult
from leap.config.loader import (
    AppConfig,
    DatasetConfig,
    GenerationConfig as GenerationSettings,
    get_model_id,
    load_runtime_config,
)

import logging
# shut off llm logging in case not important
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_SERVER_DEV_MODE"] = "1"
CONFIG_PATH = Path(os.environ.get("LEAP_CONFIG_PATH", "configs/default.yaml"))


@dataclass(frozen=True)
class RuntimeContext:
    config: AppConfig
    prompt_builder: PromptBuilder
    tokenizer: AutoTokenizer
    dataset: Any


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

    prompt_builder = PromptBuilder(tokenizer=tokenizer, is_instruct=app_config.model.instruct)
    dataset = load_dataset_from_config(app_config.dataset)
    return RuntimeContext(
        config=app_config,
        prompt_builder=prompt_builder,
        tokenizer=tokenizer,
        dataset=dataset,
    )


def write_results_to_jsonl(results: list[InferenceResult], output_file, generation_config: GenerationSettings):
    """Write results to JSONL file with execution accuracy metrics"""
    with open(output_file, 'w', encoding='utf-8') as f:
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
                "id": f"nt-{i+1}",
                "question": result.question,
                "ground_truth_answers": result.ground_truth_answers,
                "actions": actions,
                "execution_accuracy": result.execution_accuracy,
                "execution_metrics": result.execution_metrics.to_dict(),  # Use to_dict() method
                "metadata": {
                    "num_steps": len(actions),
                    "generation_mode": get_generation_mode_string(generation_config),
                    "evaluation_method": "wikitablequestions_logic_with_dataset_answers"
                }
            }
            f.write(json.dumps(entry) + '\n')
    
    print(f"Results written to {output_file}")


def get_generation_mode_string(generation_config: GenerationSettings):
    """Get a descriptive string for the current generation mode"""
    if generation_config.use_chain_of_table:
        constraint_desc = "with_constraints" if generation_config.use_constraints else "without_constraints"
        return f"chain_of_table_{constraint_desc}"
    elif generation_config.use_constraints:
        if generation_config.use_global_constraints:
            return "constrained_with_global"
        else:
            return "constrained_local_only"
    else:
        return "unconstrained_with_postprocessing"


def main():
    """Main function using the modular vLLM server with comprehensive logging"""
    print("Setting up modular vLLM server with integrated logging...")

    runtime = build_runtime(CONFIG_PATH)
    app_config = runtime.config
    model_settings = app_config.model
    generation_settings = app_config.generation
    
    iterative_strategy = IterativeGenerationStrategy(prompt_builder=runtime.prompt_builder)
    cot_strategy = ChainOfTableGenerationStrategy(prompt_builder=runtime.prompt_builder)
    
    # Create logging configuration
    logging_settings = app_config.logging
    logging_config = create_logging_config(
        enable_logging=logging_settings.enable_logging,
        log_dir=logging_settings.log_dir,
        save_readable_tables=logging_settings.save_readable_tables,
        compress_logs=logging_settings.compress_logs,
        log_format=logging_settings.log_format,
        max_table_chars=logging_settings.max_table_chars,
    )
    
      # Create generation configuration with logging
    generation_config = create_generation_config(
        use_constraints=generation_settings.use_constraints,
        use_cot=generation_settings.use_chain_of_table,
        use_global_constraints=generation_settings.use_global_constraints,
        tokenizer_config=model_settings.tokenizer_config,
        generation_functions={
            'iterative_generation': iterative_strategy.generate_instance,
            'cot_generation': cot_strategy.generate_instance
        },
        logging_config=logging_config,
        tensor_parallel_size=model_settings.hardware.tensor_parallel_size
    )
    
    # Initialize the server
    num_workers = model_settings.hardware.num_workers
    server = ProcessParallelVLLM(
        model_id=model_settings.id,
        num_workers=model_settings.hardware.num_workers,
        gpu_allocation=model_settings.hardware.gpu_allocation,
        generation_config=generation_config
    )
    
    try:
        # Start server
        print(f"Starting server with {num_workers} workers...")
        print(f"Logging configuration: {server.get_logging_stats()}")
        
        if not server.start_workers():
            print("Failed to start all workers. Exiting.")
            return
        
        # Prepare requests
        requests = []
        run_config = app_config.run
        max_examples = run_config.max_examples
        dataset = runtime.dataset
        subset_size = len(dataset) if max_examples is None else min(max_examples, len(dataset))

        for i, example in enumerate(dataset):
            if i >= subset_size:
                break

            # Use typed request builder
            inference_request = InferenceRequest.from_example(example, index=i)

            # Convert to dict for queue
            requests.append(inference_request.to_dict())
        
        print(f"Processing {len(requests)} questions...")
        print(f"Generation mode: {get_generation_mode_string(generation_settings)}")
        
        # Generate responses with comprehensive logging
        start_time = time.time()
        results_dicts = server.generate_batch(requests)
        end_time = time.time()

        # Convert dicts to typed results
        results = [InferenceResult.from_dict(r) for r in results_dicts]

        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Average time per request: {(end_time - start_time) / len(requests):.2f} seconds")

        # Analyze results
        analyze_execution_accuracy(results)

        # Write results
        write_results_to_jsonl(
            results,
            model_settings.results_file,
            generation_settings,
        )

        # Print sample results
        print_sample_results(results)
        
        # Demonstrate logging analysis
        demonstrate_logging_analysis(server, results)
        
    finally:
        # Shutdown will automatically generate summary report
        server.shutdown()


def analyze_execution_accuracy(results: list[InferenceResult]):
    """Analyze and print execution accuracy metrics"""
    execution_accuracies = []
    answer_found_rates = []
    proper_termination_rates = []
    baseline_rates = []

    print(f"\n{'='*80}")
    print("EXECUTION ACCURACY ANALYSIS")
    print(f"{'='*80}")

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
        
        print(f"Overall Execution Accuracy: {overall_execution_accuracy:.3f} ({overall_execution_accuracy*100:.1f}%)")
        print(f"Answer Found in Final Table Rate: {overall_answer_found_rate:.3f} ({overall_answer_found_rate*100:.1f}%)")
        print(f"Answer Found in Original Table Rate (Baseline): {overall_baseline_rate:.3f} ({overall_baseline_rate*100:.1f}%)")
        print(f"Proper Termination Rate: {overall_termination_rate:.3f} ({overall_termination_rate*100:.1f}%)")
        
        if overall_baseline_rate > 0:
            improvement = overall_answer_found_rate - overall_baseline_rate
            improvement_pct = (improvement / overall_baseline_rate) * 100 if overall_baseline_rate > 0 else 0
            print(f"Improvement over baseline: {improvement:+.3f} ({improvement_pct:+.1f}%)")
        
        success_cases = sum(1 for acc in execution_accuracies if acc == 1.0)
        print(f"Successful Cases: {success_cases}/{len(execution_accuracies)} ({success_cases/len(execution_accuracies)*100:.1f}%)")


def print_sample_results(results: list[InferenceResult]):
    """Print sample results for inspection"""
    print(f"\n{'='*80}")
    print("SAMPLE RESULTS")
    print(f"{'='*80}")

    for i in range(min(5, len(results))):
        result = results[i]
        metrics = result.execution_metrics

        print(f"\nExample {i+1}:")
        print("Question:", result.question)
        print("Ground Truth Answers:", result.ground_truth_answers)
        print("Actions:")
        for j, action in enumerate(result.action_history):
            print(f"  Step {j+1}: {action}")

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
    if not server.get_logging_stats().get('enabled', False):
        print("Logging not enabled - skipping analysis demonstration")
        return
    
    print(f"\n{'='*80}")
    print("LOGGING ANALYSIS DEMONSTRATION")
    print(f"{'='*80}")
    
    # Show logging statistics
    stats = server.get_logging_stats()
    print(f"Logging Statistics:")
    print(f"  Directory: {stats.get('log_dir', 'N/A')}")
    print(f"  Requests logged: {stats.get('requests_logged', 0)}")
    print(f"  Total log entries: {stats.get('total_entries', 0)}")
    print(f"  Save readable tables: {stats.get('save_readable_tables', False)}")
    
    # Try to analyze logs for first few requests (they would have been logged during processing)
    print(f"\nSample request analysis:")
    
    # Note: In a real scenario, we would have the actual request IDs from the processing
    # For demonstration, we show what the analysis would look like
    print("  (Request-specific logs would be available after processing)")
    print("  Use server.analyze_request_logs(request_id) to analyze specific requests")
    print("  Use server.write_summary_report() to generate comprehensive reports")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()