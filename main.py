import json
import multiprocessing as mp
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from vllm_server import ProcessParallelVLLM, create_generation_config, create_logging_config

from generation_strategies import (
    ChainOfTableGenerationStrategy,
    IterativeGenerationStrategy,
    parse_action_string,
)
from prompt_builder import PromptBuilder

import logging
# shut off llm logging in case not important
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_SERVER_DEV_MODE"] = "1"

CONFIG_PATH = Path(os.environ.get("LEAP_CONFIG_PATH", "configs/default.yaml"))


def load_app_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_presets(presets_path: Path) -> Dict[str, Any]:
    with open(presets_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Model presets file {presets_path} must define a mapping")
    return data.get("models", data)


def build_model_settings(model_section: Dict[str, Any], presets: Dict[str, Any]) -> Dict[str, Any]:
    model_id = model_section["id"]
    preset = deepcopy(presets.get(model_id, {}))
    if not preset:
        raise ValueError(f"No model preset found for id '{model_id}'")

    log_dir = model_section.get("log_dir", preset.get("log_dir", f"table_logs_{model_id.replace('/', '_')}"))
    instruct = model_section.get("instruct", preset.get("instruct", False))

    hardware = preset.get("hardware", {}).copy()
    override_hw = model_section.get("hardware", {})
    if override_hw:
        hardware.update(override_hw)

    required_hw_keys = {"num_workers", "tensor_parallel_size", "gpu_allocation"}
    missing = [key for key in required_hw_keys if key not in hardware]
    if missing:
        raise ValueError(f"Missing hardware fields {missing} for model '{model_id}'")

    return {
        "log_dir": log_dir,
        "instruct": instruct,
        "hardware": hardware,
    }


def resolve_logging_config(raw_logging_config: dict, model_log_dir: str, model_id: str) -> dict:
    log_dir_template = raw_logging_config.get("log_dir", "./logs/{model_log_dir}")
    log_dir = log_dir_template.format(
        model=model_id.replace("/", "_"),
        model_log_dir=model_log_dir
    )

    return {
        'enable_logging': raw_logging_config.get('enable_logging', True),
        'log_dir': log_dir,
        'save_readable_tables': raw_logging_config.get('save_readable_tables', False),
        'compress_logs': raw_logging_config.get('compress_logs', False),
        'log_format': raw_logging_config.get('log_format', 'readable'),
        'max_table_chars': raw_logging_config.get('max_table_chars', 10000)
    }


def load_dataset_from_config(dataset_config: dict):
    loader = dataset_config.get("loader", "huggingface").lower()

    if loader == "huggingface":
        name = dataset_config["name"]
        split = dataset_config.get("split")
        kwargs = {}
        if split:
            kwargs["split"] = split
        if dataset_config.get("trust_remote_code") is not None:
            kwargs["trust_remote_code"] = dataset_config["trust_remote_code"]
        return load_dataset(name, **kwargs)

    if loader == "json":
        data_files = dataset_config.get("data_files")
        if not data_files:
            raise ValueError("JSON dataset loader requires 'data_files'")
        split = dataset_config.get("split")
        return load_dataset("json", data_files=data_files, split=split)

    if loader == "disk":
        path = dataset_config.get("path")
        if not path:
            raise ValueError("Disk dataset loader requires 'path'")
        return load_from_disk(path)

    raise ValueError(f"Unsupported dataset loader: {loader}")


app_config = load_app_config(CONFIG_PATH)

model_section = app_config.get("model")
if not model_section or "id" not in model_section:
    raise ValueError("Configuration must define 'model.id'")

model_id = model_section["id"]
presets_path = Path(model_section.get("presets_path", "configs/models.yaml"))
model_presets = load_model_presets(presets_path)
model_settings = build_model_settings(model_section, model_presets)

tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt_builder = PromptBuilder(tokenizer=tokenizer, is_instruct=model_settings["instruct"])

output_file = model_section.get("results_file", "./logs/results.jsonl")

GENERATION_CONFIG = app_config.get("generation", {})
USE_GENERATION_CONSTRAINTS = GENERATION_CONFIG.get("use_constraints", False)
USE_GLOBAL_CONSTRAINTS = GENERATION_CONFIG.get("use_global_constraints", False)
USE_CHAIN_OF_TABLE = GENERATION_CONFIG.get("use_chain_of_table", False)

LOGGING_CONFIG = resolve_logging_config(app_config.get("logging", {}), model_settings["log_dir"], model_id)

dataset_config = app_config.get("dataset")
if not dataset_config:
    raise ValueError("Configuration must include a 'dataset' section")

dataset = load_dataset_from_config(dataset_config)

def write_results_to_jsonl(results, examples, output_file):
    """Write results to JSONL file with execution accuracy metrics"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (result, example) in enumerate(zip(results, examples)):
            action_history = result.get('action_history', [])
            metrics = result.get('execution_accuracy_metrics', {})
            
            actions = []
            for action_str in action_history:
                parsed = parse_action_string(action_str)
                if parsed:
                    action_name, args = parsed
                    actions.append({"action": action_name, "args": args})
                else:
                    actions.append({"action": "invalid", "args": [action_str]})
            
            # Create entry with WikiTableQuestions logic-based execution accuracy metrics
            entry = {
                "id": f"nt-{i+1}",
                "question": example['question'],
                "ground_truth_answers": example['answers'],
                "actions": actions,
                "execution_accuracy": metrics.get('execution_accuracy', 0.0),
                "execution_metrics": {
                    "answer_found_in_final": metrics.get('answer_found_in_final', False),
                    "answer_found_in_original": metrics.get('answer_found_in_original', False),
                    "terminated_properly": metrics.get('terminated_properly', False),
                    "matched_answers_final": metrics.get('matched_answers_final', []),
                    "matched_answers_original": metrics.get('matched_answers_original', []),
                    "num_actions": metrics.get('num_actions', 0),
                    "final_table_size": metrics.get('final_table_size'),
                    "execution_error": metrics.get('execution_error'),
                    "evaluation_method": metrics.get('evaluation_method', 'wikitablequestions_logic')
                },
                "metadata": {
                    "num_steps": len(actions),
                    "generation_mode": get_generation_mode_string(),
                    "evaluation_method": "wikitablequestions_logic_with_dataset_answers"
                }
            }
            f.write(json.dumps(entry) + '\n')
    
    print(f"Results written to {output_file}")


def get_generation_mode_string():
    """Get a descriptive string for the current generation mode"""
    if USE_CHAIN_OF_TABLE:
        constraint_desc = "with_constraints" if USE_GENERATION_CONSTRAINTS else "without_constraints"
        return f"chain_of_table_{constraint_desc}"
    elif USE_GENERATION_CONSTRAINTS:
        if USE_GLOBAL_CONSTRAINTS:
            return "constrained_with_global"
        else:
            return "constrained_local_only"
    else:
        return "unconstrained_with_postprocessing"


def main():
    """Main function using the modular vLLM server with comprehensive logging"""
    print("Setting up modular vLLM server with integrated logging...")
    
    iterative_strategy = IterativeGenerationStrategy(prompt_builder=prompt_builder)
    cot_strategy = ChainOfTableGenerationStrategy(prompt_builder=prompt_builder)
    
    # Create logging configuration
    logging_config = create_logging_config(
        enable_logging=LOGGING_CONFIG['enable_logging'],
        log_dir=LOGGING_CONFIG['log_dir'],
        save_readable_tables=LOGGING_CONFIG['save_readable_tables'],
        compress_logs=LOGGING_CONFIG['compress_logs'],
        log_format=LOGGING_CONFIG['log_format'],
        max_table_chars=LOGGING_CONFIG['max_table_chars']
    )
    
      # Create generation configuration with logging
    generation_config = create_generation_config(
        use_constraints=USE_GENERATION_CONSTRAINTS,
        use_cot=USE_CHAIN_OF_TABLE,
        use_global_constraints=USE_GLOBAL_CONSTRAINTS,
        generation_functions={
            'iterative_generation': iterative_strategy.generate_instance,
            'cot_generation': cot_strategy.generate_instance
        },
        logging_config=logging_config,
        tensor_parallel_size=model_settings["hardware"]["tensor_parallel_size"]
    )
    
    # Initialize the server
    num_workers = model_settings["hardware"]["num_workers"]
    server = ProcessParallelVLLM(
        model_id=model_id,
        num_workers=model_settings["hardware"]["num_workers"],
        gpu_allocation=model_settings["hardware"]["gpu_allocation"],
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
        examples = []
        run_config = app_config.get("run", {})
        max_examples = run_config.get("max_examples")
        subset_size = len(dataset) if max_examples is None else min(max_examples, len(dataset))
        
        for i, example in enumerate(dataset):
            if i >= subset_size:
                break
            
            table = {
                'columns': example['table']['header'],
                'rows': example['table']['rows']
            }
            
            request = {
                'question': example['question'],
                'table': table,
                'ground_truth_answers': example['answers']
            }
            
            requests.append(request)
            examples.append(example)
        
        print(f"Processing {len(requests)} questions...")
        print(f"Generation mode: {get_generation_mode_string()}")
        
        # Generate responses with comprehensive logging
        start_time = time.time()
        results = server.generate_batch(requests)
        end_time = time.time()
        
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Average time per request: {(end_time - start_time) / len(requests):.2f} seconds")
        
        # Analyze results
        analyze_execution_accuracy(results)
        
        # Write results
        write_results_to_jsonl(results, examples, output_file)
        
        # Print sample results
        print_sample_results(results, examples)
        
        # Demonstrate logging analysis
        demonstrate_logging_analysis(server, results)
        
    finally:
        # Shutdown will automatically generate summary report
        server.shutdown()


def analyze_execution_accuracy(results):
    """Analyze and print execution accuracy metrics"""
    execution_accuracies = []
    answer_found_rates = []
    proper_termination_rates = []
    baseline_rates = []
    
    print(f"\n{'='*80}")
    print("EXECUTION ACCURACY ANALYSIS")
    print(f"{'='*80}")
    
    for result in results:
        metrics = result.get('execution_accuracy_metrics', {})
        execution_accuracies.append(metrics.get('execution_accuracy', 0.0))
        answer_found_rates.append(1.0 if metrics.get('answer_found_in_final', False) else 0.0)
        proper_termination_rates.append(1.0 if metrics.get('terminated_properly', False) else 0.0)
        baseline_rates.append(1.0 if metrics.get('answer_found_in_original', False) else 0.0)
    
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


def print_sample_results(results, examples):
    """Print sample results for inspection"""
    print(f"\n{'='*80}")
    print("SAMPLE RESULTS")
    print(f"{'='*80}")
    
    for i in range(min(5, len(results))):
        result = results[i]
        action_history = result.get('action_history', [])
        metrics = result.get('execution_accuracy_metrics', {})
        
        print(f"\nExample {i+1}:")
        print("Question:", examples[i]['question'])
        print("Ground Truth Answers:", examples[i]['answers'])
        print("Actions:")
        for j, action in enumerate(action_history):
            print(f"  Step {j+1}: {action}")
        
        print(f"Execution Accuracy: {metrics.get('execution_accuracy', 0.0):.1f}")
        print(f"Answer Found in Final: {metrics.get('answer_found_in_final', False)}")
        print(f"Answer Found in Original: {metrics.get('answer_found_in_original', False)}")
        print(f"Terminated Properly: {metrics.get('terminated_properly', False)}")
        
        if metrics.get('matched_answers_final'):
            print(f"Matched Answers: {metrics['matched_answers_final']}")
        
        if metrics.get('final_table_size'):
            rows, cols = metrics['final_table_size']
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