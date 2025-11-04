import ast
import json
import multiprocessing as mp
import os
import time

from datasets import load_dataset, load_from_disk
from typing import List, Tuple, Optional

from eval import to_value_list, check_denotation
from generate import generate_action_arguments, generate_action_selection, generate_single_action
from table import apply_action, extract_table_values_for_eval, serialize_table_to_csv
from vllm_server import ProcessParallelVLLM, create_generation_config, create_logging_config
from tokenizer_config import ModelConfig

from tokenizer_config import tokenizer_config

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_SERVER_DEV_MODE"] = "1"

# Configuration
model_id = "meta-llama/Llama-2-70b-hf"
# model_id = "gpt2"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_id = "mistralai/Mixtral-8x7B-v0.1"
# model_id = "openai/gpt-oss-120b"
# model_id = "openai/gpt-oss-20b"

model_config = ModelConfig(model_id)

output_file = "parallel_results.jsonl"

# Configuration flags
USE_GENERATION_CONSTRAINTS = True
USE_GLOBAL_CONSTRAINTS = True
USE_CHAIN_OF_TABLE = False
COT_ACTION_TEMPERATURE = 0.3
COT_ARGS_TEMPERATURE = 0.7

# Logging configuration
LOGGING_CONFIG = {
    'enable_logging': True,
    'log_dir': model_config.log_dir,
    'save_readable_tables': False,
    'compress_logs': False,
    'log_format': 'readable',
    'max_table_chars': 10000
}

# Load dataset
# dataset = load_dataset('wikitablequestions', split='train[:300]', trust_remote_code=True)
# dataset = load_from_disk("./answerable_questions/train")
dataset = load_dataset('json', data_files='dataset_simple.json', split='train')

def parse_action_string(action_str: str) -> Optional[Tuple[str, List]]:
    """Parse action string into (action_name, args) tuple"""
    try:
        # Handle 'end' action separately
        action_str = action_str.strip()
        if action_str == "end" or action_str.startswith("end("):
            return "end", []
        
        # Parse actions with arguments
        if '(' not in action_str:
            return None
            
        action_name, args_str = action_str.split('(', 1)
        action_name = action_name.strip()
        args_str = args_str.rstrip(')').replace('row ', '').strip()
        # Extract list arguments
        if args_str.startswith('[') and args_str.endswith(']'):
            args_str = args_str.replace("\\", "\\\\")

            args_list = ast.literal_eval(args_str)
            return action_name, args_list
        
        # Handle simple arguments
        if ',' in args_str:
            args_list = [arg.strip() for arg in args_str.split(',')]
        else:
            args_list = [args_str]
        
        return action_name, args_list
    except:
        return None

def calculate_execution_accuracy_with_dataset_answers(action_history, final_table, ground_truth_answers, original_table):
    """Calculate execution accuracy using WikiTableQuestions evaluator logic"""
    result = {
        'execution_accuracy': 0.0,
        'terminated_properly': False,
        'answer_found_in_final': False,
        'answer_found_in_original': False,
        'final_table_values': [],
        'original_table_values': [],
        'matched_answers_final': [],
        'matched_answers_original': [],
        'execution_error': None,
        'num_actions': len(action_history),
        'final_table_size': None,
        'evaluation_method': 'wikitablequestions_logic'
    }
    
    try:
        # Check if sequence terminated properly
        result['terminated_properly'] = (
            len(action_history) > 1 and  # Must have more than just one action
            action_history[-1].startswith('end') and  # Last action must be end
            not all(action.startswith('end') for action in action_history[:-1])  # Not all prior actions are end
        )
        
        # Convert ground truth answers to Value objects using evaluator logic
        target_values = to_value_list(ground_truth_answers)
        
        # Extract and evaluate original table
        result['original_table_values'] = extract_table_values_for_eval(original_table)
        original_predicted_values = to_value_list(result['original_table_values'])
        
        # Check if original table contains the answer using evaluator logic
        result['answer_found_in_original'] = check_denotation(target_values, original_predicted_values)
        if result['answer_found_in_original']:
            # Find which answers matched in original table
            result['matched_answers_original'] = find_matching_answers(target_values, original_predicted_values)
        
        # Check final table
        if final_table:
            result['final_table_size'] = (len(final_table['rows']), len(final_table['columns']))
            result['final_table_values'] = extract_table_values_for_eval(final_table)
            final_predicted_values = to_value_list(result['final_table_values'])
            
            # Check if final table contains the answer using evaluator logic
            result['answer_found_in_final'] = check_denotation(target_values, final_predicted_values)
            if result['answer_found_in_final']:
                # Find which answers matched in final table
                result['matched_answers_final'] = find_matching_answers(target_values, final_predicted_values)
            
            # Calculate execution accuracy
            if result['terminated_properly'] and result['answer_found_in_final']:
                result['execution_accuracy'] = 1.0
            else:
                result['execution_accuracy'] = 0.0
        else:
            result['execution_error'] = "No final table produced"
    
    except Exception as e:
        result['execution_error'] = str(e)
    
    return result

def find_matching_answers(target_values, predicted_values):
    """Find which target answers have matches in predicted values"""
    matched = []
    for target in target_values:
        for predicted in predicted_values:
            if target.match(predicted):
                # Get the normalized string representation
                matched.append(target.normalized)
                break
    return matched

# Generation functions with integrated logging via callback
async def iterative_generation_function(request, worker, state_machines, logging_callback=None):
    """Generate actions iteratively with comprehensive logging via callback"""
    question = request['question']
    table = request['table']
    ground_truth_answers = request['ground_truth_answers']
    request_id = request['request_id']
    
    current_table = table
    action_history = []
    failures = 0
    validity_failures = 0
    max_failures = 3
    max_validity_failures = 3
    step = 0
    max_steps = 10
    
    generation_mode = worker._get_generation_mode_string()
    
    # Log initial table state
    if logging_callback:
        logging_callback(request_id, 0, "initial", current_table, generation_mode=generation_mode)
    
    while (failures < max_failures and 
           validity_failures < max_validity_failures and 
           step < max_steps):
        
        step_id = f"{request_id}_step{step}"
        
        # Build prompt
        max_chars = 1500 if step == 0 else 1000
        table_str = serialize_table_to_csv(current_table, max_chars)
        
        step_prompt = f"Table:\n{table_str}\n\n"
        step_prompt += f"Question: {question}\n"
        
        if action_history:
            step_prompt += "Actions taken so far:\n"
            for i, action in enumerate(action_history):
                step_prompt += f"{i+1}. {action}\n"
            step_prompt += "\n"
        
        if worker.use_constraints:
            instruction_prompt = "Next action: "
        else:
            instruction_prompt = "What should be the next action to answer this question? Choose from: select_row([row_indices]), select_column([\"column_names\"]), or end(). Next action: "
        # Truncate if needed
        estimated_length = len(step_prompt) // 4
        if estimated_length > worker.max_model_len - 100:
            table_str = serialize_table_to_csv(current_table, 500)
            question_short = question[:100] + "..." if len(question) > 100 else question
            step_prompt = f"Table:\n{table_str}\n\nQuestion: {question_short}\n"
            if worker.use_constraints:
                instruction_prompt = "Next action: "
            else:
                instruction_prompt = "What should be the next action? Choose from: select_row([row_indices]), select_column([\"column_names\"]), or end(). Next action: "
        
        step_prompt = model_config.add_instruct_tokens_for_instruct_models(step_prompt, instruction_prompt)        
        
        try:
            # Generate action
            action_str = await generate_single_action(worker, step_prompt, current_table, step_id, state_machines, action_history)
            
            # Parse action
            if worker.use_constraints:
                parsed_action = parse_action_string(action_str)
            else:
                parsed_action = parse_action_string(action_str)
                # Could add extraction fallback here
            if not parsed_action:
                validity_failures += 1
                print(f"Step {step}: Failed to generate valid action from: {action_str}")
                if logging_callback:
                    logging_callback(
                        request_id, step + 1, f"validity_failed:{action_str}", 
                        current_table, success=False, failure_type="validity_failure",
                        generation_mode=generation_mode
                    )
                continue
                
            action_name, args = parsed_action
            
            if action_name == "end":
                action_history.append("end()")
                if logging_callback:
                    logging_callback(request_id, step + 1, "end()", current_table, generation_mode=generation_mode)
                break
                
            # Apply action
            new_table = apply_action(current_table, action_name, args)
            
            if not new_table:
                validity_failures += 1
                print(f"Step {step}: Failed to apply action: {action_name}({args})")
                if logging_callback:
                    logging_callback(
                        request_id, step + 1, f"{action_name}({args})", 
                        current_table, success=False, failure_type="validity_failure",
                        generation_mode=generation_mode
                    )
                continue
                
            # Action successful
            current_table = new_table
            action_history.append(f"{action_name}({args})")
            failures = 0
            validity_failures = 0
            step += 1
            print(f"Step {step}: Applied {action_name}({args})")
            
            # Log successful transformation
            if logging_callback:
                logging_callback(
                    request_id, step, f"{action_name}({args})", 
                    current_table, success=True,
                    generation_mode=generation_mode
                )
            
        except Exception as e:
            failures += 1
            print(f"Step {step}: Generation error: {str(e)}")
            if logging_callback:
                logging_callback(
                    request_id, step + 1, f"generation_error:{str(e)}", 
                    current_table, success=False, failure_type="generation_error",
                    generation_mode=generation_mode
                )
    
    # Calculate accuracy
    accuracy_metrics = calculate_execution_accuracy_with_dataset_answers(
        action_history, current_table, ground_truth_answers, table
    )
    
    return {
        'action_history': action_history,
        'final_table': current_table,
        'execution_accuracy_metrics': accuracy_metrics
    }


async def cot_generation_function(request, worker, state_machines, logging_callback=None):
    """Chain-of-Table generation function with comprehensive logging via callback"""
    question = request['question']
    table = request['table']
    ground_truth_answers = request['ground_truth_answers']
    request_id = request['request_id']
    
    current_table = table
    action_history = []
    failures = 0
    validity_failures = 0
    max_failures = 3
    max_validity_failures = 3
    step = 0
    max_steps = 10
    
    generation_mode = "CoT"
    
    # Log initial table state
    if logging_callback:
        logging_callback(request_id, 0, "initial", current_table, generation_mode=generation_mode)
    
    while (failures < max_failures and 
           validity_failures < max_validity_failures and 
           step < max_steps):
        
        try:
            # Step 1: Dynamic Plan - Select action
            action_name = await generate_action_selection(model_config, worker, question, current_table, action_history, request_id, state_machines, step, COT_ACTION_TEMPERATURE)
            
            if not action_name:
                validity_failures += 1
                print(f"Step {step}: Failed to select valid action")
                if logging_callback:
                    logging_callback(
                        request_id, step + 1, "action_selection_failed", 
                        current_table, success=False, failure_type="validity_failure",
                        generation_mode=generation_mode
                    )
                continue
            
            if action_name == "end":
                action_history.append("end()")
                if logging_callback:
                    logging_callback(request_id, step + 1, "end()", current_table, generation_mode=generation_mode)
                break
            
            # Step 2: Generate Args
            args = await generate_action_arguments(model_config, worker, question, current_table, action_name, action_history, request_id, state_machines, step, COT_ARGS_TEMPERATURE)
            
            if args is None:
                validity_failures += 1
                print(f"Step {step}: Failed to generate valid arguments for {action_name}")
                if logging_callback:
                    logging_callback(
                        request_id, step + 1, f"{action_name}_args_failed", 
                        current_table, success=False, failure_type="validity_failure",
                        generation_mode=generation_mode
                    )
                continue
            
            # Apply action
            new_table = apply_action(current_table, action_name, args)
            
            if not new_table:
                validity_failures += 1
                print(f"Step {step}: Failed to apply action: {action_name}({args})")
                if logging_callback:
                    logging_callback(
                        request_id, step + 1, f"{action_name}({args})", 
                        current_table, success=False, failure_type="validity_failure",
                        generation_mode=generation_mode
                    )
                continue
            
            # Action successful
            current_table = new_table
            action_history.append(f"{action_name}({args})")
            failures = 0
            validity_failures = 0
            step += 1
            print(f"Step {step}: Applied {action_name}({args}) [CoT]")
            
            # Log successful transformation
            if logging_callback:
                logging_callback(
                    request_id, step, f"{action_name}({args})", 
                    current_table, success=True,
                    generation_mode=generation_mode
                )
            
        except Exception as e:
            failures += 1
            print(f"Step {step}: Generation error in CoT: {str(e)}")
            if logging_callback:
                logging_callback(
                    request_id, step + 1, f"cot_generation_error:{str(e)}", 
                    current_table, success=False, failure_type="generation_error",
                    generation_mode=generation_mode
                )
    
    # Calculate accuracy
    accuracy_metrics = calculate_execution_accuracy_with_dataset_answers(
        action_history, current_table, ground_truth_answers, table
    )
    
    return {
        'action_history': action_history,
        'final_table': current_table,
        'execution_accuracy_metrics': accuracy_metrics
    }


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
            'iterative_generation': iterative_generation_function,
            'cot_generation': cot_generation_function
        },
        logging_config=logging_config,
        tensor_parallel_size=model_config.tensor_parallel_size
    )
    
    # Initialize the server
    num_workers = 1
    server = ProcessParallelVLLM(
        model_id=model_id,
        num_workers=model_config.num_workers,
        gpu_allocation=model_config.gpu_allocation,
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
        subset_size = min(1000, len(dataset))
        
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