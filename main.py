import json
import time
import multiprocessing as mp
import ast
import os
import re
from typing import List, Tuple, Dict, Any, Optional
from datasets import load_dataset
from vllm import SamplingParams
from vllm_server import ProcessParallelVLLM, create_generation_config, create_logging_config
from constraints import create_action_only_constraint_processor, create_constraint_logits_processor
from eval import to_value_list, check_denotation

# Initialize environment
os.environ["VLLM_USE_V1"] = "0"


# Configuration
# model_id = "meta-llama/Llama-2-70b-hf"
model_id = "gpt2"
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
    'log_dir': 'table_logs_llama' if model_id == "meta-llama/Llama-2-70b-hf" else 'table_logs',
    'save_readable_tables': False,
    'compress_logs': False,
    'log_format': 'readable',
    'max_table_chars': 10000
}

# Load dataset
dataset = load_dataset('wikitablequestions', split='train[:1000]', trust_remote_code=True)


# Utility functions (table manipulation, parsing, etc.)
def serialize_table_to_csv(table, max_chars=1500):
    """Convert table to CSV string with limited characters"""
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(table['columns'])
    
    char_count = len(','.join(table['columns']))
    for i, row in enumerate(table['rows']):
        if i >= 10 or char_count > max_chars:
            break
        row_str = ','.join(str(cell) for cell in row)
        if char_count + len(row_str) > max_chars:
            break
        writer.writerow(row)
        char_count += len(row_str)
    return output.getvalue().strip()


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
        args_str = args_str.rstrip(')').strip()
        
        # Extract list arguments
        if args_str.startswith('[') and args_str.endswith(']'):
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


def apply_action(table: Dict[str, Any], action: str, args: List) -> Optional[Dict[str, Any]]:
    """Apply action to table and return new table state"""
    if action == "select_row":
        # Validate row indices
        valid_indices = []
        for idx in args:
            if isinstance(idx, int) and 0 <= idx < len(table['rows']):
                valid_indices.append(idx)
            elif isinstance(idx, str) and idx.isdigit():
                idx_int = int(idx)
                if 0 <= idx_int < len(table['rows']):
                    valid_indices.append(idx_int)
        
        if not valid_indices:
            return None
            
        # Create new table with selected rows
        new_rows = [table['rows'][i] for i in valid_indices]
        return {'columns': table['columns'], 'rows': new_rows}
    
    elif action == "select_column":
        valid_columns = [col for col in args if col in table['columns']]
        
        if not valid_columns:
            return None
            
        # Create new table with selected columns
        col_indices = [table['columns'].index(col) for col in valid_columns]
        new_rows = []
        for row in table['rows']:
            new_rows.append([row[i] for i in col_indices])
        
        return {'columns': valid_columns, 'rows': new_rows}
    
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


def extract_table_values_for_eval(table):
    """Extract all values from a table as a flat list of strings for evaluation"""
    if not table or not table.get('rows'):
        return []
    
    values = []
    for row in table['rows']:
        for cell in row:
            if cell is not None and str(cell).strip():
                values.append(str(cell).strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_values = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    
    return unique_values


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
            step_prompt += "Next action: "
        else:
            step_prompt += "What should be the next action to answer this question? Choose from: select_row([row_indices]), select_column([\"column_names\"]), or end(). Next action: "
        
        # Truncate if needed
        estimated_length = len(step_prompt) // 4
        if estimated_length > worker.max_model_len - 100:
            table_str = serialize_table_to_csv(current_table, 500)
            question_short = question[:100] + "..." if len(question) > 100 else question
            step_prompt = f"Table:\n{table_str}\n\nQuestion: {question_short}\n"
            if worker.use_constraints:
                step_prompt += "Next action: "
            else:
                step_prompt += "What should be the next action? Choose from: select_row([row_indices]), select_column([\"column_names\"]), or end(). Next action: "
        
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
            action_name = await generate_action_selection(worker, question, current_table, action_history, request_id, state_machines, step)
            
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
            args = await generate_action_arguments(worker, question, current_table, action_name, action_history, request_id, state_machines, step)
            
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


async def cot_generation_function(request, worker, state_machines, logging_context=None):
    """Chain-of-Table generation function with comprehensive logging"""
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
    if logging_context:
        logging_context.log_step("initial", current_table, generation_mode=generation_mode)
    
    while (failures < max_failures and 
           validity_failures < max_validity_failures and 
           step < max_steps):
        
        try:
            # Step 1: Dynamic Plan - Select action
            action_name = await generate_action_selection(worker, question, current_table, action_history, request_id, state_machines, step)
            
            if not action_name:
                validity_failures += 1
                print(f"Step {step}: Failed to select valid action")
                if logging_context:
                    logging_context.log_step(
                        "action_selection_failed", 
                        current_table, 
                        success=False, 
                        failure_type="validity_failure",
                        generation_mode=generation_mode
                    )
                continue
            
            if action_name == "end":
                action_history.append("end()")
                if logging_context:
                    logging_context.log_step("end()", current_table, generation_mode=generation_mode)
                break
            
            # Step 2: Generate Args
            args = await generate_action_arguments(worker, question, current_table, action_name, action_history, request_id, state_machines, step)
            
            if args is None:
                validity_failures += 1
                print(f"Step {step}: Failed to generate valid arguments for {action_name}")
                if logging_context:
                    logging_context.log_step(
                        f"{action_name}_args_failed", 
                        current_table, 
                        success=False, 
                        failure_type="validity_failure",
                        generation_mode=generation_mode
                    )
                continue
            
            # Apply action
            new_table = apply_action(current_table, action_name, args)
            
            if not new_table:
                validity_failures += 1
                print(f"Step {step}: Failed to apply action: {action_name}({args})")
                if logging_context:
                    logging_context.log_step(
                        f"{action_name}({args})", 
                        current_table, 
                        success=False, 
                        failure_type="validity_failure",
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
            if logging_context:
                logging_context.log_step(
                    f"{action_name}({args})", 
                    current_table, 
                    success=True,
                    generation_mode=generation_mode
                )
            
        except Exception as e:
            failures += 1
            print(f"Step {step}: Generation error in CoT: {str(e)}")
            if logging_context:
                logging_context.log_step(
                    f"cot_generation_error:{str(e)}", 
                    current_table, 
                    success=False, 
                    failure_type="generation_error",
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

async def generate_single_action(worker, prompt, table, request_id, state_machines, action_history=None):
    """
    Generate single action with or without constraints
    
    Args:
        worker: The worker process
        prompt: The prompt text
        table: The table data
        request_id: Unique request identifier
        state_machines: Dictionary of state machines
        action_history: List of previously executed actions (for global constraints)
    """
    try:
        if worker.use_constraints:
            # Get global constraints setting from worker's generation config
            use_global_constraints = worker.generation_config.get('use_global_constraints', True)
            
            # Pass action_history to the constraint processor
            constraint_processor = create_constraint_logits_processor(
                table, worker.tokenizer, request_id, state_machines, 
                action_history, use_global_constraints
            )
            
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=50,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                logits_processors=[constraint_processor]
            )
        else:
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=100,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                stop=["\n", "Next", "Step"]
            )
        
        return await worker.generate_text(prompt, request_id, sampling_params)
        
    except Exception as e:
        print(f"Generation error in worker {worker.worker_id}: {e}")
        return ""


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
            step_prompt += "Next action: "
        else:
            step_prompt += "What should be the next action to answer this question? Choose from: select_row([row_indices]), select_column([\"column_names\"]), or end(). Next action: "
        
        # Truncate if needed
        estimated_length = len(step_prompt) // 4
        if estimated_length > worker.max_model_len - 100:
            table_str = serialize_table_to_csv(current_table, 500)
            question_short = question[:100] + "..." if len(question) > 100 else question
            step_prompt = f"Table:\n{table_str}\n\nQuestion: {question_short}\n"
            if worker.use_constraints:
                step_prompt += "Next action: "
            else:
                step_prompt += "What should be the next action? Choose from: select_row([row_indices]), select_column([\"column_names\"]), or end(). Next action: "
        
        try:
            # Generate action - pass action_history for global constraints
            action_str = await generate_single_action(
                worker, step_prompt, current_table, step_id, state_machines, action_history
            )
            
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

async def generate_action_selection(worker, question, table, action_history, request_id, state_machines, step):
    """CoT Step 1: Dynamic Plan - Select which action to perform"""
    step_id = f"{request_id}_action_step{step}"
    
    prompt = build_dynamic_plan_prompt(question, table, action_history)
    
    estimated_length = len(prompt) // 4
    if estimated_length > worker.max_model_len - 50:
        table_str = serialize_table_to_csv(table, 500)
        question_short = question[:80] + "..." if len(question) > 80 else question
        prompt = f"Table:\n{table_str}\n\nQuestion: {question_short}\n\n"
        prompt += "Available actions: select_row, select_column, end\n"
        prompt += "What action should be performed next?\nAction: "
    
    try:
        if worker.use_constraints:
            constraint_processor = create_action_only_constraint_processor(
                worker.tokenizer, step_id, state_machines
            )
            
            sampling_params = SamplingParams(
                temperature=COT_ACTION_TEMPERATURE,
                max_tokens=20,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                logits_processors=[constraint_processor]
            )
        else:
            sampling_params = SamplingParams(
                temperature=COT_ACTION_TEMPERATURE,
                max_tokens=30,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                stop=["\n", "Arguments", "Next"]
            )
        
        action_text = await worker.generate_text(prompt, step_id, sampling_params)
        return parse_action_name(action_text)
        
    except Exception as e:
        print(f"Action selection error in worker {worker.worker_id}: {e}")
        return None


async def generate_action_arguments(worker, question, table, action_name, action_history, request_id, state_machines, step):
    """CoT Step 2: Generate Args - Generate arguments for the selected action"""
    step_id = f"{request_id}_args_step{step}"
    
    if action_name == "end":
        return []
    
    prompt = build_generate_args_prompt(question, table, action_name, action_history)
    
    estimated_length = len(prompt) // 4
    if estimated_length > worker.max_model_len - 100:
        table_str = serialize_table_to_csv(table, 600)
        question_short = question[:80] + "..." if len(question) > 80 else question
        
        prompt = f"Table:\n{table_str}\n\nQuestion: {question_short}\n\n"
        prompt += f"Selected action: {action_name}\n"
        
        if action_name == "select_row":
            prompt += f"Which row indices (0 to {len(table['rows'])-1})?\nRow indices: "
        else:  # select_column
            prompt += f"Which columns from {table['columns'][:3]}...?\nColumn names: "
    
    try:
        sampling_params = SamplingParams(
            temperature=COT_ARGS_TEMPERATURE,
            max_tokens=100,
            stop_token_ids=[worker.tokenizer.eos_token_id],
            stop=["\n", "Next", "Step"]
        )
        
        args_text = await worker.generate_text(prompt, step_id, sampling_params)
        return extract_arguments_from_text(args_text, action_name, table)
        
    except Exception as e:
        print(f"Argument generation error in worker {worker.worker_id}: {e}")
        return None


def parse_action_name(action_str: str) -> Optional[str]:
    """Parse action string and return only the action name"""
    action_str = action_str.strip().lower()
    
    if action_str in ["select_row", "select_column", "end"]:
        return action_str
    
    if "select_row" in action_str or "row" in action_str:
        return "select_row"
    elif "select_column" in action_str or "column" in action_str:
        return "select_column"
    elif "end" in action_str or "finish" in action_str or "done" in action_str:
        return "end"
    
    return None


def extract_arguments_from_text(text: str, action_name: str, table: Dict[str, Any]) -> Optional[List]:
    """Extract arguments for a specific action from free-form text"""
    text = text.strip()
    
    if action_name == "end":
        return []
    
    elif action_name == "select_row":
        patterns = [
            r'\[([0-9,\s]+)\]',
            r'(\d+(?:\s*,\s*\d+)*)',
            r'rows?\s+(\d+(?:\s*,\s*\d+)*)',
            r'indices?\s+(\d+(?:\s*,\s*\d+)*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    indices_str = match.group(1)
                    indices = [int(x.strip()) for x in indices_str.split(',')]
                    valid_indices = [idx for idx in indices if 0 <= idx < len(table['rows'])]
                    if valid_indices:
                        return valid_indices
                except:
                    continue
        
        numbers = re.findall(r'\b(\d+)\b', text)
        if numbers:
            try:
                indices = [int(x) for x in numbers]
                valid_indices = [idx for idx in indices if 0 <= idx < len(table['rows'])]
                if valid_indices:
                    return valid_indices[:5]
            except:
                pass
    
    elif action_name == "select_column":
        patterns = [
            r'\[(["\'][^"\']+["\'](?:\s*,\s*["\'][^"\']+["\'])*)\]',
            r'["\']([^"\']+)["\'](?:\s*,\s*["\']([^"\']+)["\'])*',
            r'columns?\s+(["\'][^"\']+["\'](?:\s*,\s*["\'][^"\']+["\'])*)',
        ]
        
        mentioned_columns = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        for col in match:
                            if col and col.strip('"\'') in table['columns']:
                                mentioned_columns.append(col.strip('"\''))
                    else:
                        col_matches = re.findall(r'["\']([^"\']+)["\']', match)
                        for col in col_matches:
                            if col in table['columns']:
                                mentioned_columns.append(col)
        
        if mentioned_columns:
            return list(set(mentioned_columns))
        
        for col in table['columns']:
            if col.lower() in text.lower():
                mentioned_columns.append(col)
        
        if mentioned_columns:
            return list(set(mentioned_columns[:3]))
    
    return None


def build_dynamic_plan_prompt(question: str, table: Dict[str, Any], action_history: List[str]) -> str:
    """Build prompt for dynamic_plan step in CoT"""
    max_chars = 1000
    table_str = serialize_table_to_csv(table, max_chars)
    
    prompt = f"Table:\n{table_str}\n\n"
    prompt += f"Question: {question}\n\n"
    
    if action_history:
        prompt += "Actions taken so far:\n"
        for i, action in enumerate(action_history):
            prompt += f"{i+1}. {action}\n"
        prompt += "\n"
    
    prompt += "Available actions: select_row, select_column, end\n"
    prompt += "What action should be performed next to answer the question?\n"
    prompt += "Action: "
    
    return prompt


def build_generate_args_prompt(question: str, table: Dict[str, Any], action_name: str, action_history: List[str]) -> str:
    """Build prompt for generate_args step in CoT"""
    max_chars = 1200
    table_str = serialize_table_to_csv(table, max_chars)
    
    prompt = f"Table:\n{table_str}\n\n"
    prompt += f"Question: {question}\n\n"
    
    if action_history:
        prompt += "Actions taken so far:\n"
        for i, action in enumerate(action_history):
            prompt += f"{i+1}. {action}\n"
        prompt += "\n"
    
    prompt += f"Selected action: {action_name}\n"
    
    if action_name == "select_row":
        prompt += "Which row indices should be selected? Provide the indices as a list, e.g., [0, 1, 2]\n"
        prompt += f"Available rows: 0 to {len(table['rows'])-1}\n"
        prompt += "Row indices: "
    elif action_name == "select_column":
        prompt += "Which columns should be selected? Provide the column names as a list, e.g., [\"Name\", \"Age\"]\n"
        prompt += f"Available columns: {table['columns']}\n"
        prompt += "Column names: "
    else:
        prompt += "No arguments needed for end action.\n"
        prompt += "Arguments: "
    
    return prompt


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
        logging_config=logging_config
    )
    
    # Initialize the server
    num_workers = 4
    server = ProcessParallelVLLM(
        model_id=model_id,
        num_workers=num_workers,
        gpu_allocation=[2, 3, 4, 5],
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