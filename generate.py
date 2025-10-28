
from typing import List, Tuple, Dict, Any, Optional
import re

from vllm import SamplingParams

from constraints import create_action_only_constraint_processor, create_constraint_logits_processor
from table import serialize_table_to_csv

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
                max_tokens=1000, # increased for more row params
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


async def generate_action_selection(worker, question, table, action_history, request_id, state_machines, step, temperature):
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
                temperature,
                max_tokens=20,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                logits_processors=[constraint_processor]
            )
        else:
            sampling_params = SamplingParams(
                temperature,
                max_tokens=30,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                stop=["\n", "Arguments", "Next"]
            )
        
        action_text = await worker.generate_text(prompt, step_id, sampling_params)
        return parse_action_name(action_text)
        
    except Exception as e:
        print(f"Action selection error in worker {worker.worker_id}: {e}")
        return None

async def generate_action_arguments(worker, question, table, action_name, action_history, request_id, state_machines, step, temperature):
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
            temperature,
            max_tokens=100,
            stop_token_ids=[worker.tokenizer.eos_token_id],
            stop=["\n", "Next", "Step"]
        )
        
        args_text = await worker.generate_text(prompt, step_id, sampling_params)
        return extract_arguments_from_text(args_text, action_name, table)
        
    except Exception as e:
        print(f"Argument generation error in worker {worker.worker_id}: {e}")
        return None
