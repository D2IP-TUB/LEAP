
from typing import List, Tuple, Dict, Any, Optional
import re

from vllm import SamplingParams

from constraints import create_action_only_constraint_processor, create_constraint_logits_processor

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
                max_tokens=900, # increased for more row params
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


async def generate_action_selection(
    worker,
    question,
    table,
    action_history,
    request_id,
    state_machines,
    step,
    temperature,
    prompt_builder,
):
    """CoT Step 1: Dynamic Plan - Select which action to perform"""
    step_id = f"{request_id}_action_step{step}"
    prompt = prompt_builder.build_cot_action_prompt(
        question=question,
        table=table,
        action_history=action_history,
        worker=worker,
    )
    
    try:
        if worker.use_constraints:
            constraint_processor = create_action_only_constraint_processor(
                worker.tokenizer, step_id, state_machines
            )
            
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=20,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                logits_processors=[constraint_processor]
            )
        else:
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=30,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                stop=["\n", "Arguments", "Next"]
            )
        
        action_text = await worker.generate_text(prompt, step_id, sampling_params)
        return parse_action_name(action_text)
        
    except Exception as e:
        print(f"Action selection error in worker {worker.worker_id}: {e}")
        return None

async def generate_action_arguments(
    worker,
    question,
    table,
    action_name,
    action_history,
    request_id,
    state_machines,
    step,
    temperature,
    prompt_builder,
):
    """CoT Step 2: Generate Args - Generate arguments for the selected action"""
    step_id = f"{request_id}_args_step{step}"
    
    if action_name == "end":
        return []
    
    prompt = prompt_builder.build_cot_arguments_prompt(
        question=question,
        table=table,
        action_name=action_name,
        action_history=action_history,
        worker=worker,
    )
    
    try:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=900,
            stop_token_ids=[worker.tokenizer.eos_token_id],
            stop=["\n", "Next", "Step"]
        )
        
        args_text = await worker.generate_text(prompt, step_id, sampling_params)
        return extract_arguments_from_text(args_text, action_name, table)
        
    except Exception as e:
        print(f"Argument generation error in worker {worker.worker_id}: {e}")
        return None
