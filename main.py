import json
from datasets import load_dataset
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import csv
import io
import asyncio
import time
import uuid
import multiprocessing as mp
import queue
import ast
import os
import re
from typing import List, Tuple, Dict, Any, Optional
from constraints import create_action_only_constraint_processor, create_constraint_logits_processor
from eval import to_value_list, check_denotation

# Initialize environment
os.environ["VLLM_USE_V1"] = "0"  # Set this explicitly to avoid conflicts

# Load dataset
dataset = load_dataset('wikitablequestions', split='train[:1000]')
model_id = "gpt2"
output_file = "parallel_results.jsonl"

# Configuration flags
USE_GENERATION_CONSTRAINTS = True  # Set to False to disable constraints and use post-processing
ENABLE_TABLE_LOGGING = True
TABLE_LOG_DIR = "table_logs"
COMPRESS_TABLE_LOGS = False  # Disabled for easy reading
LOG_FORMAT = "readable"  # Options: "json", "csv", "readable", "pickle"
SAVE_READABLE_TABLES = True  # Save tables as easy-to-read CSV files
 
USE_CHAIN_OF_TABLE = False  # Set to True to enable CoT-style generation (dynamic_plan + generate_args)
COT_ACTION_TEMPERATURE = 0.3  # Lower temperature for action selection
COT_ARGS_TEMPERATURE = 0.7   # Higher temperature for argument generation

def setup_logging_directory():
    """Create logging directory if it doesn't exist"""
    if ENABLE_TABLE_LOGGING and not os.path.exists(TABLE_LOG_DIR):
        os.makedirs(TABLE_LOG_DIR)
        print(f"Created table logging directory: {TABLE_LOG_DIR}")

def serialize_table_to_csv(table, max_chars=1500):
    """Convert table to CSV string with limited characters"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(table['columns'])
    
    char_count = len(','.join(table['columns']))
    for i, row in enumerate(table['rows']):
        if i >= 10 or char_count > max_chars:
            break
        row_str = ','.join(row)
        if char_count + len(row_str) > max_chars:
            break
        writer.writerow(row)
        char_count += len(row_str)
    return output.getvalue().strip()

def get_table_summary(table):
    """Get a compact summary of table dimensions"""
    return {
        "num_rows": len(table['rows']),
        "num_columns": len(table['columns']),
        "columns": table['columns'][:5] + ["..."] if len(table['columns']) > 5 else table['columns']
    }

def log_table_state(request_id, step, action, table, success=True, failure_type=None, generation_mode=None):
    """Log table state in human-readable format with failure type tracking"""
    if not ENABLE_TABLE_LOGGING:
        return
    
    try:
        log_entry = {
            "request_id": request_id,
            "step": step,
            "action": action,
            "timestamp": time.time(),
            "success": success,
            "failure_type": failure_type,
            "generation_mode": generation_mode,  # NEW: Track which generation mode was used
            "table_summary": get_table_summary(table)
        }
        
        # Create readable table representation
        if SAVE_READABLE_TABLES:
            # Save table as CSV for easy viewing
            table_csv = serialize_table_to_csv(table, max_chars=10000)  # Larger limit for logging
            log_entry["table_preview"] = table_csv.split('\n')[:6]  # First 5 rows + header for JSON log
            
            # Save full table as separate CSV file
            clean_action = action.replace('(', '_').replace(')', '').replace('[', '').replace(']', '').replace('"', '').replace(',', '_').replace(' ', '_')
            clean_action = clean_action[:50]  # Limit length
            table_filename = f"{request_id}_step{step:02d}_{clean_action}.csv"
            table_path = os.path.join(TABLE_LOG_DIR, table_filename)
            
            with open(table_path, 'w', encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(table['columns'])
                for row in table['rows']:
                    writer.writerow(row)
        
        # Write to main log file
        log_filename = f"{request_id}_log.json"
        log_path = os.path.join(TABLE_LOG_DIR, log_filename)
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, indent=2) + '\n' + '-'*80 + '\n')
                
    except Exception as e:
        print(f"Warning: Failed to log table state: {e}")

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

# NEW: Parse action name only (for CoT dynamic_plan step)
def parse_action_name(action_str: str) -> Optional[str]:
    """Parse action string and return only the action name"""
    action_str = action_str.strip().lower()
    
    # Direct action name matches
    if action_str in ["select_row", "select_column", "end"]:
        return action_str
    
    # Pattern matching for partial matches
    if "select_row" in action_str or "row" in action_str:
        return "select_row"
    elif "select_column" in action_str or "column" in action_str:
        return "select_column"
    elif "end" in action_str or "finish" in action_str or "done" in action_str:
        return "end"
    
    return None

def extract_action_from_text(text: str, table: Dict[str, Any]) -> Optional[Tuple[str, List]]:
    """
    Extract action from free-form text using regex patterns and table context.
    This is used when generation constraints are disabled.
    """
    text = text.strip()
    
    # Pattern 1: Direct action format like "select_row([0, 1, 2])" or "select_column(["col1", "col2"])"
    action_pattern = r'(select_row|select_column|end)\s*\(\s*(\[.*?\]|\d+|".*?")\s*\)'
    match = re.search(action_pattern, text, re.IGNORECASE)
    
    if match:
        action_name = match.group(1).lower()
        args_str = match.group(2)
        
        try:
            if args_str.startswith('[') and args_str.endswith(']'):
                args_list = ast.literal_eval(args_str)
            elif args_str.isdigit():
                args_list = [int(args_str)]
            elif args_str.startswith('"') and args_str.endswith('"'):
                args_list = [args_str.strip('"')]
            else:
                args_list = [args_str]
            
            return action_name, args_list
        except:
            pass
    
    # Pattern 2: Natural language patterns
    text_lower = text.lower()
    
    # Check for "end" action
    if any(phrase in text_lower for phrase in ["end", "finish", "done", "stop", "complete"]):
        return "end", []
    
    # Check for row selection
    row_patterns = [
        r'select\s+row[s]?\s*(\d+(?:\s*,\s*\d+)*)',
        r'row[s]?\s*(\d+(?:\s*,\s*\d+)*)',
        r'keep\s+row[s]?\s*(\d+(?:\s*,\s*\d+)*)',
        r'filter\s+row[s]?\s*(\d+(?:\s*,\s*\d+)*)'
    ]
    
    for pattern in row_patterns:
        match = re.search(pattern, text_lower)
        if match:
            row_indices_str = match.group(1)
            try:
                row_indices = [int(x.strip()) for x in row_indices_str.split(',')]
                # Validate row indices
                valid_indices = [idx for idx in row_indices if 0 <= idx < len(table['rows'])]
                if valid_indices:
                    return "select_row", valid_indices
            except:
                pass
    
    # Check for column selection
    col_patterns = [
        r'select\s+column[s]?\s*["\']([^"\']+)["\']',
        r'column[s]?\s*["\']([^"\']+)["\']',
        r'keep\s+column[s]?\s*["\']([^"\']+)["\']',
        r'filter\s+column[s]?\s*["\']([^"\']+)["\']'
    ]
    
    for pattern in col_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Validate column names
            valid_cols = [col for col in matches if col in table['columns']]
            if valid_cols:
                return "select_column", valid_cols
    
    # Pattern 3: Try to find column names mentioned in the text
    mentioned_columns = []
    for col in table['columns']:
        if col.lower() in text_lower:
            mentioned_columns.append(col)
    
    if mentioned_columns:
        # If multiple columns mentioned, assume column selection
        if len(mentioned_columns) > 1:
            return "select_column", mentioned_columns
        # If single column and some selection keywords, assume column selection
        elif any(word in text_lower for word in ["select", "choose", "pick", "keep", "filter"]):
            return "select_column", mentioned_columns
    
    # Pattern 4: Try to find row numbers mentioned
    row_numbers = re.findall(r'\b(\d+)\b', text)
    if row_numbers:
        try:
            row_indices = [int(x) for x in row_numbers]
            valid_indices = [idx for idx in row_indices if 0 <= idx < len(table['rows'])]
            if valid_indices and any(word in text_lower for word in ["row", "select", "choose", "pick", "keep", "filter"]):
                return "select_row", valid_indices
        except:
            pass
    
    return None

# NEW: Extract arguments for a given action from text (for CoT generate_args step)
def extract_arguments_from_text(text: str, action_name: str, table: Dict[str, Any]) -> Optional[List]:
    """
    Extract arguments for a specific action from free-form text.
    This is used in CoT generate_args step.
    """
    text = text.strip()
    
    if action_name == "end":
        return []
    
    elif action_name == "select_row":
        # Look for row indices in various formats
        patterns = [
            r'\[([0-9,\s]+)\]',  # [1, 2, 3]
            r'(\d+(?:\s*,\s*\d+)*)',  # 1, 2, 3
            r'rows?\s+(\d+(?:\s*,\s*\d+)*)',  # rows 1, 2, 3
            r'indices?\s+(\d+(?:\s*,\s*\d+)*)',  # indices 1, 2, 3
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    indices_str = match.group(1)
                    indices = [int(x.strip()) for x in indices_str.split(',')]
                    # Validate indices
                    valid_indices = [idx for idx in indices if 0 <= idx < len(table['rows'])]
                    if valid_indices:
                        return valid_indices
                except:
                    continue
        
        # Look for individual numbers
        numbers = re.findall(r'\b(\d+)\b', text)
        if numbers:
            try:
                indices = [int(x) for x in numbers]
                valid_indices = [idx for idx in indices if 0 <= idx < len(table['rows'])]
                if valid_indices:
                    return valid_indices[:5]  # Limit to first 5 to avoid too many
            except:
                pass
    
    elif action_name == "select_column":
        # Look for column names in various formats
        patterns = [
            r'\[(["\'][^"\']+["\'](?:\s*,\s*["\'][^"\']+["\'])*)\]',  # ["col1", "col2"]
            r'["\']([^"\']+)["\'](?:\s*,\s*["\']([^"\']+)["\'])*',  # "col1", "col2"
            r'columns?\s+(["\'][^"\']+["\'](?:\s*,\s*["\'][^"\']+["\'])*)',  # columns "col1", "col2"
        ]
        
        mentioned_columns = []
        
        # Try structured patterns first
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        for col in match:
                            if col and col.strip('"\'') in table['columns']:
                                mentioned_columns.append(col.strip('"\''))
                    else:
                        # Parse the match string
                        col_matches = re.findall(r'["\']([^"\']+)["\']', match)
                        for col in col_matches:
                            if col in table['columns']:
                                mentioned_columns.append(col)
        
        if mentioned_columns:
            return list(set(mentioned_columns))  # Remove duplicates
        
        # Fallback: look for any column names mentioned in text
        for col in table['columns']:
            if col.lower() in text.lower():
                mentioned_columns.append(col)
        
        if mentioned_columns:
            return list(set(mentioned_columns[:3]))  # Limit to first 3 unique columns
    
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
        return {
            'columns': table['columns'],
            'rows': new_rows
        }
    
    elif action == "select_column":
        # Validate column names
        valid_columns = []
        for col in args:
            if col in table['columns']:
                valid_columns.append(col)
        
        if not valid_columns:
            return None
            
        # Create new table with selected columns
        col_indices = [table['columns'].index(col) for col in valid_columns]
        new_rows = []
        for row in table['rows']:
            new_rows.append([row[i] for i in col_indices])
        
        return {
            'columns': valid_columns,
            'rows': new_rows
        }
    
    return None

def calculate_execution_accuracy_with_dataset_answers(action_history, final_table, ground_truth_answers, original_table):
    """
    Calculate execution accuracy using WikiTableQuestions evaluator logic with dataset answers
    
    Args:
        action_history: List of action strings
        final_table: Final table state after applying actions  
        ground_truth_answers: List of ground truth answer strings from dataset
        original_table: Original table before transformations
    
    Returns:
        Dict with accuracy metrics
    """
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
            len(action_history) > 0 and 
            action_history[-1].startswith('end')
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
    """
    Find which target answers have matches in predicted values
    
    Args:
        target_values: List of Value objects (ground truth)
        predicted_values: List of Value objects (from table)
    
    Returns:
        List of original answer strings that were matched
    """
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

# NEW: Chain-of-Table prompt builders
def build_dynamic_plan_prompt(question: str, table: Dict[str, Any], action_history: List[str]) -> str:
    """Build prompt for dynamic_plan step in CoT"""
    max_chars = 1000  # Shorter table for action selection
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
    max_chars = 1200  # Slightly longer table for argument generation
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
    else:  # end
        prompt += "No arguments needed for end action.\n"
        prompt += "Arguments: "
    
    return prompt

class VLLMWorkerProcess(mp.Process):
    """Individual worker process for vLLM inference"""
    
    def __init__(self, worker_id: int, gpu_ids: List[int], model_id: str, 
                 input_queue: mp.Queue, output_queue: mp.Queue, use_constraints: bool = True, use_cot: bool = False):
        super().__init__()
        self.worker_id = worker_id
        self.gpu_ids = gpu_ids
        self.model_id = model_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.use_constraints = use_constraints
        self.use_cot = use_cot  # NEW: Chain-of-Table flag
        self.engine = None
        self.tokenizer = None
        self.model_loaded = mp.Event()
        
    def run(self):
        """Main worker process loop"""
        try:
            # Set GPU visibility for this process
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
            
            generation_mode = "CoT" if self.use_cot else ("constrained" if self.use_constraints else "unconstrained")
            print(f"Worker {self.worker_id} starting with GPUs: {self.gpu_ids}, mode: {generation_mode}")
            
            # Initialize engine
            self._init_engine()
            
            # Signal model loaded
            self.model_loaded.set()
            print(f"Worker {self.worker_id}: Model loaded successfully")
            
            # Send ready signal
            self.output_queue.put(('ready', self.worker_id, None))
            
            # Process requests
            asyncio.run(self._process_requests())
            
        except Exception as e:
            print(f"Worker {self.worker_id} failed: {e}")
            self.output_queue.put(('error', self.worker_id, str(e)))
    
    def _init_engine(self):
        """Initialize the vLLM engine"""
        print(f"Worker {self.worker_id}: Starting model loading...")
        start_time = time.time()
        
        engine_args = AsyncEngineArgs(
            model=self.model_id,
            trust_remote_code=True,
            max_model_len=1024,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=1,  # Single GPU per worker for simplicity
            max_num_batched_tokens=8192,
            max_num_seqs=32,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        tokenizer_group = self.engine.engine.tokenizer
        self.tokenizer = tokenizer_group.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_model_len = self.engine.engine.model_config.max_model_len
        
        load_time = time.time() - start_time
        print(f"Worker {self.worker_id}: Model loaded in {load_time:.2f} seconds, max_model_len={self.max_model_len}")
    
    async def _process_requests(self):
        """Process incoming requests"""
        state_machines = {}
        
        while True:
            try:
                # Get request from queue
                try:
                    request = self.input_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                
                if request is None:  # Shutdown signal
                    break
                
                
                
                # Unpack request to get ground truth answers
                request_id, question, table, ground_truth_answers = request

                if self.use_cot:
                    action_history, final_table = await self._generate_cot_actions(question, table, request_id, state_machines)
                else:
                    action_history, final_table = await self._generate_iterative_actions(question, table, request_id, state_machines)

                # Calculate execution accuracy using evaluator logic with dataset answers
                accuracy_metrics = calculate_execution_accuracy_with_dataset_answers(
                    action_history, final_table, ground_truth_answers, table
                )

                result = {
                    'action_history': action_history,
                    'final_table': final_table,
                    'execution_accuracy_metrics': accuracy_metrics
                }

                # Send result back
                self.output_queue.put(('result', request_id, result))
                
                # Clean up state machines
                to_delete = [key for key in state_machines if key.startswith(request_id)]
                for key in to_delete:
                    del state_machines[key]
                    
            except Exception as e:
                print(f"Worker {self.worker_id} error processing request: {e}")
                self.output_queue.put(('error', self.worker_id, str(e)))
    
    # NEW: Chain-of-Table generation method
    async def _generate_cot_actions(self, question, table, request_id, state_machines):
        """Generate actions using Chain-of-Table approach (dynamic_plan + generate_args)"""
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
        log_table_state(request_id, 0, "initial", current_table, generation_mode=generation_mode)
        
        while (failures < max_failures and 
               validity_failures < max_validity_failures and 
               step < max_steps):
            
            try:
                # Step 1: Dynamic Plan - Select action
                action_name = await self._generate_action_selection(question, current_table, action_history, request_id, state_machines, step)
                
                if not action_name:
                    validity_failures += 1
                    print(f"Step {step}: Failed to select valid action")
                    log_table_state(request_id, step + 1, "action_selection_failed", current_table, success=False, failure_type="validity_failure", generation_mode=generation_mode)
                    continue
                
                # Handle end action
                if action_name == "end":
                    action_history.append("end()")
                    log_table_state(request_id, step + 1, "end()", current_table, generation_mode=generation_mode)
                    break
                
                # Step 2: Generate Args - Get arguments for the selected action
                args = await self._generate_action_arguments(question, current_table, action_name, action_history, request_id, state_machines, step)
                
                if args is None:
                    validity_failures += 1
                    print(f"Step {step}: Failed to generate valid arguments for {action_name}")
                    log_table_state(request_id, step + 1, f"{action_name}_args_failed", current_table, success=False, failure_type="validity_failure", generation_mode=generation_mode)
                    continue
                
                # Apply action to table
                new_table = apply_action(current_table, action_name, args)
                
                if not new_table:
                    validity_failures += 1
                    print(f"Step {step}: Failed to apply action: {action_name}({args})")
                    log_table_state(request_id, step + 1, f"{action_name}({args})", current_table, success=False, failure_type="validity_failure", generation_mode=generation_mode)
                    continue
                
                # Action successful
                current_table = new_table
                action_history.append(f"{action_name}({args})")
                failures = 0
                validity_failures = 0
                step += 1
                print(f"Step {step}: Applied {action_name}({args}) [CoT]")
                
                # Log successful table transformation
                log_table_state(request_id, step, f"{action_name}({args})", current_table, generation_mode=generation_mode)
                
            except Exception as e:
                failures += 1
                print(f"Step {step}: Generation error in CoT: {str(e)}")
                log_table_state(request_id, step + 1, f"cot_generation_error:{str(e)}", current_table, success=False, failure_type="generation_error", generation_mode=generation_mode)
        
        return action_history, current_table
    
    async def _generate_action_selection(self, question, table, action_history, request_id, state_machines, step):
        """CoT Step 1: Dynamic Plan - Select which action to perform"""
        step_id = f"{request_id}_action_step{step}"
        
        # Build prompt for action selection
        prompt = build_dynamic_plan_prompt(question, table, action_history)
        
        # Truncate if needed
        estimated_length = len(prompt) // 4
        if estimated_length > self.max_model_len - 50:
            # More aggressive truncation for action selection
            table_str = serialize_table_to_csv(table, 500)
            question_short = question[:80] + "..." if len(question) > 80 else question
            prompt = f"Table:\n{table_str}\n\nQuestion: {question_short}\n\n"
            prompt += "Available actions: select_row, select_column, end\n"
            prompt += "What action should be performed next?\nAction: "
        
        try:
            if self.use_constraints:
                # Use action-only constraints
                constraint_processor = create_action_only_constraint_processor(
                    self.tokenizer, step_id, state_machines
                )
                
                sampling_params = SamplingParams(
                    temperature=COT_ACTION_TEMPERATURE,
                    max_tokens=20,
                    stop_token_ids=[self.tokenizer.eos_token_id],
                    logits_processors=[constraint_processor]
                )
            else:
                # Free generation for action selection
                sampling_params = SamplingParams(
                    temperature=COT_ACTION_TEMPERATURE,
                    max_tokens=30,
                    stop_token_ids=[self.tokenizer.eos_token_id],
                    stop=["\n", "Arguments", "Next"]
                )
            
            # Generate action
            result_generator = self.engine.generate(prompt, sampling_params, step_id)
            
            final_result = None
            async for result in result_generator:
                final_result = result
            
            if final_result and final_result.outputs:
                action_text = final_result.outputs[0].text.strip()
                return parse_action_name(action_text)
            
            return None
            
        except Exception as e:
            print(f"Action selection error in worker {self.worker_id}: {e}")
            return None
    
    async def _generate_action_arguments(self, question, table, action_name, action_history, request_id, state_machines, step):
        """CoT Step 2: Generate Args - Generate arguments for the selected action"""
        step_id = f"{request_id}_args_step{step}"
        
        # Handle end action (no arguments needed)
        if action_name == "end":
            return []
        
        # Build prompt for argument generation
        prompt = build_generate_args_prompt(question, table, action_name, action_history)
        
        # Truncate if needed
        estimated_length = len(prompt) // 4
        if estimated_length > self.max_model_len - 100:
            table_str = serialize_table_to_csv(table, 600)
            question_short = question[:80] + "..." if len(question) > 80 else question
            
            prompt = f"Table:\n{table_str}\n\nQuestion: {question_short}\n\n"
            prompt += f"Selected action: {action_name}\n"
            
            if action_name == "select_row":
                prompt += f"Which row indices (0 to {len(table['rows'])-1})?\nRow indices: "
            else:  # select_column
                prompt += f"Which columns from {table['columns'][:3]}...?\nColumn names: "
        
        try:
            # Always use free generation for argument generation (more flexible)
            sampling_params = SamplingParams(
                temperature=COT_ARGS_TEMPERATURE,
                max_tokens=100,
                stop_token_ids=[self.tokenizer.eos_token_id],
                stop=["\n", "Next", "Step"]
            )
            
            # Generate arguments
            result_generator = self.engine.generate(prompt, sampling_params, step_id)
            
            final_result = None
            async for result in result_generator:
                final_result = result
            
            if final_result and final_result.outputs:
                args_text = final_result.outputs[0].text.strip()
                return extract_arguments_from_text(args_text, action_name, table)
            
            return None
            
        except Exception as e:
            print(f"Argument generation error in worker {self.worker_id}: {e}")
            return None
    
    async def _generate_iterative_actions(self, question, table, request_id, state_machines):
        """Generate actions iteratively and process table with enhanced logging (original method)"""
        current_table = table
        action_history = []
        failures = 0
        validity_failures = 0
        max_failures = 3
        max_validity_failures = 3
        step = 0
        max_steps = 10
        
        generation_mode = "constrained" if self.use_constraints else "unconstrained"
        
        # Log initial table state
        log_table_state(request_id, 0, "initial", current_table, generation_mode=generation_mode)
        
        while (failures < max_failures and 
               validity_failures < max_validity_failures and 
               step < max_steps):
            step_id = f"{request_id}_step{step}"
            
            # Build step-specific prompt with progressively shorter tables
            max_chars = 1500 if step == 0 else 1000
            table_str = serialize_table_to_csv(current_table, max_chars)
            
            step_prompt = f"Table:\n{table_str}\n\n"
            step_prompt += f"Question: {question}\n"
            
            if action_history:
                step_prompt += "Actions taken so far:\n"
                for i, action in enumerate(action_history):
                    step_prompt += f"{i+1}. {action}\n"
                step_prompt += "\n"
            
            if self.use_constraints:
                step_prompt += "Next action: "
            else:
                step_prompt += "What should be the next action to answer this question? Choose from: select_row([row_indices]), select_column([\"column_names\"]), or end(). Next action: "
            
            # Estimate token length (conservative)
            estimated_length = len(step_prompt) // 4
            
            # Truncate prompt if needed
            if estimated_length > self.max_model_len - 100:
                table_str = serialize_table_to_csv(current_table, 500)
                question_short = question[:100] + "..." if len(question) > 100 else question
                step_prompt = f"Table:\n{table_str}\n\n"
                step_prompt += f"Question: {question_short}\n"
                if self.use_constraints:
                    step_prompt += "Next action: "
                else:
                    step_prompt += "What should be the next action? Choose from: select_row([row_indices]), select_column([\"column_names\"]), or end(). Next action: "
                print(f"Worker {self.worker_id}: Truncated prompt for step {step} (estimated: {estimated_length} tokens)")
            
            try:
                # Generate action for current step
                action_str = await self._generate_single_action(step_prompt, current_table, step_id, state_machines)
                
                # Parse the action (either from constrained generation or post-processing)
                if self.use_constraints:
                    parsed_action = parse_action_string(action_str)
                else:
                    # Try structured parsing first, then fallback to text extraction
                    parsed_action = parse_action_string(action_str)
                    if not parsed_action:
                        parsed_action = extract_action_from_text(action_str, current_table)
                
                # Both parse failures and extraction failures are validity failures
                if not parsed_action:
                    validity_failures += 1
                    print(f"Step {step}: Failed to generate valid action from: {action_str}")
                    log_table_state(request_id, step + 1, f"validity_failed:{action_str}", current_table, success=False, failure_type="validity_failure", generation_mode=generation_mode)
                    continue
                    
                action_name, args = parsed_action
                
                # Handle end action
                if action_name == "end":
                    action_history.append("end()")
                    log_table_state(request_id, step + 1, "end()", current_table, generation_mode=generation_mode)
                    break
                    
                # Apply action to table
                new_table = apply_action(current_table, action_name, args)
                
                if not new_table:
                    validity_failures += 1
                    print(f"Step {step}: Failed to apply action: {action_name}({args})")
                    log_table_state(request_id, step + 1, f"{action_name}({args})", current_table, success=False, failure_type="validity_failure", generation_mode=generation_mode)
                    continue
                    
                # Action successful
                current_table = new_table
                action_history.append(f"{action_name}({args})")
                failures = 0
                validity_failures = 0
                step += 1
                print(f"Step {step}: Applied {action_name}({args})")
                
                # Log successful table transformation
                log_table_state(request_id, step, f"{action_name}({args})", current_table, generation_mode=generation_mode)
                
            except Exception as e:
                failures += 1
                print(f"Step {step}: Generation error: {str(e)}")
                log_table_state(request_id, step + 1, f"generation_error:{str(e)}", current_table, success=False, failure_type="generation_error", generation_mode=generation_mode)
            
        return action_history, current_table
    
    async def _generate_single_action(self, prompt, table, request_id, state_machines):
        """Generate single action with or without constraints (original method)"""
        try:
            if self.use_constraints:
                # Create constraint processor
                constraint_processor = create_constraint_logits_processor(
                    table, self.tokenizer, request_id, state_machines
                )
                
                sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=50,
                    stop_token_ids=[self.tokenizer.eos_token_id],
                    logits_processors=[constraint_processor]
                )
            else:
                # No constraints, allow free generation
                sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=100,
                    stop_token_ids=[self.tokenizer.eos_token_id],
                    stop=["\n", "Next", "Step"]
                )
            
            # Generate response
            result_generator = self.engine.generate(prompt, sampling_params, request_id)
            
            final_result = None
            async for result in result_generator:
                final_result = result
            
            if final_result and final_result.outputs:
                return final_result.outputs[0].text.strip()
            else:
                return ""
                
        except Exception as e:
            print(f"Generation error in worker {self.worker_id}: {e}")
            return ""

class ProcessParallelVLLM:
    """Process-based parallel vLLM engine"""
    
    def __init__(self, model_id: str, num_workers: int = 4, use_constraints: bool = True, use_cot: bool = False):
        self.model_id = model_id
        self.num_workers = num_workers
        self.use_constraints = use_constraints
        self.use_cot = use_cot  # NEW: Chain-of-Table flag
        self.workers = []
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        
        # GPU allocation
        available_gpus = [1, 2, 3, 4, 5, 6, 7]
        
        # Create worker processes
        for worker_id in range(num_workers):
            gpu_id = [available_gpus[worker_id % len(available_gpus)]]
            worker = VLLMWorkerProcess(
                worker_id=worker_id,
                gpu_ids=gpu_id,
                model_id=model_id,
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                use_constraints=use_constraints,
                use_cot=use_cot  # NEW: Pass CoT flag to workers
            )
            self.workers.append(worker)
    
    def start_workers(self):
        """Start all worker processes and wait for models to load"""
        print(f"Starting {self.num_workers} worker processes...")
        
        # Determine generation mode for logging
        if self.use_cot:
            mode_desc = "Chain-of-Table (dynamic_plan + generate_args)"
        elif self.use_constraints:
            mode_desc = "Constrained generation"
        else:
            mode_desc = "Unconstrained generation with post-processing"
        
        print(f"Generation mode: {mode_desc}")
        
        for worker in self.workers:
            worker.start()
        
        # Wait for all workers to load models and be ready
        ready_count = 0
        start_time = time.time()
        timeout = 300  # 5 minutes timeout for model loading
        
        while ready_count < self.num_workers:
            try:
                msg_type, worker_id, data = self.output_queue.get(timeout=timeout)
                if msg_type == 'ready':
                    print(f"Worker {worker_id} is ready")
                    ready_count += 1
                elif msg_type == 'error':
                    print(f"Worker {worker_id} failed: {data}")
            except queue.Empty:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    print(f"Timeout waiting for workers to load models ({timeout}s)")
                    break
        
        if ready_count == self.num_workers:
            print(f"All {self.num_workers} workers are ready")
            return True
        else:
            print(f"Only {ready_count}/{self.num_workers} workers ready")
            return False
    
    def generate_batch(self, questions_tables_and_answers: List[Tuple[str, Dict, List[str]]]) -> List[Dict]:
        """Generate responses for batch of questions"""
        # Send all requests to queue
        request_ids = []
        for i, (question, table, ground_truth_answers) in enumerate(questions_tables_and_answers):
            request_id = f"req_{i}_{uuid.uuid4().hex}"
            request_ids.append(request_id)
            self.input_queue.put((request_id, question, table, ground_truth_answers))
        
        # Collect results
        results = {}
        completed = 0
        total_requests = len(request_ids)
        
        while completed < total_requests:
            try:
                msg_type, req_id, data = self.output_queue.get(timeout=180)
                if msg_type == 'result':
                    results[req_id] = data
                    completed += 1
                    print(f"Completed {completed}/{total_requests} requests")
                elif msg_type == 'error':
                    print(f"Error: {data}")
            except queue.Empty:
                print("Timeout waiting for results")
                break
        
        # Return results in original order
        return [results.get(req_id, {'action_history': [], 'final_table': None, 'execution_accuracy_metrics': {}}) for req_id in request_ids]
    
    def shutdown(self):
        """Shutdown all worker processes"""
        print("Shutting down workers...")
        
        # Send shutdown signals
        for _ in self.workers:
            self.input_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=30)
            if worker.is_alive():
                print(f"Force terminating worker {worker.worker_id}")
                worker.terminate()
                worker.join()

def write_results_to_jsonl(results, examples, output_file):
    """Write results to JSONL file with WikiTableQuestions logic-based execution accuracy metrics"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (result, example) in enumerate(zip(results, examples)):
            action_history = result.get('action_history', [])
            metrics = result.get('execution_accuracy_metrics', {})
            
            # Convert action history to list of action objects
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
                    "has_table_logs": ENABLE_TABLE_LOGGING,
                    "generation_mode": get_generation_mode_string(),
                    "evaluation_method": "wikitablequestions_logic_with_dataset_answers",
                    "log_file": f"req_{i}_{uuid.uuid4().hex}_log.json" if ENABLE_TABLE_LOGGING else None,
                    "table_files": f"req_{i}_***.csv files in {TABLE_LOG_DIR}/" if SAVE_READABLE_TABLES else None
                }
            }
            f.write(json.dumps(entry) + '\n')
    
    print(f"Results with execution accuracy (WikiTableQuestions logic + dataset answers) written to {output_file}")

# NEW: Helper function to get generation mode string
def get_generation_mode_string():
    """Get a descriptive string for the current generation mode"""
    if USE_CHAIN_OF_TABLE:
        constraint_desc = "with_constraints" if USE_GENERATION_CONSTRAINTS else "without_constraints"
        return f"chain_of_table_{constraint_desc}"
    elif USE_GENERATION_CONSTRAINTS:
        return "constrained"
    else:
        return "unconstrained_with_postprocessing"

def create_table_summary_report():
    """Create a summary report of all table transformations with failure type analysis"""
    if not ENABLE_TABLE_LOGGING or not os.path.exists(TABLE_LOG_DIR):
        return
    
    summary_file = os.path.join(TABLE_LOG_DIR, "summary_report.json")
    summary_data = {
        "total_requests": 0,
        "total_transformations": 0,
        "action_counts": {},
        "failure_type_counts": {},
        "generation_mode_counts": {},  # NEW: Track different generation modes used
        "average_steps_per_request": 0.0,
        "table_size_changes": [],
        "incomplete_requests": [],
        "generation_mode": get_generation_mode_string()  # NEW: Overall mode
    }
    
    try:
        log_files = [f for f in os.listdir(TABLE_LOG_DIR) if f.endswith('_log.json')]
        
        total_actions = 0
        request_step_counts = []
        
        for log_file in log_files:
            if log_file == "summary_report.json":
                continue
                
            log_path = os.path.join(TABLE_LOG_DIR, log_file)
            request_id = log_file.replace('_log.json', '')
            
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split by separator and parse each JSON block
                    blocks = content.split('-'*80)
                    
                    request_steps = 0
                    has_end_action = False
                    initial_size = None
                    final_size = None
                    
                    for block in blocks:
                        block = block.strip()
                        if block:
                            try:
                                entry = json.loads(block)
                                action = entry.get('action', '')
                                summary = entry.get('table_summary', {})
                                success = entry.get('success', True)
                                failure_type = entry.get('failure_type', None)
                                generation_mode = entry.get('generation_mode', 'unknown')  # NEW
                                
                                # Track generation modes used
                                if generation_mode != 'unknown':
                                    summary_data["generation_mode_counts"][generation_mode] = summary_data["generation_mode_counts"].get(generation_mode, 0) + 1
                                
                                if action == 'initial':
                                    initial_size = (summary.get('num_rows', 0), summary.get('num_columns', 0))
                                elif action.startswith('end'):
                                    has_end_action = True
                                    final_size = (summary.get('num_rows', 0), summary.get('num_columns', 0))
                                elif not action.startswith('initial'):
                                    if success:
                                        total_actions += 1
                                        request_steps += 1
                                        
                                        # Count action types
                                        action_type = action.split('(')[0] if '(' in action else action
                                        summary_data["action_counts"][action_type] = summary_data["action_counts"].get(action_type, 0) + 1
                                        
                                        # Track size changes
                                        current_size = (summary.get('num_rows', 0), summary.get('num_columns', 0))
                                        if action_type not in ['end', 'parse_failed', 'generation_error']:
                                            final_size = current_size
                                    else:
                                        # Count failure types
                                        if failure_type:
                                            summary_data["failure_type_counts"][failure_type] = summary_data["failure_type_counts"].get(failure_type, 0) + 1
                                        
                            except json.JSONDecodeError:
                                continue
                    
                    if request_steps > 0 or not has_end_action:  # Count requests that had any activity
                        summary_data["total_requests"] += 1
                        
                        if request_steps > 0:
                            request_step_counts.append(request_steps)
                        
                        # Track incomplete requests
                        if not has_end_action:
                            summary_data["incomplete_requests"].append(request_id)
                        
                        # Track table size changes
                        if initial_size and final_size:
                            size_change = {
                                "request_id": request_id,
                                "initial_size": initial_size,
                                "final_size": final_size,
                                "row_change": final_size[0] - initial_size[0],
                                "col_change": final_size[1] - initial_size[1]
                            }
                            summary_data["table_size_changes"].append(size_change)
                        
            except Exception as e:
                print(f"Error processing log file {log_file}: {e}")
        
        summary_data["total_transformations"] = total_actions
        summary_data["average_steps_per_request"] = sum(request_step_counts) / len(request_step_counts) if request_step_counts else 0.0
        summary_data["completion_rate"] = 1.0 - (len(summary_data["incomplete_requests"]) / summary_data["total_requests"]) if summary_data["total_requests"] > 0 else 0.0
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Table transformation summary written to {summary_file}")
        print(f"Generation mode: {summary_data['generation_mode']}")
        print(f"Total requests: {summary_data['total_requests']}")
        print(f"Total transformations: {summary_data['total_transformations']}")
        print(f"Completion rate: {summary_data['completion_rate']:.2%}")
        print(f"Average steps per request: {summary_data['average_steps_per_request']:.1f}")
        print(f"Incomplete requests: {len(summary_data['incomplete_requests'])}")
        
        # NEW: Show generation mode breakdown
        if summary_data["generation_mode_counts"]:
            print("Generation mode breakdown:")
            for mode, count in summary_data["generation_mode_counts"].items():
                print(f"  {mode}: {count}")
        
        if summary_data["failure_type_counts"]:
            print("Failure breakdown:")
            total_failures = sum(summary_data["failure_type_counts"].values())
            for failure_type, count in summary_data["failure_type_counts"].items():
                percentage = (count / total_failures) * 100 if total_failures > 0 else 0
                print(f"  {failure_type}: {count} ({percentage:.1f}%)")
            
            # Calculate validity rate (inverse of validity failures)
            validity_failures = summary_data["failure_type_counts"].get("validity_failure", 0)
            total_attempts = summary_data["total_transformations"] + total_failures
            validity_rate = ((total_attempts - validity_failures) / total_attempts) * 100 if total_attempts > 0 else 0
            print(f"Validity rate (parseable actions): {validity_rate:.1f}%")
        
    except Exception as e:
        print(f"Error creating summary report: {e}")

def analyze_table_logs(request_id):
    """Analyze table logs for a specific request (utility function)"""
    if not ENABLE_TABLE_LOGGING:
        print("Table logging is disabled")
        return
    
    log_file = f"{request_id}_log.json"
    log_path = os.path.join(TABLE_LOG_DIR, log_file)
    
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            blocks = content.split('-'*80)
            
            print(f"\nTable transformation log for {request_id}:")
            print("=" * 80)
            
            for block_num, block in enumerate(blocks):
                block = block.strip()
                if block:
                    try:
                        entry = json.loads(block)
                        step = entry.get('step', 0)
                        action = entry.get('action', '')
                        summary = entry.get('table_summary', {})
                        preview = entry.get('table_preview', [])
                        success = entry.get('success', True)
                        failure_type = entry.get('failure_type', None)
                        generation_mode = entry.get('generation_mode', 'unknown')  # NEW
                        
                        status_str = "SUCCESS" if success else f"FAILED ({failure_type})"
                        print(f"\nStep {step}: {action} - {status_str} [{generation_mode}]")  # NEW: Show mode
                        print(f"Table: {summary.get('num_rows', 0)} rows × {summary.get('num_columns', 0)} columns")
                        
                        if preview:
                            print("Preview:")
                            for line in preview:
                                print(f"  {line}")
                        
                        # Mention CSV file location
                        if step > 0:  # Don't mention for initial step
                            csv_files = [f for f in os.listdir(TABLE_LOG_DIR) 
                                       if f.startswith(f"{request_id}_step{step:02d}") and f.endswith('.csv')]
                            if csv_files:
                                print(f"Full table saved as: {csv_files[0]}")
                        
                        print("-" * 40)
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing block {block_num}: {e}")
                    
    except Exception as e:
        print(f"Error analyzing log file: {e}")

def list_table_files_for_request(request_id):
    """List all table files for a specific request"""
    if not ENABLE_TABLE_LOGGING or not os.path.exists(TABLE_LOG_DIR):
        return []
    
    files = []
    for filename in os.listdir(TABLE_LOG_DIR):
        if filename.startswith(request_id) and filename.endswith('.csv'):
            files.append(filename)
    
    return sorted(files)

def main():
    """Main function using process-based parallelism with enhanced logging and configurable generation modes"""
    # Setup logging
    setup_logging_directory()
    
    # Initialize process-based engine
    print("Setting up process-based parallel vLLM engine...")
    
    # Determine generation mode
    generation_mode = get_generation_mode_string()
    print(f"Generation mode: {generation_mode}")
    
    num_workers = 4
    
    parallel_engine = ProcessParallelVLLM(
        model_id=model_id,
        num_workers=num_workers,
        use_constraints=USE_GENERATION_CONSTRAINTS,
        use_cot=USE_CHAIN_OF_TABLE  # NEW: Pass CoT flag
    )
    
    try:
        # Start all workers and wait for models to load
        all_ready = parallel_engine.start_workers()
        
        if not all_ready:
            print("Fatal: Not all workers loaded models successfully. Exiting.")
            return
        
        # Prepare batch data
        questions_tables_and_answers = []
        examples = []

        subset_size = min(1000, len(dataset))

        for i, example in enumerate(dataset):
            if i >= subset_size:
                break
                
            table = {
                'columns': example['table']['header'],
                'rows': example['table']['rows']
            }
            question = example['question']
            answers = example['answers']  # Get answers from dataset
            
            questions_tables_and_answers.append((question, table, answers))
            examples.append(example)
        
        print(f"Processing {len(questions_tables_and_answers)} questions with {num_workers} workers...")
        print(f"Generation mode: {generation_mode}")
        if ENABLE_TABLE_LOGGING:
            print(f"Table logging enabled: {TABLE_LOG_DIR} (compression: {COMPRESS_TABLE_LOGS})")
        
        start_time = time.time()
        
        # Generate using process-based parallelism
        results = parallel_engine.generate_batch(questions_tables_and_answers)
        
        end_time = time.time()
        print(f"Total time taken: {end_time - start_time:.2f} seconds")

        if len(questions_tables_and_answers) > 0:  # Note: changed variable name
            print(f"Average time per request: {(end_time - start_time) / len(questions_tables_and_answers):.2f} seconds")

        # ADD THE EXECUTION ACCURACY ANALYSIS HERE:
        execution_accuracies = []
        answer_found_rates = []
        proper_termination_rates = []
        baseline_rates = []

        print(f"\n{'='*80}")
        print("EXECUTION ACCURACY ANALYSIS (WikiTableQuestions Logic with Dataset Answers)")
        print(f"{'='*80}")

        for i, result in enumerate(results):
            metrics = result.get('execution_accuracy_metrics', {})
            execution_accuracies.append(metrics.get('execution_accuracy', 0.0))
            answer_found_rates.append(1.0 if metrics.get('answer_found_in_final', False) else 0.0)
            proper_termination_rates.append(1.0 if metrics.get('terminated_properly', False) else 0.0)
            baseline_rates.append(1.0 if metrics.get('answer_found_in_original', False) else 0.0)

        # Calculate and print overall metrics
        overall_execution_accuracy = sum(execution_accuracies) / len(execution_accuracies) if execution_accuracies else 0.0
        overall_answer_found_rate = sum(answer_found_rates) / len(answer_found_rates) if answer_found_rates else 0.0
        overall_baseline_rate = sum(baseline_rates) / len(baseline_rates) if baseline_rates else 0.0
        overall_termination_rate = sum(proper_termination_rates) / len(proper_termination_rates) if proper_termination_rates else 0.0

        print(f"Overall Execution Accuracy: {overall_execution_accuracy:.3f} ({overall_execution_accuracy*100:.1f}%)")
        print(f"Answer Found in Final Table Rate: {overall_answer_found_rate:.3f} ({overall_answer_found_rate*100:.1f}%)")
        print(f"Answer Found in Original Table Rate (Baseline): {overall_baseline_rate:.3f} ({overall_baseline_rate*100:.1f}%)")
        print(f"Proper Termination Rate: {overall_termination_rate:.3f} ({overall_termination_rate*100:.1f}%)")

        # Show improvement over baseline
        if overall_baseline_rate > 0:
            improvement = overall_answer_found_rate - overall_baseline_rate
            improvement_pct = (improvement / overall_baseline_rate) * 100 if overall_baseline_rate > 0 else 0
            print(f"Improvement over baseline: {improvement:+.3f} ({improvement_pct:+.1f}%)")

        # Show detailed breakdown
        success_cases = sum(1 for acc in execution_accuracies if acc == 1.0)
        print(f"Successful Cases: {success_cases}/{len(execution_accuracies)} ({success_cases/len(execution_accuracies)*100:.1f}%)")
        print(f"Evaluation Method: WikiTableQuestions Logic with Dataset Answers")
        # END OF EXECUTION ACCURACY ANALYSIS

        # Write results to JSONL file
        write_results_to_jsonl(results, examples, output_file)
        
        # Create summary report
        if ENABLE_TABLE_LOGGING:
            create_table_summary_report()
        
        # Print sample results
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
            print(f"Answer Found in Original (Baseline): {metrics.get('answer_found_in_original', False)}")
            print(f"Terminated Properly: {metrics.get('terminated_properly', False)}")
            
            if metrics.get('matched_answers_final'):
                print(f"Matched Answers in Final: {metrics['matched_answers_final']}")
            
            if metrics.get('final_table_size'):
                rows, cols = metrics['final_table_size']
                print(f"Final Table Size: {rows} rows × {cols} columns")
            
            print("-" * 80)
            
        # Example: Analyze logs for first request (demonstration)
        if ENABLE_TABLE_LOGGING and results:
            print("\nSample table transformation analysis:")
            # Find an actual request ID from the logs
            log_files = [f for f in os.listdir(TABLE_LOG_DIR) if f.endswith('_log.json') and f != 'summary_report.json']
            if log_files:
                sample_request_id = log_files[0].replace('_log.json', '')
                analyze_table_logs(sample_request_id)
            
    finally:
        # Always shutdown workers
        parallel_engine.shutdown()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()