import torch

from tokenizer_config import get_logic_token_ids, is_llama_tokenizer

class ConstraintStateMachine:
    """State machine for constraint processing with optional global action constraints"""
    def __init__(self, table, tokenizer, action_history=None, use_global_constraints=True):
        self.tokenizer = tokenizer
        self.table = table
        self.use_global_constraints = use_global_constraints
        self.logic_token_ids = get_logic_token_ids(self.tokenizer)
        self.llama_tokenizer = is_llama_tokenizer(self.tokenizer)
        
        # Parse action history to determine previously used action types (only if global constraints are enabled)
        if self.use_global_constraints:
            self.previously_used_actions = self._parse_action_history(action_history)
        else:
            self.previously_used_actions = set()
        
        self.reset()
        
        num_rows = min(500, len(table['rows']))
        self.valid_params = {
            "select_row": [str(i) for i in range(num_rows)],
            "select_column": table['columns'],
            "end": []
        }
        
        self.column_token_map = {}
        for col in self.valid_params["select_column"]:
            quoted = f'"{col}"'
            tokens = tokenizer.encode(quoted, add_special_tokens=False)
            if tokens:
                self.column_token_map[col] = tokens

    def _parse_action_history(self, action_history):
        """Parse action history to extract previously used action types"""
        used_actions = set()
        if action_history:
            for action_str in action_history:
                if "select_row" in action_str:
                    used_actions.add("select_row")
                elif "select_column" in action_str:
                    used_actions.add("select_column")
                elif "end" in action_str:
                    used_actions.add("end")
        return used_actions

    def reset(self):
        self.state = "start"
        self.current_action = None
        self.generated_tokens = []
        self.finished = False
        self.selected_params = set()
        self.current_param = []
        self.has_parameter = False
        self.param_complete = False
        self.expecting_parameter = False
        
        # Apply constraints based on configuration
        if self.use_global_constraints:
            self.possible_actions = self._get_allowed_actions_with_global_constraints()
        else:
            self.possible_actions = self._get_allowed_actions_without_global_constraints()
        
        self.action_prefix = []
        self.current_column = None

    def _get_allowed_actions_with_global_constraints(self):
        """Determine which actions are allowed based on global constraints"""
        allowed = []
        
        # If no actions have been taken yet, can't use 'end'
        if not self.previously_used_actions:
            # Only allow select_row and select_column for first action
            allowed.extend(["select_row", "select_column"])
        else:
            # Check which actions haven't been used yet
            if "select_row" not in self.previously_used_actions:
                allowed.append("select_row")
            if "select_column" not in self.previously_used_actions:
                allowed.append("select_column")
            
            # 'end' is allowed only if at least one other action has been taken
            # and no other actions are available
            if not allowed:  # No other actions available
                allowed.append("end")
        
        return allowed

    def _get_allowed_actions_without_global_constraints(self):
        """Get all actions without global constraints (original behavior)"""
        return ["select_row", "select_column", "end"]

    def update_state(self, token):
        if self.finished:
            return
            
        self.generated_tokens.append(token)
        
        if self.state == "start":
            self._handle_start(token)
        elif self.state == "in_action":
            self._handle_action(token)
        elif self.state == "after_action":
            self._handle_after_action(token)
        elif self.state == "in_paren_open":
            self._handle_paren_open(token)
        elif self.state == "in_params":
            self._handle_params(token)
        elif self.state == "in_paren_close":
            self._handle_paren_close(token)

    def _handle_start(self, token):
        
        
        matching_actions = []
        for action in self.possible_actions:  # Use filtered possible_actions
            tokens = self.logic_token_ids["action_tokens"][action]
            if tokens and token == tokens[0]:
                matching_actions.append(action)
        
        if matching_actions:
            self.state = "in_action"
            self.possible_actions = matching_actions
            self.action_prefix = [token]

    def _handle_action(self, token):
        
        
        self.action_prefix.append(token)
        
        completed_actions = []
        for action in self.possible_actions:
            tokens = self.logic_token_ids["action_tokens"][action]
            if tokens and self.action_prefix == tokens:
                completed_actions.append(action)
        
        if completed_actions:
            self.current_action = completed_actions[0]
            if self.current_action == "end":
                self.finished = True
                self.state = "finish"
            else:
                self.state = "after_action"
            return
        
        next_possible = []
        for action in self.possible_actions:
            tokens = self.logic_token_ids["action_tokens"][action]
            if tokens and len(self.action_prefix) < len(tokens) and tokens[:len(self.action_prefix)] == self.action_prefix:
                next_possible.append(action)
        
        if not next_possible:
            self.finished = True
        else:
            self.possible_actions = next_possible

    def _handle_after_action(self, token):
        if self.current_action == "end":
            self.finished = True
            return
            
        
        if token == self.logic_token_ids["paren_open_id"]:
            self.state = "in_paren_open"

    def _handle_paren_open(self, token):
       
        if token == self.logic_token_ids["list_open_id"]:
            self.state = "in_params"
            self.current_param = []
            self.param_complete = False
            self.expecting_parameter = True
            self.current_column = None

    def _handle_params(self, token):
        if self.current_action == "select_column":
            self._handle_column_param(token)
        else:
            self._handle_row_param(token)

    def _handle_column_param(self, token):

        if not self.current_param and self.expecting_parameter:
            if token == self.logic_token_ids["quote_id"]:
                self.current_param.append(token)
                self.current_column = []
            return
        
        if self.current_param and self.current_param[0] == self.logic_token_ids["quote_id"]:
            if token in self.logic_token_ids["closing_quotes_tokens"]:
            # if token == closing_quote_id:
                self.current_param.append(token)
                param_text = self.tokenizer.decode(self.current_param)
                clean_param = param_text.strip('"')
                
                is_valid = False
                for col, tokens in self.column_token_map.items():
                    exp = ((self.llama_tokenizer and clean_param == col) or self.current_param == tokens) and col not in self.selected_params
                    if exp:
                        self.selected_params.add(col)
                        self.has_parameter = True
                        self.param_complete = True
                        self.expecting_parameter = False
                        is_valid = True
                        break
                
                if not is_valid:
                    self.param_complete = False
                
                self.current_param = []
                self.current_column = None
            else:
                self.current_param.append(token)
                if self.current_column is None:
                    self.current_column = []
                self.current_column.append(token)
        elif self.param_complete:
            if token == self.logic_token_ids["comma_id"]:
                self.param_complete = False
                self.expecting_parameter = True
                self.current_column = None
            elif token == self.logic_token_ids["list_close_id"]:
                self.state = "in_paren_close"

    def _handle_row_param(self, token):
        if not self.current_param and self.expecting_parameter:
            if token in self.logic_token_ids["digit_tokens"]:
                num_str = self.logic_token_ids["digit_token_map"].get(token, str(token))
                if (num_str in self.valid_params["select_row"] and 
                    num_str not in self.selected_params):
                    self.selected_params.add(num_str)
                    self.has_parameter = True
                    self.param_complete = True
                    self.expecting_parameter = False
                else:
                    self.param_complete = False
            return
        
        if self.param_complete:
            if token == self.logic_token_ids["comma_id"]:
                self.param_complete = False
                self.expecting_parameter = True
            elif token == self.logic_token_ids["list_close_id"]:
                self.state = "in_paren_close"

    def _handle_paren_close(self, token):
        
        if token == self.logic_token_ids["paren_close_id"]:
            self.state = "finish"
            self.finished = True

    def allowed_tokens(self):
        if self.finished:
            return [self.tokenizer.eos_token_id]
        
        if self.state == "start":
            allowed = set()
            # Only allow tokens for actions that are permitted (with or without global constraints)
            for action in self.possible_actions:
                tokens = self.logic_token_ids["action_tokens"].get(action, [])
                if tokens:
                    allowed.add(tokens[0])
            return list(allowed)
        
        elif self.state == "in_action":
            allowed = set()
            for action in self.possible_actions:
                tokens = self.logic_token_ids["action_tokens"].get(action, [])
                if tokens and len(self.action_prefix) < len(tokens):
                    allowed.add(tokens[len(self.action_prefix)])
            return list(allowed) if allowed else [self.tokenizer.eos_token_id]
        
        elif self.state == "after_action":
            if self.current_action == "end":
                return [self.tokenizer.eos_token_id]
            return [self.logic_token_ids["paren_open_id"]]
        
        elif self.state == "in_paren_open":
            return [self.logic_token_ids["list_open_id"]]
        
        elif self.state == "in_params":
            allowed = set()
            remaining_params = set(self.valid_params[self.current_action]) - self.selected_params
            
            if self.current_action == "select_column":
                if not self.current_param and self.expecting_parameter:
                    if remaining_params:
                        allowed.add(self.logic_token_ids["quote_id"])
                elif self.current_param and self.current_param[0] == self.logic_token_ids["quote_id"]:
                    for col in remaining_params:
                        full_seq = self.column_token_map[col]
                        if len(self.current_param) < len(full_seq) and full_seq[:len(self.current_param)] == self.current_param:
                            allowed.add(full_seq[len(self.current_param)])
                    
                    candidate = self.current_param + [self.logic_token_ids["quote_id"]]
                    for col in remaining_params:
                        if self.column_token_map[col] == candidate:
                            allowed.add(self.logic_token_ids["quote_id"])
                            break
                elif self.param_complete:
                    if remaining_params:
                        allowed.add(self.logic_token_ids["comma_id"])
                    allowed.add(self.logic_token_ids["list_close_id"])
            
            else:  # select_row
                if not self.current_param and self.expecting_parameter:
                    for digit in self.logic_token_ids["digit_tokens"]:
                        num_str = self.logic_token_ids["digit_token_map"].get(digit, str(digit))
                        if num_str in remaining_params:
                            allowed.add(digit)
                elif self.param_complete:
                    if remaining_params:
                        allowed.add(self.logic_token_ids["comma_id"])
                    allowed.add(self.logic_token_ids["list_close_id"])
            
            if self.expecting_parameter:
                if self.logic_token_ids["list_close_id"] in allowed:
                    allowed.remove(self.logic_token_ids["list_close_id"])
            
            return list(allowed) if allowed else [self.tokenizer.eos_token_id]
        
        elif self.state == "in_paren_close":
            return [self.logic_token_ids["paren_close_id"]]
        
        return [self.tokenizer.eos_token_id]

# NEW: Action constraint for CoT dynamic_plan step
class ActionOnlyConstraintStateMachine:
    """Simplified state machine that only allows action selection (no parameters)"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.logic_token_ids = get_logic_token_ids(self.tokenizer)
        self.reset()
        
    def reset(self):
        self.state = "start"
        self.generated_tokens = []
        self.finished = False
        self.possible_actions = ["select_row", "select_column", "end"]
        self.action_prefix = []
        
    def update_state(self, token):
        if self.finished:
            return
            
        self.generated_tokens.append(token)
        
        if self.state == "start":
            self._handle_start(token)
        elif self.state == "in_action":
            self._handle_action(token)
            
    def _handle_start(self, token):

        
        matching_actions = []
        for action in self.possible_actions:
            tokens = self.logic_token_ids["action_tokens"][action]
            if tokens and token == tokens[0]:
                matching_actions.append(action)
        
        if matching_actions:
            self.state = "in_action"
            self.possible_actions = matching_actions
            self.action_prefix = [token]
            
    def _handle_action(self, token):
        
        
        self.action_prefix.append(token)
        
        # Check if any action is completed
        for action in self.possible_actions:
            tokens = self.logic_token_ids["action_tokens"][action]
            if tokens and self.action_prefix == tokens:
                self.finished = True
                return
        
        # Filter possible actions
        next_possible = []
        for action in self.possible_actions:
            tokens = self.logic_token_ids["action_tokens"][action]
            if tokens and len(self.action_prefix) < len(tokens) and tokens[:len(self.action_prefix)] == self.action_prefix:
                next_possible.append(action)
        
        if not next_possible:
            self.finished = True
        else:
            self.possible_actions = next_possible
            
    def allowed_tokens(self):
        if self.finished:
            return [self.tokenizer.eos_token_id]
            
        
        
        if self.state == "start":
            allowed = set()
            for action in self.possible_actions:
                tokens = self.logic_token_ids["action_tokens"][action]
                if tokens:
                    allowed.add(tokens[0])
            return list(allowed)
            
        elif self.state == "in_action":
            allowed = set()
            for action in self.possible_actions:
                tokens = self.logic_token_ids["action_tokens"][action]
                if tokens and len(self.action_prefix) < len(tokens):
                    allowed.add(tokens[len(self.action_prefix)])
            return list(allowed) if allowed else [self.tokenizer.eos_token_id]
            
        return [self.tokenizer.eos_token_id]



def create_constraint_logits_processor(table, tokenizer, request_id, state_machines_dict, 
                                      action_history=None, use_global_constraints=True):
    """
    Create a logits processor function for a specific table with external state storage
    
    Args:
        table: The table data
        tokenizer: The tokenizer
        request_id: Unique request identifier
        state_machines_dict: Dictionary to store state machines
        action_history: List of previously executed actions (for global constraints)
        use_global_constraints: Whether to apply global action constraints (default: True)
    """

    
    def constraint_logits_processor(prompt_token_ids, generated_token_ids, logits):
        # Get or create state machine for this request with action history
        if request_id not in state_machines_dict:
            state_machines_dict[request_id] = ConstraintStateMachine(
                table, tokenizer, action_history, use_global_constraints
            )
        
        sm = state_machines_dict[request_id]
        
        # Update state machine with generated tokens
        if len(generated_token_ids) > len(sm.generated_tokens):
            new_tokens = generated_token_ids[len(sm.generated_tokens):]
            for token in new_tokens:
                sm.update_state(token)
        
        # Get allowed tokens and mask logits
        try:
            allowed_tokens = sm.allowed_tokens()
        except Exception as e:
            print(f"Error in constraint processing: {e}")
            return logits
        
        # Create mask
        mask = torch.full_like(logits, float('-inf'))
        for token_id in allowed_tokens:
            if token_id < len(logits):
                mask[token_id] = 0.0
        
        return logits + mask
    
    return constraint_logits_processor

# NEW: Action-only constraint processor for CoT dynamic_plan
def create_action_only_constraint_processor(tokenizer, request_id, state_machines_dict):
    """Create a logits processor that only allows action selection (no parameters)"""
    
    def action_constraint_processor(prompt_token_ids, generated_token_ids, logits):
        # Get or create state machine for this request
        if request_id not in state_machines_dict:
            state_machines_dict[request_id] = ActionOnlyConstraintStateMachine(tokenizer)
        
        sm = state_machines_dict[request_id]
        
        # Update state machine with generated tokens
        if len(generated_token_ids) > len(sm.generated_tokens):
            new_tokens = generated_token_ids[len(sm.generated_tokens):]
            for token in new_tokens:
                sm.update_state(token)
        
        # Get allowed tokens and mask logits
        try:
            allowed_tokens = sm.allowed_tokens()
        except Exception as e:
            print(f"Error in action constraint processing: {e}")
            return logits
        
        # Create mask
        mask = torch.full_like(logits, float('-inf'))
        for token_id in allowed_tokens:
            if token_id < len(logits):
                mask[token_id] = 0.0
        
        return logits + mask
    
    return action_constraint_processor
