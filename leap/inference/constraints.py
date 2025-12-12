import torch

from leap.config.loader import TokenizerConfig
from leap.core import Table
from leap.core.actions import REGISTRY


class ConstraintStateMachine:
    """State machine for constraint processing with optional global action constraints"""

    def __init__(
        self,
        table: Table,
        tokenizer,
        tokenizer_config: TokenizerConfig,
        action_history=None,
        use_global_constraints=True,
    ):
        self.tokenizer = tokenizer
        self.table = table
        self.use_global_constraints = use_global_constraints
        self.tokenizer_config = tokenizer_config

        # Parse action history to determine previously used action types (only if global constraints are enabled)
        if self.use_global_constraints:
            self.previously_used_actions = self._parse_action_history(action_history)
        else:
            self.previously_used_actions = set()

        self.reset()

        # Generate valid parameters from registry
        self.valid_params = REGISTRY.generate_constraint_params(table)

        # Limit rows for performance (max 500)
        if "select_row" in self.valid_params:
            max_rows = min(500, len(self.valid_params["select_row"]))
            self.valid_params["select_row"] = self.valid_params["select_row"][:max_rows]

        self.column_token_map = {}
        if "select_column" in self.valid_params:
            for col in self.valid_params["select_column"]:
                quoted = f'"{col}"'
                tokens = tokenizer.encode(quoted, add_special_tokens=False)
                if tokens:
                    self.column_token_map[col] = tokens

        self.row_token_map = {}
        if "select_row" in self.valid_params:
            for row in self.valid_params["select_row"]:
                quoted = f'"{row}"'
                tokens = tokenizer.encode(quoted, add_special_tokens=False)
                if tokens:
                    self.row_token_map[row] = tokens

    def _parse_action_history(self, action_history):
        """Parse action history to extract previously used action types"""
        used_actions = set()
        if action_history:
            # Get all enabled action names from registry
            enabled_actions = REGISTRY.get_enabled_names()
            for action_str in action_history:
                # Check each enabled action
                for action_name in enabled_actions:
                    if action_name in action_str:
                        used_actions.add(action_name)
                        break  # Found the action, move to next history item
        return used_actions

    def reset(self):
        self.state = "start"
        self.current_action = None
        self.generated_tokens = []
        self.finished = False
        self.selected_params = set()
        self.current_param = []
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
        # Get enabled actions from registry
        enabled_actions = REGISTRY.get_enabled_names()

        # Terminating action: only 'end' (direct_query is applied automatically after end)
        terminating_actions = {"end"}

        # Non-terminating actions (table transformations)
        transformation_actions = [a for a in enabled_actions if a not in terminating_actions]

        allowed = []

        # If no actions have been taken yet, can't use terminating actions
        if not self.previously_used_actions:
            # Only allow transformation actions for first action
            allowed.extend(transformation_actions)
        else:
            # Check which transformation actions haven't been used yet
            for action in transformation_actions:
                if action not in self.previously_used_actions:
                    allowed.append(action)

            # Terminating actions are allowed only if:
            # - At least one other action has been taken
            # - No other transformation actions are available
            if not allowed:  # No transformation actions available
                # Add all enabled terminating actions
                allowed.extend([a for a in terminating_actions if a in enabled_actions])

        return allowed

    def _get_allowed_actions_without_global_constraints(self):
        """Get all enabled actions without global constraints"""
        return REGISTRY.get_enabled_names()

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
            tokens = self.tokenizer_config.action_tokens[action]
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
            tokens = self.tokenizer_config.action_tokens[action]
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
            tokens = self.tokenizer_config.action_tokens[action]
            if tokens and len(self.action_prefix) < len(tokens) and tokens[: len(self.action_prefix)] == self.action_prefix:
                next_possible.append(action)

        if not next_possible:
            self.finished = True
        else:
            self.possible_actions = next_possible

    def _handle_after_action(self, token):
        if self.current_action == "end":
            self.finished = True
            return

        if token == self.tokenizer_config.paren_open_id:
            self.state = "in_paren_open"

    def _handle_paren_open(self, token):
        if token == self.tokenizer_config.list_open_id:
            self.state = "in_params"
            self.current_param = []
            self.param_complete = False
            self.expecting_parameter = True

    def _handle_params(self, token):
        if not self.current_param and self.expecting_parameter:
            if token == self.tokenizer_config.quote_id:
                self.current_param.append(token)
            return

        if self.current_param and self.current_param[0] == self.tokenizer_config.quote_id:
            if token in self.tokenizer_config.closing_quotes_tokens:
                self.current_param.append(token)
                param_text = self.tokenizer.decode(self.current_param)
                clean_param = param_text.strip('"')

                is_valid = False
                token_map = self.column_token_map if self.current_action == "select_column" else self.row_token_map

                for param, tokens in token_map.items():
                    exp = (
                        (self.tokenizer_config.is_llama_tokenizer and clean_param == param) or self.current_param == tokens
                    ) and param not in self.selected_params
                    if exp:
                        self.selected_params.add(param)
                        self.param_complete = True
                        self.expecting_parameter = False
                        is_valid = True
                        break

                if not is_valid:
                    self.param_complete = False

                self.current_param = []
            else:
                self.current_param.append(token)
        elif self.param_complete:
            if token == self.tokenizer_config.comma_id:
                self.param_complete = False
                self.expecting_parameter = True
            elif token == self.tokenizer_config.list_close_id:
                self.state = "in_paren_close"

    def _handle_paren_close(self, token):
        if token == self.tokenizer_config.paren_close_id:
            self.state = "finish"
            self.finished = True

    def allowed_tokens(self):
        if self.finished:
            return [self.tokenizer.eos_token_id]

        if self.state == "start":
            allowed = set()
            # Only allow tokens for actions that are permitted (with or without global constraints)
            for action in self.possible_actions:
                tokens = self.tokenizer_config.action_tokens.get(action, [])
                if tokens:
                    allowed.add(tokens[0])
            return list(allowed)

        elif self.state == "in_action":
            allowed = set()
            for action in self.possible_actions:
                tokens = self.tokenizer_config.action_tokens.get(action, [])
                if tokens and len(self.action_prefix) < len(tokens):
                    allowed.add(tokens[len(self.action_prefix)])
            return list(allowed) if allowed else [self.tokenizer.eos_token_id]

        elif self.state == "after_action":
            if self.current_action == "end":
                return [self.tokenizer.eos_token_id]
            return [self.tokenizer_config.paren_open_id]

        elif self.state == "in_paren_open":
            return [self.tokenizer_config.list_open_id]

        elif self.state == "in_params":
            allowed = set()
            remaining_params = set(self.valid_params[self.current_action]) - self.selected_params
            token_map = self.column_token_map if self.current_action == "select_column" else self.row_token_map

            # if self.current_action == "select_column":
            if not self.current_param and self.expecting_parameter:
                if remaining_params:
                    allowed.add(self.tokenizer_config.quote_id)
            elif self.current_param and self.current_param[0] == self.tokenizer_config.quote_id:
                for param in remaining_params:
                    full_seq = token_map[param]
                    if len(self.current_param) < len(full_seq) and full_seq[: len(self.current_param)] == self.current_param:
                        allowed.add(full_seq[len(self.current_param)])

                candidate = self.current_param + [self.tokenizer_config.quote_id]
                for param in remaining_params:
                    if token_map[param] == candidate:
                        allowed.add(self.tokenizer_config.quote_id)
                        break
            elif self.param_complete:
                if remaining_params:
                    allowed.add(self.tokenizer_config.comma_id)
                allowed.add(self.tokenizer_config.list_close_id)

            if allowed:
                return list(allowed)
            return [self.tokenizer.eos_token_id]

        elif self.state == "in_paren_close":
            return [self.tokenizer_config.paren_close_id]

        return [self.tokenizer.eos_token_id]


def create_constraint_logits_processor(
    table,
    tokenizer,
    tokenizer_config: TokenizerConfig,
    request_id,
    state_machines_dict,
    action_history=None,
    use_global_constraints=True,
):
    """
    Create a logits processor function for a specific table with external state storage

    Args:
        table: The table data
        tokenizer: The tokenizer
        tokenizer_config: TokenizerConfig instance
        request_id: Unique request identifier
        state_machines_dict: Dictionary to store state machines
        action_history: List of previously executed actions (for global constraints)
        use_global_constraints: Whether to apply global action constraints (default: True)
    """

    def constraint_logits_processor(prompt_token_ids, generated_token_ids, logits):
        # Get or create state machine for this request with action history
        if request_id not in state_machines_dict:
            state_machines_dict[request_id] = ConstraintStateMachine(
                table,
                tokenizer,
                tokenizer_config,
                action_history,
                use_global_constraints,
            )

        sm = state_machines_dict[request_id]

        # Update state machine with generated tokens
        if len(generated_token_ids) > len(sm.generated_tokens):
            new_tokens = generated_token_ids[len(sm.generated_tokens) :]
            for token in new_tokens:
                sm.update_state(token)

        # Get allowed tokens and mask logits
        try:
            allowed_tokens = sm.allowed_tokens()
        except Exception as e:
            print(f"Error in constraint processing: {e}")
            return logits

        # Create mask
        mask = torch.full_like(logits, float("-inf"))
        for token_id in allowed_tokens:
            if token_id < len(logits):
                mask[token_id] = 0.0

        return logits + mask

    return constraint_logits_processor


# NEW: Action constraint for CoT dynamic_plan step
class ActionOnlyConstraintStateMachine:
    """Simplified state machine that only allows action selection (no parameters)"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Pre-compute action tokens from enabled actions
        self.action_tokens = {}
        for action_name in REGISTRY.get_enabled_names():
            self.action_tokens[action_name] = tokenizer.encode(action_name, add_special_tokens=False)
        self.reset()

    def reset(self):
        self.state = "start"
        self.generated_tokens = []
        self.finished = False
        # Get enabled action names from registry
        self.possible_actions = REGISTRY.get_enabled_names()
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
            tokens = self.action_tokens[action]
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
            tokens = self.action_tokens[action]
            if tokens and self.action_prefix == tokens:
                self.finished = True
                return

        # Filter possible actions
        next_possible = []
        for action in self.possible_actions:
            tokens = self.action_tokens[action]
            if tokens and len(self.action_prefix) < len(tokens) and tokens[: len(self.action_prefix)] == self.action_prefix:
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
                tokens = self.action_tokens[action]
                if tokens:
                    allowed.add(tokens[0])
            return list(allowed)

        elif self.state == "in_action":
            allowed = set()
            for action in self.possible_actions:
                tokens = self.action_tokens[action]
                if tokens and len(self.action_prefix) < len(tokens):
                    allowed.add(tokens[len(self.action_prefix)])
            return list(allowed) if allowed else [self.tokenizer.eos_token_id]

        return [self.tokenizer.eos_token_id]


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
            new_tokens = generated_token_ids[len(sm.generated_tokens) :]
            for token in new_tokens:
                sm.update_state(token)

        # Get allowed tokens and mask logits
        try:
            allowed_tokens = sm.allowed_tokens()
        except Exception as e:
            print(f"Error in action constraint processing: {e}")
            return logits

        # Create mask
        mask = torch.full_like(logits, float("-inf"))
        for token_id in allowed_tokens:
            if token_id < len(logits):
                mask[token_id] = 0.0

        return logits + mask

    return action_constraint_processor


# NEW: Arguments-only constraint for CoT arguments step
class ArgumentsOnlyConstraintStateMachine:
    """
    Specialized state machine for constraining only the arguments part.

    For two-phase generation where action name is already determined,
    we only need to constrain the arguments like: [ "row 0", "row 1" ]

    This avoids the model generating the action name again.
    """

    def __init__(self, table: Table, tokenizer, tokenizer_config: TokenizerConfig, action_name: str):
        self.tokenizer = tokenizer
        self.tokenizer_config = tokenizer_config
        self.action_name = action_name
        self.table = table
        self.reset()

        # Set up valid parameters based on action type
        if action_name == "select_row":
            num_rows = min(500, len(table.rows))
            self.valid_params = [f"row {i}" for i in range(num_rows)]
        elif action_name == "select_column":
            self.valid_params = list(table.columns)
        else:  # end
            self.valid_params = []

        # Build token maps
        self.param_token_map = {}
        for param in self.valid_params:
            quoted = f'"{param}"'
            tokens = tokenizer.encode(quoted, add_special_tokens=False)
            if tokens:
                self.param_token_map[param] = tokens

    def reset(self):
        self.state = "start"
        self.generated_tokens = []
        self.finished = False
        self.selected_params = set()
        self.current_param = []
        self.param_complete = False
        self.expecting_parameter = False

    def update_state(self, token):
        if self.finished:
            return

        self.generated_tokens.append(token)

        if self.state == "start":
            self._handle_start(token)
        elif self.state == "in_params":
            self._handle_params(token)
        elif self.state == "in_paren_close":
            self._handle_paren_close(token)

    def _handle_start(self, token):
        # Expect opening bracket: [
        if token == self.tokenizer_config.list_open_id:
            self.state = "in_params"
            self.current_param = []
            self.param_complete = False
            self.expecting_parameter = True

    def _handle_params(self, token):
        if not self.current_param and self.expecting_parameter:
            if token == self.tokenizer_config.quote_id:
                self.current_param.append(token)
            return

        if self.current_param and self.current_param[0] == self.tokenizer_config.quote_id:
            if token in self.tokenizer_config.closing_quotes_tokens:
                self.current_param.append(token)
                param_text = self.tokenizer.decode(self.current_param)
                clean_param = param_text.strip('"')

                is_valid = False
                for param, tokens in self.param_token_map.items():
                    exp = (
                        (self.tokenizer_config.is_llama_tokenizer and clean_param == param) or self.current_param == tokens
                    ) and param not in self.selected_params
                    if exp:
                        self.selected_params.add(param)
                        self.param_complete = True
                        self.expecting_parameter = False
                        is_valid = True
                        break

                if not is_valid:
                    self.param_complete = False

                self.current_param = []
            else:
                self.current_param.append(token)
        elif self.param_complete:
            if token == self.tokenizer_config.comma_id:
                self.param_complete = False
                self.expecting_parameter = True
            elif token == self.tokenizer_config.list_close_id:
                self.state = "finish"
                self.finished = True

    def _handle_paren_close(self, token):
        # Not used in arguments-only mode
        pass

    def allowed_tokens(self):
        if self.finished:
            return [self.tokenizer.eos_token_id]

        if self.state == "start":
            # Must start with [
            return [self.tokenizer_config.list_open_id]

        elif self.state == "in_params":
            allowed = set()
            remaining_params = set(self.valid_params) - self.selected_params

            if not self.current_param and self.expecting_parameter:
                if remaining_params:
                    allowed.add(self.tokenizer_config.quote_id)
            elif self.current_param and self.current_param[0] == self.tokenizer_config.quote_id:
                for param in remaining_params:
                    full_seq = self.param_token_map[param]
                    if len(self.current_param) < len(full_seq) and full_seq[: len(self.current_param)] == self.current_param:
                        allowed.add(full_seq[len(self.current_param)])

                candidate = self.current_param + [self.tokenizer_config.quote_id]
                for param in remaining_params:
                    if self.param_token_map[param] == candidate:
                        allowed.add(self.tokenizer_config.quote_id)
                        break
            elif self.param_complete:
                if remaining_params:
                    allowed.add(self.tokenizer_config.comma_id)
                allowed.add(self.tokenizer_config.list_close_id)

            if allowed:
                return list(allowed)
            return [self.tokenizer.eos_token_id]

        return [self.tokenizer.eos_token_id]


def create_arguments_only_constraint_processor(
    table, tokenizer, tokenizer_config: TokenizerConfig, action_name: str, request_id, state_machines_dict
):
    """
    Create a logits processor for constraining only arguments (not action name).

    Used in two-phase generation where action name is already determined.
    """

    def arguments_constraint_processor(prompt_token_ids, generated_token_ids, logits):
        # Get or create state machine for this request
        if request_id not in state_machines_dict:
            state_machines_dict[request_id] = ArgumentsOnlyConstraintStateMachine(table, tokenizer, tokenizer_config, action_name)

        sm = state_machines_dict[request_id]

        # Update state machine with generated tokens
        if len(generated_token_ids) > len(sm.generated_tokens):
            new_tokens = generated_token_ids[len(sm.generated_tokens) :]
            for token in new_tokens:
                sm.update_state(token)

        # Get allowed tokens and mask logits
        try:
            allowed_tokens = sm.allowed_tokens()
        except Exception as e:
            print(f"Error in arguments constraint processing: {e}")
            return logits

        # Create mask
        mask = torch.full_like(logits, float("-inf"))
        for token_id in allowed_tokens:
            if token_id < len(logits):
                mask[token_id] = 0.0

        return logits + mask

    return arguments_constraint_processor
