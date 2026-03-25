"""
Action Registry - Central registration system for table actions.

This module provides a registry pattern for defining available table actions.
Each action is defined once and the registry provides access to all components
that need action information (constraints, prompts, parsing, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from ..table import Table


class ActionDefinition(ABC):
    """
    Base class for action definitions.

    Each action should inherit from this and implement its specific logic.
    The registry uses these definitions to automatically configure the system.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Action name (e.g., 'select_row', 'direct_query')"""
        pass

    @property
    def requires_args(self) -> bool:
        """Whether this action requires arguments. Default: True unless overridden."""
        return self.name != "end"

    @property
    def is_terminating(self) -> bool:
        """
        Whether this action terminates the reasoning chain.

        Terminating action (end) should not be available as first action
        in Chain-of-Table generation, as per the original paper.
        Note: direct_query is automatically applied after end(), not a separate action.

        Default: False. Override for terminating actions.
        """
        return False

    @abstractmethod
    def generate_params(self, table: Table) -> List[str]:
        """
        Generate valid parameters for constraint system.

        For select_row: ["row 0", "row 1", ...]
        For select_column: ["col1", "col2", ...]
        For end/direct_query: []
        """
        pass

    @abstractmethod
    def parse_arguments(self, args_str: str) -> Optional[List[Any]]:
        """
        Parse arguments from LLM output string.

        Extracts structure only — no table-based validation.
        Validation against a specific table is handled by validate().
        Return None if parsing fails.
        """
        pass

    @abstractmethod
    def apply(self, table: Table, arguments: List[Any]) -> Optional[Table]:
        """
        Apply this operation to a table.

        Called by Action.apply_to_table() for this specific operation.
        Return None if operation fails.
        """
        pass

    def get_prompt_text_iterative(self) -> str:
        """
        Get prompt text for iterative generation strategy.

        Default format: "f_action_name([args])"
        Override for custom formatting.
        """
        if not self.requires_args:
            return f"f_{self.name}()"
        return f"f_{self.name}([args])"

    def get_prompt_text_cot(self) -> str:
        """
        Get prompt text for CoT generation strategy.

        Default: just the action name with f_ prefix.
        Override if needed.
        """
        return f"f_{self.name}"

    def get_fuzzy_match_keywords(self) -> List[str]:
        """
        Get keywords for fuzzy matching in parse_name_only().

        Default: [self.name]
        Override to add additional keywords (e.g., ["row", "select_row"])
        """
        return [self.name]

    def get_description(self) -> str:
        """
        Get human-readable description of what this action does.

        This is shown to the language model in prompts to help it understand
        available actions (as per Chain-of-Table paper, Figure 9).

        Default: Returns the action name.
        Override to provide a meaningful description.
        """
        return self.name


class ActionRegistry:
    """
    Central registry for all table actions.

    Provides single source of truth for:
    - Available actions
    - Constraint generation
    - Prompt building
    - Argument parsing
    """

    def __init__(self):
        self._actions: Dict[str, ActionDefinition] = {}
        self._enabled_actions: Optional[Set[str]] = None

    def register(self, action: ActionDefinition):
        """Register an action definition."""
        self._actions[action.name] = action

    def get(self, name: str) -> Optional[ActionDefinition]:
        """Get action definition by name."""
        return self._actions.get(name)

    def set_enabled_actions(self, enabled: List[str]):
        """
        Set which actions are enabled for this run.

        This filters actions based on experiment configuration.
        """
        self._enabled_actions = set(enabled)

    def get_enabled_names(self) -> List[str]:
        """Get list of enabled action names."""
        if self._enabled_actions is None:
            return list(self._actions.keys())
        return [name for name in self._actions.keys() if name in self._enabled_actions]

    def get_all_names(self) -> List[str]:
        """Get all registered action names (regardless of enabled status)."""
        return list(self._actions.keys())

    def is_enabled(self, name: str) -> bool:
        """Check if an action is enabled."""
        if self._enabled_actions is None:
            return name in self._actions
        return name in self._enabled_actions and name in self._actions

    def generate_constraint_params(self, table: Table) -> Dict[str, List[str]]:
        """
        Generate constraint parameters for all enabled actions.

        Returns dict like:
        {
            "select_row": ["row 0", "row 1", ...],
            "select_column": ["col1", "col2", ...],
            "end": []
        }
        """
        params = {}
        for name in self.get_enabled_names():
            action = self._actions[name]
            params[name] = action.generate_params(table)
        return params

    def get_prompt_text_iterative(self, action_history: Optional[List[str]] = None) -> str:
        """
        Generate prompt text for iterative strategy.

        Example: "Choose from: select_row([row_indices]), select_column([\"column_names\"]), or end()"

        Args:
            action_history: List of actions already taken. If provided, these actions
                          will be filtered out from the available options (except 'end').
        """
        enabled = self.get_enabled_names()

        # Filter out already-used actions (but always keep 'end' as an option)
        if action_history:
            used_actions = self._extract_action_names_from_history(action_history)
            enabled = [name for name in enabled if name not in used_actions or name == "end"]

        if not enabled:
            return ""

        action_texts = [self._actions[name].get_prompt_text_iterative() for name in enabled]

        if len(action_texts) == 1:
            return action_texts[0]
        elif len(action_texts) == 2:
            return f"{action_texts[0]} or {action_texts[1]}"
        else:
            # Multiple actions: "op1, op2, or op3"
            return ", ".join(action_texts[:-1]) + f", or {action_texts[-1]}"

    def get_prompt_text_cot(self, action_history: Optional[List[str]] = None, exclude_terminating_on_first: bool = True) -> str:
        """
        Generate prompt text for CoT strategy.

        Example: "Available actions: select_row, select_column, end"

        Args:
            action_history: List of actions already taken. If provided, these actions
                          will be filtered out from the available options (except 'end').
            exclude_terminating_on_first: If True, exclude terminating actions when action_history is empty.
                                         This follows the Chain-of-Table paper convention.
        """
        enabled = self.get_enabled_names()

        # On first step (empty history), exclude terminating actions
        if exclude_terminating_on_first and (not action_history or len(action_history) == 0):
            enabled = [name for name in enabled if not self._actions[name].is_terminating]
        # Filter out already-used actions (but always keep 'end' as an option)
        elif action_history:
            used_actions = self._extract_action_names_from_history(action_history)
            enabled = [name for name in enabled if name not in used_actions or name == "end"]

        action_texts = [self._actions[name].get_prompt_text_cot() for name in enabled]
        return ", ".join(action_texts)

    def get_action_descriptions(self, action_history: Optional[List[str]] = None, exclude_terminating_on_first: bool = True) -> str:
        """
        Generate formatted action descriptions for prompts.

        Returns a multi-line string describing each available action,
        as shown in Chain-of-Table paper (Figure 9, Appendix).

        Example output:
        "Operations:
        select_row: selects a subset of rows
        select_column: selects a subset of columns
        end: indicates completion
        direct_query: answer directly without transformation"

        Args:
            action_history: List of actions already taken. If provided, these actions
                          will be filtered out from the available options (except 'end').
            exclude_terminating_on_first: If True, exclude terminating actions when action_history is empty.
                                         This follows the Chain-of-Table paper convention.
        """
        enabled = self.get_enabled_names()

        # On first step (empty history), exclude terminating actions
        if exclude_terminating_on_first and (not action_history or len(action_history) == 0):
            enabled = [name for name in enabled if not self._actions[name].is_terminating]
        # Filter out already-used actions (but always keep 'end' as an option)
        elif action_history:
            used_actions = self._extract_action_names_from_history(action_history)
            enabled = [name for name in enabled if name not in used_actions or name == "end"]

        if not enabled:
            return ""

        lines = ["Operations:"]
        for name in enabled:
            action = self._actions[name]
            desc = action.get_description()
            lines.append(f"f_{name}: {desc}")

        return "\n".join(lines)

    def parse_action_name_fuzzy(self, text: str) -> Optional[str]:
        """
        Fuzzy match action name from text.

        Replaces Action.parse_name_only() logic.
        """

        names_and_args = text.strip().lower().split("->")
        names = [aa.strip().split("(")[0].replace("\\", "").strip() for aa in names_and_args]

        # Try fuzzy matching with keywords (strip f_ prefix if present)
        for name in names:
            lookup_name = name[2:] if name.startswith("f_") else name
            if lookup_name in self._actions and self.is_enabled(lookup_name):
                return lookup_name

        # Try keyword matching
        for name in self.get_enabled_names():
            action = self._actions[name]
            keywords = action.get_fuzzy_match_keywords()
            for keyword in keywords:
                if keyword.lower() in text.strip().lower():
                    return name

        return None

    def _extract_action_names_from_history(self, action_history: List[str]) -> set[str]:
        """
        Extract unique action names from action history.

        Args:
            action_history: List of action strings like "select_row([0, 1])", "select_column(['Name'])"

        Returns:
            Set of action names like {"select_row", "select_column"}
        """
        action_names = set()
        for action_str in action_history:
            # Extract action name from strings like "select_row([0, 1])"
            # Find the first '(' to get the action name
            paren_idx = action_str.find("(")
            if paren_idx > 0:
                action_name = action_str[:paren_idx].strip()
                # Strip f_ prefix if present (action strings may be formatted as "f_select_row(...)")
                if action_name.startswith("f_"):
                    action_name = action_name[2:]
                action_names.add(action_name)
            else:
                # Handle cases where action might not have parentheses
                # Try to match against known action names
                action_str_clean = action_str.strip().lower()
                for name in self._actions.keys():
                    if action_str_clean.startswith(name):
                        action_names.add(name)
                        break
        return action_names


# Global registry instance
REGISTRY = ActionRegistry()
