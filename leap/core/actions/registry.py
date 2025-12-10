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
    def parse_arguments(self, args_str: str, table: Table) -> Optional[List[Any]]:
        """
        Parse arguments from LLM output.

        Called by Action.parse() for this specific operation.
        Return None if parsing fails.
        """
        pass

    @abstractmethod
    def extract_arguments_from_text(self, text: str, table: Table) -> Optional[List[Any]]:
        """
        Extract arguments from free-form text (for CoT two-phase generation).

        Called by Action.extract_from_text() for this specific operation.
        Return None if extraction fails.
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

    @abstractmethod
    def validate(self, table: Table, arguments: List[Any]) -> bool:
        """
        Validate that arguments are valid for this table.

        Called by Action.is_valid_for_table() for this specific operation.
        """
        pass

    def get_prompt_text_iterative(self) -> str:
        """
        Get prompt text for iterative generation strategy.

        Default format: "action_name([args])"
        Override for custom formatting.
        """
        if not self.requires_args:
            return f"{self.name}()"
        return f"{self.name}([args])"

    def get_prompt_text_cot(self) -> str:
        """
        Get prompt text for CoT generation strategy.

        Default: just the action name.
        Override if needed.
        """
        return self.name

    def get_fuzzy_match_keywords(self) -> List[str]:
        """
        Get keywords for fuzzy matching in parse_name_only().

        Default: [self.name]
        Override to add additional keywords (e.g., ["row", "select_row"])
        """
        return [self.name]


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

    def get_prompt_text_iterative(self) -> str:
        """
        Generate prompt text for iterative strategy.

        Example: "Choose from: select_row([row_indices]), select_column([\"column_names\"]), or end()"
        """
        enabled = self.get_enabled_names()
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

    def get_prompt_text_cot(self) -> str:
        """
        Generate prompt text for CoT strategy.

        Example: "Available actions: select_row, select_column, end"
        """
        enabled = self.get_enabled_names()
        action_texts = [self._actions[name].get_prompt_text_cot() for name in enabled]
        return ", ".join(action_texts)

    def parse_action_name_fuzzy(self, text: str) -> Optional[str]:
        """
        Fuzzy match action name from text.

        Replaces Action.parse_name_only() logic.
        """
        text = text.strip().lower()

        # Try exact match first
        if text in self._actions and self.is_enabled(text):
            return text

        # Try fuzzy matching with keywords
        for name in self.get_enabled_names():
            action = self._actions[name]
            keywords = action.get_fuzzy_match_keywords()
            for keyword in keywords:
                if keyword.lower() in text:
                    return name

        return None


# Global registry instance
REGISTRY = ActionRegistry()
