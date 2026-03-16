"""Action abstraction - unified action parsing and validation"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..table import Table


@dataclass(frozen=True)
class Action:
    """
    Immutable action representation.

    Single source of truth for action parsing and validation.
    """

    name: str  # e.g. 'select_row', 'select_column', 'end'
    arguments: tuple[Any, ...]  # Immutable tuple

    def __init__(self, name: str, arguments: List[Any]):
        """Initialize with mutable list, convert to immutable internally"""
        object.__setattr__(self, "name", name.lower())
        object.__setattr__(self, "arguments", tuple(arguments))

    @classmethod
    def parse(cls, action_str: str) -> Optional["Action"]:
        """
        Parse action string into Action object.

        Delegates argument parsing to the ActionDefinition in the registry,
        keeping Action free of action-specific logic.

        Formats supported:
        - "end" or "end()"
        - "select_row([0, 1, 2])"
        - "select_column(['col1', 'col2'])"
        - "action(arg1, arg2, arg3)"
        """
        try:
            action_str = action_str.strip()

            # Handle "end" special case (with or without f_ prefix)
            if action_str in ("end", "f_end") or action_str.startswith("end(") or action_str.startswith("f_end("):
                return cls("end", [])

            # Require parentheses for non-end actions
            if "(" not in action_str:
                return None

            # Split into action name and arguments string
            action_name, args_str = action_str.split("(", 1)
            action_name = action_name.strip().lower()
            # Strip f_ prefix if present (LLM generates f_action_name)
            if action_name.startswith("f_"):
                action_name = action_name[2:]
            args_str = args_str.rstrip(")").strip()

            # Delegate to ActionDefinition
            from .registry import REGISTRY

            action_def = REGISTRY.get(action_name)
            if action_def is None:
                return None

            arguments = action_def.parse_arguments(args_str)
            if arguments is None:
                return None

            # Normalize to list (parse_arguments may return list, tuple, or scalar)
            if isinstance(arguments, list):
                return cls(action_name, arguments)
            elif isinstance(arguments, tuple):
                return cls(action_name, list(arguments))
            else:
                return cls(action_name, [arguments])

        except Exception:
            print(f"[ERROR]: Failed to parse action: {action_str}")
            return None

    @classmethod
    def parse_name_only(cls, action_str: str) -> Optional[str]:
        """
        Parse action string and return only the action name.

        Delegates to registry for fuzzy matching.
        Used for fuzzy matching when full parsing fails.
        """
        from .registry import REGISTRY

        return REGISTRY.parse_action_name_fuzzy(action_str)

    def to_string(self) -> str:
        """
        Canonical string serialization.

        Format: "action_name(arg1, arg2, arg3)"
        """
        if self.name == "end" or not self.arguments:
            return f"{self.name}()"

        args_str = ", ".join(repr(arg) for arg in self.arguments)
        return f"{self.name}({args_str})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {"action": self.name, "args": list(self.arguments)}

    def apply_to_table(self, table: Table) -> Optional[Table]:
        """
        Apply this action to a table and return new table.

        Delegates to registry action definitions.
        """
        from .registry import REGISTRY

        action_def = REGISTRY.get(self.name)
        if action_def is None:
            return None

        return action_def.apply(table, self.arguments)

    def is_valid_for_table(self, table: Table) -> bool:
        """
        Check whether this action's arguments are valid for the given table.

        Delegates to the ActionDefinition's validate() method.
        """
        from .registry import REGISTRY

        action_def = REGISTRY.get(self.name)
        if action_def is None:
            return False

        if not hasattr(action_def, "validate"):
            return True

        return action_def.validate(table, self.arguments)

    def __repr__(self) -> str:
        return f"Action({self.name}, args={list(self.arguments)})"
