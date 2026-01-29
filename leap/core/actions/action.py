"""Action abstraction - unified action parsing and validation"""

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..table import Table


@dataclass(frozen=True)
class Action:
    """
    Immutable action representation.

    Single source of truth for action parsing and validation.
    """

    name: str  # 'select_row', 'select_column', 'end'
    arguments: tuple[Any, ...]  # Immutable tuple

    def __init__(self, name: str, arguments: List[Any]):
        """Initialize with mutable list, convert to immutable internally"""
        object.__setattr__(self, "name", name.lower())
        object.__setattr__(self, "arguments", tuple(arguments))

    @classmethod
    def parse(cls, action_str: str) -> Optional["Action"]:
        """
        Parse action string into Action object.

        Formats supported:
        - "end" or "end()"
        - "select_row([0, 1, 2])"
        - "select_column(['col1', 'col2'])"
        - "action(arg1, arg2, arg3)"

        For select_row, string digits like "0" are automatically converted to int.
        """
        try:
            action_str = action_str.strip()

            # Handle "end" special case
            if action_str == "end" or action_str.startswith("end("):
                return cls("end", [])

            # Require parentheses for non-end actions
            if "(" not in action_str:
                return None

            # Split into action name and arguments
            action_name, args_str = action_str.split("(", 1)
            action_name = action_name.strip()
            args_str = args_str.rstrip(")").replace("row ", "").strip()

            # Parse arguments
            if not args_str:
                return cls(action_name, [])

            # Try list literal format: [1, 2, 3]
            if args_str.startswith("[") and args_str.endswith("]"):
                args_str = args_str.replace("\\", "\\\\")
                args_list = ast.literal_eval(args_str)

                # Convert string digits to ints for select_row
                if action_name.lower() == "select_row":
                    args_list = cls._normalize_row_indices(args_list)

                return cls(action_name, args_list)

            if action_name.lower() == "add_column" and "[" in args_str:
                args = args_str.split("[")
                column = args[0].strip(",\"").strip()
                value_string = "[" + args[1].strip()

                try:
                    values = ast.literal_eval(value_string)
                except Exception:
                    values = value_string.strip().replace("[", "").replace("]", "").strip().split(",")
                    values = [v.replace("\"", "").strip()for v in values]


                return cls(action_name, [column, values])

            if "," in args_str:
                args_list = [arg.strip() for arg in args_str.split(",")]
            else:
                args_list = [args_str]

            # Convert string digits to ints for select_row
            if action_name.lower() == "select_row":
                args_list = cls._normalize_row_indices(args_list)

            return cls(action_name, args_list)

        except Exception:
            print(f"[ERROR]: Failed to parse action: {action_str}")
            return None

    @staticmethod
    def _normalize_row_indices(args_list: List) -> List:
        """
        Normalize row indices by converting string digits to ints.

        This handles the constraint system output format where indices
        come as strings like ["0", "1"] and converts them to [0, 1].

        Args:
            args_list: List of arguments that may contain string digits

        Returns:
            List with string digits converted to ints
        """
        normalized = []
        for arg in args_list:
            if isinstance(arg, str) and arg.isdigit():
                normalized.append(int(arg))
            elif isinstance(arg, int):
                normalized.append(arg)
            else:
                # Non-numeric string - keep as is (will fail validation later)
                normalized.append(arg)
        return normalized

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

    def __repr__(self) -> str:
        return f"Action({self.name}, args={list(self.arguments)})"
