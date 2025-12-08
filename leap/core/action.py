"""Action abstraction - unified action parsing and validation"""

import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .table import Table


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

            # Comma-separated format: arg1, arg2, arg3
            if "," in args_str:
                args_list = [arg.strip() for arg in args_str.split(",")]
            else:
                args_list = [args_str]

            # Convert string digits to ints for select_row
            if action_name.lower() == "select_row":
                args_list = cls._normalize_row_indices(args_list)

            return cls(action_name, args_list)

        except Exception:
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

        Replaces generate.parse_action_name()

        Used for fuzzy matching when full parsing fails.
        """
        action_str = action_str.strip().lower()

        # Exact matches
        if action_str in ["select_row", "select_column", "end"]:
            return action_str

        # Fuzzy matching
        if "select_row" in action_str or "row" in action_str:
            return "select_row"
        elif "select_column" in action_str or "column" in action_str:
            return "select_column"
        elif "end" in action_str or "finish" in action_str or "done" in action_str:
            return "end"

        return None

    @classmethod
    def extract_from_text(cls, text: str, action_name: str, table: Table) -> Optional["Action"]:
        """
        Extract action from free-form text (used in Chain-of-Table).

        Replaces generate.extract_arguments_from_text()

        This handles the two-phase CoT generation where action name and
        arguments are generated separately.
        """
        arguments = cls._extract_arguments_from_text(text, action_name, table)
        if arguments is None:
            return None
        return cls(action_name, arguments)

    @staticmethod
    def _extract_arguments_from_text(text: str, action_name: str, table: Table) -> Optional[List]:
        """
        Internal helper: extract arguments for a specific action from free-form text.

        This is the logic from generate.extract_arguments_from_text()
        """
        text = text.strip()

        if action_name == "end":
            return []

        elif action_name == "select_row":
            # Try various patterns for row indices
            patterns = [
                r"\[([0-9,\s]+)\]",
                r"(\d+(?:\s*,\s*\d+)*)",
                r"rows?\s+(\d+(?:\s*,\s*\d+)*)",
                r"indices?\s+(\d+(?:\s*,\s*\d+)*)",
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        indices_str = match.group(1)
                        indices = [int(x.strip()) for x in indices_str.split(",")]
                        valid_indices = [idx for idx in indices if 0 <= idx < len(table.rows)]
                        if valid_indices:
                            return valid_indices
                    except Exception:
                        continue

            # Fallback: extract all numbers
            numbers = re.findall(r"\b(\d+)\b", text)
            if numbers:
                try:
                    indices = [int(x) for x in numbers]
                    valid_indices = [idx for idx in indices if 0 <= idx < len(table.rows)]
                    if valid_indices:
                        return valid_indices[:5]  # Limit to 5
                except Exception:
                    pass

        elif action_name == "select_column":
            # Try various patterns for column names
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
                                if col and col.strip("\"'") in table.columns:
                                    mentioned_columns.append(col.strip("\"'"))
                        else:
                            col_matches = re.findall(r'["\']([^"\']+)["\']', match)
                            for col in col_matches:
                                if col in table.columns:
                                    mentioned_columns.append(col)

            if mentioned_columns:
                return list(set(mentioned_columns))

            # Fallback: check if column names appear in text
            for col in table.columns:
                if col.lower() in text.lower():
                    mentioned_columns.append(col)

            if mentioned_columns:
                return list(set(mentioned_columns[:3]))  # Limit to 3

        return None

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

        Replaces table.apply_action()
        """
        if self.name == "select_row":
            return table.select_rows(list(self.arguments))
        elif self.name == "select_column":
            return table.select_columns(list(self.arguments))
        elif self.name == "end":
            return table
        return None

    def is_valid_for_table(self, table: Table) -> bool:
        """
        Validate that this action can be applied to the given table.
        """
        if self.name == "end":
            return True
        elif self.name == "select_row":
            # Check if all indices are valid
            for idx in self.arguments:
                if isinstance(idx, int):
                    if idx < 0 or idx >= len(table.rows):
                        return False
                elif isinstance(idx, str) and idx.isdigit():
                    idx_int = int(idx)
                    if idx_int < 0 or idx_int >= len(table.rows):
                        return False
                else:
                    return False
            return len(self.arguments) > 0
        elif self.name == "select_column":
            # Check if all columns exist
            for col in self.arguments:
                if col not in table.columns:
                    return False
            return len(self.arguments) > 0
        return False

    def __repr__(self) -> str:
        return f"Action({self.name}, args={list(self.arguments)})"
