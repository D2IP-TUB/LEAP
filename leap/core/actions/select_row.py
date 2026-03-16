"""
Select Row Action - Select specific rows by index.
"""

import ast
from typing import Any, List, Optional

from ..table import Table
from .registry import ActionDefinition


class SelectRowAction(ActionDefinition):
    """
    Select rows by index.

    Usage: select_row([0, 1, 2])
    """

    @property
    def name(self) -> str:
        return "select_row"

    def generate_params(self, table: Table) -> List[str]:
        """Generate row parameters for constraint system."""
        num_rows = len(table.rows)
        return [f"row {i}" for i in range(num_rows)]

    def parse_arguments(self, args_str: str) -> Optional[List[Any]]:
        """
        Parse row indices from argument string.

        Handles formats like:
        - "[0, 1, 2]"
        - "0, 1, 2"
        - "row 0, row 1" (from constraints)
        - '["0", "1"]' (string digits from constraint system)
        """
        args_str = args_str.strip()

        if not args_str or args_str == "[]":
            return []

        # Remove "row " prefix if present (constraint output format)
        args_str_clean = args_str.replace("row ", "")

        # Handle wildcard [*] — select all rows (expanded later in validate/apply via table)
        if args_str_clean.strip() in ("[*]", "*"):
            return ["*"]

        try:
            # Extract bracketed list if present (ignore extra text after closing bracket)
            bracket_start = args_str_clean.find("[")
            bracket_end = args_str_clean.rfind("]")
            if bracket_start >= 0 and bracket_end > bracket_start:
                args_str_clean = args_str_clean[bracket_start : bracket_end + 1]
                args_str_clean = args_str_clean.replace("\\", "\\\\")
                indices = ast.literal_eval(args_str_clean)
            else:
                indices = [x.strip() for x in args_str_clean.split(",")]

            return self._normalize_row_indices(indices)
        except Exception:
            return None

    @staticmethod
    def _normalize_row_indices(args_list: List) -> List:
        """Convert string digits to ints (constraint system outputs indices as strings)."""
        normalized = []
        for arg in args_list:
            if isinstance(arg, str):
                arg = arg.replace("row ", "").strip()
                if arg.isdigit():
                    normalized.append(int(arg))
                else:
                    normalized.append(arg)  # Keep as-is; will fail validation
            else:
                normalized.append(arg)
        return normalized

    def apply(self, table: Table, arguments: List[Any]) -> Optional[Table]:
        """Apply row selection to table."""
        if list(arguments) == ["*"]:
            return table.select_rows(list(range(len(table.rows))))
        return table.select_rows(list(arguments))

    def validate(self, table: Table, arguments: List[Any]) -> bool:
        """Validate row indices."""
        if len(arguments) == 0:
            return False

        if list(arguments) == ["*"]:
            return len(table.rows) > 0

        for idx in arguments:
            if isinstance(idx, int):
                if idx < 0 or idx >= len(table.rows):
                    return False
            elif isinstance(idx, str) and idx.isdigit():
                idx_int = int(idx)
                if idx_int < 0 or idx_int >= len(table.rows):
                    return False
            else:
                return False

        return True

    def get_prompt_text_iterative(self) -> str:
        """Prompt text for iterative generation."""
        return "f_select_row([row_indices])"

    def get_fuzzy_match_keywords(self) -> List[str]:
        """Keywords for fuzzy matching."""
        return ["select_row", "row"]

    def get_description(self) -> str:
        """Action description for prompts."""
        return "selects a subset of rows from the table"
