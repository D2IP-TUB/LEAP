"""
Select Row Action - Select specific rows by index.
"""

import re
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

        Only two formats are accepted:
        - "[row 0, row 1, row 2]"  — normal selection
        - "[*]"                    — select all rows
        """
        args_str = args_str.strip()

        if not args_str:
            return None

        # Handle wildcard [*]
        if args_str.strip() == "[*]":
            return ["*"]

        pattern = r'\[\s*(?:"row\s+\d+"(?:\s*,\s*"row\s+\d+")*|row\s+\d+(?:\s*,\s*row\s+\d+)*)\s*\]'
        if not re.fullmatch(pattern, args_str):
            return None

        try:
            indices = [int(x) for x in re.findall(r"row\s+(\d+)", args_str)]
            return indices
        except Exception:
            return None

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
            if not isinstance(idx, int) or idx < 0 or idx >= len(table.rows):
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
