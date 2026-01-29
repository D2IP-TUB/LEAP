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


    def apply(self, table: Table, arguments: List[Any]) -> Optional[Table]:
        """Apply row selection to table."""
        return table.select_rows(list(arguments))

    def validate(self, table: Table, arguments: List[Any]) -> bool:
        """Validate row indices."""
        if len(arguments) == 0:
            return False

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
        return "select_row([row_indices])"

    def get_description(self) -> str:
        """Action description for prompts."""
        return "selects a subset of rows from the table"
