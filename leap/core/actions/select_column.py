"""
Select Column Action - Select specific columns by name.
"""

import re
from typing import Any, List, Optional

from ..table import Table
from .registry import ActionDefinition


class SelectColumnAction(ActionDefinition):
    """
    Select columns by name.

    Usage: select_column(["col1", "col2"])
    """

    @property
    def name(self) -> str:
        return "select_column"

    def generate_params(self, table: Table) -> List[str]:
        """Generate column parameters for constraint system."""
        return list(table.columns)



    def apply(self, table: Table, arguments: List[Any]) -> Optional[Table]:
        """Apply column selection to table."""
        return table.select_columns(list(arguments))

    def get_prompt_text_iterative(self) -> str:
        """Prompt text for iterative generation."""
        return 'select_column(["column_names"])'

    def get_description(self) -> str:
        """Action description for prompts."""
        return "selects a subset of columns from the table"
