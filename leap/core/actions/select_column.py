"""
Select Column Action - Select specific columns by name.
"""

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

    def parse_arguments(self, args_str: str) -> Optional[List[Any]]:
        """
        Parse column names from argument string.

        Handles formats like:
        - '["col1", "col2"]'
        - '"col1", "col2"'
        - "col1, col2"
        """
        args_str = args_str.strip()
        # Normalize unicode/fancy quotes to plain ASCII
        args_str = args_str.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")

        if not args_str or args_str == "[]":
            return []

        # Try to extract column names
        try:
            import ast

            # Extract bracketed list if present (ignore extra text after closing bracket)
            bracket_start = args_str.find("[")
            bracket_end = args_str.rfind("]")
            if bracket_start >= 0 and bracket_end > bracket_start:
                bracketed = args_str[bracket_start : bracket_end + 1]
                try:
                    columns = ast.literal_eval(bracketed)
                except Exception:
                    # Unquoted column names inside brackets, e.g. [Name in English, Depth]
                    inner = bracketed[1:-1]
                    columns = [col.strip().strip('"').strip("'") for col in inner.split(",")]
            else:
                # Comma-separated, possibly quoted — take only up to any sentence-ending punctuation
                columns = [col.strip().strip('"').strip("'") for col in args_str.split(",")]

            # Normalize newlines to spaces — table columns have \n replaced with space at load time
            return [str(col).replace("\n", " ") for col in columns]
        except Exception:
            return None

    def apply(self, table: Table, arguments: List[Any]) -> Optional[Table]:
        """Apply column selection to table."""
        return table.select_columns(list(arguments))

    def validate(self, table: Table, arguments: List[Any]) -> bool:
        """Validate column names."""
        if len(arguments) == 0:
            return False

        for col in arguments:
            if table.resolve_column(col) is None:
                return False

        return True

    def get_prompt_text_iterative(self) -> str:
        """Prompt text for iterative generation."""
        return 'f_select_column(["column_names"])'

    def get_description(self) -> str:
        """Action description for prompts."""
        return "selects a subset of columns from the table"
