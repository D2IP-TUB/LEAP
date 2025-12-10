"""
Select Column Action - Select specific columns by name.
"""

import re
from typing import Any, List, Optional

from leap.core.action_registry import ActionDefinition
from leap.core.table import Table


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

    def parse_arguments(self, args_str: str, table: Table) -> Optional[List[Any]]:
        """
        Parse column names from argument string.

        Handles formats like:
        - '["col1", "col2"]'
        - '"col1", "col2"'
        - "col1, col2"
        """
        args_str = args_str.strip()

        if not args_str or args_str == "[]":
            return []

        # Try to extract column names
        try:
            if args_str.startswith("[") and args_str.endswith("]"):
                # List format
                import ast

                columns = ast.literal_eval(args_str)
            else:
                # Comma-separated, possibly quoted
                columns = [col.strip().strip('"').strip("'") for col in args_str.split(",")]

            return columns
        except Exception:
            return None

    def extract_arguments_from_text(self, text: str, table: Table) -> Optional[List[Any]]:
        """
        Extract column names from free-form text (CoT).

        Tries various patterns to find column names.
        """
        text = text.strip()

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

    def apply(self, table: Table, arguments: List[Any]) -> Optional[Table]:
        """Apply column selection to table."""
        return table.select_columns(list(arguments))

    def validate(self, table: Table, arguments: List[Any]) -> bool:
        """Validate column names."""
        if len(arguments) == 0:
            return False

        for col in arguments:
            if col not in table.columns:
                return False

        return True

    def get_prompt_text_iterative(self) -> str:
        """Prompt text for iterative generation."""
        return 'select_column(["column_names"])'

    def get_fuzzy_match_keywords(self) -> List[str]:
        """Keywords for fuzzy matching."""
        return ["select_column", "column"]
