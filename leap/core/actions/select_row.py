"""
Select Row Action - Select specific rows by index.
"""

import re
from typing import Any, List, Optional

from leap.core.action_registry import ActionDefinition
from leap.core.table import Table


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

    def parse_arguments(self, args_str: str, table: Table) -> Optional[List[Any]]:
        """
        Parse row indices from argument string.

        Handles formats like:
        - "[0, 1, 2]"
        - "0, 1, 2"
        - "row 0, row 1" (from constraints)
        """
        # This logic is handled by Action.parse() already,
        # but we provide it here for completeness
        args_str = args_str.strip()

        if not args_str or args_str == "[]":
            return []

        # Remove "row " prefix if present (constraint output format)
        args_str = args_str.replace("row ", "")

        # Try to extract numbers
        try:
            if args_str.startswith("[") and args_str.endswith("]"):
                # List format
                import ast

                indices = ast.literal_eval(args_str)
            else:
                # Comma-separated format
                indices = [int(x.strip()) for x in args_str.split(",")]

            return indices
        except Exception:
            return None

    def extract_arguments_from_text(self, text: str, table: Table) -> Optional[List[Any]]:
        """
        Extract row indices from free-form text (CoT).

        Tries various patterns to find row indices.
        """
        text = text.strip()

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
                    # Filter valid indices
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

        return None

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

    def get_fuzzy_match_keywords(self) -> List[str]:
        """Keywords for fuzzy matching."""
        return ["select_row", "row"]
