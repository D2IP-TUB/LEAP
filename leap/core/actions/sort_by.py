"""
Sort By Action - Sort table rows by column values.

This is a core operation from the Chain-of-Table paper for ordering/comparison questions.
Sorts rows by a specified column in ascending or descending order.

Reference: Chain-of-Table paper (arXiv:2401.04398v2), Appendix A
"""

import re
from typing import List, Optional, Tuple

from ..table import Table
from .registry import ActionDefinition


class SortByAction(ActionDefinition):
    """
    Sort rows by column value.

    Usage: sort_by(column_name, order)

    This operation:
    1. Sorts all rows by the specified column
    2. Order can be "ascending"/"asc" or "descending"/"desc"
    3. Returns a new table with sorted rows

    Example:
        Input table:
            Rank | Cyclist      | Points
            1    | Alejandro    | 40
            2    | Alexandr     | 30
            3    | Davide       | 25

        sort_by(Points, descending) produces:
            Rank | Cyclist      | Points
            1    | Alejandro    | 40
            2    | Alexandr     | 30
            3    | Davide       | 25

        sort_by(Points, ascending) produces:
            Rank | Cyclist      | Points
            3    | Davide       | 25
            2    | Alexandr     | 30
            1    | Alejandro    | 40

    Common use cases:
    - "What was the highest score?" → sort_by(Score, descending)
    - "Who finished first?" → sort_by(Time, ascending)
    - "What was the last year?" → sort_by(Year, descending)
    """

    @property
    def name(self) -> str:
        return "sort_by"

    @property
    def requires_args(self) -> bool:
        return True

    def generate_params(self, table: Table) -> List[str]:
        """
        Generate parameters for constraint system.

        Returns all column names as possible sorting columns.
        """
        return list(table.columns)


    def apply(self, table: Table, arguments: Tuple[str, str]) -> Optional[Table]:
        """
        Apply sort_by to table.

        Args:
            table: Input table
            arguments: Tuple of (column_name, order)

        Returns:
            New table with sorted rows, or None if invalid
        """
        if not isinstance(arguments, tuple) or len(arguments) != 2:
            return None

        column_name, order = arguments
        # Validate column exists
        if column_name not in table.columns:
            return None

        # Validate order
        if order not in ["asc", "desc"]:
            return None

        # Get column index
        col_idx = table.columns.index(column_name)
        # Sort rows
        def sort_key(row):
            value = row[col_idx]
            # Try to convert to number for proper numerical sorting
            try:
                return float(value)
            except (ValueError, TypeError):
                # Fall back to string comparison
                return str(value)

        sorted_rows = sorted(table.rows, key=sort_key, reverse=(order == "desc"))

        return Table(columns=list(table.columns), rows=sorted_rows)

    def get_prompt_text_iterative(self) -> str:
        """Prompt text for iterative generation."""
        return "sort_by(column_name, order)"

    def get_prompt_text_cot(self) -> str:
        """Prompt text for CoT generation."""
        return "sort_by"

    def get_description(self) -> str:
        """Action description for prompts."""
        return "sorts rows by column values in ascending or descending order"
