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

    def parse_arguments(self, args_str: str, table: Table) -> Optional[Tuple[str, str]]:
        """
        Parse column name and order from argument string.

        Expected formats:
        - "Points, descending"
        - "Year, asc"
        - "column_name, order"

        Returns:
            Tuple of (column_name, order) or None if parsing fails
            Order is normalized to "asc" or "desc"
        """
        args_str = args_str.strip()

        if not args_str or args_str == "[]":
            return None

        # Try to split on comma
        parts = args_str.split(",")
        if len(parts) != 2:
            return None

        column_name = parts[0].strip().strip('"').strip("'")
        order_str = parts[1].strip().strip('"').strip("'").lower()

        # Validate column exists
        if column_name not in table.columns:
            return None

        # Normalize order
        order = self._normalize_order(order_str)
        if order is None:
            return None

        return (column_name, order)

    def _normalize_order(self, order_str: str) -> Optional[str]:
        """
        Normalize order string to 'asc' or 'desc'.

        Accepts:
        - ascending, asc, a
        - descending, desc, d
        - from-small-to-large
        - from-large-to-small
        """
        order_str = order_str.lower().strip()

        # Ascending patterns
        if any(pattern in order_str for pattern in ["asc", "small-to-large", "low-to-high", "increasing"]):
            return "asc"

        # Descending patterns
        if any(pattern in order_str for pattern in ["desc", "large-to-small", "high-to-low", "decreasing"]):
            return "desc"

        return None

    def extract_arguments_from_text(self, text: str, table: Table) -> Optional[Tuple[str, str]]:
        """
        Extract column name and order from CoT text output.

        From paper (Figure 14, page 20), expected format:
        "Therefore, the answer is: f_sort_by(Position), the order is 'large to small'."

        Patterns to match:
        - "f_sort_by(ColumnName), the order is 'order'"
        - "sort_by(ColumnName), order"
        """
        text = text.strip()

        # Pattern 1: sort_by(ColumnName)...order is "..."
        pattern1 = r"sort_by\s*\(\s*([^)]+)\s*\).*?order\s+is\s+['\"]?([^'\",.]+)['\"]?"
        match = re.search(pattern1, text, re.IGNORECASE | re.DOTALL)

        if match:
            column_name = match.group(1).strip().strip('"').strip("'")
            order_str = match.group(2).strip()

            if column_name in table.columns:
                order = self._normalize_order(order_str)
                if order:
                    return (column_name, order)

        # Pattern 2: sort_by(ColumnName, order)
        pattern2 = r"sort_by\s*\(\s*([^,)]+)\s*,\s*([^)]+)\s*\)"
        match = re.search(pattern2, text, re.IGNORECASE)

        if match:
            column_name = match.group(1).strip().strip('"').strip("'")
            order_str = match.group(2).strip().strip('"').strip("'")

            if column_name in table.columns:
                order = self._normalize_order(order_str)
                if order:
                    return (column_name, order)

        # Pattern 3: Just column name, look for order separately
        pattern3 = r"sort_by\s*\(\s*([^)]+)\s*\)"
        match = re.search(pattern3, text, re.IGNORECASE)

        if match:
            column_name = match.group(1).strip().strip('"').strip("'")

            if column_name in table.columns:
                # Look for order in surrounding text
                if "descend" in text.lower() or "large-to-small" in text.lower():
                    return (column_name, "desc")
                elif "ascend" in text.lower() or "small-to-large" in text.lower():
                    return (column_name, "asc")
                else:
                    # Default to descending (most common in paper examples)
                    return (column_name, "desc")

        return None

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

    def validate(self, table: Table, arguments: Tuple[str, str]) -> bool:
        """
        Validate sort_by arguments.

        Checks:
        - Arguments is a tuple of (str, str)
        - Column name exists
        - Order is valid
        - Table has at least one row
        """
        if not isinstance(arguments, tuple) or len(arguments) != 2:
            return False

        column_name, order = arguments

        # Check column name
        if not isinstance(column_name, str) or not column_name:
            return False

        if column_name not in table.columns:
            return False

        # Check order
        if order not in ["asc", "desc"]:
            return False

        if len(table.rows) == 0:
            return False

        return True

    def get_prompt_text_iterative(self) -> str:
        """Prompt text for iterative generation."""
        return "sort_by(column_name, order)"

    def get_prompt_text_cot(self) -> str:
        """Prompt text for CoT generation."""
        return "sort_by"

    def get_fuzzy_match_keywords(self) -> List[str]:
        """Keywords for fuzzy matching."""
        return ["sort_by", "sort", "order"]

    def get_description(self) -> str:
        """Action description for prompts."""
        return "sorts rows by column values in ascending or descending order"
