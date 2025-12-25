"""
Add Column Action - Add a new column with LLM-generated values.

This is a generative action from the Chain-of-Table paper. Unlike select_row/select_column,
this action requires the LLM to generate new data: both the column name and the values
for each row.

Similar to direct_query in being generative, but different in placement:
- direct_query: generates final answer (terminating)
- add_column: generates intermediate column (non-terminating)

Reference: Chain-of-Table paper (arXiv:2401.04398v2), Appendix A
"""

import re
from typing import List, Optional, Tuple

from ..table import Table
from .registry import ActionDefinition


class AddColumnAction(ActionDefinition):
    """
    Add a new column with computed/extracted values.

    Usage: add_column(column_name, [value1, value2, ...])

    This is a generative operation where the LLM:
    1. Determines what column to add (e.g., "Country", "Year")
    2. Generates a value for each row in the table

    Common use cases (from paper):
    - Extract structured data: "Country" from "Cyclist (Country)" column
    - Parse dates: "Year" from "Date" column
    - Create numerical columns: "Attendance Number" from text
    - Extract categorical data for grouping/filtering
    """

    @property
    def name(self) -> str:
        return "add_column"

    @property
    def requires_args(self) -> bool:
        return True

    def generate_params(self, table: Table) -> List[str]:
        """
        Generate parameters for constraint system.

        For add_column, there are no fixed constraints - the LLM generates
        both the column name and values freely based on the question.
        """
        return []  # No constraints - this is a generative action

    def parse_arguments(self, args_str: str, table: Table) -> Optional[Tuple[str, List[str]]]:
        """
        Parse column name and values from argument string.

        Expected formats (from paper prompts):
        - "Country, [ESP, RUS, ITA, ...]"
        - "Year | [2001, 2002, 2005]"
        - Column name followed by list of values

        Returns:
            Tuple of (column_name, values_list) or None if parsing fails
        """
        args_str = args_str.strip()

        if not args_str:
            return None

        # Try to split on common separators: comma, pipe, colon
        for separator in [",", "|", ":"]:
            if separator in args_str:
                parts = args_str.split(separator, 1)
                if len(parts) == 2:
                    column_name = parts[0].strip()
                    values_str = parts[1].strip()

                    # Parse values list
                    values = self._parse_values_list(values_str)
                    if values is not None:
                        # Validate length matches table
                        if len(values) == len(table.rows):
                            return (column_name, values)

        return None

    def _parse_values_list(self, values_str: str) -> Optional[List[str]]:
        """
        Parse a list of values from string.

        Handles formats:
        - "[ESP, RUS, ITA]"
        - "ESP | RUS | ITA"
        - "ESP, RUS, ITA"
        """
        values_str = values_str.strip()

        # Remove brackets if present
        if values_str.startswith("[") and values_str.endswith("]"):
            values_str = values_str[1:-1].strip()

        if not values_str:
            return None

        # Try different separators
        for sep in ["|", ","]:
            if sep in values_str:
                values = [v.strip() for v in values_str.split(sep)]
                if values:
                    return values

        # Single value
        return [values_str]

    def extract_arguments_from_text(self, text: str, table: Table) -> Optional[Tuple[str, List[str]]]:
        """
        Extract column name and values from CoT text output.

        From paper (Figure 10, page 19), expected format:
        "Therefore, the answer is: f_add_column(Country). The value: ESP | RUS | ITA"

        Patterns to match:
        - "f_add_column(ColumnName). The value: val1 | val2 | ..."
        - "add_column(ColumnName). Values: [val1, val2, ...]"
        """
        text = text.strip()

        # Pattern 1: "f_add_column(Name). The value: ..."
        pattern1 = r"add_column\s*\(\s*([^)]+)\s*\).*?(?:value|values?)\s*:?\s*(.+)"
        match = re.search(pattern1, text, re.IGNORECASE | re.DOTALL)

        if match:
            column_name = match.group(1).strip()
            values_str = match.group(2).strip()

            values = self._parse_values_list(values_str)
            if values and len(values) == len(table.rows):
                return (column_name, values)

        # Pattern 2: Just column name, look for values elsewhere
        pattern2 = r"add_column\s*\(\s*([^)]+)\s*\)"
        match = re.search(pattern2, text, re.IGNORECASE)

        if match:
            column_name = match.group(1).strip()

            # Look for values after the match
            remaining_text = text[match.end() :]
            value_patterns = [
                r"(?:value|values?)\s*:?\s*(.+)",
                r"\[([^\]]+)\]",
            ]

            for vpattern in value_patterns:
                vmatch = re.search(vpattern, remaining_text, re.IGNORECASE)
                if vmatch:
                    values_str = vmatch.group(1).strip()
                    values = self._parse_values_list(values_str)
                    if values and len(values) == len(table.rows):
                        return (column_name, values)

        return None

    def apply(self, table: Table, arguments: Tuple[str, List[str]]) -> Optional[Table]:
        """
        Apply add_column to table.

        Args:
            table: Input table
            arguments: Tuple of (column_name, values)

        Returns:
            New table with added column, or None if invalid
        """
        if not isinstance(arguments, tuple) or len(arguments) != 2:
            return None

        column_name, values = arguments

        # Validate
        if not column_name or not values:
            return None

        if len(values) != len(table.rows):
            return None

        # Create new table with added column
        new_columns = list(table.columns) + [column_name]
        new_rows = []

        for i, row in enumerate(table.rows):
            new_row = list(row) + [values[i]]
            new_rows.append(new_row)

        return Table(columns=new_columns, rows=new_rows)

    def validate(self, table: Table, arguments: Tuple[str, List[str]]) -> bool:
        """
        Validate add_column arguments.

        Checks:
        - Arguments is a tuple of (str, list)
        - Column name is non-empty
        - Values list length matches table row count
        - Column name doesn't already exist
        """
        if not isinstance(arguments, tuple) or len(arguments) != 2:
            return False

        column_name, values = arguments

        # Check column name
        if not column_name or not isinstance(column_name, str):
            return False

        # Check if column already exists
        if column_name in table.columns:
            return False

        # Check values
        if not isinstance(values, list):
            return False

        if len(values) != len(table.rows):
            return False

        # All values should be strings (or convertible)
        for val in values:
            if val is None:
                return False

        return True

    def get_prompt_text_iterative(self) -> str:
        """Prompt text for iterative generation."""
        return "add_column(column_name, [values])"

    def get_prompt_text_cot(self) -> str:
        """Prompt text for CoT generation."""
        return "add_column"

    def get_fuzzy_match_keywords(self) -> List[str]:
        """Keywords for fuzzy matching."""
        return ["add_column", "add", "column", "create"]

    def get_description(self) -> str:
        """Action description for prompts."""
        return "adds a new column with computed or extracted values to the table"
