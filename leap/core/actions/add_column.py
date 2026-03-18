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

import ast
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

    def parse_arguments(self, args_str: str) -> Optional[Tuple[str, List[str]]]:
        """
        Parse column name and values from argument string.

        Expected formats (from paper prompts):
        - "Country, [ESP, RUS, ITA, ...]"   (unquoted values — from LLM output)
        - "Country, ['ESP', 'RUS', 'ITA']"  (quoted values — valid Python literal)
        - "Year | [2001, 2002, 2005]"
        - Column name followed by list of values

        Returns:
            Tuple of (column_name, values_list) or None if parsing fails.
            Row count validation is handled by validate().
        """
        args_str = args_str.strip()

        if not args_str:
            return None

        # Normalize unicode/fancy quotes to plain ASCII
        args_str = args_str.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")

        # If model generated "explanation\nThe answer is: add_column(col, [vals])"
        # extract just the arguments from inside the add_column() call
        import re as _re

        match = _re.search(r"add_column\((.+)\)", args_str, _re.DOTALL)
        if match:
            args_str = match.group(1).strip()

        # Primary format: "ColumnName, [val1, val2, ...]"
        # Split on first comma only to separate column name from values list.
        if "[" in args_str and not args_str.startswith("["):
            col_part, values_part = args_str.split(",", 1)
            column_name = col_part.strip().strip('"').strip("'")
            values = self._parse_bracket_values(values_part.strip())
            if values is not None:
                return (column_name, values)

        # Fallback: other separators like pipe or colon
        for separator in ["|", ":"]:
            if separator in args_str:
                parts = args_str.split(separator, 1)
                if len(parts) == 2:
                    column_name = parts[0].strip().strip('"').strip("'")
                    values = self._parse_values_list(parts[1].strip())
                    if values is not None:
                        return (column_name, values)

        return None

    def _parse_bracket_values(self, values_str: str) -> Optional[List[str]]:
        """
        Parse a bracketed list, handling both quoted and unquoted values.

        Tries ast.literal_eval first (handles quoted strings and numbers),
        then falls back to splitting on commas for unquoted strings like [ESP, RUS, ITA].
        """
        values_str = values_str.strip()
        # Normalize unicode/fancy quotes to plain ASCII
        values_str = values_str.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        if not (values_str.startswith("[") and values_str.endswith("]")):
            # Try to find brackets anywhere in the string
            bracket_start = values_str.find("[")
            bracket_end = values_str.rfind("]")
            if bracket_start >= 0 and bracket_end > bracket_start:
                values_str = values_str[bracket_start : bracket_end + 1]
            else:
                return None

        # Try standard Python literal first (quoted strings, numbers)
        try:
            parsed = ast.literal_eval(values_str)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (ValueError, SyntaxError):
            pass

        # Fallback: unquoted values like [ESP, RUS, ITA]
        inner = values_str[1:-1].strip()
        if not inner:
            return None
        return [v.strip().strip('"').strip("'") for v in inner.split(",")]

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

    def apply(self, table: Table, arguments: List) -> Optional[Table]:
        """
        Apply add_column to table.

        Args:
            table: Input table
            arguments: Tuple of (column_name, values)

        Returns:
            New table with added column, or None if invalid
        """

        if len(arguments) != 2:
            return None
        column_name, values = arguments

        if not column_name or not values:
            return None

        if len(values) != len(table.rows):
            print(f"[WARNING]: Values length does not match table rows: {len(values)} != {len(table.rows)}. Skipping...")
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
        - Arguments is a tuple or list of (str, list)
        - Column name is non-empty
        - Values list length matches table row count
        - Column name doesn't already exist
        """
        if not isinstance(arguments, (tuple, list)) or len(arguments) != 2:
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
        return "f_add_column(column_name, [values])"

    def get_prompt_text_cot(self) -> str:
        """Prompt text for CoT generation."""
        return "f_add_column"

    def get_description(self) -> str:
        """Action description for prompts."""
        return "adds a new column with computed or extracted values to the table"
