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

    def apply(self, table: Table, arguments: Tuple[str, List]) -> Optional[Table]:
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


    def get_prompt_text_iterative(self) -> str:
        """Prompt text for iterative generation."""
        return "add_column(column_name, [values])"

    def get_prompt_text_cot(self) -> str:
        """Prompt text for CoT generation."""
        return "add_column"

    def get_description(self) -> str:
        """Action description for prompts."""
        return "adds a new column with computed or extracted values to the table"
