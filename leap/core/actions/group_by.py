"""
Group By Action - Group rows by column value and count occurrences.

This is a core operation from the Chain-of-Table paper for aggregation questions.
Groups rows by a specified column and outputs a summary table with counts.

Reference: Chain-of-Table paper (arXiv:2401.04398v2), Appendix A
"""

from collections import Counter
from typing import List, Optional

from ..table import Table
from .registry import ActionDefinition


class GroupByAction(ActionDefinition):
    """
    Group rows by column value and provide counts.

    Usage: group_by(column_name)

    This operation:
    1. Groups all rows by unique values in the specified column
    2. Counts occurrences of each value
    3. Returns a new table with columns: [Group, column_name, Count]

    Example:
        Input table:
            Rank | Cyclist      | Country
            1    | Alejandro    | ESP
            2    | Alexandr     | RUS
            3    | Davide       | ITA
            4    | Paolo        | ITA

        group_by(Country) produces:
            Group | Country | Count
            1     | ESP     | 1
            2     | RUS     | 1
            3     | ITA     | 2

    Common use cases:
    - "Which country had the most cyclists?" → group_by(Country)
    - "How many athletes per country?" → group_by(Country)
    - Aggregation/counting questions
    """

    @property
    def name(self) -> str:
        return "group_by"

    @property
    def requires_args(self) -> bool:
        return True

    def generate_params(self, table: Table) -> List[str]:
        """
        Generate parameters for constraint system.

        Returns all column names as possible grouping columns.
        """
        return list(table.columns)

    def parse_arguments(self, args_str: str) -> Optional[str]:
        """
        Parse column name from argument string.


        Returns:
            Column name (str) or None if parsing fails
        """
        args_str = args_str.strip()

        if not args_str or args_str == "[]":
            return None

        column_name = args_str.strip('"').strip("'").strip()
        # Stop at comma or semicolon (model may add extra prose), but NOT space (column names can have spaces)
        for sep in [",", ";"]:
            if sep in column_name:
                column_name = column_name.split(sep)[0].strip()
        return column_name.strip('"').strip("'")

    def apply(self, table: Table, arguments) -> Optional[Table]:
        """
        Apply group_by to table.

        Args:
            table: Input table
            arguments: Column name to group by (str or single-element list)

        Returns:
            New table with grouped results, or None if invalid
        """
        if isinstance(arguments, (list, tuple)):
            if len(arguments) != 1:
                return None
            arguments = arguments[0]

        if not isinstance(arguments, str):
            return None

        column_name = table.resolve_column(arguments) or arguments

        # Validate column exists
        if column_name not in table.columns:
            return None

        # Get column index
        col_idx = table.columns.index(column_name)

        # Count occurrences of each value
        value_counts = Counter()
        for row in table.rows:
            value = row[col_idx]
            value_counts[value] += 1

        # Create grouped table
        # Columns: [Group, column_name, Count]
        new_columns = ["Group", column_name, "Count"]
        new_rows = []

        for group_id, (value, count) in enumerate(value_counts.items(), 1):
            new_rows.append([group_id, value, count])

        return Table(columns=new_columns, rows=new_rows)

    def validate(self, table: Table, arguments) -> bool:
        """
        Validate group_by arguments.

        Checks:
        - Argument is a string (column name)
        - Column exists in table
        - Table has at least one row
        """
        if isinstance(arguments, (list, tuple)):
            if len(arguments) != 1:
                return False
            arguments = arguments[0]

        if not isinstance(arguments, str):
            return False

        if not arguments:
            return False

        if table.resolve_column(arguments) is None:
            return False

        if len(table.rows) == 0:
            return False

        return True

    def get_prompt_text_iterative(self) -> str:
        """Prompt text for iterative generation."""
        return "f_group_by(column_name)"

    def get_prompt_text_cot(self) -> str:
        """Prompt text for CoT generation."""
        return "f_group_by"

    def get_description(self) -> str:
        """Action description for prompts."""
        return "groups rows by column value and counts occurrences"
