"""
Direct Query Action - Answer directly from table without transformation.

This action allows the LLM to answer questions directly from the current table
state without applying any transformations. Useful for simple questions that don't
require intermediate reasoning steps.
"""

from typing import Any, List, Optional

from leap.core.action_registry import ActionDefinition
from leap.core.table import Table


class DirectQueryAction(ActionDefinition):
    """
    Direct query operation - answers directly without table transformation.

    Usage: direct_query()

    This operation:
    - Takes no arguments
    - Returns the table unchanged
    - Signals that the answer can be derived directly from current table state
    """

    @property
    def name(self) -> str:
        return "direct_query"

    @property
    def requires_args(self) -> bool:
        return False  # No arguments needed

    def generate_params(self, table: Table) -> List[str]:
        """Direct query takes no parameters."""
        return []

    def parse_arguments(self, args_str: str, table: Table) -> Optional[List[Any]]:
        """
        Direct query has no arguments.

        Accepts empty string or empty list.
        """
        args_str = args_str.strip()
        if not args_str or args_str == "[]":
            return []
        return None  # Invalid if arguments provided

    def extract_arguments_from_text(self, text: str, table: Table) -> Optional[List[Any]]:
        """
        Extract arguments from CoT text output.

        Since direct_query takes no arguments, always return empty list.
        """
        return []

    def apply(self, table: Table, arguments: List[Any]) -> Optional[Table]:
        """
        Apply direct_query - returns table unchanged.

        The action signals that the answer should be generated
        directly from this table without further transformations.
        """
        return table

    def validate(self, table: Table, arguments: List[Any]) -> bool:
        """
        Validate direct_query action.

        Always valid - can query any table directly.
        """
        return len(arguments) == 0

    def get_prompt_text_iterative(self) -> str:
        """Prompt text for iterative generation."""
        return "direct_query()"

    def get_prompt_text_cot(self) -> str:
        """Prompt text for CoT generation."""
        return "direct_query"

    def get_fuzzy_match_keywords(self) -> List[str]:
        """Keywords for fuzzy matching."""
        return ["direct_query", "direct", "query"]
