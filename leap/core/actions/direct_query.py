"""
Direct Query Action - Answer directly from table without transformation.

This is a terminating action from the Chain-of-Table paper (Figure 15, Appendix).
It signals that the question should be answered directly from the current table
without further transformations. Like 'end', this terminates the action chain,
but semantically indicates a direct answer strategy rather than completion after
transformations.

Reference: Chain-of-Table paper (arXiv:2401.04398v2)
"""

from typing import Any, List, Optional

from ..table import Table
from .registry import ActionDefinition


class DirectQueryAction(ActionDefinition):
    """
    Direct query operation - terminating action for direct answers.

    Usage: direct_query()

    This is a terminating operation from Chain-of-Table (Figure 15):
    - Takes no arguments
    - Returns the table unchanged (to preserve state for answer generation)
    - Signals that the answer should be generated directly from current table
    - Can only be used as the final operation in the chain
    - Semantically different from 'end': indicates direct answering vs. completion
    """

    @property
    def name(self) -> str:
        return "direct_query"

    @property
    def requires_args(self) -> bool:
        return False  # No arguments needed

    @property
    def is_terminating(self) -> bool:
        """Direct query is a terminating action."""
        return True

    def generate_params(self, table: Table) -> List[str]:
        """Direct query takes no parameters."""
        return []

    def parse_arguments(self, args_str: str) -> Optional[List[Any]]:
        """
        Direct query has no arguments.

        Accepts empty string or empty list.
        """
        args_str = args_str.strip()
        if not args_str or args_str == "[]":
            return []
        return None  # Invalid if arguments provided

    def apply(self, table: Table, arguments: List[Any]) -> Optional[Table]:
        """
        Apply direct_query - returns table unchanged.

        The action signals that the answer should be generated
        directly from this table without further transformations.
        """
        return table

    def get_prompt_text_iterative(self) -> str:
        """Prompt text for iterative generation."""
        return "f_direct_query()"

    def get_prompt_text_cot(self) -> str:
        """Prompt text for CoT generation."""
        return "f_direct_query"

    def get_description(self) -> str:
        """Action description for prompts."""
        return "generates the answer directly from the current table state (terminating action)"
