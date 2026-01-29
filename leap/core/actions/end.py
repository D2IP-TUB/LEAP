"""
End Action - Signals completion of the reasoning chain.
"""

from typing import Any, List, Optional

from ..table import Table
from .registry import ActionDefinition


class EndAction(ActionDefinition):
    """
    End operation - signals completion of reasoning chain.

    Usage: end()

    This operation:
    - Takes no arguments
    - Returns the table unchanged
    - Signals that the reasoning chain is complete and answer can be generated
    """

    @property
    def name(self) -> str:
        return "end"

    @property
    def requires_args(self) -> bool:
        return False

    @property
    def is_terminating(self) -> bool:
        """End is a terminating action."""
        return True

    def generate_params(self, table: Table) -> List[str]:
        """End takes no parameters."""
        return []

    def apply(self, table: Table, arguments: List[Any]) -> Optional[Table]:
        """
        Apply end - returns table unchanged.

        Signals that the reasoning chain is complete.
        """
        return table

    def get_prompt_text_iterative(self) -> str:
        """Prompt text for iterative generation."""
        return "end()"

    def get_prompt_text_cot(self) -> str:
        """Prompt text for CoT generation."""
        return "end"

    def get_description(self) -> str:
        """Action description for prompts."""
        return "indicates that the table is ready and ends the transformation chain"
