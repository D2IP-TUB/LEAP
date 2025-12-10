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

    def generate_params(self, table: Table) -> List[str]:
        """End takes no parameters."""
        return []

    def parse_arguments(self, args_str: str, table: Table) -> Optional[List[Any]]:
        """
        End has no arguments.

        Accepts empty string or empty list.
        """
        args_str = args_str.strip()
        if not args_str or args_str == "[]":
            return []
        return None  # Invalid if arguments provided

    def extract_arguments_from_text(self, text: str, table: Table) -> Optional[List[Any]]:
        """
        Extract arguments from CoT text output.

        Since end takes no arguments, always return empty list.
        """
        return []

    def apply(self, table: Table, arguments: List[Any]) -> Optional[Table]:
        """
        Apply end - returns table unchanged.

        Signals that the reasoning chain is complete.
        """
        return table

    def validate(self, table: Table, arguments: List[Any]) -> bool:
        """
        Validate end action.

        Always valid - can end reasoning at any table state.
        """
        return len(arguments) == 0

    def get_prompt_text_iterative(self) -> str:
        """Prompt text for iterative generation."""
        return "end()"

    def get_prompt_text_cot(self) -> str:
        """Prompt text for CoT generation."""
        return "end"

    def get_fuzzy_match_keywords(self) -> List[str]:
        """Keywords for fuzzy matching."""
        return ["end", "finish", "done"]

    def get_description(self) -> str:
        """Action description for prompts."""
        return "indicates that the table is ready and ends the transformation chain"
