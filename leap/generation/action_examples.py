"""
Action Examples System - Manages examples for prompt building.

This module provides:
1. Example DTOs using existing Table class
2. External YAML storage for examples
3. Cached prompt template building (build once, reuse many times)
4. Easy integration with PromptBuilder
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from leap.core import Table


@dataclass(frozen=True)
class ActionExample:
    """
    A single example for an action.

    Stores table, question, and any additional metadata needed for the prompt.
    Uses the existing Table DTO for consistency.
    """

    table: Table
    question: str
    explanation: Optional[str] = None
    answer: Optional[str] = None
    # Additional fields for specific actions (e.g., select_column needs these)
    similar_words: Optional[List[str]] = None
    column_value_links: Optional[List[str]] = None
    semantic_sentence_links: Optional[List[str]] = None
    # For f_add_column - stores the actual values added
    added_column_values: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionExample":
        """Create ActionExample from dictionary (loaded from YAML)."""
        # Parse table data
        table_data = data["table"]
        if isinstance(table_data, str):
            # Table is stored as CSV string - parse it
            table = cls._parse_table_from_csv(table_data)
        elif isinstance(table_data, dict):
            # Table is stored as structured dict
            table = Table.from_dict(table_data)
        else:
            raise ValueError(f"Invalid table format: {type(table_data)}")

        return cls(
            table=table,
            question=data["question"],
            explanation=data.get("explanation"),
            answer=data.get("answer"),
            similar_words=data.get("similar_words"),
            column_value_links=data.get("column_value_links"),
            semantic_sentence_links=data.get("semantic_sentence_links"),
            added_column_values=data.get("added_column_values"),
        )

    @staticmethod
    def _parse_table_from_csv(csv_str: str) -> Table:
        """Parse table from CSV string format (matching Table.to_csv() output)."""
        import csv as csv_module
        import io

        # Parse as CSV (matching Table.to_csv() format)
        lines = csv_str.strip().split("\n")
        if not lines:
            return Table(columns=[], rows=[])

        reader = csv_module.reader(io.StringIO(csv_str))
        rows_data = list(reader)

        if not rows_data:
            return Table(columns=[], rows=[])

        # First row is header: [" ", "Col1", "Col2", ...]
        header = rows_data[0]
        columns = header[1:] if len(header) > 1 else []

        # Remaining rows: ["row 0", "val1", "val2", ...]
        rows = []
        for row in rows_data[1:]:
            if len(row) > 1:
                rows.append(row[1:])

        return Table(columns=columns, rows=rows)

    def format_table_for_prompt(self) -> str:
        """Format table using the standard Table.to_csv() method for consistency."""
        # Use the same serialization as everywhere else in the system
        return self.table.to_csv(max_chars=5000, crop=False)


class ActionExamplesManager:
    """
    Manages loading and caching of action examples.

    Features:
    - Loads examples from external YAML file
    - Caches parsed examples (parse once, use many times)
    - Provides easy access by action name
    """

    def __init__(self, examples_path: Optional[Path] = None):
        """
        Initialize the examples manager.

        Args:
            examples_path: Path to YAML file with examples.
                          If None, uses default: configs/action_examples.yaml
        """
        if examples_path is None:
            # Default path relative to project root
            examples_path = Path(__file__).parent.parent.parent / "configs" / "action_examples.yaml"

        self.examples_path = examples_path
        self._examples_cache: Optional[Dict[str, List[ActionExample]]] = None

    def get_examples(self, action_name: str) -> List[ActionExample]:
        """
        Get examples for a specific action.

        Loads and caches on first call, then reuses cache.

        Args:
            action_name: Name of the action (e.g., "direct_query", "select_row")

        Returns:
            List of ActionExample objects for this action
        """
        if self._examples_cache is None:
            self._load_examples()

        return self._examples_cache.get(action_name, [])

    def _load_examples(self):
        """Load examples from YAML file and cache them."""
        if not self.examples_path.exists():
            raise FileNotFoundError(f"Action examples file not found: {self.examples_path}")

        with open(self.examples_path) as f:
            data = yaml.safe_load(f)

        # Parse all examples
        self._examples_cache = {}
        for action_name, examples_data in data.get("examples", {}).items():
            self._examples_cache[action_name] = [ActionExample.from_dict(ex) for ex in examples_data]

    def clear_cache(self):
        """Clear the examples cache (useful for testing or reloading)."""
        self._examples_cache = None


class ActionPromptTemplate:
    """
    Pre-built prompt template for an action.

    Built once during initialization, then reused for all instances.
    Only the instance-specific parts (current table, question) are filled in later.
    """

    def __init__(self, action_name: str, examples: List[ActionExample], instruction: str):
        """
        Create a prompt template.

        Args:
            action_name: Name of the action
            examples: List of examples for this action
            instruction: Instruction text explaining what to do
        """
        self.action_name = action_name
        self.examples = examples
        self.instruction = instruction
        self._template = self._build_template()

    def _build_template(self) -> str:
        """Build the static part of the prompt (just examples in original format)."""
        parts = []

        # Add examples in the same format as the original prompt
        # Only include: Table, Question, and Answer
        for i, example in enumerate(self.examples, 1):
            # Add "Table:" prefix to match original format
            parts.append("Table:")
            parts.append(example.format_table_for_prompt())
            parts.append("")
            parts.append(f"Question: {example.question}")
            parts.append("")

            # Only add the answer (no explanation or other fields)
            if example.answer:
                parts.append(f"The answer is : {example.answer}")
                parts.append("")

        return "\n".join(parts)

    def get_examples_only(self) -> str:
        """
        Get just the examples part (without current instance).

        Returns:
            Examples formatted in the original prompt style
        """
        return self._template


class ActionPromptBuilder:
    """
    Builds and caches prompt templates for all actions.

    This is the main class that PromptBuilder will use.
    Build templates once during initialization, then reuse for all inferences.
    """

    def __init__(self, examples_manager: Optional[ActionExamplesManager] = None):
        """
        Initialize the prompt builder.

        Args:
            examples_manager: Manager for loading examples. If None, creates default.
        """
        self.examples_manager = examples_manager or ActionExamplesManager()
        self._templates: Dict[str, ActionPromptTemplate] = {}
        self._build_all_templates()

    def _build_all_templates(self):
        """Build templates for all actions (called once during init)."""
        # Define instructions for each action
        instructions = {
            "direct_query": "This action generates an answer directly from the current table without transformation.",
            "select_row": "Using f_select_row() to select relevant rows in the given table that support or oppose the statement.\nPlease use f_select_row([*]) to select all rows in the table.",
            "select_column": "Use f_select_column() to filter out useless columns in the table according to information in the statement and the table.",
            "add_column": "To answer the question, we can first use f_add_column() to add more columns to the table.\n\nThe added columns should have these data types:\n1. Numerical: the numerical strings that can be used in sort, sum\n2. Datetype: the strings that describe a date, such as year, month, day\n3. String: other strings",
            "group_by": "To answer the question, we can first use f_group_by() to group the values in a column.",
            "sort_by": 'To answer the question, we can first use f_sort_by() to sort the values in a column to get the order of the items. The order can be "large to small" or "small to large".\n\nThe column to sort should have these data types:\n1. Numerical: the numerical strings that can be used in sort\n2. DateType: the strings that describe a date, such as year, month, day\n3. String: other strings',
            "action_selection": "",  # No instruction needed for action selection examples
        }

        # Build template for each action
        for action_name, instruction in instructions.items():
            examples = self.examples_manager.get_examples(action_name)
            if examples:  # Only build template if examples exist
                self._templates[action_name] = ActionPromptTemplate(action_name, examples, instruction)

    def get_examples(self, action_name: str) -> Optional[str]:
        """
        Get just the examples for an action (without current instance).

        This is the main method called by PromptBuilder.

        Args:
            action_name: Name of the action

        Returns:
            Examples string in original prompt format, or None if action not found
        """
        template = self._templates.get(action_name)
        if template is None:
            return None

        return template.get_examples_only()

    def has_prompt(self, action_name: str) -> bool:
        """Check if a prompt template exists for this action."""
        return action_name in self._templates
