"""
Action Examples System - Manages examples for prompt building.

This module provides:
1. Example DTOs using existing Table class
2. External YAML storage for examples
3. Cached prompt template building (build once, reuse many times)
4. Easy integration with PromptBuilder
"""

from __future__ import annotations

import csv as csv_module
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    # For action_selection mid-chain examples - actions already taken
    action_history: Optional[List[str]] = None

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
            action_history=data.get("action_history"),
        )

    @staticmethod
    def _parse_table_from_csv(csv_str: str) -> Table:
        """Parse table from CSV string format (matching Table.to_csv() output)."""

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
        table_str = self.table.to_csv(max_chars=5000, crop=False)
        return f"Table:\n{table_str}"


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
        self._system_rules_cache: Optional[Dict[str, str]] = None
        self._action_descriptions_cache: Optional[List[Dict[str, str]]] = None

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

    def get_action_descriptions(self) -> List[Dict[str, str]]:
        """Return the list of action description dicts from the YAML top-level key."""
        if self._action_descriptions_cache is None:
            self._load_examples()
        return self._action_descriptions_cache or []

    def get_system_rules(self, action_name: str) -> Optional[str]:
        """
        Get optional system rules string for a specific action.

        Returns:
            The system rules string if defined in the yaml, otherwise None.
        """
        if self._system_rules_cache is None:
            self._load_examples()

        return self._system_rules_cache.get(action_name)

    def _load_examples(self):
        """Load examples from YAML file and cache them.

        Supports two formats per action:
        - Bare list (legacy): action_name: [example, ...]
        - Dict with optional system key: action_name: {system: "...", examples: [...]}
        """
        if not self.examples_path.exists():
            raise FileNotFoundError(f"Action examples file not found: {self.examples_path}")

        with open(self.examples_path) as f:
            data = yaml.safe_load(f)

        self._examples_cache = {}
        self._system_rules_cache = {}
        self._action_descriptions_cache = data.get("action_descriptions", [])
        for action_name, action_data in data.get("examples", {}).items():
            if isinstance(action_data, list):
                # Legacy format: bare list of examples
                examples_list = action_data
                system_rules = None
            else:
                # New format: dict with optional "system" key and "examples" list
                examples_list = action_data.get("examples", [])
                system_rules = action_data.get("system")

            self._examples_cache[action_name] = [ActionExample.from_dict(ex) for ex in examples_list]
            if system_rules:
                self._system_rules_cache[action_name] = system_rules

    def clear_cache(self):
        """Clear the examples cache (useful for testing or reloading)."""
        self._examples_cache = None
        self._system_rules_cache = None


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
        self._instructions = instruction
        self._examples_template = self._build_examples_template()

    def _build_examples_template(self) -> Tuple[List[str], List[str]]:
        """Build the static part of the prompt (just examples in original format)."""
        examples = []
        answers = []

        # Add examples in the same format as the original prompt
        # Only include: Table, Question, and Answer
        for i, example in enumerate(self.examples, 1):
            example_parts = []
            example_parts.append(example.format_table_for_prompt())
            example_parts.append("")
            example_parts.append(f"Question: {example.question}")

            examples.append("\n".join(example_parts))
            answers.append(example.answer)

        return examples, answers

    def get_examples_template(self) -> Tuple[List[str], List[str]]:
        return self._examples_template

    def get_instructions(self) -> str:
        return self._instructions


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
        self._templates: Dict[Tuple[List[str], List[str]], ActionPromptTemplate] = {}
        self._build_all_templates()

    def _build_all_templates(self):
        """Build templates for all actions (called once during init).

        Instructions are sourced entirely from the "system" field in action_examples.yaml.
        """
        for action_name in ("select_row", "select_column", "add_column", "group_by", "sort_by", "action_selection"):
            examples = self.examples_manager.get_examples(action_name)
            if examples:
                instruction = self.examples_manager.get_system_rules(action_name) or ""
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

        return template.get_examples_template()

    def get_instruction(self, action_name: str) -> str:
        template = self._templates.get(action_name)
        if template is None:
            return None
        return template.get_instructions()

    def has_prompt(self, action_name: str) -> bool:
        """Check if a prompt template exists for this action."""
        return action_name in self._templates
