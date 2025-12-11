"""
Tests for PromptBuilder to verify action filtering behavior in generated prompts.
"""

from unittest.mock import MagicMock

import pytest
from transformers import AutoTokenizer

from leap.core import Table
from leap.core.actions import REGISTRY
from leap.generation.prompt_builder import CotPromptSettings, IterativePromptSettings, PromptBuilder


@pytest.fixture(scope="module")
def tokenizer():
    """GPT2 tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def sample_table():
    """Create a simple test table."""
    return Table(
        columns=["Name", "Age", "City"],
        rows=[
            ["Alice", "25", "NYC"],
            ["Bob", "30", "LA"],
            ["Charlie", "35", "SF"],
        ],
    )


@pytest.fixture
def mock_worker():
    """Create a mock worker object."""
    worker = MagicMock()
    worker.use_constraints = False
    worker.max_model_len = 4096
    return worker


@pytest.fixture
def prompt_builder_non_instruct(tokenizer):
    """Create a prompt builder for non-instruct models."""
    return PromptBuilder(
        tokenizer=tokenizer,
        is_instruct=False,
        iterative_settings=IterativePromptSettings(),
        cot_settings=CotPromptSettings(),
    )


@pytest.fixture
def setup_registry():
    """Set up the registry with all actions enabled."""
    REGISTRY.set_enabled_actions(["select_row", "select_column", "end", "direct_query"])
    yield
    # Reset after test
    REGISTRY.set_enabled_actions(["select_row", "select_column", "end"])


class TestIterativePromptFiltering:
    """Test iterative prompt generation with action filtering."""

    def test_no_action_history_shows_all_actions(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """With no action history, all actions should be shown."""
        prompt = prompt_builder_non_instruct.build_iterative_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=[],
            worker=mock_worker,
            step=0,
        )

        # All actions should be present
        assert "select_row" in prompt
        assert "select_column" in prompt
        assert "end()" in prompt
        assert "direct_query()" in prompt

        # Should show all operation descriptions
        assert "f_select_row:" in prompt
        assert "f_select_column:" in prompt
        assert "f_end:" in prompt
        assert "f_direct_query:" in prompt

    def test_filters_out_used_select_row(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """After using select_row, it should be filtered from available actions."""
        action_history = ["select_row([0, 1])"]

        prompt = prompt_builder_non_instruct.build_iterative_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=action_history,
            worker=mock_worker,
            step=1,
        )

        # select_row should NOT be in the available actions
        # Check in the "Choose from:" section
        choose_from_section = prompt.split("Choose from:")[-1].split("Next action:")[0]
        assert "select_row" not in choose_from_section

        # Other actions should still be present
        assert "select_column" in choose_from_section
        assert "end()" in choose_from_section
        assert "direct_query()" in choose_from_section

        # select_row should also not be in operations descriptions
        operations_section = prompt.split("Operations:")[1].split("What should be")[0]
        assert "f_select_row:" not in operations_section

    def test_filters_multiple_used_actions(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """After using multiple actions, all should be filtered."""
        action_history = [
            "select_row([0, 1])",
            "select_column(['Name', 'Age'])",
        ]

        prompt = prompt_builder_non_instruct.build_iterative_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=action_history,
            worker=mock_worker,
            step=2,
        )

        choose_from_section = prompt.split("Choose from:")[-1].split("Next action:")[0]

        # Both used actions should be filtered
        assert "select_row" not in choose_from_section
        assert "select_column" not in choose_from_section

        # Unused actions should remain
        assert "end()" in choose_from_section
        assert "direct_query()" in choose_from_section

    def test_end_action_always_available(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """The 'end' action should always be available, even if used before."""
        action_history = [
            "select_row([0, 1])",
            "end()",  # Even though end was used
        ]

        prompt = prompt_builder_non_instruct.build_iterative_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=action_history,
            worker=mock_worker,
            step=2,
        )

        # end should still be available
        choose_from_section = prompt.split("Choose from:")[-1].split("Next action:")[0]
        assert "end()" in choose_from_section

    def test_action_history_shown_in_prompt(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """Action history should be displayed in the prompt."""
        action_history = [
            "select_row([0, 1])",
            "select_column(['Name'])",
        ]

        prompt = prompt_builder_non_instruct.build_iterative_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=action_history,
            worker=mock_worker,
            step=2,
        )

        # History should be shown
        assert "Actions taken so far:" in prompt
        assert "1. select_row([0, 1])" in prompt
        assert "2. select_column(['Name'])" in prompt


class TestCoTPromptFiltering:
    """Test Chain-of-Table prompt generation with action filtering."""

    def test_cot_action_prompt_no_history(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """CoT action prompt with no history shows non-terminating actions only (first step)."""
        prompt = prompt_builder_non_instruct.build_cot_action_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=[],
            worker=mock_worker,
        )

        # On first step, only non-terminating actions should be shown
        assert "Available actions:" in prompt
        available_section = prompt.split("Available actions:")[1].split("\n")[0]
        assert "select_row" in available_section
        assert "select_column" in available_section
        # Terminating actions should NOT be shown on first step
        assert "end" not in available_section
        assert "direct_query" not in available_section

    def test_cot_action_prompt_filters_used_actions(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """CoT action prompt filters out already-used actions."""
        action_history = ["select_row([0, 1])"]

        prompt = prompt_builder_non_instruct.build_cot_action_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=action_history,
            worker=mock_worker,
        )

        # select_row should be filtered from available actions
        available_section = prompt.split("Available actions:")[1].split("\n")[0]
        assert "select_row" not in available_section
        assert "select_column" in available_section
        assert "end" in available_section
        assert "direct_query" in available_section

        # Also check operations section
        operations_section = prompt.split("Operations:")[1].split("Available actions:")[0]
        assert "f_select_row:" not in operations_section

    def test_cot_arguments_prompt(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """CoT arguments prompt should work correctly."""
        action_history = ["select_row([0, 1])"]

        prompt = prompt_builder_non_instruct.build_cot_arguments_prompt(
            question="What is the average age?",
            table=sample_table,
            action_name="select_column",
            action_history=action_history,
            worker=mock_worker,
        )

        # Should show selected action
        assert "Selected action: select_column" in prompt

        # Should show available columns
        assert "Available columns:" in prompt
        assert "['Name', 'Age', 'City']" in prompt

        # Should ask for column names
        assert "Column names:" in prompt


class TestConstraintMode:
    """Test prompt generation with constraints enabled."""

    def test_constraint_mode_still_filters_actions(self, prompt_builder_non_instruct, sample_table, setup_registry):
        """Even with constraints enabled, action descriptions should be filtered."""
        worker = MagicMock()
        worker.use_constraints = True
        worker.max_model_len = 4096

        action_history = ["select_row([0, 1])"]

        prompt = prompt_builder_non_instruct.build_iterative_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=action_history,
            worker=worker,
            step=1,
        )

        # With constraints, only operations are shown (no "Choose from")
        # But select_row should still be filtered from operations
        operations_section = prompt.split("Operations:")[1].split("Next action:")[0]
        assert "f_select_row:" not in operations_section
        assert "f_select_column:" in operations_section
        assert "f_end:" in operations_section


class TestEdgeCases:
    """Test edge cases in prompt building."""

    def test_empty_action_history(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """Empty action history should work the same as None."""
        prompt = prompt_builder_non_instruct.build_iterative_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=[],
            worker=mock_worker,
            step=0,
        )

        # All actions should be available
        assert "select_row" in prompt
        assert "select_column" in prompt
        assert "end()" in prompt

    def test_action_history_with_whitespace(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """Action history with extra whitespace should be handled correctly."""
        action_history = [
            " select_row([0, 1]) ",
            "  select_column(['Name'])  ",
        ]

        prompt = prompt_builder_non_instruct.build_iterative_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=action_history,
            worker=mock_worker,
            step=2,
        )

        # Both actions should be filtered despite whitespace
        choose_from_section = prompt.split("Choose from:")[-1].split("Next action:")[0]
        assert "select_row" not in choose_from_section
        assert "select_column" not in choose_from_section

    def test_only_end_and_direct_query_remaining(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """When only terminating actions remain, they should be shown."""
        action_history = [
            "select_row([0, 1])",
            "select_column(['Name', 'Age'])",
        ]

        prompt = prompt_builder_non_instruct.build_iterative_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=action_history,
            worker=mock_worker,
            step=2,
        )

        choose_from_section = prompt.split("Choose from:")[-1].split("Next action:")[0]

        # Only terminating actions should remain
        assert "end()" in choose_from_section
        assert "direct_query()" in choose_from_section

        # Format should be "action1 or action2" for two actions
        assert " or " in choose_from_section


class TestPromptStructure:
    """Test the overall structure of generated prompts."""

    def test_iterative_prompt_has_required_sections(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """Iterative prompt should have all required sections."""
        prompt = prompt_builder_non_instruct.build_iterative_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=["select_row([0, 1])"],
            worker=mock_worker,
            step=1,
        )

        # Required sections
        assert "Table:" in prompt
        assert "Question:" in prompt
        assert "Actions taken so far:" in prompt
        assert "Operations:" in prompt
        assert "Choose from:" in prompt
        assert "Next action:" in prompt

    def test_cot_action_prompt_has_required_sections(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """CoT action prompt should have all required sections."""
        prompt = prompt_builder_non_instruct.build_cot_action_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=["select_row([0, 1])"],
            worker=mock_worker,
        )

        # Required sections
        assert "Table:" in prompt
        assert "Question:" in prompt
        assert "Actions taken so far:" in prompt
        assert "Operations:" in prompt
        assert "Available actions:" in prompt
        assert "Action:" in prompt

    def test_print_example_prompts(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry, capsys):
        """Print example prompts for visual inspection (run with -s flag)."""
        print("\n" + "=" * 80)
        print("EXAMPLE: Iterative prompt with no action history")
        print("=" * 80)
        prompt1 = prompt_builder_non_instruct.build_iterative_prompt(
            question="What is the total population?",
            table=sample_table,
            action_history=[],
            worker=mock_worker,
            step=0,
        )
        print(prompt1)

        print("\n" + "=" * 80)
        print("EXAMPLE: Iterative prompt after select_row was used")
        print("=" * 80)
        prompt2 = prompt_builder_non_instruct.build_iterative_prompt(
            question="What is the total population?",
            table=sample_table,
            action_history=["select_row([0, 1])"],
            worker=mock_worker,
            step=1,
        )
        print(prompt2)

        print("\n" + "=" * 80)
        print("EXAMPLE: CoT action prompt after select_row and select_column were used")
        print("=" * 80)
        prompt3 = prompt_builder_non_instruct.build_cot_action_prompt(
            question="What is the total population?",
            table=sample_table,
            action_history=["select_row([0, 1])", "select_column(['Name'])"],
            worker=mock_worker,
        )
        print(prompt3)
        print("=" * 80)


class TestCoTFirstActionRestrictions:
    """Test that CoT action prompts exclude terminating actions on first step."""

    def test_cot_first_action_excludes_terminating_actions(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """On the first action (empty history), 'end' and 'direct_query' should NOT be available."""
        prompt = prompt_builder_non_instruct.build_cot_action_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=[],
            worker=mock_worker,
        )

        # Check available actions section
        available_section = prompt.split("Available actions:")[1].split("\n")[0]

        # Terminating actions should NOT be available on first step
        assert "end" not in available_section
        assert "direct_query" not in available_section

        # Non-terminating actions should be available
        assert "select_row" in available_section
        assert "select_column" in available_section

        # Also check operations section
        operations_section = prompt.split("Operations:")[1].split("Available actions:")[0]
        assert "f_end:" not in operations_section
        assert "f_direct_query:" not in operations_section
        assert "f_select_row:" in operations_section
        assert "f_select_column:" in operations_section

    def test_cot_subsequent_actions_include_terminating_actions(
        self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry
    ):
        """After first action, 'end' and 'direct_query' should be available."""
        action_history = ["select_row([0, 1])"]

        prompt = prompt_builder_non_instruct.build_cot_action_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=action_history,
            worker=mock_worker,
        )

        # Check available actions section
        available_section = prompt.split("Available actions:")[1].split("\n")[0]

        # After first action, terminating actions should be available
        assert "end" in available_section
        assert "direct_query" in available_section

        # select_row should be filtered (already used)
        assert "select_row" not in available_section

        # select_column should still be available
        assert "select_column" in available_section


class TestCoTActionHistory:
    """Test that CoT prompts properly display action history."""

    def test_cot_action_prompt_shows_action_history(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """CoT action prompt should display the action history."""
        action_history = [
            "select_row([0, 1])",
            "select_column(['Name', 'Age'])",
        ]

        prompt = prompt_builder_non_instruct.build_cot_action_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=action_history,
            worker=mock_worker,
        )

        # Action history should be displayed
        assert "Actions taken so far:" in prompt
        assert "1. select_row([0, 1])" in prompt
        assert "2. select_column(['Name', 'Age'])" in prompt

    def test_cot_action_prompt_no_history_section_when_empty(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """When action history is empty, no history section should appear."""
        prompt = prompt_builder_non_instruct.build_cot_action_prompt(
            question="What is the average age?",
            table=sample_table,
            action_history=[],
            worker=mock_worker,
        )

        # No action history section should appear
        assert "Actions taken so far:" not in prompt

    def test_cot_arguments_prompt_shows_action_history(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry):
        """CoT arguments prompt should also display the action history."""
        action_history = ["select_row([0, 1])"]

        prompt = prompt_builder_non_instruct.build_cot_arguments_prompt(
            question="What is the average age?",
            table=sample_table,
            action_name="select_column",
            action_history=action_history,
            worker=mock_worker,
        )

        # Action history should be displayed
        assert "Actions taken so far:" in prompt
        assert "1. select_row([0, 1])" in prompt

    def test_print_cot_first_action_example(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry, capsys):
        """Print example of first CoT action prompt (run with -s flag)."""
        print("\n" + "=" * 80)
        print("[PHASE 1 PROMPT - ACTION SELECTION - FIRST STEP]")
        print("=" * 80)
        prompt = prompt_builder_non_instruct.build_cot_action_prompt(
            question="what was the last year where this team was a part of the usl a-league?",
            table=sample_table,
            action_history=[],
            worker=mock_worker,
        )
        print(prompt)
        print("=" * 80)

    def test_print_cot_subsequent_action_example(self, prompt_builder_non_instruct, sample_table, mock_worker, setup_registry, capsys):
        """Print example of subsequent CoT action prompt with history (run with -s flag)."""
        print("\n" + "=" * 80)
        print("[PHASE 1 PROMPT - ACTION SELECTION - WITH HISTORY]")
        print("=" * 80)
        prompt = prompt_builder_non_instruct.build_cot_action_prompt(
            question="what was the last year where this team was a part of the usl a-league?",
            table=sample_table,
            action_history=["select_row([0, 1, 2, 3])"],
            worker=mock_worker,
        )
        print(prompt)
        print("=" * 80)
