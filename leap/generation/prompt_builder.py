from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from leap.core import Table
from leap.core.actions import REGISTRY
from leap.generation.action_examples import ActionPromptBuilder


@dataclass
class IterativePromptSettings:
    initial_table_chars: int = 1500
    step_table_chars: int = 1000
    fallback_table_chars: int = 500
    question_truncation: int = 100
    safety_margin_tokens: int = 100


@dataclass
class CotPromptSettings:
    action_table_chars: int = 1000
    action_fallback_table_chars: int = 500
    action_question_truncation: int = 80
    action_safety_margin_tokens: int = 50
    args_table_chars: int = 1200
    args_fallback_table_chars: int = 600
    args_question_truncation: int = 80
    args_safety_margin_tokens: int = 100


class PromptBuilder:
    """Centralized helper for constructing model prompts."""

    def __init__(
        self,
        tokenizer,
        is_instruct: bool,
        iterative_settings: IterativePromptSettings | None = None,
        cot_settings: CotPromptSettings | None = None,
        use_action_examples: bool = True,
    ):
        self.tokenizer = tokenizer
        self.is_instruct = is_instruct
        self.iterative_settings = iterative_settings or IterativePromptSettings()
        self.cot_settings = cot_settings or CotPromptSettings()

        # Initialize action examples system (builds templates once)
        self.action_examples: Optional[ActionPromptBuilder] = None
        if use_action_examples:
            try:
                self.action_examples = ActionPromptBuilder()
            except FileNotFoundError:
                # If examples file doesn't exist, fall back to old behavior
                self.action_examples = None

    @staticmethod
    def _format_table(table: Table, max_chars: int, caption: Optional[str] = None) -> str:
        """Format table CSV with optional caption prefix."""
        table_str = table.to_csv(max_chars=max_chars)
        if caption:
            return f"Table caption: {caption}\nTable:\n{table_str}"
        return f"Table:\n{table_str}"

    @staticmethod
    def _format_row_csv(table: Table, row: tuple, row_idx: int) -> str:
        """Format a single row as CSV with the correct row index in the label."""
        import csv as _csv
        import io

        out = io.StringIO()
        writer = _csv.writer(out)
        writer.writerow([" "] + list(table.columns))
        writer.writerow([f"row {row_idx}"] + list(row))
        return out.getvalue().rstrip()

    def build_iterative_prompt(
        self,
        *,
        question: str,
        table: Table,
        action_history: Sequence[str],
        worker,
        step: int,
        table_caption: Optional[str] = None,
    ) -> str:
        """Create a constraint-aware prompt for iterative generation."""
        max_chars = self.iterative_settings.initial_table_chars if step == 0 else self.iterative_settings.step_table_chars
        table_str = self._format_table(table, max_chars, table_caption)

        step_prompt = f"{table_str}\n\nQuestion: {question}\n"
        # if action_history:
        #     step_prompt += "Actions taken so far:\n"
        #     for idx, action in enumerate(action_history):
        #         step_prompt += f"{idx + 1}. {action}\n"
        #     step_prompt += "\n"

        instruction_prompt = self._build_iterative_instruction(worker, action_history)

        estimated_length = len(step_prompt) // 4
        if estimated_length > worker.max_model_len - self.iterative_settings.safety_margin_tokens:
            table_str = self._format_table(table, self.iterative_settings.fallback_table_chars, table_caption)
            question_short = self._truncate_text(question, self.iterative_settings.question_truncation)
            step_prompt = f"{table_str}\n\nQuestion: {question_short}\n"
            instruction_prompt = self._build_iterative_instruction(worker, action_history, fallback=True)

        return self._append_instruction(step_prompt, instruction_prompt)

    def _build_iterative_instruction(self, worker, action_history: Sequence[str] = None, fallback: bool = False) -> str:
        """Instruction text for iterative generation."""
        action_descriptions = REGISTRY.get_action_descriptions(action_history, exclude_terminating_on_first=False)

        if worker.use_constraints:
            return f"{action_descriptions}\n\nNext action: "

        actions_text = REGISTRY.get_prompt_text_iterative(action_history)

        question_prefix = "What should be the next action? " if fallback else "What should be the next action to answer this question? "
        base = f"{action_descriptions}\n\n{question_prefix}Choose from: {actions_text}.\nNext action: "
        return base

    @staticmethod
    def _append_instruction(step_prompt: str, instruction_prompt: str) -> str:
        return step_prompt + instruction_prompt

    @staticmethod
    def _to_display_action(action_str: str) -> str:
        """Add f_ prefix to action name for display to LLM."""
        if not action_str.startswith("f_"):
            return "f_" + action_str
        return action_str

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        return text[:limit] + "..." if len(text) > limit else text

    def _build_action_selection_body(
        self,
        table_str: str,
        question: str,
        action_history: Sequence[str],
        available_label: str,
        question_suffix: str,
    ) -> str:
        """Build the user-message body for an action-selection turn."""
        prompt = f"{table_str}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Actions taken so far:\n"
        if action_history:
            for idx, action in enumerate(action_history):
                prompt += f"{idx + 1}. {self._to_display_action(action)}\n"
        else:
            prompt += "None\n"
        prompt += "\n"
        actions_text = REGISTRY.get_prompt_text_cot(action_history, exclude_terminating_on_first=True)
        prompt += f"{available_label}: {actions_text}\n"
        prompt += question_suffix
        return prompt

    def _build_action_selection_example_message(self, example) -> str:
        """Build the user-message content for a single few-shot action-selection example."""
        action_history = example.action_history or []
        example_actions_text = REGISTRY.get_prompt_text_cot(action_history, exclude_terminating_on_first=True)
        prompt = f"{example.format_table_for_prompt()}\n\n"
        prompt += f"Question: {example.question}\n\n"
        prompt += "Actions taken so far:\n"
        if action_history:
            for idx, action in enumerate(action_history):
                prompt += f"{idx + 1}. {self._to_display_action(action)}\n"
        else:
            prompt += "None\n"
        prompt += "\n"
        prompt += f"Available actions: {example_actions_text}\n"
        prompt += "What actions should be performed next?"
        return prompt

    def build_cot_action_prompt(
        self,
        *,
        question: str,
        table: Table,
        action_history: Sequence[str],
        worker,
        table_caption: Optional[str] = None,
    ) -> str:
        """Prompt for CoT action selection (dynamic plan)."""
        table_str = self._format_table(table, self.cot_settings.action_table_chars, table_caption)
        available_label = "Available actions"
        question_suffix = "What actions should be performed next?"
        instruction_prompt = self._build_action_selection_body(table_str, question, action_history, available_label, question_suffix)

        estimated_length = len(instruction_prompt) // 4
        if estimated_length > worker.max_model_len - self.cot_settings.action_safety_margin_tokens:
            table_str = self._format_table(table, self.cot_settings.action_fallback_table_chars, table_caption)
            available_label = "The next operation must be one of the following"
            question_suffix = "What actions should be performed next?"
            question_short = self._truncate_text(question, self.cot_settings.action_question_truncation)
            instruction_prompt = self._build_action_selection_body(
                table_str, question_short, action_history, available_label, question_suffix
            )

        if not (self.action_examples and self.action_examples.has_prompt("action_selection")):
            return self._append_instruction("", instruction_prompt)

        examples = self.action_examples.examples_manager.get_examples("action_selection")
        if not (examples and self.is_instruct):
            return self._append_instruction("", instruction_prompt)

        system = (
            self.action_examples.examples_manager.get_system_rules("action_selection")
            or "You are a helpful table question answering assistant"
        )
        messages = [{"role": "system", "content": system}]

        for example in examples:
            messages.append({"role": "user", "content": self._build_action_selection_example_message(example)})
            messages.append({"role": "assistant", "content": example.answer})

        messages.append({"role": "user", "content": instruction_prompt})

        if self.is_instruct:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            result = ""
            for msg in messages:
                result += msg["content"] + "\n" if msg["role"] == "user" else msg["content"] + "\n\n"
            return result

    def build_cot_arguments_prompt(
        self,
        *,
        question: str,
        table: Table,
        action_name: str,
        action_history: Sequence[str],
        worker,
        table_caption: Optional[str] = None,
    ) -> str:
        """Prompt for CoT argument generation."""

        messages = []

        instruction = self.action_examples.get_instruction(action_name)
        examples, example_answers = self.action_examples.get_examples(action_name)

        if instruction:
            messages.append({"role": "system", "content": instruction})

        for example, example_answer in zip(examples, example_answers):
            messages.append({"role": "user", "content": example})
            messages.append({"role": "assistant", "content": example_answer})

        if action_name == "add_column" and len(table.rows) > 3:
            table = Table(columns=list(table.columns), rows=[list(r) for r in table.rows[:3]])

        table_str = self._format_table(table, self.cot_settings.args_table_chars, table_caption)
        final_prompt = f"{table_str}\n\n"
        final_prompt += f"Question: {question}\n"

        messages.append({"role": "user", "content": final_prompt})

        if self.is_instruct:
            instruct_result = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return instruct_result
        else:
            # For non-instruct models, just concatenate the messages
            result = ""
            for msg in messages:
                if msg["role"] == "user":
                    result += msg["content"] + "\n"
                else:
                    result += msg["content"] + "\n\n"

            return result

    def build_add_column_per_row_prompt(
        self,
        *,
        table: Table,
        column_name: str,
        target_row: tuple,
        target_row_idx: int,
        seed_values: list,
        explanation: str,
    ) -> str:
        """Prompt to generate the add_column value for a single row.

        Chat-template format: task description in each user turn, each seed row
        as a user/assistant example, target row as the final user message.

            User:  <task description>\\n\\n<row CSV>\\n\\n<prompt line>
            Asst:  <seed value>
            ...
            User:  <task description>\\n\\n<target row CSV>\\n\\n<prompt line>
        """
        # Extract reasoning text — everything before "Therefore the answer is:"
        for marker in ("Therefore, the answer is:", "Therefore the answer is:"):
            if marker in explanation:
                task_description = explanation[: explanation.index(marker)].strip()
                break
        else:
            task_description = explanation.strip()

        prompt_line = f"We need to determine the value for column '{column_name}'. The value:"

        messages = []

        if self.is_instruct:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Output only the cell value for the '{column_name}' column. "
                        "Reply with a single word, number, or short phrase only. "
                        "No explanation, no sentences, no punctuation at the end."
                    ),
                }
            )

        # One user/assistant shot per seed row
        for i, (row, value) in enumerate(zip(table.rows[:3], seed_values[:3])):
            row_csv = self._format_row_csv(table, row, i)
            messages.append({"role": "user", "content": f"{task_description}\n\n{row_csv}\n\n{prompt_line}"})
            messages.append({"role": "assistant", "content": str(value)})

        # Target row — final user message
        target_csv = self._format_row_csv(table, target_row, target_row_idx)
        messages.append({"role": "user", "content": f"{task_description}\n\n{target_csv}\n\n{prompt_line}"})

        if self.is_instruct:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            result = ""
            for msg in messages:
                result += msg["content"] + "\n" if msg["role"] == "user" else msg["content"] + "\n\n"
            return result

    def build_query_prompt(
        self,
        *,
        question: str,
        table: Table,
        action_history: Sequence[str],
        worker,
        table_caption: Optional[str] = None,
    ) -> str:
        """
        Build Query(T,Q) prompt for answer generation.

        This follows Chain-of-Table paper Section 3.4 and Appendix E.3.
        The final table from the operation chain is used to generate the answer.
        """
        messages = []

        # Load examples and system prompt from YAML
        query_examples = self.action_examples.examples_manager.get_examples("query_answer")
        system_prompt = (
            self.action_examples.examples_manager.get_system_rules("query_answer")
            or "Here is the table to answer this question. Please understand the table and answer the question:"
        )
        messages.append({"role": "system", "content": system_prompt})

        for example in query_examples:
            example_instruction = f"{example.format_table_for_prompt()}\n\nQuestion: {example.question}\n"
            messages.append({"role": "user", "content": example_instruction})
            messages.append({"role": "assistant", "content": f"Answer:\n{example.answer}"})

        # Add current query
        table_str = self._format_table(table, 2000, table_caption)
        current_instruction = f"{table_str}\n\n"
        current_instruction += f"Question: {question}\n"

        messages.append({"role": "user", "content": current_instruction})

        # Use tokenizer to format the conversation - model-agnostic!
        if self.is_instruct:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "Answer:"
        else:
            # For non-instruct models, just concatenate the messages
            result = ""
            for msg in messages:
                if msg["role"] == "user":
                    result += msg["content"] + "\n"
                else:
                    result += msg["content"] + "\n\n"
            return result
