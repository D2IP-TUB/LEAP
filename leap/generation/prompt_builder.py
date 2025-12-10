from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from leap.core import Table
from leap.core.actions import REGISTRY


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
    ):
        self.tokenizer = tokenizer
        self.is_instruct = is_instruct
        self.iterative_settings = iterative_settings or IterativePromptSettings()
        self.cot_settings = cot_settings or CotPromptSettings()

    def build_iterative_prompt(
        self,
        *,
        question: str,
        table: Table,
        action_history: Sequence[str],
        worker,
        step: int,
    ) -> str:
        """Create a constraint-aware prompt for iterative generation."""
        max_chars = self.iterative_settings.initial_table_chars if step == 0 else self.iterative_settings.step_table_chars
        table_str = table.to_csv(max_chars=max_chars)

        step_prompt = f"Table:\n{table_str}\n\nQuestion: {question}\n"
        if action_history:
            step_prompt += "Actions taken so far:\n"
            for idx, action in enumerate(action_history):
                step_prompt += f"{idx + 1}. {action}\n"
            step_prompt += "\n"

        instruction_prompt = self._build_iterative_instruction(worker)

        estimated_length = len(step_prompt) // 4
        if estimated_length > worker.max_model_len - self.iterative_settings.safety_margin_tokens:
            table_str = table.to_csv(max_chars=self.iterative_settings.fallback_table_chars)
            question_short = self._truncate_text(question, self.iterative_settings.question_truncation)
            step_prompt = f"Table:\n{table_str}\n\nQuestion: {question_short}\n"
            instruction_prompt = self._build_iterative_instruction(worker, fallback=True)

        return self._append_instruction(step_prompt, instruction_prompt)

    def _build_iterative_instruction(self, worker, fallback: bool = False) -> str:
        """Instruction text for iterative generation."""
        if worker.use_constraints:
            return "Next action: "

        # Get action prompt text from registry
        actions_text = REGISTRY.get_prompt_text_iterative()

        question_prefix = "What should be the next action? " if fallback else "What should be the next action to answer this question? "
        base = f"{question_prefix}Choose from: {actions_text}. Next action: "
        return base

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        return text[:limit] + "..." if len(text) > limit else text

    def build_cot_action_prompt(
        self,
        *,
        question: str,
        table: Table,
        action_history: Sequence[str],
        worker,
    ) -> str:
        """Prompt for CoT action selection (dynamic plan)."""
        table_str = table.to_csv(max_chars=self.cot_settings.action_table_chars)
        prompt = f"Table:\n{table_str}\n\n"
        prompt += f"Question: {question}\n\n"

        if action_history:
            prompt += "Actions taken so far:\n"
            for idx, action in enumerate(action_history):
                prompt += f"{idx + 1}. {action}\n"
            prompt += "\n"

        # Get action list from registry
        actions_text = REGISTRY.get_prompt_text_cot()
        prompt += f"Available actions: {actions_text}\n"
        instruction_prompt = "What action should be performed next to answer the question?\n"
        instruction_prompt += "Action: "

        estimated_length = len(prompt) // 4
        if estimated_length > worker.max_model_len - self.cot_settings.action_safety_margin_tokens:
            table_str = table.to_csv(max_chars=self.cot_settings.action_fallback_table_chars)
            question_short = self._truncate_text(question, self.cot_settings.action_question_truncation)
            prompt = f"Table:\n{table_str}\n\nQuestion: {question_short}\n\n"
            prompt += f"Available actions: {actions_text}\n"
            instruction_prompt = "What action should be performed next?\nAction: "

        return self._append_instruction(prompt, instruction_prompt)

    def build_cot_arguments_prompt(
        self,
        *,
        question: str,
        table: Table,
        action_name: str,
        action_history: Sequence[str],
        worker,
    ) -> str:
        """Prompt for CoT argument generation."""
        table_str = table.to_csv(max_chars=self.cot_settings.args_table_chars)
        prompt = f"Table:\n{table_str}\n\n"
        prompt += f"Question: {question}\n\n"

        if action_history:
            prompt += "Actions taken so far:\n"
            for idx, action in enumerate(action_history):
                prompt += f"{idx + 1}. {action}\n"
            prompt += "\n"

        prompt += f"Selected action: {action_name}\n"

        if action_name == "select_row":
            prompt += f"Available rows: 0 to {len(table.rows) - 1}\n"
            instruction_prompt = "Which row indices should be selected? Provide the indices as a list, e.g., [0, 1, 2]\n"
            instruction_prompt += "Row indices: "
        elif action_name == "select_column":
            prompt += f"Available columns: {list(table.columns)}\n"
            instruction_prompt = 'Which columns should be selected? Provide the column names as a list, e.g., ["Name", "Age"]\n'
            instruction_prompt += "Column names: "
        else:
            prompt += "No arguments needed for end action.\n"
            instruction_prompt = "Arguments: "

        estimated_length = len(prompt) // 4
        if estimated_length > worker.max_model_len - self.cot_settings.args_safety_margin_tokens:
            table_str = table.to_csv(max_chars=self.cot_settings.args_fallback_table_chars)
            question_short = self._truncate_text(question, self.cot_settings.args_question_truncation)
            prompt = f"Table:\n{table_str}\n\nQuestion: {question_short}\n\n"
            prompt += f"Selected action: {action_name}\n"

            if action_name == "select_row":
                instruction_prompt = f"Which row indices (0 to {len(table.rows) - 1})?\nRow indices: "
            else:
                sample_cols = list(table.columns)[:3]
                instruction_prompt = f"Which columns from {sample_cols}...?\nColumn names: "

        return self._append_instruction(prompt, instruction_prompt)

    def _append_instruction(self, prompt: str, instruction_prompt: str) -> str:
        if self.is_instruct:
            message = [{"role": "user", "content": instruction_prompt}]
            addition = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False).strip("<s> ")
            return prompt + addition
        return prompt + instruction_prompt
