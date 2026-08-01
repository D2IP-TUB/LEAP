from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from leap.core import Table
from leap.core.actions import REGISTRY
from leap.generation.action_examples import ActionPromptBuilder, PromptCatalog


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


class PromptBuilder:
    """Centralized helper for constructing model prompts."""

    def __init__(
        self,
        tokenizer,
        is_instruct: bool,
        iterative_settings: IterativePromptSettings | None = None,
        cot_settings: CotPromptSettings | None = None,
        use_action_examples: bool = True,
        prompt_catalog: PromptCatalog | None = None,
    ):
        self.tokenizer = tokenizer
        self.is_instruct = is_instruct
        self.iterative_settings = iterative_settings or IterativePromptSettings()
        self.cot_settings = cot_settings or CotPromptSettings()

        self.prompt_catalog = prompt_catalog or PromptCatalog()

        # Initialize CoT action examples system (builds templates once)
        self.action_examples: Optional[ActionPromptBuilder] = None
        if use_action_examples:
            self.action_examples = ActionPromptBuilder(self.prompt_catalog.cot_examples_manager)

    @staticmethod
    def _format_table(table: Table, max_chars: int, crop: bool = False) -> str:
        """Format table CSV."""
        table_str = table.to_csv(max_chars=max_chars, crop=crop)
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
    ) -> str:
        """Create a single-phase prompt with operation guidance and chain examples."""
        max_chars = self.iterative_settings.initial_table_chars if step == 0 else self.iterative_settings.step_table_chars
        table_str = self._format_table(table, max_chars)

        prompt = self._compose_iterative_prompt(
            question=question,
            table_str=table_str,
            action_history=action_history,
            worker=worker,
        )

        estimated_length = len(prompt) // 4
        if estimated_length > worker.max_model_len - self.iterative_settings.safety_margin_tokens:
            table_str = self._format_table(table, self.iterative_settings.fallback_table_chars, True)
            question_short = self._truncate_text(question, self.iterative_settings.question_truncation)
            prompt = self._compose_iterative_prompt(
                question=question_short,
                table_str=table_str,
                action_history=action_history,
                worker=worker,
            )

        return prompt

    def _compose_iterative_prompt(
        self,
        *,
        question: str,
        table_str: str,
        action_history: Sequence[str],
        worker,
    ) -> str:
        """Render the complete iterative prompt for one current table state."""
        excluded_actions = self._excluded_actions(worker)
        add_column_available = not excluded_actions
        use_global_constraints = getattr(worker, "use_global_constraints", False) is True
        action_descriptions = REGISTRY.get_action_descriptions(
            action_history,
            exclude_terminating_on_first=False,
            excluded_actions=excluded_actions,
            use_global_constraints=use_global_constraints,
        )
        actions_text = REGISTRY.get_prompt_text_iterative(
            action_history,
            excluded_actions=excluded_actions,
            use_global_constraints=use_global_constraints,
        )

        messages = [
            {
                "role": "system",
                "content": self.prompt_catalog.iterative_system(add_column_available=add_column_available),
            }
        ]
        if self.action_examples is not None:
            example_template = self.prompt_catalog.iterative_template("example_turn")
            for example in self.prompt_catalog.iterative_examples:
                messages.append(
                    {
                        "role": "user",
                        "content": self.prompt_catalog.render(
                            example_template,
                            table=example.format_table_for_prompt(),
                            question=example.question,
                        ),
                    }
                )
                answer = example.answer if add_column_available else example.answer_without_add_column
                messages.append({"role": "assistant", "content": self._example_answer(example, answer, worker)})

        history = "\n".join(f"{idx + 1}. {self._to_display_action(action)}" for idx, action in enumerate(action_history)) or "None"
        available_operations = f"{action_descriptions}\n{actions_text}".strip()
        messages.append(
            {
                "role": "user",
                "content": self.prompt_catalog.render(
                    self.prompt_catalog.iterative_template("current_turn"),
                    table=table_str,
                    question=question,
                    action_history=history,
                    available_operations=available_operations,
                ),
            }
        )
        return self._serialize_messages(messages)

    def _serialize_messages(self, messages) -> str:
        if self.is_instruct:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = ""
        for message in messages:
            result += message["content"] + "\n" if message["role"] == "user" else message["content"] + "\n\n"
        return result

    @staticmethod
    def _example_answer(example, answer: str, worker) -> str:
        generation_config = getattr(worker, "generation_config", None)
        include_explanations = getattr(generation_config, "include_prompt_explanations", False) is True
        if include_explanations and example.explanation:
            return f"Explanation: {example.explanation}\nAnswer: {answer}"
        return answer

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

    @staticmethod
    def _add_column_available(worker) -> bool:
        """Whether this run may expose add_column to the model."""
        if not REGISTRY.is_enabled("add_column"):
            return False
        return not (
            getattr(worker, "use_constraints", False) is True
            and getattr(worker, "constraint_backend", "legacy_state_machine") == "xgrammar"
        )

    @classmethod
    def _excluded_actions(cls, worker) -> set[str]:
        return set() if cls._add_column_available(worker) else {"add_column"}

    @staticmethod
    def _remove_add_column_rule(system: str) -> str:
        """Return the existing action-selection rules with only add_column material removed."""
        system = system.replace(", f_add_column", "")
        rule_start = "If the table lacks a column to answer the question, use f_add_column()"
        next_rule = "Often only a subset of columns is required to answer a question."
        start = system.find(rule_start)
        end = system.find(next_rule, start)
        if start >= 0 and end >= 0:
            system = system[:start] + system[end:]
        return system

    def _build_action_selection_body(
        self,
        table_str: str,
        question: str,
        action_history: Sequence[str],
        available_label: str,
        question_suffix: str,
        excluded_actions: set[str],
        use_global_constraints: bool,
    ) -> str:
        """Build the user-message body for an action-selection turn."""
        history = "\n".join(f"{idx + 1}. {self._to_display_action(action)}" for idx, action in enumerate(action_history)) or "None"
        actions_text = REGISTRY.get_prompt_text_cot(
            action_history,
            exclude_terminating_on_first=True,
            excluded_actions=excluded_actions,
            use_global_constraints=use_global_constraints,
        )
        return self.prompt_catalog.render(
            self.prompt_catalog.cot_template("action_selection_turn"),
            table=table_str,
            question=question,
            action_history=history,
            available_label=available_label,
            available_actions=actions_text,
            question_suffix=question_suffix,
        )

    def _build_action_selection_example_message(self, example, excluded_actions: set[str], use_global_constraints: bool) -> str:
        """Build the user-message content for a single few-shot action-selection example."""
        action_history = example.action_history or []
        example_actions_text = REGISTRY.get_prompt_text_cot(
            action_history,
            exclude_terminating_on_first=True,
            excluded_actions=excluded_actions,
            use_global_constraints=use_global_constraints,
        )
        history = "\n".join(f"{idx + 1}. {self._to_display_action(action)}" for idx, action in enumerate(action_history)) or "None"
        return self.prompt_catalog.render(
            self.prompt_catalog.cot_template("action_selection_turn"),
            table=example.format_table_for_prompt(),
            question=example.question,
            action_history=history,
            available_label="Available actions",
            available_actions=example_actions_text,
            question_suffix="",
        )

    def build_cot_action_prompt(
        self,
        *,
        question: str,
        table: Table,
        action_history: Sequence[str],
        worker,
    ) -> str:
        """Prompt for CoT action selection (dynamic plan)."""
        table_str = self._format_table(table, self.cot_settings.action_table_chars)
        excluded_actions = self._excluded_actions(worker)
        use_global_constraints = getattr(worker, "use_global_constraints", False) is True
        available_label = "Available actions"
        question_suffix = ""
        instruction_prompt = self._build_action_selection_body(
            table_str,
            question,
            action_history,
            available_label,
            question_suffix,
            excluded_actions,
            use_global_constraints,
        )

        estimated_length = len(instruction_prompt) // 4
        if estimated_length > worker.max_model_len - self.cot_settings.action_safety_margin_tokens:
            table_str = self._format_table(table, self.cot_settings.action_fallback_table_chars, True)
            available_label = "The next operation must be one of the following"
            question_suffix = ""
            question_short = self._truncate_text(question, self.cot_settings.action_question_truncation)
            instruction_prompt = self._build_action_selection_body(
                table_str,
                question_short,
                action_history,
                available_label,
                question_suffix,
                excluded_actions,
                use_global_constraints,
            )

        if not (self.action_examples and self.action_examples.has_prompt("action_selection")):
            return self._append_instruction("", instruction_prompt)

        examples = self.action_examples.examples_manager.get_examples("action_selection")
        if not (examples and self.is_instruct):
            return self._append_instruction("", instruction_prompt)

        system = self.action_examples.examples_manager.get_system_rules("action_selection")
        add_column_available = not excluded_actions
        if not add_column_available:
            system = self._remove_add_column_rule(system)
        messages = [{"role": "system", "content": system}]

        for example in examples:
            answer = example.answer if add_column_available else example.answer_without_add_column
            if not answer:
                continue
            messages.append(
                {
                    "role": "user",
                    "content": self._build_action_selection_example_message(example, excluded_actions, use_global_constraints),
                }
            )
            messages.append({"role": "assistant", "content": self._example_answer(example, answer, worker)})

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
    ) -> str:
        """Prompt for CoT argument generation."""

        if action_name == "add_column" and not self._add_column_available(worker):
            raise ValueError("add_column is not available for this generation configuration")

        messages = []

        instruction = self.prompt_catalog.cot_examples_manager.get_system_rules(action_name)
        examples = self.prompt_catalog.cot_examples_manager.get_examples(action_name)

        if instruction:
            messages.append({"role": "system", "content": instruction})

        for example in examples:
            example_prompt = self.prompt_catalog.render(
                self.prompt_catalog.cot_template("argument_turn"),
                table=example.format_table_for_prompt(),
                question=example.question,
            )
            messages.append({"role": "user", "content": example_prompt})
            messages.append({"role": "assistant", "content": self._example_answer(example, example.answer, worker)})

        if action_name == "add_column" and len(table.rows) > 3:
            table = Table(columns=list(table.columns), rows=[list(r) for r in table.rows[:3]])

        table_str = self._format_table(table, self.cot_settings.args_table_chars)

        estimated_length = len(instruction) // 4
        if estimated_length > worker.max_model_len - self.cot_settings.args_table_chars:
            table_str = self._format_table(table, self.cot_settings.action_fallback_table_chars, True)

        final_prompt = self.prompt_catalog.render(
            self.prompt_catalog.cot_template("argument_turn"),
            table=table_str,
            question=question,
        )

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

        messages = []

        if self.is_instruct:
            messages.append(
                {
                    "role": "system",
                    "content": self.prompt_catalog.render(
                        self.prompt_catalog.cot_template("add_column_row_system"),
                        column_name=column_name,
                    ),
                }
            )

        # One user/assistant shot per seed row
        for i, (row, value) in enumerate(zip(table.rows[:3], seed_values[:3])):
            row_csv = self._format_row_csv(table, row, i)
            messages.append(
                {
                    "role": "user",
                    "content": self.prompt_catalog.render(
                        self.prompt_catalog.cot_template("add_column_row_turn"),
                        task_description=task_description,
                        row=row_csv,
                        column_name=column_name,
                    ),
                }
            )
            messages.append({"role": "assistant", "content": str(value)})

        # Target row — final user message
        target_csv = self._format_row_csv(table, target_row, target_row_idx)
        messages.append(
            {
                "role": "user",
                "content": self.prompt_catalog.render(
                    self.prompt_catalog.cot_template("add_column_row_turn"),
                    task_description=task_description,
                    row=target_csv,
                    column_name=column_name,
                ),
            }
        )

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
    ) -> str:
        """
        Build Query(T,Q) prompt for answer generation.

        This follows Chain-of-Table paper Section 3.4 and Appendix E.3.
        The final table from the operation chain is used to generate the answer.
        """
        messages = []

        messages.append({"role": "system", "content": self.prompt_catalog.direct_query["system"].rstrip()})

        for example in self.prompt_catalog.direct_query_examples:
            example_instruction = self.prompt_catalog.render(
                self.prompt_catalog.direct_query_template("turn"),
                table=example.format_table_for_prompt(),
                question=example.question,
            )
            messages.append({"role": "user", "content": example_instruction})
            messages.append({"role": "assistant", "content": self._example_answer(example, example.answer, worker)})

        # Add current query
        table_str = self._format_table(table, 2000)

        current_instruction = self.prompt_catalog.render(
            self.prompt_catalog.direct_query_template("turn"),
            table=table_str,
            question=question,
        )

        messages.append({"role": "user", "content": current_instruction})

        # Use tokenizer to format the conversation - model-agnostic!
        if self.is_instruct:
            # return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "Answer:"
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # For non-instruct models, just concatenate the messages
            result = ""
            for msg in messages:
                if msg["role"] == "user":
                    result += msg["content"] + "\n"
                else:
                    result += msg["content"] + "\n\n"
            return result
