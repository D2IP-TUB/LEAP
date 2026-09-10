from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from leap.core import Action, Table
from leap.core.actions import REGISTRY
from leap.generation.action_examples import ActionPromptBuilder, PromptCatalog
from leap.inference.json_constraints import JsonActionCodec, available_json_actions


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
        prompt_catalog: PromptCatalog | None = None,
        output_format: str = "function",
    ):
        self.tokenizer = tokenizer
        self.is_instruct = is_instruct
        self.iterative_settings = iterative_settings or IterativePromptSettings()
        self.cot_settings = cot_settings or CotPromptSettings()
        if output_format not in {"function", "json"}:
            raise ValueError("output_format must be 'function' or 'json'")
        self.output_format = output_format

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
            excluded_actions = self._excluded_actions(worker) | {"add_column"}
            prompt = self._compose_iterative_prompt(
                question=question_short,
                table_str=table_str,
                action_history=action_history,
                worker=worker,
                excluded_actions=excluded_actions,
            )

        return prompt

    def _compose_iterative_prompt(
        self,
        *,
        question: str,
        table_str: str,
        action_history: Sequence[str],
        worker,
        excluded_actions: set[str] | None = None,
    ) -> str:
        """Render the complete iterative prompt for one current table state."""
        excluded_actions = self._excluded_actions(worker) if excluded_actions is None else excluded_actions
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
        if self.output_format == "json":
            available_names = available_json_actions(action_history, use_global_constraints=use_global_constraints)
            available_names = [name for name in available_names if name not in excluded_actions]
            action_descriptions = "\n".join(f"- {name}: {REGISTRY.get(name).get_description()}" for name in available_names)
            actions_text = "\n".join(self.prompt_catalog.iterative_json["operation_shapes"][name] for name in available_names)

        messages = [
            {
                "role": "system",
                "content": self._format_system_for_output(self.prompt_catalog.iterative_system(add_column_available=add_column_available)),
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
                if self.output_format == "json":
                    answer = self._chain_to_json(answer)
                messages.append({"role": "assistant", "content": answer})

        if self.output_format == "json":
            history = JsonActionCodec.history_json(action_history) if action_history else "[]"
        else:
            history = "\n".join(f"{idx + 1}. {self._to_display_action(action)}" for idx, action in enumerate(action_history)) or "None"
        available_operations = f"{action_descriptions}\n{actions_text}".strip()
        current_template = self.prompt_catalog.iterative_template("current_turn")
        if self.output_format == "json":
            marker = (
                "Return only the single next operation, including all arguments. "
                "Do not return an operation chain or an explanation.\nNext operation:"
            )
            instruction = self.prompt_catalog.iterative_json["instruction"] + "\nNext JSON operation:"
            current_template = current_template.replace(marker, instruction)
        messages.append(
            {
                "role": "user",
                "content": self.prompt_catalog.render(
                    current_template,
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
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        result = ""
        for message in messages:
            result += message["content"] + "\n" if message["role"] == "user" else message["content"] + "\n\n"
        return result

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
        return True

    def _format_system_for_output(self, text: str) -> str:
        if self.output_format != "json":
            return text

        def replace_call(match: re.Match[str]) -> str:
            action = Action.parse(match.group(0))
            if action:
                return JsonActionCodec.dumps(action)
            name = match.group(0).split("(", 1)[0].removeprefix("f_")
            return self.prompt_catalog.cot_json["operation_shapes"].get(name, match.group(0))

        text = re.sub(r"f_[a-z_]+\([^\n]*\)", replace_call, text)
        for name, shape in self.prompt_catalog.cot_json["operation_shapes"].items():
            text = text.replace(f"f_{name}", shape)
        replacement_name = "action field"
        replacement_call = "complete JSON operation object"
        return (
            text.replace("complete function call", replacement_call)
            .replace("function name or outer parentheses", replacement_name)
            .replace("function name", replacement_name)
        )

    @staticmethod
    def _chain_to_json(answer: str) -> str:
        actions = [Action.parse(part.strip()) for part in answer.split("->")]
        if not actions or any(action is None for action in actions):
            raise ValueError(f"Invalid function-chain prompt example: {answer!r}")
        return json.dumps([JsonActionCodec.to_dict(action) for action in actions], ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _argument_answer_to_json(action_name: str, answer: str) -> str:
        action = Action.parse(f"{action_name}({answer})")
        if action is None:
            raise ValueError(f"Invalid {action_name} prompt example: {answer!r}")
        return JsonActionCodec.dumps(action, include_action=False)

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
        if self.output_format == "json":
            history = JsonActionCodec.history_json(action_history) if action_history else "[]"
            names = available_json_actions(action_history, use_global_constraints=use_global_constraints)
            names = [name for name in names if name not in excluded_actions]
            actions_text = ", ".join(json.dumps({"action": name}, separators=(",", ":")) for name in names)
            question_suffix = self.prompt_catalog.cot_json["action_instruction"]
        else:
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
        if self.output_format == "json":
            names = available_json_actions(action_history, use_global_constraints=use_global_constraints)
            names = [name for name in names if name not in excluded_actions]
            example_actions_text = ", ".join(json.dumps({"action": name}, separators=(",", ":")) for name in names)
            history = JsonActionCodec.history_json(action_history) if action_history else "[]"
            question_suffix = self.prompt_catalog.cot_json["action_instruction"]
        else:
            example_actions_text = REGISTRY.get_prompt_text_cot(
                action_history,
                exclude_terminating_on_first=True,
                excluded_actions=excluded_actions,
                use_global_constraints=use_global_constraints,
            )
            history = "\n".join(f"{idx + 1}. {self._to_display_action(action)}" for idx, action in enumerate(action_history)) or "None"
            question_suffix = ""
        return self.prompt_catalog.render(
            self.prompt_catalog.cot_template("action_selection_turn"),
            table=example.format_table_for_prompt(),
            question=example.question,
            action_history=history,
            available_label="Available actions",
            available_actions=example_actions_text,
            question_suffix=question_suffix,
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
        if self.output_format == "json":
            suffix = self.prompt_catalog.cot_json["action_instruction"]
            system = self._format_system_for_output(system) + "\n\n" + suffix
        messages = [{"role": "system", "content": system}]

        for example in examples:
            answer = example.answer if add_column_available else example.answer_without_add_column
            if not answer:
                continue
            if self.output_format == "json":
                first_action = Action.parse(answer.split("->", 1)[0].strip())
                if first_action is None:
                    raise ValueError(f"Invalid action-selection prompt example: {answer!r}")
                answer = json.dumps({"action": first_action.name}, separators=(",", ":"))
            messages.append(
                {
                    "role": "user",
                    "content": self._build_action_selection_example_message(example, excluded_actions, use_global_constraints),
                }
            )
            messages.append({"role": "assistant", "content": answer})

        messages.append({"role": "user", "content": instruction_prompt})

        if self.is_instruct:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
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

        if self.output_format == "json":
            instruction = self._format_system_for_output(instruction or "")
            suffix = self.prompt_catalog.cot_json["argument_instructions"][action_name]
            instruction += "\n\n" + suffix

        if instruction:
            messages.append({"role": "system", "content": instruction})

        for example in examples:
            example_prompt = self.prompt_catalog.render(
                self.prompt_catalog.cot_template("argument_turn"),
                table=example.format_table_for_prompt(),
                question=example.question,
            )
            messages.append({"role": "user", "content": example_prompt})
            example_answer = example.answer
            if self.output_format == "json":
                example_answer = self._argument_answer_to_json(action_name, example_answer)
            messages.append({"role": "assistant", "content": example_answer})

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
            instruct_result = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
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
            messages.append({"role": "assistant", "content": example.answer})

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
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            # For non-instruct models, just concatenate the messages
            result = ""
            for msg in messages:
                if msg["role"] == "user":
                    result += msg["content"] + "\n"
                else:
                    result += msg["content"] + "\n\n"
            return result
