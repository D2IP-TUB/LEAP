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

        instruction_prompt = self._build_iterative_instruction(worker, action_history)

        estimated_length = len(step_prompt) // 4
        if estimated_length > worker.max_model_len - self.iterative_settings.safety_margin_tokens:
            table_str = table.to_csv(max_chars=self.iterative_settings.fallback_table_chars)
            question_short = self._truncate_text(question, self.iterative_settings.question_truncation)
            step_prompt = f"Table:\n{table_str}\n\nQuestion: {question_short}\n"
            instruction_prompt = self._build_iterative_instruction(worker, action_history, fallback=True)

        return self._append_instruction(step_prompt, instruction_prompt)

    def _build_iterative_instruction(self, worker, action_history: Sequence[str] = None, fallback: bool = False) -> str:
        """Instruction text for iterative generation."""
        # Get action descriptions from registry (as per Chain-of-Table paper Figure 9)
        # Pass action_history to filter out already-used actions
        # For iterative mode, we don't exclude terminating actions on first step
        action_descriptions = REGISTRY.get_action_descriptions(action_history, exclude_terminating_on_first=False)

        if worker.use_constraints:
            # When using constraints, still show action descriptions
            return f"{action_descriptions}\n\nNext action: "

        # Get action prompt text from registry
        # Pass action_history to filter out already-used actions
        actions_text = REGISTRY.get_prompt_text_iterative(action_history)

        question_prefix = "What should be the next action? " if fallback else "What should be the next action to answer this question? "
        base = f"{action_descriptions}\n\n{question_prefix}Choose from: {actions_text}.\nNext action: "
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
        # Build current instance prompt
        table_str = table.to_csv(max_chars=self.cot_settings.action_table_chars)
        instruction_prompt = f"Table:\n{table_str}\n\n"
        instruction_prompt += f"Question: {question}\n\n"

        if action_history:
            instruction_prompt += "Actions taken so far:\n"
            for idx, action in enumerate(action_history):
                instruction_prompt += f"{idx + 1}. {action}\n"
            instruction_prompt += "\n"

        actions_text = REGISTRY.get_prompt_text_cot(action_history, exclude_terminating_on_first=True)
        
        instruction_prompt += f"Available actions: {actions_text}\n"
        instruction_prompt += "What actions should be performed next to answer the question?\n"
        instruction_prompt += "Action: "

        if self.action_examples and self.action_examples.has_prompt("action_selection"):
            examples = self.action_examples.examples_manager.get_examples("action_selection")

        if not examples:
            return self._append_instruction("", instruction_prompt)

        # Build conversation history with examples using messages format
        messages = []

        # Add instruction as system message if available
        instruction = self.action_examples.get_instruction("action_selection")
        if instruction:
            messages.append({"role": "system", "content": instruction})

        for example in examples:
            # Format each example as a conversation turn (user message + assistant response)
            example_table_str = example.table.to_csv(max_chars=2000, crop=False)
            example_prompt = (
                f"Table:\n{example_table_str}\n\n"
                f"Question: {example.question}\n"
            )
            if example.explanation:
                example_prompt += f"Explanation: {example.explanation}\n"
            example_prompt += "Action: "
            
            messages.append({"role": "user", "content": example_prompt})
            messages.append({"role": "assistant", "content": example.answer})

        # Add current instance as final user message
        messages.append({"role": "user", "content": instruction_prompt})

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


    def build_cot_arguments_prompt(
        self,
        # *,
        question: str,
        table: Table,
        action_name: str,
        worker,
    ) -> str:
        """Prompt for CoT argument generation."""

        messages = []

        prompt = self.action_examples.get_instruction(action_name)
        examples, example_answers = self.action_examples.get_examples(action_name)

        for example, example_answer in zip(examples, example_answers):
            prompt += example
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": example_answer})
            prompt = ""

        table_str = table.to_csv(max_chars=self.cot_settings.args_table_chars)
        final_prompt = f"Table:\n{table_str}\n\n"
        final_prompt += f"Question: {question}\n\n"
        final_prompt += self.action_examples.get_final_instructions(action_name, table)

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
        # Use similar max_chars as final query in paper
        table_str = table.to_csv(max_chars=2000)

        # Build messages for conversation with 1-shot example
        messages = []

        # Load example from YAML
        query_examples = self.action_examples.examples_manager.get_examples("query_answer")
        example = query_examples[0]
        example_table = example.format_table_for_prompt()
        example_question = example.question
        example_answer = example.answer

        # Build example instruction with format guidelines
        example_instruction = "Here is the table to answer this question. Please understand the table and answer the question:\n\n"
        example_instruction += "Provide your answer(s) as a Python list of strings.\n"
        example_instruction += "Examples:\n"
        example_instruction += '- Single answer: ["Italy"]\n'
        example_instruction += '- Multiple answers: ["Italy", "Spain", "France"]\n'
        example_instruction += '- Yes/no: ["yes"] or ["no"]\n\n'
        example_instruction += f"Table:\n{example_table}\n\n"
        example_instruction += f"Question: {example_question}\n"

        messages.append({"role": "user", "content": example_instruction})
        messages.append({"role": "assistant", "content": f"Answer:{example_answer}"})

        # Add current query
        current_instruction = "Here is the table to answer this question. Please understand the table and answer the question:\n\n"
        current_instruction += "Provide your answer(s) as a Python list of strings.\n\n"
        current_instruction += f"Table:\n{table_str}\n\n"
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
