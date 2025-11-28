
from typing import Any, Callable, Dict, List, Optional, Tuple

from Evaluation import calculate_execution_accuracy_with_dataset_answers
from generate import (
    generate_action_arguments,
    generate_action_selection,
    generate_single_action,
)
from table import apply_action, serialize_table_to_csv

DEFAULT_COT_ACTION_TEMPERATURE = 0.3
DEFAULT_COT_ARGS_TEMPERATURE = 0.7


class BaseGenerationStrategy:
    """Shared scaffolding for strategies that operate on a single request."""

    def __init__(
        self,
        model_config,
        *,
        max_failures: int = 3,
        max_validity_failures: int = 3,
        max_steps: int = 10,
    ) -> None:
        self.model_config = model_config
        self.max_failures = max_failures
        self.max_validity_failures = max_validity_failures
        self.max_steps = max_steps

    async def generate_instance(
        self,
        request: Dict[str, Any],
        worker,
        state_machines,
        logging_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class IterativeGenerationStrategy(BaseGenerationStrategy):
    """Iterative action generation with constraint-aware prompting."""

    def __init__(self, model_config, **kwargs) -> None:
        super().__init__(model_config, **kwargs)

    async def generate_instance(
        self,
        request: Dict[str, Any],
        worker,
        state_machines,
        logging_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        question = request["question"]
        table = request["table"]
        ground_truth_answers = request["ground_truth_answers"]
        request_id = request["request_id"]

        current_table = table
        action_history: List[str] = []
        failures = 0
        validity_failures = 0
        step = 0

        generation_mode = worker._get_generation_mode_string()

        if logging_callback:
            logging_callback(
                request_id, 0, "initial", current_table, generation_mode=generation_mode
            )

        while (
            failures < self.max_failures
            and validity_failures < self.max_validity_failures
            and step < self.max_steps
        ):
            step_id = f"{request_id}_step{step}"

            max_chars = 1500 if step == 0 else 1000
            table_str = serialize_table_to_csv(current_table, max_chars)
            step_prompt = f"Table:\n{table_str}\n\nQuestion: {question}\n"

            if action_history:
                step_prompt += "Actions taken so far:\n"
                for i, action in enumerate(action_history):
                    step_prompt += f"{i+1}. {action}\n"
                step_prompt += "\n"

            if worker.use_constraints:
                instruction_prompt = "Next action: "
            else:
                instruction_prompt = (
                    'What should be the next action to answer this question? Choose from: '
                    'select_row([row_indices]), select_column(["column_names"]), or end(). Next action: '
                )

            estimated_length = len(step_prompt) // 4
            if estimated_length > worker.max_model_len - 100:
                table_str = serialize_table_to_csv(current_table, 500)
                question_short = (
                    question[:100] + "..." if len(question) > 100 else question
                )
                step_prompt = f"Table:\n{table_str}\n\nQuestion: {question_short}\n"
                instruction_prompt = (
                    "Next action: "
                    if worker.use_constraints
                    else 'What should be the next action? Choose from: select_row([row_indices]), '
                    'select_column(["column_names"]), or end(). Next action: '
                )

            step_prompt = self.model_config.add_instruct_tokens_for_instruct_models(
                step_prompt, instruction_prompt
            )

            try:
                action_str = await generate_single_action(
                    worker,
                    step_prompt,
                    current_table,
                    step_id,
                    state_machines,
                    action_history,
                )

                parsed_action = parse_action_string(action_str)
                if not parsed_action:
                    validity_failures += 1
                    print(
                        f"Step {step}: Failed to generate valid action from: {action_str}"
                    )
                    if logging_callback:
                        logging_callback(
                            request_id,
                            step + 1,
                            f"validity_failed:{action_str}",
                            current_table,
                            success=False,
                            failure_type="validity_failure",
                            generation_mode=generation_mode,
                        )
                    continue

                action_name, args = parsed_action

                if action_name == "end":
                    action_history.append("end()")
                    if logging_callback:
                        logging_callback(
                            request_id,
                            step + 1,
                            "end()",
                            current_table,
                            generation_mode=generation_mode,
                        )
                    break

                new_table = apply_action(current_table, action_name, args)

                if not new_table:
                    validity_failures += 1
                    print(
                        f"Step {step}: Failed to apply action: {action_name}({args})"
                    )
                    if logging_callback:
                        logging_callback(
                            request_id,
                            step + 1,
                            f"{action_name}({args})",
                            current_table,
                            success=False,
                            failure_type="validity_failure",
                            generation_mode=generation_mode,
                        )
                    continue

                current_table = new_table
                action_history.append(f"{action_name}({args})")
                failures = 0
                validity_failures = 0
                step += 1
                print(f"Step {step}: Applied {action_name}({args})")

                if logging_callback:
                    logging_callback(
                        request_id,
                        step,
                        f"{action_name}({args})",
                        current_table,
                        success=True,
                        generation_mode=generation_mode,
                    )

            except Exception as exc:
                failures += 1
                print(f"Step {step}: Generation error: {str(exc)}")
                if logging_callback:
                    logging_callback(
                        request_id,
                        step + 1,
                        f"generation_error:{str(exc)}",
                        current_table,
                        success=False,
                        failure_type="generation_error",
                        generation_mode=generation_mode,
                    )

        accuracy_metrics = calculate_execution_accuracy_with_dataset_answers(
            action_history, current_table, ground_truth_answers, table
        )

        return {
            "action_history": action_history,
            "final_table": current_table,
            "execution_accuracy_metrics": accuracy_metrics,
        }


class ChainOfTableGenerationStrategy(BaseGenerationStrategy):
    """Two-step Chain-of-Table strategy (action selection + args)."""

    def __init__(
        self,
        model_config,
        *,
        action_temperature: float = DEFAULT_COT_ACTION_TEMPERATURE,
        args_temperature: float = DEFAULT_COT_ARGS_TEMPERATURE,
        **kwargs,
    ) -> None:
        super().__init__(model_config, **kwargs)
        self.action_temperature = action_temperature
        self.args_temperature = args_temperature

    async def generate_instance(
        self,
        request: Dict[str, Any],
        worker,
        state_machines,
        logging_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        question = request["question"]
        table = request["table"]
        ground_truth_answers = request["ground_truth_answers"]
        request_id = request["request_id"]

        current_table = table
        action_history: List[str] = []
        failures = 0
        validity_failures = 0
        step = 0

        generation_mode = "CoT"

        if logging_callback:
            logging_callback(
                request_id, 0, "initial", current_table, generation_mode=generation_mode
            )

        while (
            failures < self.max_failures
            and validity_failures < self.max_validity_failures
            and step < self.max_steps
        ):
            try:
                action_name = await generate_action_selection(
                    self.model_config,
                    worker,
                    question,
                    current_table,
                    action_history,
                    request_id,
                    state_machines,
                    step,
                    self.action_temperature,
                )

                if not action_name:
                    validity_failures += 1
                    print(f"Step {step}: Failed to select valid action")
                    if logging_callback:
                        logging_callback(
                            request_id,
                            step + 1,
                            "action_selection_failed",
                            current_table,
                            success=False,
                            failure_type="validity_failure",
                            generation_mode=generation_mode,
                        )
                    continue

                if action_name == "end":
                    action_history.append("end()")
                    if logging_callback:
                        logging_callback(
                            request_id,
                            step + 1,
                            "end()",
                            current_table,
                            generation_mode=generation_mode,
                        )
                    break

                args = await generate_action_arguments(
                    self.model_config,
                    worker,
                    question,
                    current_table,
                    action_name,
                    action_history,
                    request_id,
                    state_machines,
                    step,
                    self.args_temperature,
                )

                if args is None:
                    validity_failures += 1
                    print(
                        f"Step {step}: Failed to generate valid arguments for {action_name}"
                    )
                    if logging_callback:
                        logging_callback(
                            request_id,
                            step + 1,
                            f"{action_name}_args_failed",
                            current_table,
                            success=False,
                            failure_type="validity_failure",
                            generation_mode=generation_mode,
                        )
                    continue

                new_table = apply_action(current_table, action_name, args)

                if not new_table:
                    validity_failures += 1
                    print(
                        f"Step {step}: Failed to apply action: {action_name}({args})"
                    )
                    if logging_callback:
                        logging_callback(
                            request_id,
                            step + 1,
                            f"{action_name}({args})",
                            current_table,
                            success=False,
                            failure_type="validity_failure",
                            generation_mode=generation_mode,
                        )
                    continue

                current_table = new_table
                action_history.append(f"{action_name}({args})")
                failures = 0
                validity_failures = 0
                step += 1
                print(f"Step {step}: Applied {action_name}({args}) [CoT]")

                if logging_callback:
                    logging_callback(
                        request_id,
                        step,
                        f"{action_name}({args})",
                        current_table,
                        success=True,
                        generation_mode=generation_mode,
                    )

            except Exception as exc:
                failures += 1
                print(f"Step {step}: Generation error in CoT: {str(exc)}")
                if logging_callback:
                    logging_callback(
                        request_id,
                        step + 1,
                        f"cot_generation_error:{str(exc)}",
                        current_table,
                        success=False,
                        failure_type="generation_error",
                        generation_mode=generation_mode,
                    )

        accuracy_metrics = calculate_execution_accuracy_with_dataset_answers(
            action_history, current_table, ground_truth_answers, table
        )

        return {
            "action_history": action_history,
            "final_table": current_table,
            "execution_accuracy_metrics": accuracy_metrics,
        }


def parse_action_string(action_str: str) -> Optional[Tuple[str, List]]:
    """Parse action string into (action_name, args) tuple."""
    try:
        action_str = action_str.strip()
        if action_str == "end" or action_str.startswith("end("):
            return "end", []

        if "(" not in action_str:
            return None

        action_name, args_str = action_str.split("(", 1)
        action_name = action_name.strip()
        args_str = args_str.rstrip(")").replace("row ", "").strip()

        if args_str.startswith("[") and args_str.endswith("]"):
            args_str = args_str.replace("\\", "\\\\")
            args_list = ast_literal_eval(args_str)
            return action_name, args_list

        if "," in args_str:
            args_list = [arg.strip() for arg in args_str.split(",")]
        else:
            args_list = [args_str]

        return action_name, args_list
    except Exception:
        return None


def ast_literal_eval(args_str: str):
    """Wrapper to keep literal_eval localized for easier mocking/testing."""
    import ast

    return ast.literal_eval(args_str)
