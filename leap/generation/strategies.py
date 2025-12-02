
from typing import Any, Callable, Dict, List, Optional

from leap.evaluation.evaluator import calculate_execution_accuracy_with_dataset_answers
from leap.generation.generate import (
    generate_action_arguments,
    generate_action_selection,
    generate_single_action,
)
from leap.generation.prompt_builder import PromptBuilder
from leap.core import Action, Table, InferenceRequest, InferenceResult

DEFAULT_COT_ACTION_TEMPERATURE = 0.3
DEFAULT_COT_ARGS_TEMPERATURE = 0.7


class BaseGenerationStrategy:
    """Shared scaffolding for strategies that operate on a single request."""

    def __init__(
        self,
        *,
        prompt_builder: PromptBuilder,
        max_failures: int = 3,
        max_validity_failures: int = 3,
        max_steps: int = 10,
    ) -> None:
        self.max_failures = max_failures
        self.max_validity_failures = max_validity_failures
        self.max_steps = max_steps
        self.prompt_builder = prompt_builder

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

    def __init__(self, *, prompt_builder: PromptBuilder, **kwargs) -> None:
        super().__init__(prompt_builder=prompt_builder, **kwargs)

    async def generate_instance(
        self,
        request: InferenceRequest,
        worker,
        state_machines,
        logging_callback: Optional[Callable] = None,
    ) -> InferenceResult:
        question = request.question
        original_table: Table = request.table
        current_table: Table = original_table
        ground_truth_answers = request.ground_truth_answers
        request_id = request.request_id

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
            step_prompt = self.prompt_builder.build_iterative_prompt(
                question=question,
                table=current_table,
                action_history=action_history,
                worker=worker,
                step=step,
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

                action = Action.parse(action_str)
                if not action:
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

                if action.name == "end":
                    action_history.append(action.to_string())
                    if logging_callback:
                        logging_callback(
                            request_id,
                            step + 1,
                            action.to_string(),
                            current_table,
                            generation_mode=generation_mode,
                        )
                    break

                new_table = action.apply_to_table(current_table)

                if not new_table:
                    validity_failures += 1
                    print(
                        f"Step {step}: Failed to apply action: {action.to_string()}"
                    )
                    if logging_callback:
                        logging_callback(
                            request_id,
                            step + 1,
                            action.to_string(),
                            current_table,
                            success=False,
                            failure_type="validity_failure",
                            generation_mode=generation_mode,
                        )
                    continue

                current_table = new_table
                action_history.append(action.to_string())
                failures = 0
                validity_failures = 0
                step += 1
                print(f"Step {step}: Applied {action.to_string()}")

                if logging_callback:
                    logging_callback(
                        request_id,
                        step,
                        action.to_string(),
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

                # If this is a critical error (like max_model_len exceeded), break the loop
                error_str = str(exc).lower()
                if "max_model_len" in error_str or "maximum model length" in error_str:
                    print(f"Critical error detected: {exc}. Stopping generation for this instance.")
                    break

        accuracy_metrics = calculate_execution_accuracy_with_dataset_answers(
            action_history, current_table, ground_truth_answers, original_table
        )

        return InferenceResult(
            action_history=action_history,
            final_table=current_table,
            execution_metrics=accuracy_metrics,
            request_id=request_id,
            question=question,
            ground_truth_answers=ground_truth_answers,
        )


class ChainOfTableGenerationStrategy(BaseGenerationStrategy):
    """Two-step Chain-of-Table strategy (action selection + args)."""

    def __init__(
        self,
        *,
        prompt_builder: PromptBuilder,
        action_temperature: float = DEFAULT_COT_ACTION_TEMPERATURE,
        args_temperature: float = DEFAULT_COT_ARGS_TEMPERATURE,
        **kwargs,
    ) -> None:
        super().__init__(prompt_builder=prompt_builder, **kwargs)
        self.action_temperature = action_temperature
        self.args_temperature = args_temperature

    async def generate_instance(
        self,
        request: InferenceRequest,
        worker,
        state_machines,
        logging_callback: Optional[Callable] = None,
    ) -> InferenceResult:
        question = request.question
        original_table: Table = request.table
        current_table: Table = original_table
        ground_truth_answers = request.ground_truth_answers
        request_id = request.request_id

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
                    worker,
                    question,
                    current_table,
                    action_history,
                    request_id,
                    state_machines,
                    step,
                    self.action_temperature,
                    self.prompt_builder,
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
                    action = Action("end", [])
                    action_history.append(action.to_string())
                    if logging_callback:
                        logging_callback(
                            request_id,
                            step + 1,
                            action.to_string(),
                            current_table,
                            generation_mode=generation_mode,
                        )
                    break

                args = await generate_action_arguments(
                    worker,
                    question,
                    current_table,
                    action_name,
                    action_history,
                    request_id,
                    state_machines,
                    step,
                    self.args_temperature,
                    self.prompt_builder,
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

                # Create Action object from action_name and args
                action = Action(action_name, args)

                new_table = action.apply_to_table(current_table)

                if not new_table:
                    validity_failures += 1
                    print(
                        f"Step {step}: Failed to apply action: {action.to_string()}"
                    )
                    if logging_callback:
                        logging_callback(
                            request_id,
                            step + 1,
                            action.to_string(),
                            current_table,
                            success=False,
                            failure_type="validity_failure",
                            generation_mode=generation_mode,
                        )
                    continue

                current_table = new_table
                action_history.append(action.to_string())
                failures = 0
                validity_failures = 0
                step += 1
                print(f"Step {step}: Applied {action.to_string()} [CoT]")

                if logging_callback:
                    logging_callback(
                        request_id,
                        step,
                        action.to_string(),
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

                # If this is a critical error (like max_model_len exceeded), break the loop
                error_str = str(exc).lower()
                if "max_model_len" in error_str or "maximum model length" in error_str:
                    print(f"Critical error detected: {exc}. Stopping generation for this instance.")
                    break

        accuracy_metrics = calculate_execution_accuracy_with_dataset_answers(
            action_history, current_table, ground_truth_answers, original_table
        )

        return InferenceResult(
            action_history=action_history,
            final_table=current_table,
            execution_metrics=accuracy_metrics,
            request_id=request_id,
            question=question,
            ground_truth_answers=ground_truth_answers,
        )
