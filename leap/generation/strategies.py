from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from leap.core import Action, InferenceRequest, InferenceResult, Table
from leap.evaluation.evaluator import calculate_execution_accuracy_with_dataset_answers
from leap.generation.prompt_builder import PromptBuilder
from leap.utils.profiler import RequestProfiler

DEFAULT_COT_ACTION_TEMPERATURE = 0.3
DEFAULT_COT_ARGS_TEMPERATURE = 0.7


@dataclass
class ActionStepResult:
    """Result of a single action generation step.

    Attributes:
        action: The generated action, or None if generation failed
        metadata: Optional metadata about the generation process (e.g., sampling statistics)
    """

    action: Optional[Action]
    metadata: Optional[Any] = None


class BaseGenerationStrategy:
    """Shared scaffolding for strategies that operate on a single request."""

    def __init__(
        self,
        *,
        prompt_builder: PromptBuilder,
        sampling_layer,
        max_failures: int = 3,
        max_validity_failures: int = 3,
        max_steps: int = 10,
    ) -> None:
        self.max_failures = max_failures
        self.max_validity_failures = max_validity_failures
        self.max_steps = max_steps
        self.prompt_builder = prompt_builder
        self.sampling_layer = sampling_layer

    async def generate_action_step(
        self,
        worker,
        current_table: Table,
        action_history: List[str],
        request_id: str,
        state_machines,
        question: str,
        step: int,
    ) -> ActionStepResult:
        """
        Generate a single action for the current step.
        Must be implemented by subclasses to define generation strategy.

        Returns:
            ActionStepResult with action and optional metadata
        """
        raise NotImplementedError

    def get_generation_mode_string(self, worker) -> str:
        """Get the generation mode string for logging. Override if needed."""
        return worker._get_generation_mode_string()

    async def generate_instance(
        self,
        request: InferenceRequest,
        worker,
        state_machines,
        logging_callback: Optional[Callable] = None,
    ) -> InferenceResult:
        """
        Main generation loop - shared across all strategies.
        Subclasses only need to implement generate_action_step().
        """
        question = request.question
        original_table: Table = request.table
        current_table: Table = original_table
        ground_truth_answers = request.ground_truth_answers
        request_id = request.request_id

        # Initialize per-request profiler
        profiler = RequestProfiler(request_id)

        action_history: List[str] = []
        failures = 0
        validity_failures = 0
        step = 0
        sampling_metadata_list: List[Dict[str, Any]] = []

        generation_mode = self.get_generation_mode_string(worker)

        if logging_callback:
            logging_callback(request_id, 0, "initial", current_table, generation_mode=generation_mode)

        while failures < self.max_failures and validity_failures < self.max_validity_failures and step < self.max_steps:
            step_start = profiler.start_step()

            try:
                # Strategy-specific action generation
                result = await self.generate_action_step(
                    worker=worker,
                    current_table=current_table,
                    action_history=action_history,
                    request_id=request_id,
                    state_machines=state_machines,
                    question=question,
                    step=step,
                )

                # Extract action and metadata from result
                action = result.action
                if result.metadata:
                    sampling_metadata_list.append(result.metadata)

                if not action:
                    validity_failures += 1
                    action_display = "action_generation_failed"
                    print(f"Step {step}: Failed to generate valid action: {action_display}")
                    if logging_callback:
                        logging_callback(
                            request_id,
                            step + 1,
                            f"validity_failed:{action_display}",
                            current_table,
                            success=False,
                            failure_type="validity_failure",
                            generation_mode=generation_mode,
                        )
                    profiler.end_step(step_start, step, "validity_failed")
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
                    profiler.end_step(step_start, step, "end")
                    break

                with profiler.time_operation("table_transformation"):
                    new_table = action.apply_to_table(current_table)

                if not new_table:
                    validity_failures += 1
                    print(f"Step {step}: Failed to apply action: {action.to_string()}")
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
                    profiler.end_step(step_start, step, "apply_failed")
                    continue

                current_table = new_table
                action_history.append(action.to_string())
                failures = 0
                validity_failures = 0
                step += 1
                profiler.end_step(step_start, step - 1, action.name)
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
                profiler.end_step(step_start, step, "error")

                # If this is a critical error (like max_model_len exceeded), break the loop
                error_str = str(exc).lower()
                if "max_model_len" in error_str or "maximum model length" in error_str:
                    print(f"Critical error detected: {exc}. Stopping generation for this instance.")
                    break

        with profiler.time_operation("evaluation"):
            accuracy_metrics = calculate_execution_accuracy_with_dataset_answers(
                action_history, current_table, ground_truth_answers, original_table
            )

        # Print per-request timing summary
        total_time = profiler.get_total_time()
        print(f"Request {request_id} completed in {total_time:.2f}s - Operations: {profiler.timings}")

        # Package profiling data to send back to main process
        profiling_data = {
            "total_time": total_time,
            "operation_timings": profiler.timings,
            "num_steps": step,
        }

        return InferenceResult(
            action_history=action_history,
            final_table=current_table,
            execution_metrics=accuracy_metrics,
            request_id=request_id,
            question=question,
            ground_truth_answers=ground_truth_answers,
            profiling_data=profiling_data,
            sampling_metadata=sampling_metadata_list if sampling_metadata_list else None,
        )


class IterativeGenerationStrategy(BaseGenerationStrategy):
    """Iterative action generation with constraint-aware prompting."""

    def __init__(self, *, prompt_builder: PromptBuilder, sampling_layer, **kwargs) -> None:
        super().__init__(prompt_builder=prompt_builder, sampling_layer=sampling_layer, **kwargs)

    async def generate_action_step(
        self,
        worker,
        current_table: Table,
        action_history: List[str],
        request_id: str,
        state_machines,
        question: str,
        step: int,
    ) -> ActionStepResult:
        """Generate a single action using iterative strategy (single-call)."""
        step_id = f"{request_id}_step{step}"

        # Always use sampling layer (with n=1 when sampling is disabled)
        sampling_result = await self.sampling_layer.sample_action(
            worker=worker,
            table=current_table,
            action_history=action_history,
            request_id=step_id,
            state_machines=state_machines,
            prompt_builder=self.prompt_builder,
            question=question,
            step=step,
        )
        return ActionStepResult(action=sampling_result.action, metadata=sampling_result)


class ChainOfTableGenerationStrategy(BaseGenerationStrategy):
    """Two-step Chain-of-Table strategy (action selection + args)."""

    def __init__(
        self,
        *,
        prompt_builder: PromptBuilder,
        sampling_layer,
        action_temperature: float = DEFAULT_COT_ACTION_TEMPERATURE,
        args_temperature: float = DEFAULT_COT_ARGS_TEMPERATURE,
        **kwargs,
    ) -> None:
        super().__init__(prompt_builder=prompt_builder, sampling_layer=sampling_layer, **kwargs)
        self.action_temperature = action_temperature
        self.args_temperature = args_temperature

    def get_generation_mode_string(self, worker) -> str:
        """Override to return CoT mode string."""
        return "CoT"

    async def generate_action_step(
        self,
        worker,
        current_table: Table,
        action_history: List[str],
        request_id: str,
        state_machines,
        question: str,
        step: int,
    ) -> ActionStepResult:
        """Generate a single action using two-phase strategy (action selection + args)."""
        # Always use sampling layer (with n=1 when sampling is disabled)
        sampling_result = await self.sampling_layer.sample_action_two_phase(
            worker=worker,
            question=question,
            table=current_table,
            action_history=action_history,
            request_id=request_id,
            state_machines=state_machines,
            step=step,
            temperature_action=self.action_temperature,
            temperature_args=self.args_temperature,
            prompt_builder=self.prompt_builder,
        )
        return ActionStepResult(action=sampling_result.action, metadata=sampling_result)
