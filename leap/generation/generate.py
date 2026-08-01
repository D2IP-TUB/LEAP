from vllm import SamplingParams

from leap.core import Action, Table
from leap.inference.action_grammar import (
    ActionGrammarBuilder,
    StructuredActionParser,
    StructuredSamplingParamsFactory,
    uses_xgrammar,
)
from leap.inference.legacy.constraints import (
    create_action_only_constraint_processor,
    create_constraint_logits_processor,
)


async def generate_single_action(worker, prompt, table: Table, request_id, state_machines, action_history=None):
    """
    Generate single action with or without constraints

    Args:
        worker: The worker process
        prompt: The prompt text
        table: The Table object
        request_id: Unique request identifier
        state_machines: Dictionary of state machines
        action_history: List of previously executed actions (for global constraints)
    """
    try:
        if worker.use_constraints:
            # Get global constraints setting from worker's generation config
            use_global_constraints = worker.use_global_constraints

            if uses_xgrammar(worker):
                builder = ActionGrammarBuilder()
                spec = builder.build_spec(
                    table=table,
                    action_history=action_history,
                    use_global_constraints=use_global_constraints,
                    phase="single_step",
                )
                grammar = builder.build_single_step_grammar(spec)
                sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=300,
                    stop_token_ids=[worker.tokenizer.eos_token_id],
                    **StructuredSamplingParamsFactory.structured_outputs_kwargs(grammar),
                )
            else:
                # Pass action_history to the legacy constraint processor
                constraint_processor = create_constraint_logits_processor(
                    table,
                    worker.tokenizer,
                    worker.tokenizer_config,
                    request_id,
                    state_machines,
                    action_history,
                    use_global_constraints,
                )

                sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=900,  # increased for more row params
                    stop_token_ids=[worker.tokenizer.eos_token_id],
                    logits_processors=[constraint_processor],
                )
        else:
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=100,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                stop=["\n", "Next", "Step"],
            )

        return await worker.generate_text(prompt, request_id, sampling_params)

    except Exception as e:
        print(f"Generation error in worker {worker.worker_id}: {e}")
        return ""


async def generate_action_selection(
    worker,
    question,
    table: Table,
    action_history,
    request_id,
    state_machines,
    step,
    temperature,
    prompt_builder,
):
    """CoT Step 1: Dynamic Plan - Select which action to perform"""
    step_id = f"{request_id}_action_step{step}"
    prompt = prompt_builder.build_cot_action_prompt(
        question=question,
        table=table,
        action_history=action_history,
        worker=worker,
    )

    try:
        if uses_xgrammar(worker):
            builder = ActionGrammarBuilder()
            spec = builder.build_spec(
                table=table,
                action_history=action_history,
                use_global_constraints=worker.use_global_constraints,
                phase="action",
            )
            grammar = builder.build_action_grammar(spec)
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=60,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                **StructuredSamplingParamsFactory.structured_outputs_kwargs(grammar),
            )
        elif worker.use_constraints:
            constraint_processor = create_action_only_constraint_processor(
                worker.tokenizer,
                step_id,
                state_machines,
                worker.use_global_constraints,
            )

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=20,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                logits_processors=[constraint_processor],
            )
        else:
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=100,
                stop_token_ids=[worker.tokenizer.eos_token_id],
            )

        action_text = await worker.generate_text(prompt, step_id, sampling_params)
        return StructuredActionParser.parse_action_name(action_text) if uses_xgrammar(worker) else Action.parse_name_only(action_text)

    except Exception as e:
        print(f"Action selection error in worker {worker.worker_id}: {e}")
        return None


async def generate_action_arguments(
    worker,
    question,
    table: Table,
    action_name,
    action_history,
    request_id,
    state_machines,
    step,
    temperature,
    prompt_builder,
):
    """CoT Step 2: Generate Args - Generate arguments for the selected action"""
    step_id = f"{request_id}_args_step{step}"

    if action_name == "end":
        return []

    prompt = prompt_builder.build_cot_arguments_prompt(
        question=question,
        table=table,
        action_name=action_name,
        action_history=action_history,
        worker=worker,
    )

    try:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=900,
            stop_token_ids=[worker.tokenizer.eos_token_id],
            stop=["\n", "Next", "Step"],
        )
        if uses_xgrammar(worker):
            builder = ActionGrammarBuilder()
            spec = builder.build_spec(
                table=table,
                action_history=action_history,
                use_global_constraints=worker.use_global_constraints,
                phase="arguments",
                selected_action=action_name,
            )
            grammar = builder.build_arguments_grammar(spec)
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=300,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                **StructuredSamplingParamsFactory.structured_outputs_kwargs(grammar),
            )

        args_text = await worker.generate_text(prompt, step_id, sampling_params)
        if uses_xgrammar(worker):
            action = StructuredActionParser.parse_arguments(args_text, action_name, table)
            return list(action.arguments) if action else None
        action = Action.extract_from_text(args_text, action_name, table)
        return list(action.arguments) if action else None

    except Exception as e:
        print(f"Argument generation error in worker {worker.worker_id}: {e}")
        return None
