"""
Simple sampling layer for multi-sample generation with voting.

This module provides a minimal sampling mechanism that:
1. Generates N action candidates in parallel using vLLM's n parameter
2. Filters invalid candidates
3. Selects the most common action via voting

The design is intentionally simple but allows for extension through subclassing.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from vllm import SamplingParams

from leap.core import Action, Table
from leap.core.actions import REGISTRY
from leap.generation.prompt_builder import PromptBuilder
from leap.inference.function_constraints import (
    ActionGrammarBuilder,
    StructuredActionParser,
    StructuredSamplingParamsFactory,
    uses_xgrammar,
)
from leap.inference.json_constraints import (
    JsonActionCodec,
    JsonActionSchemaBuilder,
    JsonSchemaSamplingParamsFactory,
    available_json_actions,
    uses_json_operations,
    uses_json_schema,
)
from leap.inference.legacy.constraints import (
    create_action_only_constraint_processor,
    create_arguments_only_constraint_processor,
    create_constraint_logits_processor,
)
from leap.mcp.protocol import (
    MCP_ACTIONS,
    McpToolCallCodec,
    McpToolCallSchemaBuilder,
    uses_mcp_operations,
    uses_mcp_schema,
)


@dataclass
class SamplingConfig:
    """Configuration for sampling behavior."""

    enabled: bool = False
    n_samples: int = 1
    per_action_samples: dict[str, int] = None
    debug: bool = False  # Print detailed sampling information
    shuffle_invariant: bool = False  # Use shuffle-invariant sampling layer

    def __post_init__(self):
        if self.per_action_samples is None:
            self.per_action_samples = {}

    def get_n_samples(self, action_name: Optional[str] = None) -> int:
        """Get number of samples for a specific action type."""
        if action_name and action_name in self.per_action_samples:
            return self.per_action_samples[action_name]
        return self.n_samples


@dataclass
class SamplingResult:
    """Result of sampling with metadata."""

    action: Action
    n_requested: int
    n_generated: int
    n_valid: int
    winner_votes: int
    total_votes: int
    candidate_actions: list[str] | None = None
    valid_actions: list[str] | None = None
    fallback_reason: str | None = None


class SamplingLayer:
    """
    Minimal sampling layer that generates multiple candidates and votes.

    To extend:
    - Override transform_context() to modify table/history before prompt building
    - Override generate_candidates() to modify generation behavior
    - Override filter_candidates() to add custom validation
    - Override aggregate_candidates() to change voting logic
    """

    def __init__(self, config: SamplingConfig):
        self.config = config
        self.grammar_builder = ActionGrammarBuilder()
        self.json_schema_builder = JsonActionSchemaBuilder()
        self.mcp_schema_builder = McpToolCallSchemaBuilder()

    def transform_context(self, table: Table, action_history: List[str], sample_idx: int) -> tuple[Table, List[str]]:
        """
        Transform context before building prompt for each sample.

        Override this method to modify the table or action history before
        prompt generation. For example, to shuffle table rows for diversity.

        Args:
            table: Current table state
            action_history: Action history
            sample_idx: Index of current sample (0 to n-1)

        Returns:
            Modified (table, action_history) tuple

        Example:
            class ShuffleTableSamplingLayer(SamplingLayer):
                def transform_context(self, table, action_history, sample_idx):
                    shuffled = table.shuffle_rows(seed=sample_idx)
                    return shuffled, action_history
        """
        return table, action_history

    def _get_available_actions(self, action_history: List[str], worker=None) -> List[str]:
        """
        Get available actions based on action history.

        Filters out already-used actions (except 'end' which is always available)
        and excludes terminating actions on the first step.

        Args:
            action_history: List of action strings taken so far

        Returns:
            List of available action names
        """
        if worker is not None and (uses_json_operations(worker) or uses_mcp_operations(worker)):
            actions = available_json_actions(action_history, use_global_constraints=worker.use_global_constraints)
            return [name for name in actions if name in MCP_ACTIONS] if uses_mcp_operations(worker) else actions

        if worker is not None and uses_xgrammar(worker):
            from leap.inference.function_constraints import available_actions as grammar_available_actions

            return grammar_available_actions(action_history, use_global_constraints=worker.use_global_constraints)

        if worker is not None and getattr(worker, "use_global_constraints", False) is True:
            return REGISTRY.get_global_available_actions(action_history)

        available_actions = REGISTRY.get_enabled_names()

        if action_history:
            used_actions = REGISTRY._extract_action_names_from_history(action_history)
            # Filter out used actions but keep 'end' always available
            available_actions = [name for name in available_actions if name not in used_actions or name == "end"]

        # On first step, exclude terminating actions
        if not action_history or len(action_history) == 0:
            available_actions = [name for name in available_actions if not REGISTRY.get(name).is_terminating]

        return available_actions

    async def _generate_action_type(
        self,
        worker,
        action_history: List[str],
        table: Table,
        question: str,
        request_id: str,
        step: int,
        temperature_action: float,
        state_machines,
        prompt_builder,
    ) -> Optional[str]:
        """
        Generate the action type, optimizing for single-option scenarios.

        If only one action is available, returns it directly without calling LLM.
        Otherwise, calls LLM to generate the action type.

        Args:
            worker: Worker process with LLM
            action_history: List of actions taken so far
            table: Current table state
            question: Question being answered
            request_id: Request identifier
            step: Current step number
            temperature_action: Temperature for action generation
            state_machines: State machines for constraints
            prompt_builder: Prompt builder instance

        Returns:
            The action type name (e.g., "select_row", "end")
        """
        # Optimization: If only one action is available, skip LLM call (matches official implementation)
        available_actions = self._get_available_actions(action_history, worker)

        if len(available_actions) == 1:
            # Only one action available - skip LLM call and return it directly
            action_type = available_actions[0]
            if self.config.debug:
                print(f"[SAMPLING DEBUG] Only one action available: {action_type}, skipping LLM call")
        else:
            # Generate action type via LLM
            action_prompt = prompt_builder.build_cot_action_prompt(
                question=question, table=table, action_history=action_history, worker=worker
            )

            action_types = await self.generate_action_types(
                worker,
                action_prompt,
                1,
                request_id,
                step,
                temperature_action,
                state_machines,
                table,
                action_history,
            )

            # Empty/invalid action type generation is handled by the caller so
            # it can be reported as an error-driven fallback.
            action_type = action_types[0] if action_types else None

        if self.config.debug:
            print(f"[SAMPLING DEBUG] Selected action type: {action_type}")

        return action_type

    async def sample_action(
        self,
        worker,
        table: Table,
        action_history: List[str],
        request_id: str,
        state_machines,
        prompt_builder,
        question: str,
        step: int,
    ) -> SamplingResult:
        """Generate and vote on action candidates."""

        # Optimization: If only one action is available, skip LLM call (matches official implementation)
        available_actions = self._get_available_actions(action_history, worker)

        if len(available_actions) == 1:
            # Only one action available - skip LLM call and return it directly
            action_name = available_actions[0]
            action_def = REGISTRY.get(action_name)

            if self.config.debug:
                print(f"[SAMPLING DEBUG] Only one action available: {action_name}, skipping LLM call")

            # Return the action with empty arguments if it doesn't require args, otherwise fail
            if action_def and not action_def.requires_args:
                return SamplingResult(
                    action=Action(action_name, []),
                    n_requested=1,
                    n_generated=1,
                    n_valid=1,
                    winner_votes=1,
                    total_votes=1,
                )
            else:
                # This shouldn't happen in practice, but handle gracefully
                # Fall through to normal generation
                pass

        n_samples = self.config.n_samples

        if self.config.debug:
            print(f"\n[SAMPLING DEBUG] Generating {n_samples} candidates for {request_id}")

        # Generate N candidates with context transformation
        candidates = await self.generate_candidates(
            worker, n_samples, table, action_history, request_id, state_machines, prompt_builder, question, step
        )

        if self.config.debug:
            print(f"[SAMPLING DEBUG] Generated {len(candidates)} candidates:")
            for i, cand in enumerate(candidates):
                print(f"  [{i + 1}] {cand.to_string()}")

        # Filter invalid candidates
        valid_candidates = self.filter_candidates(candidates, table)

        if self.config.debug:
            print(f"[SAMPLING DEBUG] {len(valid_candidates)} valid after filtering")
            if len(valid_candidates) < len(candidates):
                invalid = set(c.to_string() for c in candidates) - set(c.to_string() for c in valid_candidates)
                for inv in invalid:
                    print(f"  [INVALID] {inv}")
                print(f"  [DEBUG] Table columns ({len(table.columns)}), rows ({len(table.rows)}):")
                for col in table.columns:
                    print(f"    - {col}")

        # Vote on best action
        action, winner_votes = self.aggregate_candidates(valid_candidates)

        if self.config.debug:
            if action:
                print(f"[SAMPLING DEBUG] Winner: {action.to_string()} ({winner_votes}/{len(valid_candidates)} votes)")
            else:
                print("[SAMPLING DEBUG] No valid action, using fallback")

        # Fallback if no valid actions
        if action is None:
            action = Action("end", [])
            winner_votes = 0
            fallback_reason = "no_valid_candidates"
        else:
            fallback_reason = None

        return SamplingResult(
            action=action,
            n_requested=n_samples,
            n_generated=len(candidates),
            n_valid=len(valid_candidates),
            winner_votes=winner_votes,
            total_votes=len(valid_candidates),
            candidate_actions=[self._serialize_action(c, worker) for c in candidates],
            valid_actions=[self._serialize_action(c, worker) for c in valid_candidates],
            fallback_reason=fallback_reason,
        )

    async def sample_action_two_phase(
        self,
        worker,
        question: str,
        table: Table,
        action_history: List[str],
        request_id: str,
        state_machines,
        step: int,
        temperature_action: float,
        temperature_args: float,
        prompt_builder,
    ) -> SamplingResult:
        """
        Two-phase sampling for Chain-of-Table:
        Phase 1: Generate single action type (no sampling)
        Phase 2: Generate N arguments based on per_action_samples config
        """

        # if self.config.debug:
        #     print(f"\n[SAMPLING DEBUG] Two-phase generation for {request_id} step {step}")

        # Phase 1: Generate action type (optimized to skip LLM when only one option)
        action_type = await self._generate_action_type(
            worker=worker,
            action_history=action_history,
            table=table,
            question=question,
            request_id=request_id,
            step=step,
            temperature_action=temperature_action,
            state_machines=state_machines,
            prompt_builder=prompt_builder,
        )

        if action_type is None:
            return SamplingResult(
                action=Action("end", []),
                n_requested=1,
                n_generated=0,
                n_valid=0,
                winner_votes=0,
                total_votes=0,
                candidate_actions=[],
                valid_actions=[],
                fallback_reason="action_type_generation_failed",
            )

        # Check if this action requires arguments
        # Some actions like 'end' don't need argument generation
        action_def = REGISTRY.get(action_type)

        if action_def and not action_def.requires_args:
            if self.config.debug:
                print(f"[SAMPLING DEBUG] Action '{action_type}' requires no arguments, skipping Phase 2")
            return SamplingResult(
                action=Action(action_type, []),
                n_requested=1,
                n_generated=1,
                n_valid=1,
                winner_votes=1,
                total_votes=1,
            )

        # Phase 2: Generate N argument sets based on action type
        # Use per_action_samples config to determine how many samples for this action
        n_samples = self.config.get_n_samples(action_type)

        if self.config.debug:
            print(f"[SAMPLING DEBUG] Phase 2 - Generating {n_samples} argument sets for '{action_type}'")

        args_candidates = await self.generate_arguments(
            worker,
            action_type,
            n_samples,
            table,
            action_history,
            request_id,
            step,
            temperature_args,
            state_machines,
            prompt_builder,
            question,
        )

        if self.config.debug:
            print(f"[SAMPLING DEBUG] Generated {len(args_candidates)} argument candidates:")
            for i, cand in enumerate(args_candidates):
                print(f"  [{i + 1}] {cand.to_string()}")

        # Filter and vote
        valid_candidates = self.filter_candidates(args_candidates, table)

        if self.config.debug:
            print(f"[SAMPLING DEBUG] {len(valid_candidates)} valid after filtering")
            if len(valid_candidates) < len(args_candidates):
                invalid = set(c.to_string() for c in args_candidates) - set(c.to_string() for c in valid_candidates)
                for inv in invalid:
                    print(f"  [INVALID] {inv}")
                print(f"  [DEBUG] Table columns ({len(table.columns)}), rows ({len(table.rows)}):")
                for col in table.columns:
                    print(f"    - {col}")

        action, winner_votes = self.aggregate_candidates(valid_candidates)

        if self.config.debug:
            if action:
                print(f"[SAMPLING DEBUG] Winner: {action.to_string()} ({winner_votes}/{len(valid_candidates)} votes)")
            else:
                print("[SAMPLING DEBUG] No valid action, using fallback")

        if action is None:
            action = Action("end", [])
            winner_votes = 0
            fallback_reason = "no_valid_candidates"
        else:
            fallback_reason = None

        return SamplingResult(
            action=action,
            n_requested=n_samples,
            n_generated=len(args_candidates),
            n_valid=len(valid_candidates),
            winner_votes=winner_votes,
            total_votes=len(valid_candidates),
            candidate_actions=[self._serialize_action(c, worker) for c in args_candidates],
            valid_actions=[self._serialize_action(c, worker) for c in valid_candidates],
            fallback_reason=fallback_reason,
        )

    async def generate_candidates(
        self,
        worker,
        n: int,
        table: Table,
        action_history: List[str],
        request_id: str,
        state_machines,
        prompt_builder,
        question: str,
        step: int,
    ) -> List[Action]:
        """
        Generate N action candidates with context transformation.

        For each sample:
        1. Transform context using transform_context()
        2. Build fresh prompt with transformed context
        3. Generate action

        All samples are generated concurrently for maximum throughput.

        Override to customize generation behavior.
        """

        async def _generate_single_sample(sample_idx: int) -> Optional[Action]:
            """Generate a single sample with its own context transformation."""
            try:
                # Transform context for this sample
                modified_table, modified_history = self.transform_context(table, action_history, sample_idx)

                # Build fresh prompt with modified context
                prompt = prompt_builder.build_iterative_prompt(
                    question=question,
                    table=modified_table,
                    action_history=modified_history,
                    worker=worker,
                    step=step,
                )

                # Generate single action with this prompt
                if uses_mcp_schema(worker):
                    spec = self.mcp_schema_builder.build_spec(
                        table=modified_table,
                        action_history=modified_history,
                        use_global_constraints=worker.use_global_constraints,
                        phase="single_step",
                    )
                    schema = self.mcp_schema_builder.build_single_step_schema(spec)
                    sampling_params = SamplingParams(
                        temperature=getattr(worker, "effective_temperature", lambda value: value)(0.7),
                        max_tokens=400,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        n=1,
                        **JsonSchemaSamplingParamsFactory.structured_outputs_kwargs(schema),
                    )
                elif uses_json_schema(worker):
                    spec = self.json_schema_builder.build_spec(
                        table=modified_table,
                        action_history=modified_history,
                        use_global_constraints=worker.use_global_constraints,
                        phase="single_step",
                    )
                    schema = self.json_schema_builder.build_single_step_schema(spec)
                    sampling_params = SamplingParams(
                        temperature=getattr(worker, "effective_temperature", lambda value: value)(0.7),
                        max_tokens=300,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        n=1,
                        **JsonSchemaSamplingParamsFactory.structured_outputs_kwargs(schema),
                    )
                elif uses_xgrammar(worker):
                    spec = self.grammar_builder.build_spec(
                        table=modified_table,
                        action_history=modified_history,
                        use_global_constraints=worker.use_global_constraints,
                        phase="single_step",
                    )
                    grammar = self.grammar_builder.build_single_step_grammar(spec)
                    sampling_params = SamplingParams(
                        temperature=getattr(worker, "effective_temperature", lambda value: value)(0.7),
                        max_tokens=300,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        n=1,
                        **StructuredSamplingParamsFactory.structured_outputs_kwargs(grammar),
                    )
                elif worker.use_constraints:
                    constraint_processor = create_constraint_logits_processor(
                        modified_table,
                        worker.tokenizer,
                        worker.tokenizer_config,
                        f"{request_id}_sample{sample_idx}",
                        state_machines,
                        modified_history,
                        worker.use_global_constraints,
                    )
                    sampling_params = SamplingParams(
                        temperature=getattr(worker, "effective_temperature", lambda value: value)(0.7),
                        max_tokens=900,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        logits_processors=[constraint_processor],
                        n=1,
                    )
                else:
                    sampling_params = SamplingParams(
                        temperature=getattr(worker, "effective_temperature", lambda value: value)(0.7),
                        max_tokens=900,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        stop=["\n", "Next", "Step"],
                        n=1,
                    )

                # DEBUG: Print prompt
                print(f"\n{'=' * 80}\n[PROMPT] Sample {sample_idx}\n{'=' * 80}\n{prompt}\n{'=' * 80}\n")

                result_generator = worker.engine.generate(prompt, sampling_params, f"{request_id}_sample{sample_idx}")
                final_result = None
                async for result in result_generator:
                    final_result = result

                if final_result and final_result.outputs:
                    response_text = final_result.outputs[0].text.strip()
                    # DEBUG: Print response
                    print(f"\n[RESPONSE] Sample {sample_idx}\n{'=' * 80}\n{response_text}\n{'=' * 80}\n")
                    if uses_mcp_operations(worker):
                        action = McpToolCallCodec.parse_action(response_text, modified_table)
                    elif uses_json_operations(worker):
                        action = JsonActionCodec.parse_single_step(response_text, modified_table)
                    elif uses_xgrammar(worker):
                        action = StructuredActionParser.parse_single_step(response_text, modified_table)
                    else:
                        action = Action.parse(response_text)
                    return action
                return None
            except Exception as e:
                print(f"[SAMPLING ERROR] Sample {sample_idx} failed: {type(e).__name__}: {e}")
                return None

        # Create all sample generation tasks concurrently
        tasks = [_generate_single_sample(i) for i in range(n)]

        # Execute all samples concurrently and wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None values and exceptions, keeping only valid actions
        candidates = [r for r in results if r is not None and not isinstance(r, Exception)]

        return candidates

    async def generate_action_types(
        self,
        worker,
        prompt: str,
        n: int,
        request_id: str,
        step: int,
        temperature: float,
        state_machines,
        table: Table,
        action_history: List[str],
    ) -> List[str]:
        """Generate N action types for two-phase sampling."""
        step_id = f"{request_id}_action_step{step}"

        if uses_mcp_schema(worker):
            spec = self.mcp_schema_builder.build_spec(
                table=table,
                action_history=action_history,
                use_global_constraints=worker.use_global_constraints,
                phase="action",
            )
            schema = self.mcp_schema_builder.build_action_schema(spec)
            sampling_params = SamplingParams(
                temperature=getattr(worker, "effective_temperature", lambda value: value)(temperature),
                max_tokens=200,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                n=n,
                **JsonSchemaSamplingParamsFactory.structured_outputs_kwargs(schema),
            )
        elif uses_json_schema(worker):
            spec = self.json_schema_builder.build_spec(
                table=table,
                action_history=action_history,
                use_global_constraints=worker.use_global_constraints,
                phase="action",
            )
            schema = self.json_schema_builder.build_action_schema(spec)
            sampling_params = SamplingParams(
                temperature=getattr(worker, "effective_temperature", lambda value: value)(temperature),
                max_tokens=60,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                n=n,
                **JsonSchemaSamplingParamsFactory.structured_outputs_kwargs(schema),
            )
        elif uses_xgrammar(worker):
            spec = self.grammar_builder.build_spec(
                table=table,
                action_history=action_history,
                use_global_constraints=worker.use_global_constraints,
                phase="action",
            )
            grammar = self.grammar_builder.build_action_grammar(spec)
            sampling_params = SamplingParams(
                temperature=getattr(worker, "effective_temperature", lambda value: value)(temperature),
                max_tokens=60,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                n=n,
                **StructuredSamplingParamsFactory.structured_outputs_kwargs(grammar),
            )
        elif worker.use_constraints:
            constraint_processor = create_action_only_constraint_processor(
                worker.tokenizer,
                step_id,
                state_machines,
                action_history=action_history,
                use_global_constraints=worker.use_global_constraints,
            )
            sampling_params = SamplingParams(
                temperature=getattr(worker, "effective_temperature", lambda value: value)(temperature),
                max_tokens=20,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                logits_processors=[constraint_processor],
                n=n,
            )
        else:
            sampling_params = SamplingParams(
                temperature=getattr(worker, "effective_temperature", lambda value: value)(temperature),
                max_tokens=100,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                n=n,
            )

        # DEBUG: Print Phase 1 prompt
        # print(f"\n{'=' * 80}\n[PHASE 1 PROMPT - ACTION SELECTION | {request_id} step={step}]\n{'=' * 80}\n{prompt}\n{'=' * 80}\n")

        result_generator = worker.engine.generate(prompt, sampling_params, step_id)
        final_result = None
        async for result in result_generator:
            final_result = result

        action_types = []
        if final_result:
            for output in final_result.outputs:
                response_text = output.text.strip()
                # DEBUG: Print Phase 1 response
                # print(f"\n[PHASE 1 RESPONSE | {request_id} step={step}]\n{'=' * 80}\n{response_text}\n{'=' * 80}\n")
                if uses_mcp_operations(worker):
                    allowed = self._get_available_actions(action_history, worker)
                    action_name = McpToolCallCodec.parse_action_name(response_text, allowed_actions=allowed)
                elif uses_json_operations(worker):
                    allowed = self._get_available_actions(action_history, worker)
                    action_name = JsonActionCodec.parse_action_name(response_text, allowed_actions=allowed)
                elif uses_xgrammar(worker):
                    action_name = StructuredActionParser.parse_action_name(response_text)
                else:
                    action_name = Action.parse_name_only(response_text)
                if action_name:
                    action_types.append(action_name)

        return action_types

    async def generate_arguments(
        self,
        worker,
        action_name: str,
        n: int,
        table: Table,
        action_history: List[str],
        request_id: str,
        step: int,
        temperature: float,
        state_machines,
        prompt_builder: PromptBuilder,
        question: str,
    ) -> List[Action]:
        """
        Generate N argument sets for a given action type with context transformation.

        For each sample:
        1. Transform context using transform_context()
        2. Build fresh args prompt with transformed context
        3. Generate arguments

        All samples are generated concurrently for maximum throughput.
        """

        async def _generate_single_argument_set(sample_idx: int) -> Optional[Action]:
            """Generate a single argument set with its own context transformation."""
            try:
                # Transform context for this sample
                modified_table, modified_history = self.transform_context(table, action_history, sample_idx)
                # Build fresh args prompt with modified context
                args_prompt = prompt_builder.build_cot_arguments_prompt(
                    question=question,
                    table=modified_table,
                    action_name=action_name,
                    action_history=modified_history,
                    worker=worker,
                )

                step_id = f"{request_id}_args_step{step}_sample{sample_idx}"

                # Generate single argument set
                if uses_mcp_schema(worker):
                    spec = self.mcp_schema_builder.build_spec(
                        table=modified_table,
                        action_history=modified_history,
                        use_global_constraints=worker.use_global_constraints,
                        phase="arguments",
                        selected_action=action_name,
                    )
                    schema = self.mcp_schema_builder.build_arguments_schema(spec)
                    sampling_params = SamplingParams(
                        temperature=getattr(worker, "effective_temperature", lambda value: value)(temperature),
                        max_tokens=400,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        n=1,
                        **JsonSchemaSamplingParamsFactory.structured_outputs_kwargs(schema),
                    )
                elif uses_json_schema(worker):
                    spec = self.json_schema_builder.build_spec(
                        table=modified_table,
                        action_history=modified_history,
                        use_global_constraints=worker.use_global_constraints,
                        phase="arguments",
                        selected_action=action_name,
                    )
                    schema = self.json_schema_builder.build_arguments_schema(spec)
                    sampling_params = SamplingParams(
                        temperature=getattr(worker, "effective_temperature", lambda value: value)(temperature),
                        max_tokens=300,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        n=1,
                        **JsonSchemaSamplingParamsFactory.structured_outputs_kwargs(schema),
                    )
                elif uses_xgrammar(worker):
                    spec = self.grammar_builder.build_spec(
                        table=modified_table,
                        action_history=modified_history,
                        use_global_constraints=worker.use_global_constraints,
                        phase="arguments",
                        selected_action=action_name,
                    )
                    grammar = self.grammar_builder.build_arguments_grammar(spec)
                    sampling_params = SamplingParams(
                        temperature=getattr(worker, "effective_temperature", lambda value: value)(temperature),
                        max_tokens=300,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        n=1,
                        **StructuredSamplingParamsFactory.structured_outputs_kwargs(grammar),
                    )
                elif worker.use_constraints and action_name != "add_column":
                    # Use arguments-only constraint processor for two-phase generation
                    # This prevents the model from generating the action name again
                    constraint_processor = create_arguments_only_constraint_processor(
                        modified_table,
                        worker.tokenizer,
                        worker.tokenizer_config,
                        action_name,
                        step_id,
                        state_machines,
                    )
                    sampling_params = SamplingParams(
                        temperature=getattr(worker, "effective_temperature", lambda value: value)(temperature),
                        max_tokens=900,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        logits_processors=[constraint_processor],
                        n=1,
                    )
                else:
                    sampling_params = SamplingParams(
                        temperature=getattr(worker, "effective_temperature", lambda value: value)(temperature),
                        max_tokens=900,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        stop=["\n", "Next", "Step"],
                        n=1,
                    )

                # DEBUG: Print Phase 2 prompt
                # print(
                #     f"\n{'=' * 80}\n[PHASE 2 PROMPT - ARGUMENTS | {request_id} step={step} sample={sample_idx}]\n{'=' * 80}\n{args_prompt}\n{'=' * 80}\n"  # noqa: E501
                # )

                result_generator = worker.engine.generate(args_prompt, sampling_params, step_id)
                final_result = None
                async for result in result_generator:
                    final_result = result

                if final_result and final_result.outputs:
                    args_text = final_result.outputs[0].text.strip()

                    print(f"\n[PHASE 2 RESPONSE | {request_id} step={step} sample={sample_idx}]\n{'=' * 80}\n{args_text}\n{'=' * 80}\n")  # noqa: E501

                    if uses_mcp_operations(worker):
                        return McpToolCallCodec.parse_action(args_text, modified_table, expected_action=action_name)

                    if uses_json_operations(worker):
                        validation_table = None if action_name == "add_column" else modified_table
                        action = JsonActionCodec.parse_arguments(args_text, action_name, validation_table)
                        if action and action.name == "add_column":
                            action = await self._extend_add_column_per_row(
                                action=action,
                                table=modified_table,
                                prompt_builder=prompt_builder,
                                worker=worker,
                                step_id=step_id,
                                request_id=request_id,
                                step=step,
                                explanation=args_text,
                            )
                        return action

                    if uses_xgrammar(worker):
                        return StructuredActionParser.parse_arguments(args_text, action_name, modified_table)

                    if not worker.use_constraints or action_name == "add_column":
                        args_text = self._clean_argument_text(args_text, action_name)

                    full_action_str = f"{action_name}({args_text})"
                    action = Action.parse(full_action_str)

                    if action and action.name == "add_column":
                        action = await self._extend_add_column_per_row(
                            action=action,
                            table=modified_table,
                            prompt_builder=prompt_builder,
                            worker=worker,
                            step_id=step_id,
                            request_id=request_id,
                            step=step,
                            explanation=args_text,
                        )

                    return action
                return None
            except Exception as e:
                print(f"[SAMPLING ERROR] Sample {sample_idx} failed: {type(e).__name__}: {e}")
                return None

        # Create all argument generation tasks concurrently
        tasks = [_generate_single_argument_set(i) for i in range(n)]

        # Execute all samples concurrently and wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None values and exceptions, keeping only valid actions
        candidates = [r for r in results if r is not None and not isinstance(r, Exception)]

        return candidates

    async def _extend_add_column_per_row(
        self,
        *,
        action: Action,
        table: Table,
        prompt_builder,
        worker,
        step_id: str,
        request_id: str,
        step: int,
        explanation: str,
    ) -> Action:
        """Extend an add_column action by generating any missing row values one at a time.

        If the action already has values for all rows, returns it unchanged.
        Otherwise uses the existing values as seed examples and generates the rest row-by-row.
        """
        col_name, seed_values = action.arguments
        if len(seed_values) >= len(table.rows):
            return action

        all_values = list(seed_values)

        for row_idx in range(len(seed_values), len(table.rows)):
            per_row_prompt = prompt_builder.build_add_column_per_row_prompt(
                table=table,
                column_name=col_name,
                target_row=table.rows[row_idx],
                target_row_idx=row_idx,
                seed_values=list(seed_values),
                explanation=explanation,
            )
            row_id = f"{step_id}_row{row_idx}"
            row_params = SamplingParams(
                temperature=getattr(worker, "effective_temperature", lambda value: value)(0.0),
                max_tokens=50,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                stop=["\n"],
                n=1,
            )
            # print(
            #     f"\n{'=' * 80}\n[PHASE 2 PROMPT - ADD_COLUMN ROW {row_idx} | {request_id} step={step}]\n{'=' * 80}\n{per_row_prompt}\n{'=' * 80}\n"  # noqa: E501
            # )
            row_gen = worker.engine.generate(per_row_prompt, row_params, row_id)
            row_result = None
            async for r in row_gen:
                row_result = r
            val = row_result.outputs[0].text.strip() if row_result and row_result.outputs else ""
            # Take only the first non-empty line in case the model generated extra text
            val = next((line.strip() for line in val.splitlines() if line.strip()), val)
            # If the model produced a verbose sentence instead of a bare value, extract just the value.
            # Handles patterns like "The value ... is <X>" and "The value ... is an empty string".
            import re as _re

            _verbose = _re.search(
                r"\bthe value\b.*?\bis\s+(an empty string|['\"]?(.*?)['\"]?)[,.]?\s*$",
                val,
                _re.IGNORECASE,
            )
            if _verbose:
                candidate = _verbose.group(1)
                if candidate.lower() == "an empty string":
                    val = ""
                else:
                    val = _verbose.group(2).strip() if _verbose.group(2) else candidate.strip()
            # print(f"\n[PHASE 2 RESPONSE - ADD_COLUMN ROW {row_idx} | {request_id}]\n{'=' * 80}\n{val}\n{'=' * 80}\n")
            all_values.append(val)

        return Action("add_column", [col_name, all_values])

    @staticmethod
    def _serialize_action(action: Action, worker) -> str:
        if uses_mcp_operations(worker):
            return McpToolCallCodec.dumps(action)
        if uses_json_operations(worker):
            return JsonActionCodec.dumps(action)
        return action.to_string()

    def filter_candidates(self, candidates: List[Action], table: Table, action_history: List[str] = None) -> List[Action]:
        """Filter candidates to only valid actions. Override to customize filtering."""
        used_actions = REGISTRY._extract_action_names_from_history(action_history) if action_history else set()
        return [
            action
            for action in candidates
            if action.is_valid_for_table(table) and (action.name not in used_actions or action.name == "end")
        ]

    def aggregate_candidates(self, candidates: List[Action]) -> tuple[Optional[Action], int]:
        """Vote on candidates. Override to customize aggregation logic."""
        if not candidates:
            return None, 0

        # Count votes (exact string match)
        vote_counts = {}
        action_map = {}
        for action in candidates:
            action_str = action.to_string()
            vote_counts[action_str] = vote_counts.get(action_str, 0) + 1
            action_map[action_str] = action

        # Return winner
        winner_str = max(vote_counts, key=vote_counts.get)
        return action_map[winner_str], vote_counts[winner_str]

    @staticmethod
    def _clean_argument_text(args_text: str, action_name: str) -> str:
        """
        Clean argument text by removing action name prefix if present.

        In two-phase generation, the model sometimes generates:
          "select_column ([ \"Team\" ]"
        when we only want:
          "[ \"Team\" ]"

        This happens because constraints expect full format but we're only
        generating arguments in phase 2.

        Args:
            args_text: Raw generated text from model
            action_name: The action name (e.g., "select_row", "select_column")

        Returns:
            Cleaned argument text
        """
        import re

        def strip_trailing_prompt_punctuation(text: str) -> str:
            cleaned_text = text.strip().rstrip(".")
            if len(cleaned_text) >= 2 and cleaned_text[0] in {"'", '"'} and cleaned_text[-1] == cleaned_text[0]:
                return cleaned_text
            return cleaned_text.rstrip("'\"")

        # Remove action name prefix (with optional space and opening paren)
        # Patterns to clean:
        # - "select_column ([ "Team" ]" -> "[ "Team" ]"
        # - "select_column([ "Team" ]" -> "[ "Team" ]"
        # - "select_([ "0" ]" -> "[ "0" ]"  (partial action name)
        # First, try exact action name match

        # Match "Therefore the answer is: f_action_name(...)" pattern
        marker = "therefore the answer is:"
        marker_idx = args_text.lower().find(marker)
        if marker_idx >= 0:
            after_marker = args_text[marker_idx + len(marker) :].strip()
            # Find the last ')' and extract everything up to and including it
            last_paren = after_marker.rfind(")")
            if last_paren >= 0:
                after_marker = after_marker[: last_paren + 1]
            args_text = after_marker

        # Normalize backslashes only for regex matching while retaining a map
        # back to the original offsets used for the returned value.
        normalized_chars = []
        normalized_to_original = []
        for original_idx, char in enumerate(args_text):
            if char == "\\":
                continue
            normalized_chars.append(char)
            normalized_to_original.append(original_idx)
        normalized_args_text = "".join(normalized_chars)

        # Try to extract just the args from f_action_name(args) or action_name(args)
        # Try f_-prefixed version first (LLM generates f_sort_by(...) etc.)
        for name_pattern in [rf"f_{re.escape(action_name)}", rf"\b{re.escape(action_name)}"]:
            # Find the opening paren after the action name
            open_match = re.search(rf"{name_pattern}\s*\(", normalized_args_text, re.IGNORECASE)
            if open_match:
                open_pos = open_match.end()  # position after '('
                # Find matching closing paren (accounting for nested parens)
                depth = 1
                pos = open_pos
                while pos < len(normalized_args_text) and depth > 0:
                    if normalized_args_text[pos] == "(":
                        depth += 1
                    elif normalized_args_text[pos] == ")":
                        depth -= 1
                    pos += 1
                if depth == 0:
                    close_pos = pos - 1  # position of closing ')'
                    original_open_pos = normalized_to_original[open_pos - 1] + 1
                    original_close_pos = normalized_to_original[close_pos]
                    return strip_trailing_prompt_punctuation(args_text[original_open_pos:original_close_pos])

        # Fallback: strip only the action name prefix (with or without f_).
        # Preserve parentheses in the remaining arguments because they may be
        # part of a valid column name, e.g. "Population (2005)".
        pattern = rf"^\s*(?:f_)?{re.escape(action_name)}\s*\(?\s*"
        cleaned = re.sub(pattern, "", args_text, count=1)
        return strip_trailing_prompt_punctuation(cleaned)
