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
from leap.inference.constraints import (
    create_action_only_constraint_processor,
    create_arguments_only_constraint_processor,
    create_constraint_logits_processor,
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

    def _get_available_actions(self, action_history: List[str]) -> List[str]:
        """
        Get available actions based on action history.

        Filters out already-used actions (except 'end' which is always available)
        and excludes terminating actions on the first step.

        Args:
            action_history: List of action strings taken so far

        Returns:
            List of available action names
        """
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
    ) -> str:
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
        available_actions = self._get_available_actions(action_history)

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

            action_types = await self.generate_action_types(worker, action_prompt, 1, request_id, step, temperature_action, state_machines)

            # Take the generated action type (fallback to "end" if generation failed)
            action_type = action_types[0] if action_types else "end"

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
        available_actions = self._get_available_actions(action_history)

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

        return SamplingResult(
            action=action,
            n_requested=n_samples,
            n_generated=len(candidates),
            n_valid=len(valid_candidates),
            winner_votes=winner_votes,
            total_votes=len(valid_candidates),
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

        if self.config.debug:
            print(f"\n[SAMPLING DEBUG] Two-phase generation for {request_id} step {step}")

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

        action, winner_votes = self.aggregate_candidates(valid_candidates)

        if self.config.debug:
            if action:
                print(f"[SAMPLING DEBUG] Winner: {action.to_string()} ({winner_votes}/{len(valid_candidates)} votes)")
            else:
                print("[SAMPLING DEBUG] No valid action, using fallback")

        if action is None:
            action = Action("end", [])
            winner_votes = 0

        return SamplingResult(
            action=action,
            n_requested=n_samples,
            n_generated=len(args_candidates),
            n_valid=len(valid_candidates),
            winner_votes=winner_votes,
            total_votes=len(valid_candidates),
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
                if worker.use_constraints:
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
                        temperature=0.7,
                        max_tokens=900,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        logits_processors=[constraint_processor],
                        n=1,
                    )
                else:
                    sampling_params = SamplingParams(
                        temperature=0.7,
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
                    action = Action.parse(response_text)
                    return action
                return None
            except Exception:
                return None

        # Create all sample generation tasks concurrently
        tasks = [_generate_single_sample(i) for i in range(n)]

        # Execute all samples concurrently and wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None values and exceptions, keeping only valid actions
        candidates = [r for r in results if r is not None and not isinstance(r, Exception)]

        return candidates

    async def generate_action_types(
        self, worker, prompt: str, n: int, request_id: str, step: int, temperature: float, state_machines
    ) -> List[str]:
        """Generate N action types for two-phase sampling."""
        step_id = f"{request_id}_action_step{step}"

        if worker.use_constraints:
            constraint_processor = create_action_only_constraint_processor(worker.tokenizer, step_id, state_machines)
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=20,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                logits_processors=[constraint_processor],
                n=n,
            )
        else:
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=30,
                stop_token_ids=[worker.tokenizer.eos_token_id],
                stop=["\n", "Arguments", "Next"],
                n=n,
            )

        # DEBUG: Print Phase 1 prompt
        print(f"\n{'=' * 80}\n[PHASE 1 PROMPT - ACTION SELECTION]\n{'=' * 80}\n{prompt}\n{'=' * 80}\n")

        result_generator = worker.engine.generate(prompt, sampling_params, step_id)
        final_result = None
        async for result in result_generator:
            final_result = result

        action_types = []
        if final_result:
            for output in final_result.outputs:
                response_text = output.text.strip()
                # DEBUG: Print Phase 1 response
                print(f"\n[PHASE 1 RESPONSE]\n{'=' * 80}\n{response_text}\n{'=' * 80}\n")
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
                    worker=worker,
                )

                step_id = f"{request_id}_args_step{step}_sample{sample_idx}"

                # Generate single argument set
                if worker.use_constraints:
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
                        temperature=temperature,
                        max_tokens=900,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        logits_processors=[constraint_processor],
                        n=1,
                    )
                else:
                    sampling_params = SamplingParams(
                        temperature=temperature,
                        max_tokens=900,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        stop=["\n", "Next", "Step"],
                        n=1,
                    )

                # DEBUG: Print Phase 2 prompt
                print(f"\n{'=' * 80}\n[PHASE 2 PROMPT - ARGUMENTS] Sample {sample_idx}\n{'=' * 80}\n{args_prompt}\n{'=' * 80}\n")

                result_generator = worker.engine.generate(args_prompt, sampling_params, step_id)
                final_result = None
                async for result in result_generator:
                    final_result = result

                if final_result and final_result.outputs:
                    args_text = final_result.outputs[0].text.strip()
                    # DEBUG: Print Phase 2 response
                    print(f"\n[PHASE 2 RESPONSE] Sample {sample_idx}\n{'=' * 80}\n{args_text}\n{'=' * 80}\n")

                    # Clean up: Remove action name prefix if model incorrectly generated it
                    # e.g., "select_column ([ "Team" ]" -> "[ "Team" ]"
                    # This happens when constraints force full format but we only want args
                    args_text = self._clean_argument_text(args_text, action_name)
                    full_action_str = f"{action_name}({args_text})"
                    action = Action.parse(full_action_str)
                    return action
                return None
            except Exception:
                return None

        # Create all argument generation tasks concurrently
        tasks = [_generate_single_argument_set(i) for i in range(n)]

        # Execute all samples concurrently and wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None values and exceptions, keeping only valid actions
        candidates = [r for r in results if r is not None and not isinstance(r, Exception)]

        return candidates

    def filter_candidates(self, candidates: List[Action], table: Table) -> List[Action]:
        """Filter candidates to only valid actions. Override to customize filtering."""
        valid = []
        for action in candidates:
            try:
                result = action.apply_to_table(table)
                if result is not None:
                    valid.append(action)
            except Exception:
                pass
        return valid

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

        # Remove action name prefix (with optional space and opening paren)
        # Patterns to clean:
        # - "select_column ([ "Team" ]" -> "[ "Team" ]"
        # - "select_column([ "Team" ]" -> "[ "Team" ]"
        # - "select_([ "0" ]" -> "[ "0" ]"  (partial action name)
        # First, try exact action name match

        normalized_args_text = args_text.replace("\\", "")

        if action_name not in normalized_args_text and "[" in normalized_args_text and "]" in normalized_args_text:
            args_text_cleaned = normalized_args_text.split("[")[1].split("]")[0].strip()
            return "[" + args_text_cleaned + "]"

        pattern = rf"^.*?{re.escape(action_name)}\s*\(?\s*"
        cleaned = re.sub(pattern, "", normalized_args_text, count=1)
        pattern = r"\)[^)]*$"
        cleaned = re.sub(pattern, "", cleaned)
       

        return cleaned.strip()
