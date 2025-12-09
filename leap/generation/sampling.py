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
from leap.inference.constraints import (
    create_action_only_constraint_processor,
    create_constraint_logits_processor,
)


@dataclass
class SamplingConfig:
    """Configuration for sampling behavior."""

    enabled: bool = False
    n_samples: int = 1
    per_action_samples: dict[str, int] = None
    debug: bool = False  # Print detailed sampling information

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

        # Phase 1: Generate single action type (no voting needed)
        action_prompt = prompt_builder.build_cot_action_prompt(question=question, table=table, action_history=action_history, worker=worker)

        action_types = await self.generate_action_types(worker, action_prompt, 1, request_id, step, temperature_action, state_machines)

        # Take the single generated action type
        winning_action = action_types[0] if action_types else "end"

        if self.config.debug:
            print(f"[SAMPLING DEBUG] Phase 1 - Selected action type: {winning_action}")

        if winning_action == "end":
            if self.config.debug:
                print("[SAMPLING DEBUG] Action is 'end', terminating")
            return SamplingResult(
                action=Action("end", []),
                n_requested=1,
                n_generated=1,
                n_valid=1,
                winner_votes=1,
                total_votes=1,
            )

        # Phase 2: Generate N argument sets based on action type
        # Use per_action_samples config to determine how many samples for this action
        n_samples = self.config.get_n_samples(winning_action)

        if self.config.debug:
            print(f"[SAMPLING DEBUG] Phase 2 - Generating {n_samples} argument sets for '{winning_action}'")

        args_candidates = await self.generate_arguments(
            worker,
            winning_action,
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
                        max_tokens=100,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        stop=["\n", "Next", "Step"],
                        n=1,
                    )

                result_generator = worker.engine.generate(prompt, sampling_params, f"{request_id}_sample{sample_idx}")
                final_result = None
                async for result in result_generator:
                    final_result = result

                if final_result and final_result.outputs:
                    action = Action.parse(final_result.outputs[0].text.strip())
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

        result_generator = worker.engine.generate(prompt, sampling_params, step_id)
        final_result = None
        async for result in result_generator:
            final_result = result

        action_types = []
        if final_result:
            for output in final_result.outputs:
                action_name = Action.parse_name_only(output.text.strip())
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
        prompt_builder,
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
                    action_history=modified_history,
                    action_name=action_name,
                    worker=worker,
                )

                step_id = f"{request_id}_args_step{step}_sample{sample_idx}"

                # Generate single argument set
                if worker.use_constraints:
                    constraint_processor = create_constraint_logits_processor(
                        modified_table,
                        worker.tokenizer,
                        worker.tokenizer_config,
                        step_id,
                        state_machines,
                        modified_history,
                        worker.use_global_constraints,
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
                        max_tokens=100,
                        stop_token_ids=[worker.tokenizer.eos_token_id],
                        stop=["\n", "Next", "Step"],
                        n=1,
                    )

                result_generator = worker.engine.generate(args_prompt, sampling_params, step_id)
                final_result = None
                async for result in result_generator:
                    final_result = result

                if final_result and final_result.outputs:
                    full_action_str = f"{action_name}({final_result.outputs[0].text.strip()})"
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
                _ = action.apply_to_table(table)
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
