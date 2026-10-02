"""
Shuffle-invariant sampling layer for robust table question answering.

This module implements a sampling strategy that shuffles table rows across different
samples and then normalizes the resulting actions back to the original row ordering
before voting. This approach:

1. Generates diverse predictions by presenting the same table in different row orders
2. Tests model robustness to irrelevant input perturbations
3. Maps predictions back to original indices for true shuffle-invariant voting

The key insight: if the model truly understands the content (not just position),
it should select the same rows regardless of their order in the prompt.

This implementation follows the original Monte Carlo sampling script approach with
permutation tracking and index mapping.
"""

import random
from typing import List, Optional, Tuple

from leap.core import Action, Table
from leap.generation.sampling import SamplingConfig, SamplingLayer


class ShuffleInvariantSamplingLayer(SamplingLayer):
    """
    Sampling layer that shuffles table rows and maps actions back to original indices.

    This implementation follows the original script's approach:
    1. Each sample sees a different row permutation
    2. Model generates actions based on shuffled indices
    3. Actions are mapped back to original indices before voting
    4. Voting happens on normalized (mapped + sorted) actions

    Example:
        Original table rows: [A, B, C, D]

        Sample 0 (identity):   [A, B, C, D] -> selects [0, 1] -> mapped: [0, 1]
        Sample 1 (shuffled):   [C, A, D, B] -> selects [1, 2] -> mapped: [0, 3]
        Sample 2 (shuffled):   [B, D, A, C] -> selects [2, 3] -> mapped: [0, 2]

        After mapping, votes are aggregated in original table space.
    """

    def __init__(self, config: SamplingConfig):
        super().__init__(config)
        # Store permutation mappings: (table_id, sample_idx) -> permutation
        # permutation[shuffled_idx] = original_idx
        self._permutations = {}
        # Store candidates with their sample indices for mapping
        self._candidate_sample_map = {}
        # Track current table being processed
        self._current_table_id = None
        # Track current action name being processed (for action-level control)
        self._current_action_name = None

    def transform_context(self, table: Table, action_history: List[str], sample_idx: int) -> Tuple[Table, List[str]]:
        """
        Transform context by shuffling table rows (only for select_row actions).

        Shuffling is only applied when generating select_row actions.
        For select_column and end actions, the table is returned unchanged.

        Sample 0 uses identity permutation (no shuffle) as a baseline.
        Other samples use deterministic shuffles based on sample_idx.

        The permutation mapping is stored for later use in aggregate_candidates.

        Args:
            table: Original table
            action_history: Action history (unchanged)
            sample_idx: Sample index (0 to n-1)

        Returns:
            (shuffled_table, action_history) tuple
        """
        # Only shuffle for select_row actions
        if self._current_action_name != "select_row":
            # For other actions, use identity permutation (no shuffle)
            shuffled_table, permutation = self._identity_permutation(table)
            table_id = id(table)
            self._permutations[(table_id, sample_idx)] = permutation
            self._current_table_id = table_id

            if self.config.debug:
                print(f"[SHUFFLE DEBUG] Sample {sample_idx}: No shuffle (action={self._current_action_name})")

            return shuffled_table, action_history

        # Sample 0 always uses original order (identity permutation)
        if sample_idx == 0:
            shuffled_table, permutation = self._identity_permutation(table)
        else:
            shuffled_table, permutation = self._shuffle_table_rows(table, sample_idx)

        # Store the permutation for later mapping
        table_id = id(table)
        self._permutations[(table_id, sample_idx)] = permutation
        self._current_table_id = table_id

        if self.config.debug:
            print(f"[SHUFFLE DEBUG] Sample {sample_idx}: permutation = {permutation}")

        return shuffled_table, action_history

    def _identity_permutation(self, table: Table) -> Tuple[Table, List[int]]:
        """
        Create identity permutation (no shuffle).

        Returns:
            (table, identity_mapping) where identity_mapping[i] = i
        """
        n_rows = len(table.rows)
        identity_mapping = list(range(n_rows))
        return table, identity_mapping

    def _shuffle_table_rows(self, table: Table, sample_idx: int) -> Tuple[Table, List[int]]:
        """
        Shuffle table rows deterministically based on sample index.

        Args:
            table: Original table
            sample_idx: Sample index (used as random seed)

        Returns:
            (shuffled_table, permutation_mapping) where:
            - shuffled_table: Table with rows reordered
            - permutation_mapping: permutation_mapping[shuffled_idx] = original_idx
        """
        n_rows = len(table.rows)

        # Create permutation: shuffled index -> original index
        original_indices = list(range(n_rows))

        # Deterministic shuffle based on sample_idx
        rng = random.Random(sample_idx)
        rng.shuffle(original_indices)

        # Create shuffled table
        shuffled_rows = [table.rows[i] for i in original_indices]
        shuffled_table = Table(columns=list(table.columns), rows=shuffled_rows)

        return shuffled_table, original_indices

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
        Generate N candidates with permutation tracking.

        For iterative strategy, we don't know the action type beforehand.
        We set action_name to "select_row" to enable shuffling.
        If the model generates other action types, they'll still work correctly.
        """
        # Clear mappings for this generation round
        table_id = id(table)
        keys_to_remove = [k for k in self._permutations.keys() if k[0] == table_id]
        for key in keys_to_remove:
            del self._permutations[key]
        self._candidate_sample_map.clear()
        self._current_table_id = table_id

        # For iterative strategy, assume select_row to enable shuffling
        # (model can still generate other action types)
        self._current_action_name = "select_row"

        # Generate candidates using parent method (which calls transform_context)
        candidates = await super().generate_candidates(
            worker, n, table, action_history, request_id, state_machines, prompt_builder, question, step
        )

        # Map candidates to sample indices
        # Note: We assume candidates come back in order (sample 0, 1, 2, ...)
        # This is true because asyncio.gather preserves order
        for i, candidate in enumerate(candidates):
            if candidate is not None:
                self._candidate_sample_map[id(candidate)] = i

        return candidates

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
        Generate N argument sets with permutation tracking.

        For two-phase (Chain-of-Table) strategy, we know the action type.
        Only shuffle if action_name is "select_row".
        """
        # Clear mappings for this generation round
        table_id = id(table)
        keys_to_remove = [k for k in self._permutations.keys() if k[0] == table_id]
        for key in keys_to_remove:
            del self._permutations[key]
        self._candidate_sample_map.clear()
        self._current_table_id = table_id

        # Set current action name for transform_context to check
        self._current_action_name = action_name

        # Generate candidates using parent method
        candidates = await super().generate_arguments(
            worker,
            action_name,
            n,
            table,
            action_history,
            request_id,
            step,
            temperature,
            state_machines,
            prompt_builder,
            question,
        )

        # Map candidates to sample indices
        for i, candidate in enumerate(candidates):
            if candidate is not None:
                self._candidate_sample_map[id(candidate)] = i

        return candidates

    def aggregate_candidates(self, candidates: List[Action]) -> Tuple[Optional[Action], int]:
        """
        Aggregate candidates with proper index mapping back to original positions.

        This follows the original script's approach:
        1. For each candidate, look up its sample index
        2. Get the permutation mapping for that sample
        3. Map shuffled indices back to original indices
        4. Sort for canonical representation
        5. Vote on mapped actions

        Args:
            candidates: List of valid actions (already filtered)

        Returns:
            (winning_action, vote_count) tuple
        """
        if not candidates:
            return None, 0

        # Map candidates back to original indices
        mapped_candidates = []
        failed_mappings = 0

        for candidate in candidates:
            mapped = self._map_action_to_original(candidate)
            if mapped is not None:
                mapped_candidates.append(mapped)
            else:
                failed_mappings += 1

        if self.config.debug and failed_mappings > 0:
            print(f"[SHUFFLE DEBUG] {failed_mappings} actions failed mapping")

        if not mapped_candidates:
            return None, 0

        # Vote on mapped actions using exact string matching
        vote_counts = {}
        action_map = {}

        for action in mapped_candidates:
            action_str = action.to_string()
            vote_counts[action_str] = vote_counts.get(action_str, 0) + 1
            action_map[action_str] = action

        if self.config.debug:
            print("[SHUFFLE DEBUG] Vote counts after mapping:")
            for action_str, count in sorted(vote_counts.items(), key=lambda x: -x[1]):
                print(f"  {action_str}: {count}")

        # Return winner
        winner_str = max(vote_counts, key=vote_counts.get)
        return action_map[winner_str], vote_counts[winner_str]

    def _map_action_to_original(self, action: Action) -> Optional[Action]:
        """
        Map action indices from shuffled table to original table positions.

        This follows the original script's map_action_to_original function.

        Args:
            action: Action with shuffled indices

        Returns:
            Action with original indices (sorted), or None if mapping fails
        """
        # Only select_row actions need mapping
        if action.name != "select_row":
            return action

        # Get sample index for this candidate
        candidate_id = id(action)
        sample_idx = self._candidate_sample_map.get(candidate_id)

        if sample_idx is None:
            # Fallback: if we can't find sample idx, just sort the indices
            if self.config.debug:
                print(f"[SHUFFLE DEBUG] No sample_idx found for {action.to_string()}")
            sorted_indices = tuple(sorted(action.arguments))
            return Action(action.name, sorted_indices)

        # Get permutation mapping for this sample
        table_id = self._current_table_id
        permutation_key = (table_id, sample_idx)
        permutation = self._permutations.get(permutation_key)

        if permutation is None:
            # Fallback: no permutation found, just sort
            if self.config.debug:
                print(f"[SHUFFLE DEBUG] No permutation found for sample {sample_idx}")
            sorted_indices = tuple(sorted(action.arguments))
            return Action(action.name, sorted_indices)

        # Validate indices are in bounds
        # Note: Handle both ints and string digits (for robustness, though parse should normalize)
        n_rows = len(permutation)
        validated_indices = []
        for idx in action.arguments:
            # Convert string digits to int if needed (defense in depth)
            if isinstance(idx, str) and idx.isdigit():
                idx = int(idx)
            elif not isinstance(idx, int):
                if self.config.debug:
                    print(f"[SHUFFLE DEBUG] Index {idx} is not an integer or digit string")
                return None

            # Check bounds
            if idx < 0 or idx >= n_rows:
                if self.config.debug:
                    print(f"[SHUFFLE DEBUG] Index {idx} out of bounds [0, {n_rows})")
                return None

            validated_indices.append(idx)

        # Map shuffled indices to original indices using validated indices
        try:
            original_indices = [permutation[idx] for idx in validated_indices]
            # Sort for canonical representation
            original_indices.sort()
            mapped_action = Action(action.name, tuple(original_indices))

            if self.config.debug:
                print(f"[SHUFFLE DEBUG] Mapped {action.to_string()} -> {mapped_action.to_string()} (sample {sample_idx})")

            return mapped_action
        except (IndexError, KeyError, ValueError) as e:
            if self.config.debug:
                print(f"[SHUFFLE DEBUG] Mapping failed for {action.to_string()}: {e}")
            return None
