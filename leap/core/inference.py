"""Types for the inference pipeline

This module contains the core types that define the contract for the inference pipeline:
- InferenceRequest: Input to the inference process
- InferenceResult: Output from the inference process
- ExecutionMetrics: Detailed metrics about execution quality

These types work together to provide end-to-end type safety for the inference flow.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .action import Action
from .table import Table


@dataclass(frozen=True)
class ExecutionMetrics:
    """Metrics from executing a sequence of actions on a table

    This captures both the accuracy of the final result and details about
    the execution process (termination, errors, etc.)
    """

    execution_accuracy: float
    answer_found_in_final: bool
    answer_found_in_original: bool
    terminated_properly: bool
    matched_answers_final: List[str]
    matched_answers_original: List[str]
    num_actions: int
    final_table_size: Optional[Tuple[int, int]] = None
    execution_error: Optional[str] = None
    evaluation_method: str = "wikitablequestions_logic"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionMetrics":
        """Create from dictionary (for backward compatibility during migration)

        Args:
            data: Dictionary containing metrics fields

        Returns:
            ExecutionMetrics instance
        """
        return cls(
            execution_accuracy=data.get("execution_accuracy", 0.0),
            answer_found_in_final=data.get("answer_found_in_final", False),
            answer_found_in_original=data.get("answer_found_in_original", False),
            terminated_properly=data.get("terminated_properly", False),
            matched_answers_final=data.get("matched_answers_final", []),
            matched_answers_original=data.get("matched_answers_original", []),
            num_actions=data.get("num_actions", 0),
            final_table_size=data.get("final_table_size"),
            execution_error=data.get("execution_error"),
            evaluation_method=data.get("evaluation_method", "wikitablequestions_logic"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)

        Returns:
            Dictionary representation of metrics
        """
        return {
            "execution_accuracy": self.execution_accuracy,
            "answer_found_in_final": self.answer_found_in_final,
            "answer_found_in_original": self.answer_found_in_original,
            "terminated_properly": self.terminated_properly,
            "matched_answers_final": self.matched_answers_final,
            "matched_answers_original": self.matched_answers_original,
            "num_actions": self.num_actions,
            "final_table_size": self.final_table_size,
            "execution_error": self.execution_error,
            "evaluation_method": self.evaluation_method,
        }


@dataclass(frozen=True)
class InferenceRequest:
    """Request for table reasoning inference

    Represents a single question-answering task over a table. The request_id
    is automatically generated if not provided.
    """

    question: str
    table: Table
    ground_truth_answers: List[str]
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_example(cls, example: Dict[str, Any], index: Optional[int] = None) -> "InferenceRequest":
        """Create from dataset example (WikiTableQuestions format)

        Args:
            example: Dataset example with 'question', 'table', 'answers' fields
            index: Optional index for deterministic request_id

        Returns:
            InferenceRequest instance
        """
        table = Table(columns=example["table"]["header"], rows=example["table"]["rows"])
        request_id = f"req_{index}" if index is not None else str(uuid.uuid4())
        return cls(
            question=example["question"],
            table=table,
            ground_truth_answers=example["answers"],
            request_id=request_id,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceRequest":
        """Create from dictionary (for multiprocessing queue deserialization)

        Args:
            data: Dictionary containing request fields

        Returns:
            InferenceRequest instance
        """
        table = data["table"]
        if isinstance(table, dict):
            table = Table.from_dict(table)
        return cls(
            question=data["question"],
            table=table,
            ground_truth_answers=data["ground_truth_answers"],
            request_id=data.get("request_id", str(uuid.uuid4())),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for multiprocessing queue serialization)

        Returns:
            Dictionary representation that can be passed through multiprocessing queues
        """
        return {
            "question": self.question,
            "table": self.table,  # Keep as Table object (it's a dataclass)
            "ground_truth_answers": self.ground_truth_answers,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class InferenceResult:
    """Result from table reasoning inference

    Contains the action sequence, final table state, execution quality metrics,
    and the original request context for self-contained results.
    """

    action_history: List[str]  # List of action strings (e.g., "select_row([0, 1])")
    final_table: Table
    execution_metrics: ExecutionMetrics
    request_id: Optional[str] = None
    question: Optional[str] = None  # Original question from request
    ground_truth_answers: Optional[List[str]] = None  # Expected answers from request
    profiling_data: Optional[Dict[str, Any]] = None  # Profiling timings from worker

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceResult":
        """Create from dictionary (for multiprocessing queue deserialization)

        Args:
            data: Dictionary containing result fields

        Returns:
            InferenceResult instance
        """
        # Handle error results - create a failed InferenceResult
        if "error" in data and "final_table" not in data:
            error_msg = data["error"]
            return cls(
                action_history=[],
                final_table=Table(columns=[], rows=[]),  # Empty table for failed results
                execution_metrics=ExecutionMetrics(
                    execution_accuracy=0.0,
                    answer_found_in_final=False,
                    answer_found_in_original=False,
                    terminated_properly=False,
                    matched_answers_final=[],
                    matched_answers_original=[],
                    num_actions=0,
                    execution_error=error_msg,
                ),
                request_id=data.get("request_id"),
                question=data.get("question"),
                ground_truth_answers=data.get("ground_truth_answers"),
            )

        # Convert metrics if it's a dict
        metrics = data.get("execution_accuracy_metrics", {})
        if isinstance(metrics, dict):
            metrics = ExecutionMetrics.from_dict(metrics)

        # Convert table if it's a dict
        final_table = data["final_table"]
        if isinstance(final_table, dict):
            final_table = Table.from_dict(final_table)

        return cls(
            action_history=data["action_history"],
            final_table=final_table,
            execution_metrics=metrics,
            request_id=data.get("request_id"),
            question=data.get("question"),
            ground_truth_answers=data.get("ground_truth_answers"),
            profiling_data=data.get("profiling_data"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for multiprocessing queue serialization)

        Returns:
            Dictionary representation that can be passed through multiprocessing queues
        """
        return {
            "action_history": self.action_history,
            "final_table": self.final_table,  # Keep as Table object
            "execution_accuracy_metrics": self.execution_metrics.to_dict(),
            "request_id": self.request_id,
            "question": self.question,
            "ground_truth_answers": self.ground_truth_answers,
            "profiling_data": self.profiling_data,
        }

    def get_actions(self) -> List[Action]:
        """Parse action history into Action objects

        Returns:
            List of successfully parsed Action objects (invalid actions are skipped)
        """
        actions = []
        for action_str in self.action_history:
            action = Action.parse(action_str)
            if action:
                actions.append(action)
        return actions

    @property
    def num_steps(self) -> int:
        """Number of actions in the history"""
        return len(self.action_history)

    @property
    def execution_accuracy(self) -> float:
        """Convenience property for execution accuracy"""
        return self.execution_metrics.execution_accuracy
