"""Inference runtime exports, loaded lazily to keep protocol imports lightweight."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ProcessParallelVLLM",
    "ConstraintStateMachine",
    "create_constraint_logits_processor",
    "create_action_only_constraint_processor",
]


def __getattr__(name: str) -> Any:
    if name == "ProcessParallelVLLM":
        from leap.inference.vllm_server import ProcessParallelVLLM

        return ProcessParallelVLLM
    if name in {
        "ConstraintStateMachine",
        "create_constraint_logits_processor",
        "create_action_only_constraint_processor",
    }:
        from leap.inference import constraints

        return getattr(constraints, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
