"""
Generation module for action and argument generation.
"""


# Lazy imports to avoid requiring vllm for tests that only need dataclasses
def __getattr__(name):
    """Lazy import for generation functions that require vllm."""
    if name in ("generate_action_arguments", "generate_action_selection", "generate_single_action"):
        from leap.generation.generate import (
            generate_action_arguments,
            generate_action_selection,
            generate_single_action,
        )

        return {
            "generate_action_arguments": generate_action_arguments,
            "generate_action_selection": generate_action_selection,
            "generate_single_action": generate_single_action,
        }[name]
    elif name == "PromptBuilder":
        from leap.generation.prompt_builder import PromptBuilder

        return PromptBuilder
    elif name in ("BaseGenerationStrategy", "ChainOfTableGenerationStrategy", "IterativeGenerationStrategy"):
        from leap.generation.strategies import (
            BaseGenerationStrategy,
            ChainOfTableGenerationStrategy,
            IterativeGenerationStrategy,
        )

        return {
            "BaseGenerationStrategy": BaseGenerationStrategy,
            "ChainOfTableGenerationStrategy": ChainOfTableGenerationStrategy,
            "IterativeGenerationStrategy": IterativeGenerationStrategy,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "generate_single_action",
    "generate_action_selection",
    "generate_action_arguments",
    "BaseGenerationStrategy",
    "IterativeGenerationStrategy",
    "ChainOfTableGenerationStrategy",
    "PromptBuilder",
]
