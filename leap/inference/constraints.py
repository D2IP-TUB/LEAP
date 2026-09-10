"""Compatibility imports for the legacy masked-token constraint backend."""

from leap.inference.legacy.constraints import (  # noqa: F401
    ActionOnlyConstraintStateMachine,
    ArgumentsOnlyConstraintStateMachine,
    ConstraintStateMachine,
    create_action_only_constraint_processor,
    create_arguments_only_constraint_processor,
    create_constraint_logits_processor,
)
