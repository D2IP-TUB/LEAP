"""
Table Actions - Registry of all available table transformation actions.

This module auto-registers all action definitions on import and exports
the Action class and REGISTRY for external use.
"""

# Import registry and action definitions
from .action import Action

# Import all action implementations
from .direct_query import DirectQueryAction
from .end import EndAction
from .registry import REGISTRY, ActionDefinition
from .select_column import SelectColumnAction
from .select_row import SelectRowAction

# Register all actions that the LLM can pick
REGISTRY.register(SelectRowAction())
REGISTRY.register(SelectColumnAction())
REGISTRY.register(EndAction())
# Note: DirectQueryAction is NOT registered - it's applied automatically after end()

__all__ = [
    "Action",
    "ActionDefinition",
    "REGISTRY",
    "SelectRowAction",
    "SelectColumnAction",
    "EndAction",
    "DirectQueryAction",
]
