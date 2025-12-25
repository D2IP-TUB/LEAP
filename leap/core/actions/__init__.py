"""
Table Actions - Registry of all available table transformation actions.

This module auto-registers all action definitions on import and exports
the Action class and REGISTRY for external use.
"""

# Import registry and action definitions
from .action import Action

# Import all action implementations
from .add_column import AddColumnAction
from .direct_query import DirectQueryAction
from .end import EndAction
from .group_by import GroupByAction
from .registry import REGISTRY, ActionDefinition
from .select_column import SelectColumnAction
from .select_row import SelectRowAction
from .sort_by import SortByAction

# Register all actions that the LLM can pick
REGISTRY.register(AddColumnAction())
REGISTRY.register(SelectRowAction())
REGISTRY.register(SelectColumnAction())
REGISTRY.register(GroupByAction())
REGISTRY.register(SortByAction())
REGISTRY.register(EndAction())
# Note: DirectQueryAction is NOT registered - it's applied automatically after end()

__all__ = [
    "Action",
    "ActionDefinition",
    "REGISTRY",
    "AddColumnAction",
    "SelectRowAction",
    "SelectColumnAction",
    "GroupByAction",
    "SortByAction",
    "EndAction",
    "DirectQueryAction",
]
