"""
Table Actions - Registry of all available table transformation actions.

This module auto-registers all action definitions on import.
"""

from leap.core.action_registry import REGISTRY
from leap.core.actions.direct_query import DirectQueryAction
from leap.core.actions.end import EndAction
from leap.core.actions.select_column import SelectColumnAction
from leap.core.actions.select_row import SelectRowAction

# Register all actions
REGISTRY.register(SelectRowAction())
REGISTRY.register(SelectColumnAction())
REGISTRY.register(EndAction())
REGISTRY.register(DirectQueryAction())

__all__ = [
    "SelectRowAction",
    "SelectColumnAction",
    "EndAction",
    "DirectQueryAction",
    "REGISTRY",
]
