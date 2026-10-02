from typing import List, Tuple

from leap.core import Table
from leap.core.actions import Action


def apply_single_action(initial_table: Table, action: str) -> Table:
    action = Action.parse(action)
    if not action:
        return None

    new_table = action.apply_to_table(initial_table)
    if new_table is None:
        return None
    return new_table


def apply_actions(initial_table: Table, action_history: List[str], max_steps: int | None = None) -> List[Tuple[str, Table]]:
    tables: List[Tuple[str, Table]] = []
    current = initial_table
    seen_action_names = set()

    steps = action_history if max_steps is None else action_history[:max_steps]

    for action_str in steps:
        action = Action.parse(action_str)
        if not action:
            continue

        if action.name not in ("end", "direct_query") and action.name in seen_action_names:
            break

        new_table = action.apply_to_table(current)
        if new_table is None:
            # invalid on this table; skip or break
            continue

        tables.append((action.to_string(), new_table))
        current = new_table

        if action.name not in ("end", "direct_query"):
            seen_action_names.add(action.name)

    return tables
