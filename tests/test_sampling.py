from types import SimpleNamespace

from leap.core.actions import REGISTRY
from leap.core.actions.action import Action
from leap.generation.sampling import SamplingConfig, SamplingLayer


def test_add_column_argument_cleaning_removes_echoed_function_call():
    raw_args = "f_add_column(display type, ['monochrome', 'color'])"

    cleaned = SamplingLayer._clean_argument_text(raw_args, "add_column")
    action = Action.parse(f"add_column({cleaned})")

    assert cleaned == "display type, ['monochrome', 'color']"
    assert action is not None
    assert action.name == "add_column"
    assert action.arguments == ("display type", ["monochrome", "color"])


def test_global_constraints_remove_add_column_from_action_selection():
    previous_enabled = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["add_column", "select_row", "end"])
    try:
        actions = SamplingLayer(SamplingConfig())._get_available_actions(
            [],
            SimpleNamespace(use_constraints=False, use_global_constraints=True),
        )
    finally:
        REGISTRY._enabled_actions = previous_enabled

    assert actions == ["select_row"]


def test_sort_by_argument_cleaning_preserves_parentheses_in_column_name():
    raw_args = '"Population (2005)", "desc"'

    cleaned = SamplingLayer._clean_argument_text(raw_args, "sort_by")
    action = Action.parse(f"sort_by({cleaned})")

    assert cleaned == raw_args
    assert action is not None
    assert action.arguments == ("Population (2005)", "desc")


def test_select_column_argument_cleaning_preserves_parentheses_in_column_name():
    raw_args = '["Top scorer (League)"]'

    cleaned = SamplingLayer._clean_argument_text(raw_args, "select_column")
    action = Action.parse(f"select_column({cleaned})")

    assert cleaned == raw_args
    assert action is not None
    assert action.arguments == ("Top scorer (League)",)


def test_group_by_argument_cleaning_preserves_parentheses_in_column_name():
    raw_args = '"2013 Endowment (and US rank)"'

    cleaned = SamplingLayer._clean_argument_text(raw_args, "group_by")
    action = Action.parse(f"group_by({cleaned})")

    assert cleaned == raw_args
    assert action is not None
    assert action.arguments == ("2013 Endowment (and US rank)",)


def test_echoed_function_cleaning_removes_only_outer_parentheses():
    raw_args = 'f_select_column(["Top scorer (League)"])'

    cleaned = SamplingLayer._clean_argument_text(raw_args, "select_column")
    action = Action.parse(f"select_column({cleaned})")

    assert cleaned == '["Top scorer (League)"]'
    assert action is not None
    assert action.arguments == ("Top scorer (League)",)


def test_echoed_scalar_argument_cleaning_preserves_escaped_quotes():
    raw_args = r'f_group_by("Writer \"A\" (lead)")'

    cleaned = SamplingLayer._clean_argument_text(raw_args, "group_by")
    action = Action.parse(f"group_by({cleaned})")

    assert cleaned == r'"Writer \"A\" (lead)"'
    assert action is not None
    assert action.arguments == ('Writer "A" (lead)',)


def test_echoed_list_argument_cleaning_preserves_escaped_quotes():
    raw_args = r'f_select_column(["Writer \"A\" (lead)"])'

    cleaned = SamplingLayer._clean_argument_text(raw_args, "select_column")
    action = Action.parse(f"select_column({cleaned})")

    assert cleaned == r'["Writer \"A\" (lead)"]'
    assert action is not None
    assert action.arguments == ('Writer "A" (lead)',)
