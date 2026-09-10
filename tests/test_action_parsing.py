from leap.core.actions.action import Action


def test_select_row_string_round_trips():
    for action in (Action("select_row", [0, 2]), Action("select_row", ["*"])):
        serialized = action.to_string()

        assert Action.parse(serialized) == action


def test_group_by_preserves_unquoted_trailing_parentheses():
    action = Action.parse("group_by(Population (2005))")

    assert action == Action("group_by", ["Population (2005)"])


def test_group_by_preserves_comma_in_quoted_column_name():
    action = Action.parse('group_by("GDP per capita (US$, PPP)")')

    assert action == Action("group_by", ["GDP per capita (US$, PPP)"])


def test_sort_by_preserves_comma_in_quoted_column_name():
    action = Action.parse('sort_by("GDP Growth, 2007-2011 (in %)", "asc")')

    assert action == Action("sort_by", ["GDP Growth, 2007-2011 (in %)", "asc"])


def test_sort_by_legacy_unquoted_arguments_are_unchanged():
    action = Action.parse("sort_by(Year, descending)")

    assert action == Action("sort_by", ["Year", "desc"])
