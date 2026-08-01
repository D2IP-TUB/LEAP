import pytest
from transformers import AutoTokenizer

from leap.config.loader import TokenizerConfig
from leap.core import Table
from leap.core.actions import REGISTRY
from leap.inference.legacy.constraints import (
    ActionOnlyConstraintStateMachine,
    ArgumentsOnlyConstraintStateMachine,
    ConstraintStateMachine,
)


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="module")
def tokenizer_config(gpt2_tokenizer):
    """Create tokenizer config for gpt2"""
    return TokenizerConfig(
        is_llama_tokenizer=False,
        comma_id=gpt2_tokenizer.encode(",", add_special_tokens=False)[0],
        list_open_id=gpt2_tokenizer.encode("[", add_special_tokens=False)[0],
        list_close_id=gpt2_tokenizer.encode("]", add_special_tokens=False)[0],
        paren_open_id=gpt2_tokenizer.encode("(", add_special_tokens=False)[0],
        paren_close_id=gpt2_tokenizer.encode(")", add_special_tokens=False)[0],
        quote_id=gpt2_tokenizer.encode('"', add_special_tokens=False)[0],
        closing_quotes_tokens=gpt2_tokenizer.encode('"', add_special_tokens=False),
        action_tokens={
            "select_row": gpt2_tokenizer.encode("select_row", add_special_tokens=False),
            "select_column": gpt2_tokenizer.encode("select_column", add_special_tokens=False),
            "end": gpt2_tokenizer.encode("end", add_special_tokens=False),
        },
    )


@pytest.fixture(autouse=True)
def enabled_constraint_actions():
    REGISTRY.set_enabled_actions(["select_row", "select_column", "end"])
    yield
    REGISTRY._enabled_actions = None


def make_table(num_rows=5, columns=None):
    cols = columns or ["foo", "bar", "baz"]
    return Table(columns=cols, rows=[["val" for _ in cols] for _ in range(num_rows)])


def test_initial_allowed_tokens_respect_action_history(gpt2_tokenizer, tokenizer_config):
    table = make_table()

    # With no action history and global constraints enabled (default),
    # only select_row and select_column are allowed initially.
    machine = ConstraintStateMachine(table, gpt2_tokenizer, tokenizer_config)
    allowed = set(machine.allowed_tokens())
    assert allowed == {
        machine.tokenizer_config.action_tokens["select_row"][0],
        machine.tokenizer_config.action_tokens["select_column"][0],
    }

    # If select_row has been used before, then only select_column is allowed
    row_history = ConstraintStateMachine(table, gpt2_tokenizer, tokenizer_config, action_history=["select_row(0)"])
    allowed_after_row = set(row_history.allowed_tokens())
    assert allowed_after_row == {row_history.tokenizer_config.action_tokens["select_column"][0]}

    # If both select_row and select_column were used before, only end is allowed
    end_only = ConstraintStateMachine(
        table,
        gpt2_tokenizer,
        tokenizer_config,
        action_history=["select_row(0)", 'select_column("foo")'],
    )
    allowed_end_only = set(end_only.allowed_tokens())
    assert allowed_end_only == {end_only.tokenizer_config.action_tokens["end"][0]}


def test_select_row_multiple_parameters(gpt2_tokenizer, tokenizer_config):
    table = make_table(num_rows=3)
    machine = ConstraintStateMachine(table, gpt2_tokenizer, tokenizer_config)
    row_tokens = machine.tokenizer_config.action_tokens["select_row"]

    # Type out the action token-by-token
    for idx, token in enumerate(row_tokens):
        assert token in machine.allowed_tokens()
        machine.update_state(token)
        if idx + 1 < len(row_tokens):
            expected = row_tokens[idx + 1]
            assert expected in machine.allowed_tokens()

    # After action, expect '('
    assert machine.state == "after_action"
    assert machine.allowed_tokens() == [machine.tokenizer_config.paren_open_id]

    # Then '['
    machine.update_state(machine.tokenizer_config.paren_open_id)
    machine.update_state(machine.tokenizer_config.list_open_id)
    # Next, the opening quote for a parameter
    assert machine.tokenizer_config.quote_id in machine.allowed_tokens()

    def feed_row(name):
        tokens = machine.row_token_map[name]
        for idx, token in enumerate(tokens):
            if idx == 0:
                assert token in machine.allowed_tokens()
            machine.update_state(token)

    # First parameter
    feed_row("row 0")
    allowed_after_first = set(machine.allowed_tokens())
    assert machine.tokenizer_config.list_close_id in allowed_after_first
    assert machine.tokenizer_config.comma_id in allowed_after_first

    # Second parameter
    machine.update_state(machine.tokenizer_config.comma_id)
    assert machine.selected_params == {"row 0"}
    assert machine.tokenizer_config.quote_id in machine.allowed_tokens()

    feed_row("row 1")
    allowed_after_second = set(machine.allowed_tokens())
    assert machine.tokenizer_config.list_close_id in allowed_after_second
    assert machine.tokenizer_config.comma_id in allowed_after_second

    # Third parameter
    machine.update_state(machine.tokenizer_config.comma_id)
    assert machine.selected_params == {"row 0", "row 1"}
    assert machine.tokenizer_config.quote_id in machine.allowed_tokens()

    feed_row("row 2")
    allowed_after_third = set(machine.allowed_tokens())
    assert machine.tokenizer_config.list_close_id in allowed_after_third
    assert machine.tokenizer_config.comma_id not in allowed_after_third

    # Close list and paren
    machine.update_state(machine.tokenizer_config.list_close_id)
    assert machine.state == "in_paren_close"
    assert machine.allowed_tokens() == [machine.tokenizer_config.paren_close_id]

    machine.update_state(machine.tokenizer_config.paren_close_id)
    assert machine.finished
    assert machine.allowed_tokens() == [gpt2_tokenizer.eos_token_id]
    assert machine.selected_params == {"row 0", "row 1", "row 2"}


def test_full_action_select_row_wildcard_is_exclusive(gpt2_tokenizer, tokenizer_config):
    machine = ConstraintStateMachine(make_table(), gpt2_tokenizer, tokenizer_config)

    for token in machine.tokenizer_config.action_tokens["select_row"]:
        machine.update_state(token)
    machine.update_state(machine.tokenizer_config.paren_open_id)
    machine.update_state(machine.tokenizer_config.list_open_id)

    assert machine.wildcard_tokens[0] in machine.allowed_tokens()
    for token in machine.wildcard_tokens:
        assert token in machine.allowed_tokens()
        machine.update_state(token)

    assert machine.allowed_tokens() == [machine.tokenizer_config.list_close_id]
    machine.update_state(machine.tokenizer_config.list_close_id)
    assert machine.allowed_tokens() == [machine.tokenizer_config.paren_close_id]
    machine.update_state(machine.tokenizer_config.paren_close_id)
    assert machine.finished


def test_arguments_only_select_row_wildcard_is_exclusive(gpt2_tokenizer, tokenizer_config):
    machine = ArgumentsOnlyConstraintStateMachine(make_table(), gpt2_tokenizer, tokenizer_config, "select_row")

    machine.update_state(machine.tokenizer_config.list_open_id)
    assert machine.wildcard_tokens[0] in machine.allowed_tokens()
    for token in machine.wildcard_tokens:
        assert token in machine.allowed_tokens()
        machine.update_state(token)

    assert machine.allowed_tokens() == [machine.tokenizer_config.list_close_id]
    machine.update_state(machine.tokenizer_config.list_close_id)
    assert machine.finished
    assert machine.allowed_tokens() == [gpt2_tokenizer.eos_token_id]


def test_select_row_wildcard_is_not_offered_for_empty_tables(gpt2_tokenizer, tokenizer_config):
    machine = ArgumentsOnlyConstraintStateMachine(make_table(num_rows=0), gpt2_tokenizer, tokenizer_config, "select_row")

    machine.update_state(machine.tokenizer_config.list_open_id)
    assert machine.wildcard_enabled is False
    assert machine.wildcard_tokens[0] not in machine.allowed_tokens()


def test_select_column_multiple_parameters(gpt2_tokenizer, tokenizer_config):
    table = make_table(columns=["foo", "bar", "baz"])
    machine = ConstraintStateMachine(table, gpt2_tokenizer, tokenizer_config)
    col_tokens = machine.tokenizer_config.action_tokens["select_column"]

    # Type out the action token-by-token
    for idx, token in enumerate(col_tokens):
        assert token in machine.allowed_tokens()
        machine.update_state(token)
        if idx + 1 < len(col_tokens):
            next_token = col_tokens[idx + 1]
            assert next_token in machine.allowed_tokens()

    # Open paren and list
    machine.update_state(machine.tokenizer_config.paren_open_id)
    machine.update_state(machine.tokenizer_config.list_open_id)

    def feed_column(name):
        tokens = machine.column_token_map[name]
        for position, token in enumerate(tokens):
            assert token in machine.allowed_tokens()
            machine.update_state(token)
            if position + 1 < len(tokens):
                expected = tokens[position + 1]
                assert expected in machine.allowed_tokens()

    # First column
    feed_column("foo")
    allowed_after_first = set(machine.allowed_tokens())
    assert machine.tokenizer_config.list_close_id in allowed_after_first
    assert machine.tokenizer_config.comma_id in allowed_after_first

    # Second column
    machine.update_state(machine.tokenizer_config.comma_id)
    feed_column("bar")
    allowed_after_second = set(machine.allowed_tokens())
    assert machine.tokenizer_config.list_close_id in allowed_after_second
    assert machine.tokenizer_config.comma_id in allowed_after_second

    # Third column
    machine.update_state(machine.tokenizer_config.comma_id)
    feed_column("baz")
    allowed_after_third = set(machine.allowed_tokens())
    assert machine.tokenizer_config.list_close_id in allowed_after_third
    assert machine.tokenizer_config.comma_id not in allowed_after_third

    # Close list and paren
    machine.update_state(machine.tokenizer_config.list_close_id)
    assert machine.state == "in_paren_close"
    assert machine.allowed_tokens() == [machine.tokenizer_config.paren_close_id]

    machine.update_state(machine.tokenizer_config.paren_close_id)
    assert machine.finished
    assert machine.selected_params == {"foo", "bar", "baz"}


def test_end_action_finishes_when_only_end_allowed(gpt2_tokenizer, tokenizer_config):
    table = make_table()
    machine = ConstraintStateMachine(
        table,
        gpt2_tokenizer,
        tokenizer_config,
        action_history=["select_row(0)", 'select_column("foo")'],
    )
    end_tokens = machine.tokenizer_config.action_tokens["end"]

    allowed_start = set(machine.allowed_tokens())
    assert allowed_start == {end_tokens[0]}

    machine.update_state(end_tokens[0])
    if len(end_tokens) > 1:
        assert set(machine.allowed_tokens()) == {end_tokens[1]}
        for token in end_tokens[1:]:
            machine.update_state(token)
    assert machine.allowed_tokens() == [gpt2_tokenizer.eos_token_id]


def test_add_column_bypasses_full_action_state_machine(gpt2_tokenizer, tokenizer_config):
    REGISTRY.set_enabled_actions(["add_column", "select_row", "end"])
    config = TokenizerConfig(
        **{
            **tokenizer_config.__dict__,
            "action_tokens": {
                **tokenizer_config.action_tokens,
                "add_column": gpt2_tokenizer.encode("add_column", add_special_tokens=False),
            },
        }
    )
    machine = ConstraintStateMachine(make_table(), gpt2_tokenizer, config, use_global_constraints=False)

    for token in config.action_tokens["add_column"]:
        assert token in machine.allowed_tokens()
        machine.update_state(token)

    assert machine.current_action == "add_column"
    assert machine.bypass_constraints is True


def test_global_constraints_remove_add_column_transition(gpt2_tokenizer, tokenizer_config):
    REGISTRY.set_enabled_actions(["add_column", "select_row", "end"])
    machine = ConstraintStateMachine(make_table(), gpt2_tokenizer, tokenizer_config, use_global_constraints=True)

    assert machine.possible_actions == ["select_row"]

    action_machine = ActionOnlyConstraintStateMachine(gpt2_tokenizer, use_global_constraints=True)
    assert "add_column" not in action_machine.possible_actions


def test_arguments_only_sort_by_order_uses_token_prefix(gpt2_tokenizer, tokenizer_config):
    table = make_table(columns=["foo"])
    machine = ArgumentsOnlyConstraintStateMachine(table, gpt2_tokenizer, tokenizer_config, "sort_by")

    for token in machine.param_token_map["foo"]:
        assert token in machine.allowed_tokens()
        machine.update_state(token)

    assert machine.tokenizer_config.comma_id in machine.allowed_tokens()
    machine.update_state(machine.tokenizer_config.comma_id)

    order_tokens = machine.param_token_map["asc"]
    for position, token in enumerate(order_tokens):
        assert token in machine.allowed_tokens()
        machine.update_state(token)
        if position + 1 < len(order_tokens):
            assert order_tokens[position + 1] in machine.allowed_tokens()

    assert machine.finished
    assert machine.allowed_tokens() == [gpt2_tokenizer.eos_token_id]
