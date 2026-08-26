import pytest

from leap.config.loader import _build_generation_config
from leap.core import Table
from leap.core.actions import REGISTRY
from leap.generation.sampling import SamplingConfig
from leap.inference.function_constraints import (
    ActionGrammarBuilder,
    StructuredActionParser,
    StructuredSamplingParamsFactory,
    available_actions,
)


@pytest.fixture(autouse=True)
def enabled_actions():
    REGISTRY.set_enabled_actions(["select_row", "select_column", "add_column", "group_by", "sort_by", "end"])
    yield
    REGISTRY._enabled_actions = None


def make_table(num_rows=3):
    return Table(columns=["Name", "Points"], rows=[[f"name {idx}", str(idx)] for idx in range(num_rows)])


def test_default_generation_config_uses_legacy_backend_for_old_configs():
    config = _build_generation_config(
        {"use_constraints": True, "use_global_constraints": False},
        ["select_row", "end"],
        SamplingConfig(),
    )

    assert config.constraint_backend == "legacy_state_machine"
    assert config.output_format == "function"


def test_truncated_add_column_batching_defaults_off_and_requires_boolean():
    config = _build_generation_config({}, ["add_column", "end"], SamplingConfig())
    assert config.batch_truncated_add_column is False

    opted_in = _build_generation_config(
        {"batch_truncated_add_column": True},
        ["add_column", "end"],
        SamplingConfig(),
    )
    assert opted_in.batch_truncated_add_column is True

    with pytest.raises(ValueError, match="batch_truncated_add_column"):
        _build_generation_config(
            {"batch_truncated_add_column": "yes"},
            ["add_column", "end"],
            SamplingConfig(),
        )


def test_generation_config_accepts_constrained_json_add_column():
    config = _build_generation_config(
        {"use_constraints": True, "constraint_backend": "xgrammar", "output_format": "json"},
        ["select_row", "add_column", "end"],
        SamplingConfig(),
    )
    assert config.output_format == "json"


@pytest.mark.parametrize("use_constraints", [False, True])
def test_legacy_backend_forces_function_output_format(use_constraints):
    config = _build_generation_config(
        {
            "use_constraints": use_constraints,
            "constraint_backend": "legacy_state_machine",
            "output_format": "json",
        },
        ["select_row", "end"],
        SamplingConfig(),
    )
    assert config.output_format == "function"


def test_generation_config_accepts_mcp_mode():
    config = _build_generation_config(
        {"constraint_backend": "xgrammar", "output_format": "mcp"},
        ["select_column", "end"],
        SamplingConfig(),
    )

    assert config.output_format == "mcp"


def test_generation_config_accepts_add_column_in_mcp_mode():
    config = _build_generation_config(
        {"constraint_backend": "xgrammar", "output_format": "mcp"},
        ["select_column", "add_column", "end"],
        SamplingConfig(),
    )
    assert config.output_format == "mcp"
    assert "add_column" in config.enabled_actions


def test_generation_config_rejects_unknown_output_format():
    with pytest.raises(ValueError, match="output_format"):
        _build_generation_config({"output_format": "xml"}, ["end"], SamplingConfig())


def test_generation_config_accepts_add_column_for_xgrammar():
    config = _build_generation_config(
        {"use_constraints": True, "constraint_backend": "xgrammar"},
        ["select_row", "add_column", "end"],
        SamplingConfig(),
    )
    assert config.enabled_actions == ("select_row", "add_column", "end")


def test_generation_config_accepts_legacy_state_machine_with_add_column():
    config = _build_generation_config(
        {"use_constraints": True, "constraint_backend": "legacy_state_machine"},
        ["select_row", "add_column", "end"],
        SamplingConfig(),
    )

    assert config.constraint_backend == "legacy_state_machine"


def test_action_grammar_respects_history_and_row_cap():
    table = make_table(num_rows=600)
    builder = ActionGrammarBuilder()
    spec = builder.build_spec(
        table=table,
        action_history=["select_row(0)"],
        use_global_constraints=False,
        phase="single_step",
    )
    grammar = builder.build_single_step_grammar(spec)

    assert "select_row" not in spec.allowed_actions
    assert "select_column" in spec.allowed_actions
    assert len(spec.rows) == 500
    assert '"f_select_column("' in grammar
    assert '"f_select_row("' not in grammar


def test_global_available_actions_follow_leap_transition_matrix():
    assert available_actions([], use_global_constraints=True) == ["add_column", "select_row", "select_column", "group_by", "sort_by"]
    assert available_actions(
        ["select_row(0)"],
        use_global_constraints=True,
    ) == ["select_column", "group_by", "sort_by", "end"]
    assert available_actions(
        ["select_row(0)", 'select_column("Name")'],
        use_global_constraints=True,
    ) == ["group_by", "sort_by", "end"]
    assert available_actions(['group_by("Name")'], use_global_constraints=True) == ["sort_by", "end"]
    assert available_actions(['sort_by("Points", "desc")'], use_global_constraints=True) == ["end"]


def test_global_available_actions_intersect_enabled_and_backend_supported_actions():
    REGISTRY.set_enabled_actions(["add_column", "select_row", "group_by", "end"])

    assert available_actions([], use_global_constraints=True) == ["add_column", "select_row", "group_by"]
    assert available_actions(["select_row(0)"], use_global_constraints=True) == ["group_by", "end"]


def test_registry_global_transitions_include_add_column_when_enabled():
    REGISTRY.set_enabled_actions(["add_column", "select_row", "select_column", "group_by", "sort_by", "end"])

    expected = {
        None: ["add_column", "select_row", "select_column", "group_by", "sort_by"],
        "add_column": ["select_row", "select_column", "group_by", "sort_by", "end"],
        "select_row": ["select_column", "group_by", "sort_by", "end"],
        "select_column": ["group_by", "sort_by", "end"],
        "group_by": ["sort_by", "end"],
        "sort_by": ["end"],
    }

    for previous_action, successors in expected.items():
        history = [] if previous_action is None else [f"{previous_action}()"]
        assert REGISTRY.get_global_available_actions(history) == successors


def test_argument_grammars_for_supported_actions():
    table = make_table()
    builder = ActionGrammarBuilder()

    row_spec = builder.build_spec(
        table=table,
        action_history=[],
        use_global_constraints=False,
        phase="arguments",
        selected_action="select_row",
    )
    row_grammar = builder.build_arguments_grammar(row_spec)
    assert "root ::= row_list" in row_grammar
    assert '"[*]"' in row_grammar
    assert '"\\"row 0\\""' in row_grammar
    assert '"\\"row 2\\""' in row_grammar

    sort_spec = builder.build_spec(
        table=table,
        action_history=[],
        use_global_constraints=False,
        phase="arguments",
        selected_action="sort_by",
    )
    sort_grammar = builder.build_arguments_grammar(sort_spec)
    assert 'root ::= column ", " order' in sort_grammar
    assert '"\\"Name\\""' in sort_grammar
    assert '"\\"asc\\""' in sort_grammar
    assert '"\\"desc\\""' in sort_grammar

    add_spec = builder.build_spec(
        table=table,
        action_history=[],
        use_global_constraints=False,
        phase="arguments",
        selected_action="add_column",
    )
    add_grammar = builder.build_arguments_grammar(add_spec)
    assert "root ::= json_string" in add_grammar
    assert add_grammar.count("json_string") >= len(table.rows) + 1
    assert 'value_list ::= "[" json_string ", " json_string ", " json_string "]"' in add_grammar


def test_select_row_wildcard_is_supported_only_for_non_empty_tables():
    builder = ActionGrammarBuilder()
    argument_spec = builder.build_spec(
        table=make_table(),
        action_history=[],
        use_global_constraints=False,
        phase="arguments",
        selected_action="select_row",
    )
    assert '"[*]"' in builder.build_arguments_grammar(argument_spec)

    single_step_spec = builder.build_spec(
        table=make_table(),
        action_history=[],
        use_global_constraints=False,
        phase="single_step",
    )
    assert '"[*]"' in builder.build_single_step_grammar(single_step_spec)

    empty_spec = builder.build_spec(
        table=make_table(num_rows=0),
        action_history=[],
        use_global_constraints=False,
        phase="arguments",
        selected_action="select_row",
    )
    assert '"[*]"' not in builder.build_arguments_grammar(empty_spec)


def test_action_selection_grammar_outputs_old_action_names():
    builder = ActionGrammarBuilder()
    spec = builder.build_spec(
        table=make_table(),
        action_history=[],
        use_global_constraints=False,
        phase="action",
    )
    grammar = builder.build_action_grammar(spec)

    assert 'root ::= "f_add_column"' in grammar
    assert '"f_select_row"' in grammar
    assert '"f_select_column"' in grammar
    assert '"f_end"' not in grammar


def test_structured_parser_uses_code_style_actions():
    table = make_table()

    row_action = StructuredActionParser.parse_arguments('["row 0", "row 2"]', "select_row", table)
    assert row_action.to_string() == "select_row([row 0, row 2])"

    column_action = StructuredActionParser.parse_single_step('f_select_column(["Name"])', table)
    assert column_action.to_string() == "select_column('Name')"

    invalid = StructuredActionParser.parse_arguments('["row 999"]', "select_row", table)
    assert invalid is None


def test_structured_parser_validates_select_row_wildcard():
    table = make_table()
    wildcard = StructuredActionParser.parse_arguments("[*]", "select_row", table)

    assert wildcard is not None
    assert wildcard.arguments == ("*",)
    assert wildcard.apply_to_table(table) == table
    assert StructuredActionParser.parse_arguments('[*, "row 0"]', "select_row", table) is None
    assert StructuredActionParser.parse_arguments("[*]", "select_row", make_table(num_rows=0)) is None


def test_structured_params_factory_uses_grammar():
    kwargs = StructuredSamplingParamsFactory.structured_outputs_kwargs('root ::= "f_end"\n')
    params = kwargs.get("structured_outputs")
    if params is not None:
        assert params.grammar == 'root ::= "f_end"\n'
    else:
        assert kwargs["guided_decoding"].grammar == 'root ::= "f_end"\n'
