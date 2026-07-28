import pytest

from leap.config.loader import _build_generation_config
from leap.core import Table
from leap.core.actions import REGISTRY
from leap.generation.sampling import SamplingConfig
from leap.inference.action_grammar import (
    ActionGrammarBuilder,
    StructuredActionParser,
    StructuredSamplingParamsFactory,
    available_actions,
)


@pytest.fixture(autouse=True)
def enabled_actions():
    REGISTRY.set_enabled_actions(["select_row", "select_column", "group_by", "sort_by", "end"])
    yield
    REGISTRY._enabled_actions = None


def make_table(num_rows=3):
    return Table(columns=["Name", "Points"], rows=[[f"name {idx}", str(idx)] for idx in range(num_rows)])


def test_default_generation_config_uses_xgrammar_backend():
    config = _build_generation_config(
        {"use_constraints": True, "use_global_constraints": False},
        ["select_row", "end"],
        SamplingConfig(),
    )

    assert config.constraint_backend == "xgrammar"


def test_generation_config_rejects_add_column_for_xgrammar():
    with pytest.raises(ValueError, match="add_column"):
        _build_generation_config(
            {"use_constraints": True, "constraint_backend": "xgrammar"},
            ["select_row", "add_column", "end"],
            SamplingConfig(),
        )


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


def test_global_available_actions_only_allows_end_after_transformations_exhausted():
    assert available_actions([], use_global_constraints=True) == ["select_row", "select_column", "group_by", "sort_by"]
    assert available_actions(
        ["select_row(0)", 'select_column("Name")', 'group_by("Name")', 'sort_by("Points", "desc")'],
        use_global_constraints=True,
    ) == ["end"]


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


def test_action_selection_grammar_outputs_old_action_names():
    builder = ActionGrammarBuilder()
    spec = builder.build_spec(
        table=make_table(),
        action_history=[],
        use_global_constraints=False,
        phase="action",
    )
    grammar = builder.build_action_grammar(spec)

    assert 'root ::= "f_select_row"' in grammar
    assert '"f_select_column"' in grammar
    assert '"f_end"' not in grammar


def test_structured_parser_uses_code_style_actions():
    table = make_table()

    row_action = StructuredActionParser.parse_arguments('["row 0", "row 2"]', "select_row", table)
    assert row_action.to_string() == "select_row(0, 2)"

    column_action = StructuredActionParser.parse_single_step('f_select_column(["Name"])', table)
    assert column_action.to_string() == "select_column('Name')"

    invalid = StructuredActionParser.parse_arguments('["row 999"]', "select_row", table)
    assert invalid is None


def test_structured_params_factory_uses_grammar():
    kwargs = StructuredSamplingParamsFactory.structured_outputs_kwargs('root ::= "f_end"\n')
    params = kwargs.get("structured_outputs")
    if params is not None:
        assert params.grammar == 'root ::= "f_end"\n'
    else:
        assert kwargs == {"guided_grammar": 'root ::= "f_end"\n'}
