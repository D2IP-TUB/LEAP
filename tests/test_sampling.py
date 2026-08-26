import asyncio
from types import SimpleNamespace

from leap.core import Table
from leap.core.actions import REGISTRY
from leap.core.actions.action import Action
from leap.generation.prompt_builder import CotPromptSettings, PromptBuilder
from leap.generation.sampling import AddColumnDiagnostic, SamplingConfig, SamplingLayer
from leap.generation.shuffle_invariant_sampling import ShuffleInvariantSamplingLayer


def test_add_column_argument_cleaning_removes_echoed_function_call():
    raw_args = "f_add_column(display type, ['monochrome', 'color'])"

    cleaned = SamplingLayer._clean_argument_text(raw_args, "add_column")
    action = Action.parse(f"add_column({cleaned})")

    assert cleaned == "display type, ['monochrome', 'color']"
    assert action is not None
    assert action.name == "add_column"
    assert action.arguments == ("display type", ["monochrome", "color"])


def test_add_column_token_budget_scales_with_rows_and_remaining_context():
    tokenizer = SimpleNamespace(encode=lambda prompt, add_special_tokens: list(range(len(prompt))))
    worker = SimpleNamespace(tokenizer=tokenizer, max_model_len=10_000)

    assert SamplingLayer._add_column_max_tokens(Table(columns=["Name"], rows=[["x"]] * 8), "", worker) == 1024
    assert SamplingLayer._add_column_max_tokens(Table(columns=["Name"], rows=[["x"]] * 38), "", worker) == 2752

    constrained_worker = SimpleNamespace(tokenizer=tokenizer, max_model_len=700)
    assert SamplingLayer._add_column_max_tokens(Table(columns=["Name"], rows=[["x"]] * 38), "x" * 100, constrained_worker) == 472


def test_global_transition_policy_controls_sampling_without_decoding_constraints():
    previous_enabled = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["add_column", "select_row", "select_column", "group_by", "sort_by", "end"])
    try:
        layer = SamplingLayer(SamplingConfig())
        worker = SimpleNamespace(
            use_constraints=False,
            constraint_backend="legacy_state_machine",
            use_global_constraints=True,
        )
        initial_actions = layer._get_available_actions([], worker)
        actions_after_row = layer._get_available_actions(["select_row([row 0])"], worker)
    finally:
        REGISTRY._enabled_actions = previous_enabled

    assert initial_actions == ["add_column", "select_row", "select_column", "group_by", "sort_by"]
    assert actions_after_row == ["select_column", "group_by", "sort_by", "end"]


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


def test_add_column_batches_use_existing_argument_character_budget():
    table = Table(columns=["Name"], rows=[["A" * 20], ["B" * 20], ["C" * 20]])
    builder = PromptBuilder(
        tokenizer=None,
        is_instruct=False,
        cot_settings=CotPromptSettings(args_table_chars=45),
    )

    batches = builder.split_add_column_batches(table)

    assert [len(batch.rows) for batch in batches] == [1, 1, 1]
    assert [batch.rows[0][0] for batch in batches] == ["A" * 20, "B" * 20, "C" * 20]


def test_truncated_add_column_batching_is_opt_in():
    table = Table(columns=["Name"], rows=[["A" * 20], ["B" * 20], ["C" * 20]])
    builder = PromptBuilder(
        tokenizer=None,
        is_instruct=False,
        cot_settings=CotPromptSettings(args_table_chars=45),
    )

    default_batches = SamplingLayer._get_add_column_batches(builder, table, SimpleNamespace())
    opted_in_batches = SamplingLayer._get_add_column_batches(
        builder,
        table,
        SimpleNamespace(batch_truncated_add_column=True),
    )

    assert default_batches == [table]
    assert [len(batch.rows) for batch in opted_in_batches] == [1, 1, 1]


def test_add_column_batches_are_collected_into_one_action():
    class Engine:
        def generate(self, _prompt, _params, _request_id):
            async def results():
                yield SimpleNamespace(outputs=[SimpleNamespace(text='["second, value", "third"]')])

            return results()

    full_table = Table(columns=["Name"], rows=[["A"], ["B"], ["C"]])
    batches = [
        Table(columns=["Name"], rows=[["A"]]),
        Table(columns=["Name"], rows=[["B"], ["C"]]),
    ]
    worker = SimpleNamespace(
        engine=Engine(),
        tokenizer=SimpleNamespace(eos_token_id=0),
        output_format="function",
        use_constraints=False,
        effective_temperature=lambda value: value,
    )
    builder = PromptBuilder(tokenizer=None, is_instruct=False)
    layer = SamplingLayer(SamplingConfig())

    result = asyncio.run(
        layer._complete_add_column_batches(
            action=Action("add_column", ["Derived", ["first"]]),
            full_table=full_table,
            batches=batches,
            question="Derive values",
            prompt_builder=builder,
            worker=worker,
            step_id="test",
        )
    )

    assert result == Action("add_column", ["Derived", ["first", "second, value", "third"]])


def test_add_column_argument_failures_capture_raw_output_and_parse_reason():
    class Engine:
        def generate(self, _prompt, _params, _request_id):
            async def results():
                yield SimpleNamespace(
                    outputs=[SimpleNamespace(text='{"column":"Derived","values":["a"', finish_reason="length", token_ids=[1, 2])]
                )

            return results()

    previous_enabled = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["add_column", "end"])
    try:
        layer = SamplingLayer(SamplingConfig())
        diagnostics = []
        candidates = asyncio.run(
            layer.generate_arguments(
                worker=SimpleNamespace(
                    engine=Engine(),
                    tokenizer=SimpleNamespace(eos_token_id=0),
                    output_format="json",
                    use_constraints=False,
                    batch_truncated_add_column=False,
                    max_model_len=32_000,
                    effective_temperature=lambda value: value,
                ),
                action_name="add_column",
                n=1,
                table=Table(columns=["Name"], rows=[["Ada"]]),
                action_history=[],
                request_id="request",
                step=0,
                temperature=0.7,
                state_machines=None,
                prompt_builder=PromptBuilder(tokenizer=None, is_instruct=False, output_format="json"),
                question="Who?",
                diagnostics=diagnostics,
            )
        )
    finally:
        REGISTRY._enabled_actions = previous_enabled

    assert candidates == []
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.failure_code == "invalid_json"
    assert diagnostic.parse_stage == "json_decode"
    assert diagnostic.finish_reason == "length"
    assert diagnostic.output_token_count == 2
    assert diagnostic.raw_output == '{"column":"Derived","values":["a"'


def test_add_column_empty_output_records_diagnostic():
    class Engine:
        def generate(self, _prompt, _params, _request_id):
            async def results():
                yield SimpleNamespace(outputs=[SimpleNamespace(text="", finish_reason="stop", token_ids=[])])

            return results()

    previous_enabled = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["add_column", "end"])
    try:
        diagnostics = []
        candidates = asyncio.run(
            SamplingLayer(SamplingConfig()).generate_arguments(
                worker=SimpleNamespace(
                    engine=Engine(),
                    tokenizer=SimpleNamespace(eos_token_id=0),
                    output_format="json",
                    use_constraints=False,
                    batch_truncated_add_column=False,
                    max_model_len=32_000,
                    effective_temperature=lambda value: value,
                ),
                action_name="add_column",
                n=1,
                table=Table(columns=["Name"], rows=[["Ada"]]),
                action_history=[],
                request_id="request",
                step=0,
                temperature=0.7,
                state_machines=None,
                prompt_builder=PromptBuilder(tokenizer=None, is_instruct=False, output_format="json"),
                question="Who?",
                diagnostics=diagnostics,
            )
        )
    finally:
        REGISTRY._enabled_actions = previous_enabled

    assert candidates == []
    assert diagnostics[0].failure_code == "empty_output"
    assert diagnostics[0].finish_reason == "stop"
    assert diagnostics[0].raw_output == ""


def test_add_column_argument_exceptions_capture_engine_failure():
    class Engine:
        def generate(self, _prompt, _params, _request_id):
            raise RuntimeError("engine unavailable")

    previous_enabled = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["add_column", "end"])
    try:
        layer = SamplingLayer(SamplingConfig())
        diagnostics = []
        candidates = asyncio.run(
            layer.generate_arguments(
                worker=SimpleNamespace(
                    engine=Engine(),
                    tokenizer=SimpleNamespace(eos_token_id=0),
                    output_format="json",
                    use_constraints=False,
                    batch_truncated_add_column=False,
                    max_model_len=32_000,
                    effective_temperature=lambda value: value,
                ),
                action_name="add_column",
                n=1,
                table=Table(columns=["Name"], rows=[["Ada"]]),
                action_history=[],
                request_id="request",
                step=0,
                temperature=0.7,
                state_machines=None,
                prompt_builder=PromptBuilder(tokenizer=None, is_instruct=False, output_format="json"),
                question="Who?",
                diagnostics=diagnostics,
            )
        )
    finally:
        REGISTRY._enabled_actions = previous_enabled

    assert candidates == []
    assert diagnostics[0].failure_code == "generation_exception"
    assert diagnostics[0].engine_exception == "RuntimeError: engine unavailable"


def test_add_column_batch_failure_records_batch_diagnostics():
    class Engine:
        def generate(self, _prompt, _params, _request_id):
            async def results():
                yield SimpleNamespace(outputs=[SimpleNamespace(text='["unterminated"', finish_reason="length", token_ids=[1])])

            return results()

    full_table = Table(columns=["Name"], rows=[["A"], ["B"]])
    diagnostic = AddColumnDiagnostic(
        request_id="request",
        step=0,
        sample_idx=0,
        selected_action="add_column",
        table_row_count=2,
        table_column_count=1,
        batch_enabled=True,
    )
    result = asyncio.run(
        SamplingLayer(SamplingConfig())._complete_add_column_batches(
            action=Action("add_column", ["Derived", ["first"]]),
            full_table=full_table,
            batches=[Table(columns=["Name"], rows=[["A"]]), Table(columns=["Name"], rows=[["B"]])],
            question="Derive values",
            prompt_builder=PromptBuilder(tokenizer=None, is_instruct=False),
            worker=SimpleNamespace(
                engine=Engine(),
                tokenizer=SimpleNamespace(eos_token_id=0),
                output_format="function",
                use_constraints=False,
                effective_temperature=lambda value: value,
            ),
            step_id="request_args_step0_sample0",
            diagnostic=diagnostic,
        )
    )

    assert result is None
    assert diagnostic.failure_code == "batch_invalid_json"
    assert diagnostic.batch_diagnostics[0].raw_output == '["unterminated"'
    assert diagnostic.batch_diagnostics[0].finish_reason == "length"
    assert diagnostic.batch_diagnostics[0].failure_code == "invalid_json"


def test_shuffle_sampling_restores_add_column_values_to_original_row_order():
    layer = ShuffleInvariantSamplingLayer(SamplingConfig())
    action = Action("add_column", ["Derived", ["for C", "for A", "for B"]])
    layer._candidate_sample_map[id(action)] = 1
    layer._current_table_id = 10
    layer._permutations[(10, 1)] = [2, 0, 1]

    mapped = layer._map_action_to_original(action)

    assert mapped == Action("add_column", ["Derived", ["for A", "for B", "for C"]])
