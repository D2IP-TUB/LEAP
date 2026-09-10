import json
from types import SimpleNamespace

import pytest

from leap.core import Action, Table
from leap.core.actions import REGISTRY
from leap.generation.prompt_builder import PromptBuilder
from leap.inference.json_constraints import JsonActionCodec, JsonActionSchemaBuilder


@pytest.fixture(autouse=True)
def enabled_actions():
    previous = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["select_row", "select_column", "add_column", "group_by", "sort_by", "end"])
    yield
    REGISTRY._enabled_actions = previous


@pytest.fixture
def table():
    return Table(columns=["Name", 'Year "reported"'], rows=[["Ada", "2020"], ["Grace", "2021"]])


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"action": "select_row", "rows": ["row 0", "row 1"]}, Action("select_row", [0, 1])),
        ({"action": "select_row", "rows": ["*"]}, Action("select_row", ["*"])),
        ({"action": "select_column", "columns": ["Name"]}, Action("select_column", ["Name"])),
        ({"action": "group_by", "column": "Name"}, Action("group_by", ["Name"])),
        ({"action": "sort_by", "column": "Name", "order": "desc"}, Action("sort_by", ["Name", "desc"])),
        ({"action": "add_column", "column": "Rank", "values": ["1", "2"]}, Action("add_column", ["Rank", ["1", "2"]])),
        ({"action": "end"}, Action("end", [])),
    ],
)
def test_json_codec_round_trip(payload, expected, table):
    action = JsonActionCodec.parse_single_step(json.dumps(payload), table)
    assert action == expected
    assert JsonActionCodec.parse_single_step(JsonActionCodec.dumps(action), table) == expected


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"action":"end"}\n```',
        '{"action":"end"} trailing',
        '{"action":"end","args":[]}',
        '{"action":"select_row","rows":["*","row 0"]}',
        '{"action":"select_row","rows":["row 0","row 0"]}',
        '{"action":"select_column","columns":["Name","Name"]}',
        '{"action":"sort_by","column":"Name","order":"ascending"}',
        '[{"action":"end"}]',
        '{"action":"add_column","column":"Rank","values":[1,2]}',
    ],
)
def test_json_codec_is_strict(text, table):
    assert JsonActionCodec.parse_single_step(text, table) is None


def test_json_schema_is_contextual_and_disallows_extra_properties(table):
    builder = JsonActionSchemaBuilder()
    spec = builder.build_spec(table=table, action_history=[], use_global_constraints=False, phase="single_step")
    schema = builder.build_single_step_schema(spec)
    by_action = {item["properties"]["action"]["const"]: item for item in schema["anyOf"]}
    assert "end" not in by_action
    assert by_action["select_column"]["properties"]["columns"]["items"]["enum"] == ["Name", 'Year "reported"']
    assert by_action["select_row"]["properties"]["rows"]["anyOf"][1]["maxItems"] == 1
    assert "uniqueItems" not in json.dumps(schema)
    assert all(item["additionalProperties"] is False for item in schema["anyOf"])
    assert by_action["add_column"]["properties"]["values"]["minItems"] == len(table.rows)


def test_cot_argument_schema_requires_complete_add_column(table):
    builder = JsonActionSchemaBuilder()
    spec = builder.build_spec(table=table, action_history=[], use_global_constraints=False, phase="arguments", selected_action="add_column")
    values = builder.build_arguments_schema(spec)["properties"]["values"]
    assert values["minItems"] == len(table.rows)
    assert values["maxItems"] == len(table.rows)
    assert values["items"] == {"type": "string", "maxLength": 256}
    assert builder.build_arguments_schema(spec)["properties"]["column"]["maxLength"] == 128


def test_add_column_renames_duplicate_column_header():
    table = Table(columns=["Name", "Name extracted"], rows=[["Ada"], ["Grace"]])
    action = JsonActionCodec.parse_single_step(
        json.dumps({"action": "add_column", "column": "Name", "values": ["1", "2"]}),
        table,
    )

    assert action == Action("add_column", ["Name extracted 2", ["1", "2"]])


def test_add_column_preserves_text_and_normalizes_whitespace():
    table = Table(columns=["Source"], rows=[["a"], ["b"], ["c"], ["d"]])
    payload = {
        "action": "add_column",
        "column": 'Derived\n"Name"',
        "values": ['The "Best" Award', r"C:\\data\\table", "München", "line\r\nbreak\tvalue"],
    }

    action = JsonActionCodec.parse_single_step(json.dumps(payload, ensure_ascii=False), table)

    assert action == Action(
        "add_column",
        ['Derived "Name"', ['The "Best" Award', r"C:\\data\\table", "München", "line break value"]],
    )


@pytest.mark.parametrize(
    ("payload", "failure_code"),
    [
        ({"action": "add_column", "column": "Derived", "values": ["\u0004", "ok"]}, "disallowed_control_character"),
        ({"action": "add_column", "column": "x" * 129, "values": ["a", "b"]}, "string_too_long"),
        ({"action": "add_column", "column": "Derived", "values": ["x" * 257, "b"]}, "string_too_long"),
        ({"action": "add_column", "column": "Derived", "values": ["a"]}, "incorrect_value_count"),
    ],
)
def test_add_column_inspection_reports_decoded_and_table_validation_failures(table, payload, failure_code):
    inspection = JsonActionCodec.inspect_single_step(json.dumps(payload), table)

    assert inspection.action is None
    assert inspection.failure_code == failure_code


def test_add_column_accepts_exact_string_length_limits(table):
    action = JsonActionCodec.parse_single_step(
        json.dumps({"action": "add_column", "column": "x" * 128, "values": ["y" * 256, "z"]}),
        table,
    )

    assert action is not None
    assert action.arguments[0] == "x" * 128
    assert action.arguments[1][0] == "y" * 256


def test_empty_table_schema_falls_back_to_end():
    builder = JsonActionSchemaBuilder()
    spec = builder.build_spec(table=Table(columns=[], rows=[]), action_history=[], use_global_constraints=False, phase="single_step")
    schema = builder.build_single_step_schema(spec)
    assert spec.allowed_actions == ("end",)
    assert schema["anyOf"][0]["properties"]["action"] == {"const": "end"}


class RecordingTokenizer:
    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "\n".join(message["content"] for message in messages)


def test_json_prompts_convert_all_operation_examples(table):
    tokenizer = RecordingTokenizer()
    builder = PromptBuilder(tokenizer=tokenizer, is_instruct=True, output_format="json")
    worker = SimpleNamespace(
        max_model_len=100_000,
        output_format="json",
        use_constraints=True,
        constraint_backend="legacy_state_machine",
        use_global_constraints=False,
    )
    iterative = builder.build_iterative_prompt(question="Who?", table=table, action_history=["select_row([row 0])"], worker=worker, step=1)
    assert "f_" not in iterative and "->" not in iterative
    assert all(isinstance(json.loads(m["content"]), list) for m in tokenizer.messages if m["role"] == "assistant")

    cot_action = builder.build_cot_action_prompt(question="Who?", table=table, action_history=[], worker=worker)
    assert "f_" not in cot_action and "->" not in cot_action
    assert all(set(json.loads(m["content"])) == {"action"} for m in tokenizer.messages if m["role"] == "assistant")

    cot_args = builder.build_cot_arguments_prompt(
        question="Who?", table=table, action_name="select_column", action_history=[], worker=worker
    )
    assert "f_" not in cot_args and "->" not in cot_args
    assert all(set(json.loads(m["content"])) == {"columns"} for m in tokenizer.messages if m["role"] == "assistant")
