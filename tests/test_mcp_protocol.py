import json

import pytest

from leap.core import Action, Table
from leap.core.actions import REGISTRY
from leap.mcp.protocol import McpToolCallCodec, McpToolCallSchemaBuilder


@pytest.fixture(autouse=True)
def enabled_actions():
    previous = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["select_row", "select_column", "group_by", "sort_by", "end"])
    yield
    REGISTRY._enabled_actions = previous


@pytest.fixture
def table():
    return Table(columns=["Name", "Score"], rows=[["Ada", "2"], ["Grace", "1"]])


def test_mcp_codec_parses_official_tools_call_envelope(table):
    text = McpToolCallCodec.dumps_call("select_column", {"columns": ["Name"]}, request_id="request-1")

    call = McpToolCallCodec.parse(text)
    action = McpToolCallCodec.parse_action(text, table)

    assert call is not None
    assert call.request_id == "request-1"
    assert call.name == "select_column"
    assert action == Action("select_column", ["Name"])


@pytest.mark.parametrize(
    "payload",
    [
        {"jsonrpc": "1.0", "id": "1", "method": "tools/call", "params": {"name": "end", "arguments": {}}},
        {"jsonrpc": "2.0", "id": "1", "method": "call_tool", "params": {"name": "end", "arguments": {}}},
        {"jsonrpc": "2.0", "id": "1", "method": "tools/call", "params": {"name": "add_column", "arguments": {}}},
        {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/call",
            "params": {"name": "end", "arguments": {}},
            "extra": True,
        },
    ],
)
def test_mcp_codec_rejects_non_protocol_or_unsupported_requests(payload):
    assert McpToolCallCodec.parse(json.dumps(payload)) is None


def test_cot_action_selection_requires_empty_arguments():
    selection = McpToolCallCodec.dumps_call("sort_by", {})
    complete = McpToolCallCodec.dumps_call("sort_by", {"column": "Score", "order": "desc"})

    assert McpToolCallCodec.parse_action_name(selection, allowed_actions=["sort_by"]) == "sort_by"
    assert McpToolCallCodec.parse_action_name(complete, allowed_actions=["sort_by"]) is None


def test_mcp_schema_wraps_contextual_operation_schema(table):
    builder = McpToolCallSchemaBuilder()
    spec = builder.build_spec(
        table=table,
        action_history=[],
        use_global_constraints=False,
        phase="single_step",
    )

    schema = builder.build_single_step_schema(spec)
    branches = {branch["properties"]["params"]["properties"]["name"]["const"]: branch for branch in schema["anyOf"]}
    select_column = branches["select_column"]
    arguments = select_column["properties"]["params"]["properties"]["arguments"]

    assert "add_column" not in branches
    assert arguments["properties"]["columns"]["items"]["enum"] == ["Name", "Score"]
    assert select_column["properties"]["method"] == {"const": "tools/call"}
    assert select_column["additionalProperties"] is False
