import asyncio
import subprocess
import sys

import pytest

from leap.core import Table
from leap.core.actions import REGISTRY
from leap.mcp.client import McpTableClient
from leap.mcp.protocol import McpToolCall
from leap.mcp.server import group_by, select_column, select_row, sort_by


@pytest.fixture(autouse=True)
def enabled_actions():
    previous = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["select_row", "select_column", "group_by", "sort_by", "end"])
    yield
    REGISTRY._enabled_actions = previous


@pytest.mark.parametrize(
    ("tool", "kwargs", "expected"),
    [
        (select_row, {"rows": ["row 1"]}, Table(columns=["Name", "Score"], rows=[["Grace", "1"]])),
        (select_column, {"columns": ["Name"]}, Table(columns=["Name"], rows=[["Ada"], ["Grace"]])),
        (group_by, {"column": "Score"}, Table(columns=["Group", "Score", "Count"], rows=[[1, "2", 1], [2, "1", 1]])),
        (sort_by, {"column": "Score", "order": "asc"}, Table(columns=["Name", "Score"], rows=[["Grace", "1"], ["Ada", "2"]])),
    ],
)
def test_mcp_tools_return_complete_table_states(tool, kwargs, expected):
    table = Table(columns=["Name", "Score"], rows=[["Ada", "2"], ["Grace", "1"]])

    payload = tool(table=table.to_dict(), **kwargs)

    assert Table.from_dict(payload["table"]) == expected
    assert payload["terminated"] is False


def test_server_module_does_not_import_inference_or_ml_runtimes():
    command = [
        sys.executable,
        "-c",
        (
            "import runpy, sys; "
            "runpy.run_module('leap.mcp.server', run_name='leap_mcp_import_check'); "
            "assert 'leap.inference' not in sys.modules; "
            "assert 'torch' not in sys.modules; "
            "assert 'vllm' not in sys.modules"
        ),
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)


def test_stdio_mcp_session_reuses_server_for_multiple_table_operations():
    original = Table(
        columns=["Name", "Score"],
        rows=[["Ada", "2"], ["Grace", "1"]],
    )
    client = McpTableClient()

    async def scenario():
        async with client:
            selected = await client.apply(
                McpToolCall(
                    request_id="e2e-1",
                    name="select_column",
                    arguments={"columns": ["Name"]},
                ),
                original,
            )
            ended = await client.apply(McpToolCall(request_id="e2e-2", name="end", arguments={}), selected.table)
            assert client.is_connected is True
            return selected, ended

    selected, ended = asyncio.run(scenario())

    assert selected.action.to_string() == "select_column('Name')"
    assert selected.table == Table(columns=["Name"], rows=[["Ada"], ["Grace"]])
    assert selected.terminated is False
    assert selected.timings.startup_seconds > 0
    assert ended.action.to_string() == "end()"
    assert ended.table == selected.table
    assert ended.terminated is True
    assert ended.timings.startup_seconds == 0
    assert client.is_connected is False
    assert original.columns == ("Name", "Score")
