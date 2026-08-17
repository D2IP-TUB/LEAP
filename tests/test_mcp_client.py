import asyncio
from types import SimpleNamespace

import pytest

from leap.core import Table
from leap.mcp.client import McpTableClient, McpToolError, McpTransportError
from leap.mcp.protocol import McpToolCall


class FakeSession:
    def __init__(self, *, transport_error: Exception | None = None, tool_error: bool = False):
        self.transport_error = transport_error
        self.tool_error = tool_error
        self.calls = 0
        self.active_calls = 0
        self.max_active_calls = 0

    async def call_tool(self, name, arguments):
        self.calls += 1
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            await asyncio.sleep(0)
            if self.transport_error is not None:
                raise self.transport_error
            if self.tool_error:
                return SimpleNamespace(
                    isError=True,
                    structuredContent=None,
                    content=[SimpleNamespace(type="text", text="invalid tool arguments")],
                )
            return SimpleNamespace(
                isError=False,
                structuredContent={"action": "end()", "table": arguments["table"], "terminated": True},
                content=[],
            )
        finally:
            self.active_calls -= 1


class RecordingClient(McpTableClient):
    def __init__(self, sessions):
        super().__init__()
        self.sessions = list(sessions)
        self.starts = 0
        self.closes = 0

    async def _ensure_session(self):
        if self._session is not None:
            return 0.0
        self._session = self.sessions[self.starts]
        self.starts += 1
        return 0.25

    async def _close_session(self):
        if self._session is not None:
            self.closes += 1
        self._session = None
        self._session_stack = None


def _call(request_id="1"):
    return McpToolCall(request_id=request_id, name="end", arguments={})


def _table():
    return Table(columns=["Name"], rows=[["Ada"]])


def test_client_requires_explicit_async_context():
    with pytest.raises(RuntimeError, match="async with"):
        asyncio.run(McpTableClient().apply(_call(), _table()))


def test_unused_client_never_initializes_a_session():
    client = RecordingClient([FakeSession()])

    async def scenario():
        async with client:
            assert client.is_connected is False

    asyncio.run(scenario())

    assert client.starts == 0
    assert client.closes == 0


def test_calls_reuse_one_session_and_are_serialized():
    session = FakeSession()
    client = RecordingClient([session])

    async def scenario():
        async with client:
            first, second = await asyncio.gather(client.apply(_call("1"), _table()), client.apply(_call("2"), _table()))
            assert first.timings.startup_seconds == 0.25
            assert second.timings.startup_seconds == 0.0

    asyncio.run(scenario())

    assert client.starts == 1
    assert client.closes == 1
    assert session.calls == 2
    assert session.max_active_calls == 1


def test_transport_failure_restarts_and_retries_once():
    failed = FakeSession(transport_error=RuntimeError("connection closed"))
    replacement = FakeSession()
    client = RecordingClient([failed, replacement])

    async def scenario():
        async with client:
            result = await client.apply(_call(), _table())
            assert result.terminated is True
            assert result.timings.startup_seconds == 0.5

    asyncio.run(scenario())

    assert client.starts == 2
    assert failed.calls == 1
    assert replacement.calls == 1


def test_tool_error_is_not_retried():
    rejected = FakeSession(tool_error=True)
    client = RecordingClient([rejected])

    async def scenario():
        async with client:
            with pytest.raises(McpToolError, match="invalid tool arguments"):
                await client.apply(_call(), _table())

    asyncio.run(scenario())

    assert client.starts == 1
    assert rejected.calls == 1


def test_second_transport_failure_is_propagated():
    first = FakeSession(transport_error=RuntimeError("first failure"))
    second = FakeSession(transport_error=RuntimeError("second failure"))
    client = RecordingClient([first, second])

    async def scenario():
        async with client:
            with pytest.raises(McpTransportError, match="transport failed"):
                await client.apply(_call(), _table())

    asyncio.run(scenario())

    assert client.starts == 2
    assert first.calls == 1
    assert second.calls == 1
