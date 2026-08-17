"""Persistent client adapter for the local table-transformation MCP server."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from leap.core import Action, Table
from leap.inference.json_constraints import JsonActionCodec

from .protocol import McpToolCall


class McpTransportError(RuntimeError):
    """The MCP subprocess or transport failed after one reconnect attempt."""


class McpToolError(ValueError):
    """The MCP server rejected a tool call or returned an invalid result."""


@dataclass(frozen=True)
class McpCallTimings:
    startup_seconds: float
    call_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class McpTableResult:
    action: Action
    table: Table
    terminated: bool
    timings: McpCallTimings


@dataclass
class _QueuedCall:
    call: McpToolCall
    table: Table
    future: asyncio.Future[McpTableResult]


_STOP = object()


class McpTableClient:
    """Serialize calls through one lazy, persistent MCP session.

    The broker task owns the SDK's stdio and session contexts from creation to
    cleanup. This is important because the stdio transport contains an AnyIO
    task group whose cancel scope must be exited by its owning task.
    """

    def __init__(self, server_parameters: StdioServerParameters | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.server_parameters = server_parameters or StdioServerParameters(
            command=sys.executable,
            args=["-m", "leap.mcp.server"],
            cwd=project_root,
        )
        self._queue: asyncio.Queue[_QueuedCall | object] | None = None
        self._broker_task: asyncio.Task[None] | None = None
        self._session_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._closing = False

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def __aenter__(self) -> "McpTableClient":
        if self._broker_task is not None:
            raise RuntimeError("McpTableClient is already active")
        self._queue = asyncio.Queue()
        self._closing = False
        self._broker_task = asyncio.create_task(self._run_broker(), name="leap-mcp-broker")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if self._broker_task is None or self._queue is None:
            return
        self._closing = True
        await self._queue.put(_STOP)
        try:
            await self._broker_task
        finally:
            self._broker_task = None
            self._queue = None
            self._closing = False

    async def apply(self, call: McpToolCall, table: Table) -> McpTableResult:
        if self._broker_task is None or self._queue is None or self._closing:
            raise RuntimeError("McpTableClient must be used inside 'async with'")
        total_start = time.perf_counter()
        future = asyncio.get_running_loop().create_future()
        await self._queue.put(_QueuedCall(call=call, table=table, future=future))
        result = await future
        return McpTableResult(
            action=result.action,
            table=result.table,
            terminated=result.terminated,
            timings=McpCallTimings(
                startup_seconds=result.timings.startup_seconds,
                call_seconds=result.timings.call_seconds,
                total_seconds=time.perf_counter() - total_start,
            ),
        )

    async def _run_broker(self) -> None:
        assert self._queue is not None
        try:
            while True:
                item = await self._queue.get()
                if item is _STOP:
                    break
                assert isinstance(item, _QueuedCall)
                if item.future.cancelled():
                    continue
                try:
                    result = await self._apply_with_reconnect(item.call, item.table)
                except Exception as exc:
                    if not item.future.done():
                        item.future.set_exception(exc)
                else:
                    if not item.future.done():
                        item.future.set_result(result)
        finally:
            await self._close_session()

    async def _apply_with_reconnect(self, call: McpToolCall, table: Table) -> McpTableResult:
        total_start = time.perf_counter()
        startup_seconds = 0.0
        call_seconds = 0.0
        last_error: Exception | None = None

        for attempt in range(2):
            try:
                startup_seconds += await self._ensure_session()
                assert self._session is not None
                call_start = time.perf_counter()
                try:
                    response = await self._session.call_tool(call.name, {"table": table.to_dict(), **call.arguments})
                finally:
                    call_seconds += time.perf_counter() - call_start
            except Exception as exc:
                last_error = exc
                await self._close_session()
                if attempt == 0:
                    continue
                raise McpTransportError(f"MCP transport failed while calling {call.name!r}") from exc

            action, result_table, terminated = self._parse_response(response, call, table)
            return McpTableResult(
                action=action,
                table=result_table,
                terminated=terminated,
                timings=McpCallTimings(
                    startup_seconds=startup_seconds,
                    call_seconds=call_seconds,
                    total_seconds=time.perf_counter() - total_start,
                ),
            )

        raise McpTransportError(f"MCP transport failed while calling {call.name!r}") from last_error

    async def _ensure_session(self) -> float:
        if self._session is not None:
            return 0.0

        start = time.perf_counter()
        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(stdio_client(self.server_parameters))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise

        self._session_stack = stack
        self._session = session
        return time.perf_counter() - start

    async def _close_session(self) -> None:
        stack = self._session_stack
        self._session = None
        self._session_stack = None
        if stack is not None:
            await stack.aclose()

    @classmethod
    def _parse_response(cls, result, call: McpToolCall, table: Table) -> tuple[Action, Table, bool]:
        if result.isError:
            message = cls._text_content(result.content) or f"MCP tool {call.name!r} failed"
            raise McpToolError(message)
        payload = result.structuredContent
        if payload is None:
            text = cls._text_content(result.content)
            try:
                payload = json.loads(text)
            except (json.JSONDecodeError, TypeError) as exc:
                raise McpToolError(f"MCP tool {call.name!r} returned no structured table state") from exc
        if not isinstance(payload, dict):
            raise McpToolError(f"MCP tool {call.name!r} returned an invalid payload")

        table_payload = payload.get("table")
        if not isinstance(table_payload, dict):
            raise McpToolError(f"MCP tool {call.name!r} omitted the table state")
        action = JsonActionCodec._parse_payload({"action": call.name, **call.arguments}, table=table)
        if action is None:
            raise McpToolError(f"MCP tool call contains invalid {call.name!r} arguments")
        returned_action = payload.get("action")
        if returned_action != action.to_string():
            raise McpToolError(f"MCP tool returned unexpected action {returned_action!r}")
        return action, Table.from_dict(table_payload), payload.get("terminated") is True

    @staticmethod
    def _text_content(content: list[Any]) -> str:
        return "\n".join(item.text for item in content if getattr(item, "type", None) == "text")
