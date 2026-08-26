"""Model-facing MCP tool-call envelopes and constrained-output schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from leap.core import Action, Table
from leap.inference.json_constraints import ActionParseInspection, JsonActionCodec, JsonActionSchemaBuilder, JsonActionSchemaSpec

MCP_ACTIONS = ("select_row", "select_column", "add_column", "group_by", "sort_by", "end")


@dataclass(frozen=True)
class McpToolCall:
    request_id: str | int
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class McpToolCallInspection:
    """Detailed outcome of parsing an MCP ``tools/call`` envelope."""

    call: McpToolCall | None
    stage: str
    failure_code: str | None = None
    failure_reason: str | None = None


class McpToolCallCodec:
    """Strict conversion between MCP ``tools/call`` requests and LEAP actions."""

    @classmethod
    def parse(cls, text: str) -> McpToolCall | None:
        return cls.inspect(text).call

    @classmethod
    def inspect(cls, text: str) -> McpToolCallInspection:
        try:
            payload = json.loads(text.strip())
        except (json.JSONDecodeError, TypeError) as error:
            return McpToolCallInspection(None, "mcp_json_decode", "invalid_json", str(error))
        expected_fields = {"jsonrpc", "id", "method", "params"}
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            return McpToolCallInspection(None, "mcp_envelope", "invalid_envelope_fields", "Invalid MCP envelope fields.")
        if payload["jsonrpc"] != "2.0":
            return McpToolCallInspection(None, "mcp_envelope", "invalid_jsonrpc_version", "jsonrpc must be '2.0'.")
        if payload["method"] != "tools/call":
            return McpToolCallInspection(None, "mcp_envelope", "invalid_method", "method must be 'tools/call'.")
        if not isinstance(payload["id"], (str, int)) or isinstance(payload["id"], bool):
            return McpToolCallInspection(None, "mcp_envelope", "invalid_request_id", "id must be a string or integer.")
        params = payload["params"]
        if not isinstance(params, dict) or set(params) != {"name", "arguments"}:
            return McpToolCallInspection(None, "mcp_params", "invalid_params_fields", "Invalid MCP params fields.")
        if params["name"] not in MCP_ACTIONS:
            return McpToolCallInspection(None, "mcp_params", "unsupported_tool", "Unsupported MCP tool name.")
        if not isinstance(params["arguments"], dict):
            return McpToolCallInspection(None, "mcp_params", "invalid_arguments_type", "arguments must be an object.")
        return McpToolCallInspection(McpToolCall(payload["id"], params["name"], params["arguments"]), "complete")

    @classmethod
    def parse_action_name(cls, text: str, *, allowed_actions: list[str] | tuple[str, ...] | None = None) -> str | None:
        call = cls.parse(text)
        if call is None or call.arguments:
            return None
        if allowed_actions is not None and call.name not in allowed_actions:
            return None
        return call.name

    @classmethod
    def parse_action(cls, text: str, table: Table, *, expected_action: str | None = None) -> Action | None:
        return cls.inspect_action(text, table, expected_action=expected_action).action

    @classmethod
    def inspect_action(
        cls,
        text: str,
        table: Table,
        *,
        expected_action: str | None = None,
    ) -> ActionParseInspection:
        inspection = cls.inspect(text)
        call = inspection.call
        if call is None:
            return ActionParseInspection(
                action=None,
                stage=inspection.stage,
                failure_code=inspection.failure_code,
                failure_reason=inspection.failure_reason,
            )
        if expected_action is not None and call.name != expected_action:
            return ActionParseInspection(
                action=None,
                stage="action_name",
                failure_code="unexpected_action",
                failure_reason=f"Expected action {expected_action!r}, got {call.name!r}",
            )
        return JsonActionCodec.inspect_payload({"action": call.name, **call.arguments}, table=table)

    @classmethod
    def dumps(cls, action: Action, *, request_id: str | int = "step", include_arguments: bool = True) -> str:
        arguments = JsonActionCodec.to_dict(action, include_action=False) if include_arguments else {}
        return cls.dumps_call(action.name, arguments, request_id=request_id)

    @staticmethod
    def dumps_call(name: str, arguments: dict[str, Any], *, request_id: str | int = "step") -> str:
        return json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


class McpToolCallSchemaBuilder:
    """Wrap existing contextual action schemas in an MCP JSON-RPC envelope."""

    def __init__(self) -> None:
        self._json = JsonActionSchemaBuilder()

    def build_spec(self, **kwargs) -> JsonActionSchemaSpec:
        spec = self._json.build_spec(**kwargs)
        allowed = tuple(name for name in spec.allowed_actions if name in MCP_ACTIONS)
        return JsonActionSchemaSpec(
            phase=spec.phase,
            allowed_actions=allowed,
            selected_action=spec.selected_action,
            rows=spec.rows,
            columns=spec.columns,
            table_row_count=spec.table_row_count,
        )

    def build_action_schema(self, spec: JsonActionSchemaSpec) -> dict[str, Any]:
        branches = [self._envelope(name, self._empty_object()) for name in spec.allowed_actions]
        return {"anyOf": branches}

    def build_arguments_schema(self, spec: JsonActionSchemaSpec) -> dict[str, Any]:
        if not spec.selected_action:
            raise ValueError("Argument schema requires selected_action.")
        arguments = self._json.build_arguments_schema(spec)
        return self._envelope(spec.selected_action, arguments)

    def build_single_step_schema(self, spec: JsonActionSchemaSpec) -> dict[str, Any]:
        action_schema = self._json.build_single_step_schema(spec)
        branches = []
        for action, schema in zip(spec.allowed_actions, action_schema["anyOf"]):
            properties = dict(schema["properties"])
            properties.pop("action", None)
            required = [name for name in schema["required"] if name != "action"]
            arguments = self._object(properties, required)
            branches.append(self._envelope(action, arguments))
        return {"anyOf": branches}

    @classmethod
    def _envelope(cls, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        params = cls._object(
            {"name": {"const": name}, "arguments": arguments},
            ["name", "arguments"],
        )
        return cls._object(
            {
                "jsonrpc": {"const": "2.0"},
                "id": {"type": "string"},
                "method": {"const": "tools/call"},
                "params": params,
            },
            ["jsonrpc", "id", "method", "params"],
        )

    @classmethod
    def _empty_object(cls) -> dict[str, Any]:
        return cls._object({}, [])

    @staticmethod
    def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def uses_mcp_operations(worker) -> bool:
    return getattr(worker, "output_format", "function") == "mcp"


def uses_mcp_schema(worker) -> bool:
    return uses_mcp_operations(worker) and getattr(worker, "use_constraints", False)
