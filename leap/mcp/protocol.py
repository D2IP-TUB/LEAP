"""Model-facing MCP tool-call envelopes and constrained-output schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from leap.core import Action, Table
from leap.inference.json_constraints import JsonActionCodec, JsonActionSchemaBuilder, JsonActionSchemaSpec

MCP_ACTIONS = ("select_row", "select_column", "group_by", "sort_by", "end")


@dataclass(frozen=True)
class McpToolCall:
    request_id: str | int
    name: str
    arguments: dict[str, Any]


class McpToolCallCodec:
    """Strict conversion between MCP ``tools/call`` requests and LEAP actions."""

    @classmethod
    def parse(cls, text: str) -> McpToolCall | None:
        try:
            payload = json.loads(text.strip())
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict) or set(payload) != {"jsonrpc", "id", "method", "params"}:
            return None
        if payload["jsonrpc"] != "2.0" or payload["method"] != "tools/call":
            return None
        if not isinstance(payload["id"], (str, int)) or isinstance(payload["id"], bool):
            return None
        params = payload["params"]
        if not isinstance(params, dict) or set(params) != {"name", "arguments"}:
            return None
        if params["name"] not in MCP_ACTIONS or not isinstance(params["arguments"], dict):
            return None
        return McpToolCall(payload["id"], params["name"], params["arguments"])

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
        call = cls.parse(text)
        if call is None or (expected_action is not None and call.name != expected_action):
            return None
        return JsonActionCodec._parse_payload({"action": call.name, **call.arguments}, table=table)

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
