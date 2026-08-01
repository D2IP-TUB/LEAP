"""JSON operation schemas, parsing, and state for constrained decoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from leap.core import Action, Table
from leap.core.actions import REGISTRY

JsonPhase = Literal["action", "arguments", "single_step"]
ROW_LIMIT = 500


def uses_json_operations(worker) -> bool:
    return getattr(worker, "output_format", "function") == "json"


def uses_json_schema(worker) -> bool:
    return uses_json_operations(worker) and getattr(worker, "use_constraints", False)


def available_json_actions(
    action_history: list[str] | tuple[str, ...] | None,
    *,
    use_global_constraints: bool,
) -> list[str]:
    history = list(action_history or [])
    enabled = REGISTRY.get_enabled_names()
    if use_global_constraints:
        return REGISTRY.get_global_available_actions(history)
    if history:
        used = REGISTRY._extract_action_names_from_history(history)
        enabled = [name for name in enabled if name not in used or name == "end"]
    if not history:
        enabled = [name for name in enabled if not REGISTRY.get(name).is_terminating]
    return enabled


class JsonActionCodec:
    """Strict conversion between the public JSON protocol and internal Actions."""

    _fields = {
        "select_row": {"action", "rows"},
        "select_column": {"action", "columns"},
        "group_by": {"action", "column"},
        "sort_by": {"action", "column", "order"},
        "add_column": {"action", "column", "values"},
        "end": {"action"},
    }
    _argument_fields = {name: fields - {"action"} for name, fields in _fields.items()}

    @classmethod
    def parse_single_step(cls, text: str, table: Table | None = None) -> Action | None:
        payload = cls._load_object(text)
        return cls._parse_payload(payload, table=table) if payload is not None else None

    @classmethod
    def parse_action_name(
        cls,
        text: str,
        *,
        allowed_actions: list[str] | tuple[str, ...] | None = None,
    ) -> str | None:
        payload = cls._load_object(text)
        if payload is None or set(payload) != {"action"} or not isinstance(payload["action"], str):
            return None
        name = payload["action"]
        if REGISTRY.get(name) is None or not REGISTRY.is_enabled(name):
            return None
        if allowed_actions is not None and name not in allowed_actions:
            return None
        return name

    @classmethod
    def parse_arguments(cls, text: str, action_name: str, table: Table | None = None) -> Action | None:
        payload = cls._load_object(text)
        if payload is None or set(payload) != cls._argument_fields.get(action_name):
            return None
        return cls._parse_payload({"action": action_name, **payload}, table=table)

    @classmethod
    def to_dict(cls, action: Action, *, include_action: bool = True) -> dict[str, Any]:
        args = list(action.arguments)
        if action.name == "select_row":
            values = ["*"] if args == ["*"] else [f"row {idx}" for idx in args]
            payload = {"rows": values}
        elif action.name == "select_column":
            payload = {"columns": args}
        elif action.name == "group_by":
            payload = {"column": args[0]}
        elif action.name == "sort_by":
            payload = {"column": args[0], "order": args[1]}
        elif action.name == "add_column":
            payload = {"column": args[0], "values": list(args[1])}
        elif action.name == "end":
            payload = {}
        else:
            raise ValueError(f"Unsupported JSON action: {action.name!r}")
        return {"action": action.name, **payload} if include_action else payload

    @classmethod
    def dumps(cls, action: Action, *, include_action: bool = True) -> str:
        return json.dumps(cls.to_dict(action, include_action=include_action), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def history_json(cls, action_history: list[str] | tuple[str, ...]) -> str:
        actions = [Action.parse(value) for value in action_history]
        payloads = [cls.to_dict(action) for action in actions if action is not None]
        return json.dumps(payloads, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _load_object(text: str) -> dict[str, Any] | None:
        try:
            value = json.loads(text.strip())
        except (json.JSONDecodeError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    @classmethod
    def _parse_payload(cls, payload: dict[str, Any], *, table: Table | None) -> Action | None:
        name = payload.get("action")
        if not isinstance(name, str) or set(payload) != cls._fields.get(name):
            return None
        if REGISTRY.get(name) is None or not REGISTRY.is_enabled(name):
            return None
        try:
            if name == "select_row":
                rows = payload["rows"]
                if not isinstance(rows, list) or not rows or not all(isinstance(row, str) for row in rows):
                    return None
                if len(rows) != len(set(rows)):
                    return None
                if rows == ["*"]:
                    arguments = ["*"]
                elif "*" in rows:
                    return None
                else:
                    arguments = []
                    for row in rows:
                        prefix, separator, index = row.partition(" ")
                        if prefix != "row" or separator != " " or not index.isdigit():
                            return None
                        arguments.append(int(index))
            elif name == "select_column":
                arguments = payload["columns"]
                if not isinstance(arguments, list) or not arguments or not all(isinstance(value, str) for value in arguments):
                    return None
                if len(arguments) != len(set(arguments)):
                    return None
            elif name == "group_by":
                if not isinstance(payload["column"], str):
                    return None
                arguments = [payload["column"]]
            elif name == "sort_by":
                if not isinstance(payload["column"], str) or payload["order"] not in {"asc", "desc"}:
                    return None
                arguments = [payload["column"], payload["order"]]
            elif name == "add_column":
                if not isinstance(payload["column"], str) or not payload["column"]:
                    return None
                values = payload["values"]
                if not isinstance(values, list) or not values or any(value is None or isinstance(value, (dict, list)) for value in values):
                    return None
                arguments = [payload["column"], [str(value) for value in values]]
            else:
                arguments = []
            action = Action(name, arguments)
            return action if table is None or action.is_valid_for_table(table) else None
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class JsonActionSchemaSpec:
    phase: JsonPhase
    allowed_actions: tuple[str, ...]
    selected_action: str | None
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    table_row_count: int


class JsonActionSchemaBuilder:
    def build_spec(
        self,
        *,
        table: Table,
        action_history: list[str] | tuple[str, ...] | None,
        use_global_constraints: bool,
        phase: JsonPhase,
        selected_action: str | None = None,
    ) -> JsonActionSchemaSpec:
        allowed = available_json_actions(action_history, use_global_constraints=use_global_constraints)
        if not table.rows:
            allowed = [name for name in allowed if name not in {"select_row", "add_column"}]
        if not table.columns:
            allowed = [name for name in allowed if name not in {"select_column", "group_by", "sort_by"}]
        if not allowed and REGISTRY.is_enabled("end"):
            allowed = ["end"]
        if selected_action is not None and selected_action not in REGISTRY.get_enabled_names():
            raise ValueError(f"Action {selected_action!r} is not enabled.")
        return JsonActionSchemaSpec(
            phase=phase,
            allowed_actions=tuple(allowed),
            selected_action=selected_action,
            rows=tuple(f"row {idx}" for idx in range(min(ROW_LIMIT, len(table.rows)))),
            columns=tuple(str(column) for column in table.columns),
            table_row_count=len(table.rows),
        )

    def build_action_schema(self, spec: JsonActionSchemaSpec) -> dict[str, Any]:
        if spec.phase != "action":
            raise ValueError("Action schema requires phase='action'.")
        return self._object({"action": {"type": "string", "enum": list(spec.allowed_actions)}}, ["action"])

    def build_arguments_schema(self, spec: JsonActionSchemaSpec) -> dict[str, Any]:
        if spec.phase != "arguments" or not spec.selected_action:
            raise ValueError("Argument schema requires phase='arguments' and selected_action.")
        return self._schema_for_action(spec.selected_action, spec, include_action=False)

    def build_single_step_schema(self, spec: JsonActionSchemaSpec) -> dict[str, Any]:
        if spec.phase != "single_step":
            raise ValueError("Single-step schema requires phase='single_step'.")
        # llguidance does not implement oneOf. The action const makes these
        # anyOf branches mutually exclusive in practice.
        return {"anyOf": [self._schema_for_action(action, spec, include_action=True) for action in spec.allowed_actions]}

    def _schema_for_action(self, action: str, spec: JsonActionSchemaSpec, *, include_action: bool) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        if include_action:
            properties["action"] = {"const": action}
            required.append("action")
        if action == "select_row":
            # xgrammar/vLLM does not implement JSON Schema's uniqueItems.
            # Duplicate values are rejected by JsonActionCodec after decoding.
            regular = {"type": "array", "items": {"type": "string", "enum": list(spec.rows)}, "minItems": 1}
            wildcard = {"type": "array", "prefixItems": [{"const": "*"}], "minItems": 1, "maxItems": 1}
            properties["rows"] = {"anyOf": [regular, wildcard]} if spec.rows else regular
            required.append("rows")
        elif action == "select_column":
            properties["columns"] = self._enum_array(spec.columns)
            required.append("columns")
        elif action in {"group_by", "sort_by"}:
            properties["column"] = {"type": "string", "enum": list(spec.columns)}
            required.append("column")
            if action == "sort_by":
                properties["order"] = {"type": "string", "enum": ["asc", "desc"]}
                required.append("order")
        elif action == "add_column":
            # llguidance does not implement `not`; Action validation rejects
            # an existing column name after decoding.
            properties["column"] = {"type": "string", "minLength": 1}
            values: dict[str, Any] = {"type": "array", "items": {"type": ["string", "number", "boolean"]}, "minItems": 1}
            if spec.table_row_count:
                values["maxItems"] = spec.table_row_count
                if spec.phase == "single_step":
                    values["minItems"] = spec.table_row_count
            properties["values"] = values
            required.extend(["column", "values"])
        elif action != "end":
            raise ValueError(f"Unsupported JSON action: {action!r}")
        return self._object(properties, required)

    @staticmethod
    def _enum_array(values: tuple[str, ...]) -> dict[str, Any]:
        # Keep this schema within xgrammar's supported subset. The codec
        # enforces uniqueness after structured decoding.
        return {"type": "array", "items": {"type": "string", "enum": list(values)}, "minItems": 1}

    @staticmethod
    def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


class JsonSchemaSamplingParamsFactory:
    @staticmethod
    def structured_outputs_kwargs(schema: dict[str, Any]) -> dict[str, Any]:
        from vllm.sampling_params import StructuredOutputsParams

        return {
            "structured_outputs": StructuredOutputsParams(
                json=schema,
                disable_additional_properties=True,
            )
        }
