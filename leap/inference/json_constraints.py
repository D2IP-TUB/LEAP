"""JSON operation schemas, parsing, and state for constrained decoding."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from leap.core import Action, Table
from leap.core.actions import REGISTRY

JsonPhase = Literal["action", "arguments", "single_step"]
ROW_LIMIT = 500
MAX_ADD_COLUMN_NAME_LENGTH = 128
MAX_ADD_COLUMN_VALUE_LENGTH = 256


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


@dataclass(frozen=True)
class ActionParseInspection:
    """Detailed outcome of parsing one JSON action payload."""

    action: Action | None
    stage: str
    failure_code: str | None = None
    failure_reason: str | None = None
    payload: dict[str, Any] | None = None


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
        return cls.inspect_single_step(text, table).action

    @classmethod
    def inspect_single_step(cls, text: str, table: Table | None = None) -> ActionParseInspection:
        payload, failure = cls._load_object_with_error(text)
        if payload is None:
            return failure
        return cls.inspect_payload(payload, table=table)

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
        return cls.inspect_arguments(text, action_name, table).action

    @classmethod
    def inspect_arguments(cls, text: str, action_name: str, table: Table | None = None) -> ActionParseInspection:
        payload, failure = cls._load_object_with_error(text)
        if payload is None:
            return failure
        expected_fields = cls._argument_fields.get(action_name)
        if expected_fields is None:
            return ActionParseInspection(
                action=None,
                stage="action_name",
                failure_code="unsupported_action",
                failure_reason=f"Unsupported action: {action_name!r}",
                payload=payload,
            )
        if set(payload) != expected_fields:
            return ActionParseInspection(
                action=None,
                stage="argument_fields",
                failure_code="unexpected_argument_fields",
                failure_reason=f"Expected fields {sorted(expected_fields)!r}, got {sorted(payload)!r}",
                payload=payload,
            )
        return cls.inspect_payload({"action": action_name, **payload}, table=table)

    @classmethod
    @staticmethod
    def normalize_add_column_text(value: str, *, field_name: str, max_length: int) -> tuple[str | None, str | None, str | None]:
        """Normalize permitted whitespace and reject unsafe decoded control characters."""
        normalized = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
        if len(normalized) > max_length:
            return None, "string_too_long", f"{field_name} exceeds the {max_length}-character limit."
        for position, character in enumerate(normalized):
            if unicodedata.category(character) == "Cc":
                return None, "disallowed_control_character", f"{field_name} contains a control character at position {position}."
        return normalized, None, None

    @staticmethod
    def unique_extracted_column_name(column: str, existing_columns: list[str]) -> str:
        """Return a unique derived-column name for an existing table header."""
        suffix = " extracted"
        index = 1
        while True:
            numbered_suffix = suffix if index == 1 else f"{suffix} {index}"
            candidate = f"{column[: MAX_ADD_COLUMN_NAME_LENGTH - len(numbered_suffix)]}{numbered_suffix}"
            if candidate not in existing_columns:
                return candidate
            index += 1

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
        payload, _ = JsonActionCodec._load_object_with_error(text)
        return payload

    @staticmethod
    def _load_object_with_error(text: str) -> tuple[dict[str, Any] | None, ActionParseInspection]:
        try:
            value = json.loads(text.strip())
        except (json.JSONDecodeError, TypeError) as error:
            return None, ActionParseInspection(
                action=None,
                stage="json_decode",
                failure_code="invalid_json",
                failure_reason=str(error),
            )
        if not isinstance(value, dict):
            return None, ActionParseInspection(
                action=None,
                stage="json_decode",
                failure_code="top_level_not_object",
                failure_reason=f"Expected JSON object, got {type(value).__name__}",
            )
        return value, ActionParseInspection(action=None, stage="json_decode")

    @classmethod
    def _parse_payload(cls, payload: dict[str, Any], *, table: Table | None) -> Action | None:
        return cls.inspect_payload(payload, table=table).action

    @classmethod
    def inspect_payload(cls, payload: dict[str, Any], *, table: Table | None) -> ActionParseInspection:
        name = payload.get("action")
        if not isinstance(name, str):
            return ActionParseInspection(None, "action_name", "missing_or_invalid_action", "Action must be a string.", payload)
        expected_fields = cls._fields.get(name)
        if expected_fields is None:
            return ActionParseInspection(None, "action_name", "unsupported_action", f"Unsupported action: {name!r}", payload)
        if set(payload) != expected_fields:
            return ActionParseInspection(
                None,
                "payload_fields",
                "unexpected_payload_fields",
                f"Expected fields {sorted(expected_fields)!r}, got {sorted(payload)!r}",
                payload,
            )
        if REGISTRY.get(name) is None or not REGISTRY.is_enabled(name):
            return ActionParseInspection(None, "action_name", "disabled_action", f"Action is disabled: {name!r}", payload)

        try:
            if name == "select_row":
                rows = payload["rows"]
                if not isinstance(rows, list) or not rows or not all(isinstance(row, str) for row in rows):
                    return ActionParseInspection(None, "arguments", "invalid_rows", "Rows must be a non-empty string list.", payload)
                if len(rows) != len(set(rows)):
                    return ActionParseInspection(None, "arguments", "duplicate_rows", "Rows must be unique.", payload)
                if rows == ["*"]:
                    arguments = ["*"]
                elif "*" in rows:
                    return ActionParseInspection(None, "arguments", "mixed_wildcard_rows", "Wildcard rows cannot be mixed.", payload)
                else:
                    arguments = []
                    for row in rows:
                        prefix, separator, index = row.partition(" ")
                        if prefix != "row" or separator != " " or not index.isdigit():
                            return ActionParseInspection(None, "arguments", "invalid_row_identifier", f"Invalid row: {row!r}", payload)
                        arguments.append(int(index))
            elif name == "select_column":
                arguments = payload["columns"]
                if not isinstance(arguments, list) or not arguments or not all(isinstance(value, str) for value in arguments):
                    return ActionParseInspection(None, "arguments", "invalid_columns", "Columns must be a non-empty string list.", payload)
                if len(arguments) != len(set(arguments)):
                    return ActionParseInspection(None, "arguments", "duplicate_columns", "Columns must be unique.", payload)
            elif name == "group_by":
                if not isinstance(payload["column"], str):
                    return ActionParseInspection(None, "arguments", "invalid_column", "Column must be a string.", payload)
                arguments = [payload["column"]]
            elif name == "sort_by":
                if not isinstance(payload["column"], str) or payload["order"] not in {"asc", "desc"}:
                    return ActionParseInspection(None, "arguments", "invalid_sort_arguments", "Invalid sort column or order.", payload)
                arguments = [payload["column"], payload["order"]]
            elif name == "add_column":
                if not isinstance(payload["column"], str):
                    return ActionParseInspection(None, "arguments", "invalid_column_type", "Column must be a string.", payload)
                column, error_code, error_reason = cls.normalize_add_column_text(
                    payload["column"],
                    field_name="Column name",
                    max_length=MAX_ADD_COLUMN_NAME_LENGTH,
                )
                if error_code:
                    return ActionParseInspection(None, "decoded_value_validation", error_code, error_reason, payload)
                if not column:
                    return ActionParseInspection(None, "arguments", "empty_column_name", "Column must not be empty.", payload)
                values = payload["values"]
                if not isinstance(values, list):
                    return ActionParseInspection(None, "arguments", "invalid_values_type", "Values must be a list.", payload)
                if not values:
                    return ActionParseInspection(None, "arguments", "empty_values", "Values must not be empty.", payload)
                if not all(isinstance(value, str) for value in values):
                    return ActionParseInspection(None, "arguments", "invalid_value_type", "Every value must be a string.", payload)
                normalized_values = []
                for index, value in enumerate(values):
                    normalized_value, error_code, error_reason = cls.normalize_add_column_text(
                        value,
                        field_name=f"Value {index}",
                        max_length=MAX_ADD_COLUMN_VALUE_LENGTH,
                    )
                    if error_code:
                        return ActionParseInspection(None, "decoded_value_validation", error_code, error_reason, payload)
                    normalized_values.append(normalized_value)
                if table is not None and column in table.columns:
                    column = cls.unique_extracted_column_name(column, table.columns)
                if table is not None and len(normalized_values) != len(table.rows):
                    return ActionParseInspection(
                        None,
                        "table_validation",
                        "incorrect_value_count",
                        f"Expected {len(table.rows)} values, got {len(normalized_values)}.",
                        payload,
                    )
                arguments = [column, normalized_values]
            else:
                arguments = []
            action = Action(name, arguments)
            if table is not None and not action.is_valid_for_table(table):
                return ActionParseInspection(
                    None, "table_validation", "invalid_for_table", "Action is invalid for the current table.", payload
                )
            return ActionParseInspection(action, "complete", payload=payload)
        except (KeyError, TypeError, ValueError) as error:
            return ActionParseInspection(None, "arguments", "argument_parse_error", str(error), payload)


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
            # Existing names are renamed to an `` extracted`` variant after decoding.
            properties["column"] = {"type": "string", "minLength": 1, "maxLength": MAX_ADD_COLUMN_NAME_LENGTH}
            properties["values"] = self.value_list_schema(spec.table_row_count)
            required.extend(["column", "values"])
        elif action != "end":
            raise ValueError(f"Unsupported JSON action: {action!r}")
        return self._object(properties, required)

    @staticmethod
    def value_list_schema(value_count: int) -> dict[str, Any]:
        if value_count < 1:
            raise ValueError("add_column requires at least one row")
        return {
            "type": "array",
            "items": {"type": "string", "maxLength": MAX_ADD_COLUMN_VALUE_LENGTH},
            "minItems": value_count,
            "maxItems": value_count,
        }

    @staticmethod
    def parse_value_list(text: str, expected_count: int) -> list[str] | None:
        try:
            values = json.loads(text.strip())
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(values, list) or len(values) != expected_count or not all(isinstance(value, str) for value in values):
            return None
        return values

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
