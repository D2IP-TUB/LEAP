from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from leap.core import Action, Table
from leap.core.actions import REGISTRY

GrammarPhase = Literal["action", "arguments", "single_step"]

ROW_LIMIT = 500
SUPPORTED_XGRAMMAR_ACTIONS = {"select_row", "select_column", "group_by", "sort_by", "end"}


@dataclass(frozen=True)
class ActionGrammarSpec:
    phase: GrammarPhase
    allowed_actions: tuple[str, ...]
    selected_action: str | None
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    row_limit: int = ROW_LIMIT


def get_constraint_backend(worker) -> str:
    return getattr(worker, "constraint_backend", "legacy_state_machine")


def uses_xgrammar(worker) -> bool:
    return getattr(worker, "use_constraints", False) and get_constraint_backend(worker) == "xgrammar"


def available_actions(action_history: list[str] | tuple[str, ...] | None, *, use_global_constraints: bool) -> list[str]:
    enabled = [name for name in REGISTRY.get_enabled_names() if name in SUPPORTED_XGRAMMAR_ACTIONS]

    if use_global_constraints:
        supported = set(enabled)
        return [name for name in REGISTRY.get_global_available_actions(list(action_history or [])) if name in supported]

    if action_history:
        used = REGISTRY._extract_action_names_from_history(list(action_history))
        enabled = [name for name in enabled if name not in used or name == "end"]

    if not action_history:
        enabled = [name for name in enabled if not REGISTRY.get(name).is_terminating]

    return enabled


class ActionGrammarBuilder:
    def build_spec(
        self,
        *,
        table: Table,
        action_history: list[str] | tuple[str, ...] | None,
        use_global_constraints: bool,
        phase: GrammarPhase,
        selected_action: str | None = None,
    ) -> ActionGrammarSpec:
        rows = tuple(f"row {idx}" for idx in range(min(ROW_LIMIT, len(table.rows))))
        columns = tuple(str(column) for column in table.columns)
        allowed = tuple(available_actions(action_history, use_global_constraints=use_global_constraints))
        if selected_action and selected_action not in SUPPORTED_XGRAMMAR_ACTIONS:
            raise ValueError(f"Action '{selected_action}' is not supported by the xgrammar backend.")
        return ActionGrammarSpec(
            phase=phase,
            allowed_actions=allowed,
            selected_action=selected_action,
            rows=rows,
            columns=columns,
        )

    def build_action_grammar(self, spec: ActionGrammarSpec) -> str:
        if spec.phase != "action":
            raise ValueError("Action selection grammar requires phase='action'.")
        choices = [self._literal(f"f_{action}") for action in spec.allowed_actions]
        return self._grammar("root ::= " + self._choice(choices))

    def build_arguments_grammar(self, spec: ActionGrammarSpec) -> str:
        if spec.phase != "arguments":
            raise ValueError("Argument grammar requires phase='arguments'.")
        if not spec.selected_action:
            raise ValueError("Argument grammar requires selected_action.")
        rules = self._argument_rules_for_action(spec.selected_action, spec)
        return self._grammar("\n".join(rules))

    def build_single_step_grammar(self, spec: ActionGrammarSpec) -> str:
        if spec.phase != "single_step":
            raise ValueError("Single-step grammar requires phase='single_step'.")

        call_rules = []
        root_choices = []
        shared_rules = self._shared_argument_rules(spec)
        for action in spec.allowed_actions:
            rule_name = f"{action}_call"
            root_choices.append(rule_name)
            call_rules.append(self._call_rule(rule_name, action))

        return self._grammar(
            "\n".join(
                [
                    "root ::= action_call",
                    "action_call ::= " + self._choice(root_choices),
                    *call_rules,
                    *shared_rules,
                ]
            )
        )

    def _argument_rules_for_action(self, action: str, spec: ActionGrammarSpec) -> list[str]:
        if action == "select_row":
            return ["root ::= row_list", *self._row_rules(spec.rows)]
        if action == "select_column":
            return ["root ::= column_list", *self._column_rules(spec.columns)]
        if action == "group_by":
            return ["root ::= column", *self._column_rules(spec.columns)]
        if action == "sort_by":
            return ["root ::= column " + self._literal(", ") + " order", *self._column_rules(spec.columns), self._order_rule()]
        if action == "end":
            return ["root ::= " + self._literal("")]
        raise ValueError(f"Action '{action}' is not supported by the xgrammar backend.")

    def _shared_argument_rules(self, spec: ActionGrammarSpec) -> list[str]:
        rules = []
        if any(action == "select_row" for action in spec.allowed_actions):
            rules.extend(self._row_rules(spec.rows))
        if any(action in {"select_column", "group_by", "sort_by"} for action in spec.allowed_actions):
            rules.extend(self._column_rules(spec.columns))
        if "sort_by" in spec.allowed_actions:
            rules.append(self._order_rule())
        return rules

    def _call_rule(self, rule_name: str, action: str) -> str:
        if action == "select_row":
            return f"{rule_name} ::= {self._literal('f_select_row(')} row_list {self._literal(')')}"
        if action == "select_column":
            return f"{rule_name} ::= {self._literal('f_select_column(')} column_list {self._literal(')')}"
        if action == "group_by":
            return f"{rule_name} ::= {self._literal('f_group_by(')} column {self._literal(')')}"
        if action == "sort_by":
            return f"{rule_name} ::= {self._literal('f_sort_by(')} column {self._literal(', ')} order {self._literal(')')}"
        if action == "end":
            return f"{rule_name} ::= {self._literal('f_end()')}"
        raise ValueError(f"Action '{action}' is not supported by the xgrammar backend.")

    def _row_rules(self, rows: tuple[str, ...]) -> list[str]:
        explicit_rows = self._literal("[") + " row (" + self._literal(", ") + " row)* " + self._literal("]")
        row_list_choices = [explicit_rows]
        if rows:
            row_list_choices.insert(0, self._literal("[*]"))
        return [
            "row_list ::= " + self._choice(row_list_choices),
            "row ::= " + self._quoted_value_choices(rows),
        ]

    def _column_rules(self, columns: tuple[str, ...]) -> list[str]:
        return [
            "column_list ::= " + self._literal("[") + " column (" + self._literal(", ") + " column)* " + self._literal("]"),
            "column ::= " + self._quoted_value_choices(columns),
        ]

    def _order_rule(self) -> str:
        return "order ::= " + self._choice([self._literal('"asc"'), self._literal('"desc"')])

    def _quoted_value_choices(self, values: tuple[str, ...]) -> str:
        if not values:
            return self._literal("__NO_VALID_VALUES__")
        return self._choice([self._literal(f'"{value}"') for value in values])

    @staticmethod
    def _grammar(text: str) -> str:
        return text + "\n"

    @staticmethod
    def _choice(values: list[str]) -> str:
        return " | ".join(values) if values else ActionGrammarBuilder._literal("__NO_VALID_ACTIONS__")

    @staticmethod
    def _literal(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return f'"{escaped}"'


class StructuredSamplingParamsFactory:
    @staticmethod
    def structured_outputs_kwargs(grammar: str) -> dict:
        try:
            from vllm.sampling_params import StructuredOutputsParams

            return {"structured_outputs": StructuredOutputsParams(grammar=grammar)}
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams

            return {"guided_decoding": GuidedDecodingParams.from_optional(grammar=grammar)}


class StructuredActionParser:
    @staticmethod
    def parse_action_name(text: str) -> str | None:
        return Action.parse_name_only(text)

    @staticmethod
    def parse_arguments(text: str, action_name: str, table: Table | None = None) -> Action | None:
        action = Action.parse(f"{action_name}({text.strip()})")
        return action if action and (table is None or action.is_valid_for_table(table)) else None

    @staticmethod
    def parse_single_step(text: str, table: Table | None = None) -> Action | None:
        action = Action.parse(text.strip())
        return action if action and (table is None or action.is_valid_for_table(table)) else None
