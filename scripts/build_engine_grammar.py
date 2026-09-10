from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leap.config.loader import _build_generation_config  # noqa: E402
from leap.core import Table  # noqa: E402
from leap.core.actions import REGISTRY  # noqa: E402
from leap.inference.function_constraints import ROW_LIMIT, SUPPORTED_XGRAMMAR_ACTIONS, ActionGrammarBuilder  # noqa: E402
from leap.inference.json_constraints import JsonActionSchemaBuilder  # noqa: E402

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_OUTPUT_PATH = Path("logs/engine_grammar.txt")
DEFAULT_COLUMNS = ("Name", "Points")
DEFAULT_TABLE_PREVIEW_ROWS = 20


@dataclass
class GrammarReportSamplingConfig:
    enabled: bool = False
    n_samples: int = 1
    per_action_samples: dict[str, int] | None = None
    debug: bool = False
    shuffle_invariant: bool = False


def build_engine_grammar_report(
    *,
    config_path: Path,
    columns: list[str] | None = None,
    row_count: int | None = None,
    action_history: list[str] | None = None,
    table_source: str = "config-first",
    table_preview_rows: int = DEFAULT_TABLE_PREVIEW_ROWS,
) -> str:
    raw_config = _load_yaml(config_path)
    generation_section = raw_config.get("generation", {})
    if not isinstance(generation_section, dict):
        raise ValueError("Configuration field 'generation' must be a mapping.")

    enabled_actions = generation_section.get("enabled_actions")
    if enabled_actions is not None:
        if not isinstance(enabled_actions, list) or not all(isinstance(action, str) for action in enabled_actions):
            raise ValueError("generation.enabled_actions must be a list of action names.")
        REGISTRY.set_enabled_actions(enabled_actions)
    else:
        REGISTRY._enabled_actions = None

    sampling_config = _sampling_config(generation_section.get("sampling", {}))
    generation_config = _build_generation_config(generation_section, enabled_actions, sampling_config)

    table, table_source_description = _resolve_table(
        raw_config=raw_config,
        config_path=config_path,
        table_source=table_source,
        columns=columns or list(DEFAULT_COLUMNS),
        row_count=3 if row_count is None else row_count,
    )
    builder = ActionGrammarBuilder()
    action_history = action_history or []

    if generation_config.output_format == "json":
        schema_builder = JsonActionSchemaBuilder()
        action_spec = schema_builder.build_spec(
            table=table,
            action_history=action_history,
            use_global_constraints=generation_config.use_global_constraints,
            phase="action",
        )
        single_spec = schema_builder.build_spec(
            table=table,
            action_history=action_history,
            use_global_constraints=generation_config.use_global_constraints,
            phase="single_step",
        )
        schemas = {
            "action": schema_builder.build_action_schema(action_spec),
            "single_step": schema_builder.build_single_step_schema(single_spec),
            "arguments": {},
        }
        for action_name in action_spec.allowed_actions:
            if action_name == "end":
                continue
            spec = schema_builder.build_spec(
                table=table,
                action_history=action_history,
                use_global_constraints=generation_config.use_global_constraints,
                phase="arguments",
                selected_action=action_name,
            )
            schemas["arguments"][action_name] = schema_builder.build_arguments_schema(spec)
        return "\n".join(
            [
                f"# LEAP Engine {generation_config.output_format.upper()} Schema Report",
                "",
                f"- config: {config_path}",
                f"- table_source: {table_source_description}",
                f"- parsed_columns: {list(table.columns)}",
                f"- parsed_row_count: {len(table.rows)}",
                f"- allowed_actions: {list(action_spec.allowed_actions)}",
                "",
                "## JSON Schemas",
                "```json",
                json.dumps(schemas, ensure_ascii=False, indent=2),
                "```",
                "",
                "## Parsed Table Preview",
                "```csv",
                _table_preview(table, table_preview_rows),
                "```",
            ]
        )

    action_spec = builder.build_spec(
        table=table,
        action_history=action_history,
        use_global_constraints=generation_config.use_global_constraints,
        phase="action",
    )
    single_step_spec = builder.build_spec(
        table=table,
        action_history=action_history,
        use_global_constraints=generation_config.use_global_constraints,
        phase="single_step",
    )

    action_grammar = builder.build_action_grammar(action_spec)
    single_step_grammar = builder.build_single_step_grammar(single_step_spec)
    argument_grammars = {}
    for action_name in action_spec.allowed_actions:
        spec = builder.build_spec(
            table=table,
            action_history=action_history,
            use_global_constraints=generation_config.use_global_constraints,
            phase="arguments",
            selected_action=action_name,
        )
        argument_grammars[action_name] = builder.build_arguments_grammar(spec)

    return _render_report(
        config_path=config_path,
        generation_section=generation_section,
        generation_config=generation_config,
        action_history=action_history,
        table=table,
        table_source_description=table_source_description,
        table_preview_rows=table_preview_rows,
        action_spec=action_spec,
        single_step_spec=single_step_spec,
        action_grammar=action_grammar,
        single_step_grammar=single_step_grammar,
        argument_grammars=argument_grammars,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the LEAP engine xgrammar grammar from the current generation config.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the LEAP YAML config.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Text file to write.")
    parser.add_argument(
        "--table-source",
        choices=("config-first", "sample"),
        default="config-first",
        help="Use the first table from the configured dataset, or a small synthetic sample table.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=list(DEFAULT_COLUMNS),
        help="Column names for --table-source sample.",
    )
    parser.add_argument("--row-count", type=int, default=3, help="Number of rows for --table-source sample.")
    parser.add_argument(
        "--table-preview-rows",
        type=int,
        default=DEFAULT_TABLE_PREVIEW_ROWS,
        help="Number of parsed table rows to show in the report preview.",
    )
    parser.add_argument(
        "--action-history",
        action="append",
        default=[],
        help="Previously used action, such as 'select_row(0)'. Repeat for multiple actions.",
    )
    args = parser.parse_args(argv)

    report = build_engine_grammar_report(
        config_path=args.config,
        columns=args.columns,
        row_count=args.row_count,
        action_history=args.action_history,
        table_source=args.table_source,
        table_preview_rows=args.table_preview_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote engine grammar report to {args.output}")
    return 0


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping.")
    return data


def _sampling_config(raw_sampling: Any) -> GrammarReportSamplingConfig:
    if raw_sampling is None:
        raw_sampling = {}
    if not isinstance(raw_sampling, dict):
        raise ValueError("generation.sampling must be a mapping.")
    return GrammarReportSamplingConfig(
        enabled=raw_sampling.get("enabled", False),
        n_samples=raw_sampling.get("n_samples", 1),
        per_action_samples=dict(raw_sampling.get("per_action_samples", {})),
        debug=raw_sampling.get("debug", False),
        shuffle_invariant=raw_sampling.get("shuffle_invariant", False),
    )


def _resolve_table(
    *,
    raw_config: dict[str, Any],
    config_path: Path,
    table_source: str,
    columns: list[str],
    row_count: int,
) -> tuple[Table, str]:
    if table_source == "sample":
        return _make_table(columns, row_count), f"sample table from --columns/--row-count ({len(columns)} columns, {row_count} rows)"
    if table_source != "config-first":
        raise ValueError("table_source must be either 'config-first' or 'sample'.")
    return _load_first_table_from_config(raw_config, config_path), "first table from configured dataset"


def _load_first_table_from_config(raw_config: dict[str, Any], config_path: Path) -> Table:
    dataset_section = raw_config.get("dataset", {})
    if not isinstance(dataset_section, dict):
        raise ValueError("Configuration field 'dataset' must be a mapping.")

    loader = str(dataset_section.get("loader", "huggingface")).lower()
    if loader == "huggingface":
        from datasets import load_dataset

        kwargs = {}
        if dataset_section.get("split"):
            kwargs["split"] = dataset_section["split"]
        if dataset_section.get("trust_remote_code") is not None:
            kwargs["trust_remote_code"] = dataset_section["trust_remote_code"]
        dataset = load_dataset(dataset_section["name"], **kwargs)
    elif loader == "json":
        from datasets import load_dataset

        data_files = dataset_section.get("data_files")
        if not data_files:
            raise ValueError("JSON dataset loader requires dataset.data_files.")
        dataset = load_dataset("json", data_files=_resolve_data_files(data_files, config_path.parent), split=dataset_section.get("split"))
    elif loader == "disk":
        from datasets import load_from_disk

        path = dataset_section.get("path")
        if not path:
            raise ValueError("Disk dataset loader requires dataset.path.")
        dataset = load_from_disk(str(_resolve_path(Path(path), config_path.parent)))
    else:
        raise ValueError(f"Unsupported dataset loader: {loader}")

    if isinstance(dataset, dict):
        split_name = dataset_section.get("split") or next(iter(dataset.keys()))
        dataset = dataset[split_name]

    first_row = dataset[0]
    table_data = first_row.get("table", first_row) if isinstance(first_row, dict) else first_row["table"]
    return Table.from_dict(table_data)


def _resolve_data_files(data_files: Any, base_dir: Path) -> Any:
    if isinstance(data_files, str):
        return str(_resolve_path(Path(data_files), base_dir))
    if isinstance(data_files, list):
        return [str(_resolve_path(Path(path), base_dir)) for path in data_files]
    if isinstance(data_files, dict):
        return {key: _resolve_data_files(value, base_dir) for key, value in data_files.items()}
    return data_files


def _resolve_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _make_table(columns: list[str], row_count: int) -> Table:
    if row_count < 0:
        raise ValueError("--row-count must be non-negative.")
    if not columns:
        raise ValueError("--columns must include at least one column.")
    rows = [[f"{column.lower()} {row_index}" for column in columns] for row_index in range(row_count)]
    return Table(columns=columns, rows=rows)


def _render_report(
    *,
    config_path: Path,
    generation_section: dict[str, Any],
    generation_config,
    action_history: list[str],
    table: Table,
    table_source_description: str,
    table_preview_rows: int,
    action_spec,
    single_step_spec,
    action_grammar: str,
    single_step_grammar: str,
    argument_grammars: dict[str, str],
) -> str:
    active_xgrammar = generation_config.use_constraints and generation_config.constraint_backend == "xgrammar"
    lines = [
        "# LEAP Engine Grammar",
        "",
        "## Source Config",
        f"- config_path: {config_path}",
        f"- generation.use_constraints: {generation_config.use_constraints}",
        f"- generation.constraint_backend: {generation_config.constraint_backend}",
        f"- generation.use_global_constraints: {generation_config.use_global_constraints}",
        f"- generation.strategy: {generation_config.strategy}",
        f"- generation.enabled_actions: {list(generation_config.enabled_actions or REGISTRY.get_enabled_names())}",
        f"- active_xgrammar_backend: {active_xgrammar}",
        f"- action_history: {action_history}",
        f"- table_source: {table_source_description}",
        f"- parsed_columns: {list(table.columns)}",
        f"- parsed_row_count: {len(table.rows)}",
        f"- row_limit: {ROW_LIMIT}",
        "",
    ]
    if not active_xgrammar:
        lines.extend(
            [
                (
                    "Note: this config does not currently use the xgrammar backend at runtime. The grammars below show what the "
                    "xgrammar backend would build from the same action/table settings."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Derived Options",
            f"- supported_xgrammar_actions: {sorted(SUPPORTED_XGRAMMAR_ACTIONS)}",
            f"- action_phase_allowed_actions: {list(action_spec.allowed_actions)}",
            f"- single_step_allowed_actions: {list(single_step_spec.allowed_actions)}",
            f"- grammar_rows: {len(action_spec.rows)}",
            f"- grammar_columns: {len(action_spec.columns)}",
            "",
            "## Parsed Table Preview",
            "```csv",
            _table_preview(table, table_preview_rows),
            "```",
            "",
            "## Tutorial: Options That Affect The Grammar",
            (
                "- generation.use_constraints enables constrained decoding. If it is false, normal sampling is used and the engine "
                "grammar is not applied."
            ),
            (
                "- generation.constraint_backend selects the constraint implementation. Only xgrammar uses the grammar text in this "
                "report; legacy_state_machine uses token-level logits processors instead."
            ),
            (
                "- generation.enabled_actions filters the action names before grammar construction. The xgrammar backend currently "
                "supports select_row, select_column, group_by, sort_by, and end."
            ),
            (
                "- generation.use_global_constraints applies LEAP's action-transition matrix to the latest action. It controls the "
                "available actions shown in prompts and the actions the engine can select."
            ),
            (
                "- action_history removes already-used transformation actions when global constraints are off. When they are on, the "
                "latest action selects the next transition-matrix row; end is available after any first transformation."
            ),
            (
                "- generation.strategy decides which runtime grammar shape is used. cot uses action selection plus argument grammars; "
                "iterative single-action generation uses the single-step grammar; direct_query skips table-action grammar."
            ),
            "- Table columns become the valid column string choices for select_column, group_by, and sort_by.",
            f"- Table row count becomes row choices for select_row, capped at {ROW_LIMIT} rows.",
            (
                "- The selected action controls the argument grammar. For example, sort_by allows one column plus asc or desc, while "
                "end accepts an empty argument string."
            ),
            "",
            "## Action Selection Grammar",
            "```",
            action_grammar.rstrip(),
            "```",
            "",
            "## Single-Step Grammar",
            "```",
            single_step_grammar.rstrip(),
            "```",
            "",
            "## Argument Grammars",
            "",
        ]
    )

    if argument_grammars:
        for action_name, grammar in argument_grammars.items():
            lines.extend([f"### {action_name}", "```", grammar.rstrip(), "```", ""])
    else:
        lines.extend(["No argument grammars were generated because no actions are currently allowed.", ""])

    lines.extend(["## Raw Generation Config", "```yaml", yaml.safe_dump(generation_section, sort_keys=False).rstrip(), "```", ""])
    return "\n".join(lines)


def _table_preview(table: Table, preview_rows: int) -> str:
    if preview_rows < 0:
        raise ValueError("--table-preview-rows must be non-negative.")
    max_rows = len(table.rows) if preview_rows == 0 else min(preview_rows, len(table.rows))
    preview = table.to_csv(max_rows=max_rows, crop=True, max_chars=100000)
    if max_rows < len(table.rows):
        return preview + f"\n... truncated {len(table.rows) - max_rows} rows ..."
    return preview


if __name__ == "__main__":
    raise SystemExit(main())
