from pathlib import Path

import yaml

from scripts.build_engine_grammar import build_engine_grammar_report, main


def _write_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _config(tmp_path: Path) -> Path:
    config = {
        "dataset": {
            "loader": "huggingface",
            "name": "wikitablequestions",
            "split": "train",
            "trust_remote_code": True,
        },
        "generation": {
            "strategy": "cot",
            "use_constraints": True,
            "constraint_backend": "xgrammar",
            "use_global_constraints": False,
            "enabled_actions": ["select_row", "select_column", "group_by", "sort_by", "end"],
            "sampling": {"enabled": True, "n_samples": 2},
        },
    }
    path = tmp_path / "configs" / "default.yaml"
    _write_yaml(path, config)
    return path


def test_report_uses_first_configured_wikitablequestions_table(tmp_path, monkeypatch):
    class FakeDataset:
        def __getitem__(self, index):
            assert index == 0
            return {
                "table": {
                    "header": ["Year", "Winner"],
                    "rows": [["1999", "Ada"], ["2000", "Grace"]],
                }
            }

    def fake_load_dataset(name, **kwargs):
        assert name == "wikitablequestions"
        assert kwargs == {"split": "train", "trust_remote_code": True}
        return FakeDataset()

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    report = build_engine_grammar_report(config_path=_config(tmp_path))

    assert "- table_source: first table from configured dataset" in report
    assert "- parsed_columns: ['Year', 'Winner']" in report
    assert "- parsed_row_count: 2" in report
    assert "1999,Ada" in report
    assert '"\\"Year\\""' in report
    assert '"\\"row 1\\""' in report


def test_main_writes_report_file_with_sample_table(tmp_path):
    config_path = _config(tmp_path)
    output_path = tmp_path / "grammar.txt"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--table-source",
            "sample",
            "--columns",
            "City",
            "Score",
            "--row-count",
            "1",
        ]
    )

    assert exit_code == 0
    report = output_path.read_text(encoding="utf-8")
    assert "- parsed_columns: ['City', 'Score']" in report
    assert "- parsed_row_count: 1" in report
    assert "## Tutorial: Options That Affect The Grammar" in report


def test_report_renders_json_schemas_for_json_mode(tmp_path):
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["generation"]["output_format"] = "json"
    _write_yaml(config_path, config)

    report = build_engine_grammar_report(
        config_path=config_path,
        table_source="sample",
        columns=["City", "Score"],
        row_count=2,
    )
    assert report.startswith("# LEAP Engine JSON Schema Report")
    assert '"single_step"' in report
    assert '"additionalProperties": false' in report
    assert '"City"' in report


def test_report_renders_mcp_envelope_schemas(tmp_path):
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["generation"]["output_format"] = "mcp"
    _write_yaml(config_path, config)

    report = build_engine_grammar_report(
        config_path=config_path,
        table_source="sample",
        columns=["City", "Score"],
        row_count=2,
    )

    assert report.startswith("# LEAP Engine MCP Schema Report")
    assert '"tools/call"' in report
    assert '"select_column"' in report
    assert '"arguments"' in report


def test_report_supports_add_column_for_active_xgrammar(tmp_path):
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["generation"]["enabled_actions"].append("add_column")
    _write_yaml(config_path, config)

    report = build_engine_grammar_report(config_path=config_path, table_source="sample")
    assert "f_add_column" in report
    assert "value_list" in report
