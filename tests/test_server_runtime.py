from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import server


def test_build_runtime_tool_passes_model_id_to_validation(monkeypatch):
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = object()
    config = SimpleNamespace(
        model=SimpleNamespace(id=model_id, instruct=True),
        generation=SimpleNamespace(use_constraints=True, constraint_backend="legacy_state_machine", output_format="function"),
    )
    validate = Mock(spec=server.validate_installed_runtime, wraps=server.validate_installed_runtime)
    monkeypatch.setattr(server, "get_model_id", lambda path: model_id)
    monkeypatch.setattr(server.AutoTokenizer, "from_pretrained", lambda model: tokenizer)
    monkeypatch.setattr(server, "load_runtime_config_tool", lambda path, tok: config)
    monkeypatch.setattr(server, "validate_installed_runtime", validate)
    monkeypatch.setattr(server, "PromptBuilder", Mock())
    monkeypatch.setattr("leap.vllm_runtime.installed_vllm_version", lambda: "0.10.0")
    monkeypatch.setenv("VLLM_USE_V1", "0")

    runtime = server.build_runtime_tool(Path("unused.yaml"))

    validate.assert_called_once_with(
        model_id=model_id,
        use_constraints=True,
        constraint_backend="legacy_state_machine",
        output_format="function",
    )
    assert runtime.config is config
    assert runtime.tokenizer is tokenizer
