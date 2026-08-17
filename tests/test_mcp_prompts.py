import json
from types import SimpleNamespace

import pytest

from leap.core import Table
from leap.core.actions import REGISTRY
from leap.generation.prompt_builder import PromptBuilder
from leap.mcp.protocol import McpToolCallCodec


class RecordingTokenizer:
    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "\n".join(message["content"] for message in messages)


@pytest.fixture(autouse=True)
def enabled_actions():
    previous = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["select_row", "select_column", "group_by", "sort_by", "end"])
    yield
    REGISTRY._enabled_actions = previous


@pytest.fixture
def context():
    tokenizer = RecordingTokenizer()
    builder = PromptBuilder(tokenizer=tokenizer, is_instruct=True, output_format="mcp")
    worker = SimpleNamespace(
        max_model_len=100_000,
        output_format="mcp",
        use_constraints=True,
        constraint_backend="xgrammar",
        use_global_constraints=False,
    )
    table = Table(columns=["Name", "Score"], rows=[["Ada", "2"], ["Grace", "1"]])
    return tokenizer, builder, worker, table


def test_iterative_mcp_prompt_uses_complete_tool_calls(context):
    tokenizer, builder, worker, table = context

    prompt = builder.build_iterative_prompt(question="Who?", table=table, action_history=[], worker=worker, step=0)

    assistant_answers = [message["content"] for message in tokenizer.messages if message["role"] == "assistant"]
    assert "Return only one official MCP JSON-RPC tools/call request" in prompt
    assert "Next MCP request:" in prompt
    assert builder.prompt_catalog.iterative_mcp["operation_shapes"]["select_column"] in prompt
    assert "f_" not in prompt
    assert assistant_answers
    assert all(McpToolCallCodec.parse(answer) is not None for answer in assistant_answers)
    assert all(McpToolCallCodec.parse(answer).arguments for answer in assistant_answers)


def test_iterative_mcp_current_turn_comes_from_prompt_catalog(context):
    _tokenizer, builder, worker, table = context
    builder.prompt_catalog.iterative_mcp["templates"]["current_turn"] = (
        "MCP TEMPLATE SENTINEL\n${table}\n${question}\n${action_history}\n${available_operations}"
    )

    prompt = builder.build_iterative_prompt(question="Who?", table=table, action_history=[], worker=worker, step=0)

    assert "MCP TEMPLATE SENTINEL" in prompt


def test_cot_mcp_prompts_split_selection_and_arguments(context):
    tokenizer, builder, worker, table = context

    action_prompt = builder.build_cot_action_prompt(question="Who?", table=table, action_history=[], worker=worker)
    action_answers = [message["content"] for message in tokenizer.messages if message["role"] == "assistant"]
    assert builder.prompt_catalog.cot_mcp["action_instruction"].rstrip() in action_prompt
    assert builder.prompt_catalog.cot_mcp["selection_shapes"]["select_column"] in action_prompt
    assert action_answers
    assert all(McpToolCallCodec.parse(answer).arguments == {} for answer in action_answers)

    arguments_prompt = builder.build_cot_arguments_prompt(
        question="Who?",
        table=table,
        action_name="select_column",
        action_history=[],
        worker=worker,
    )
    argument_answers = [message["content"] for message in tokenizer.messages if message["role"] == "assistant"]
    assert builder.prompt_catalog.cot_mcp["argument_instructions"]["select_column"] in arguments_prompt
    assert argument_answers
    assert all(McpToolCallCodec.parse(answer).name == "select_column" for answer in argument_answers)
    assert all(set(json.loads(answer)) == {"jsonrpc", "id", "method", "params"} for answer in argument_answers)
