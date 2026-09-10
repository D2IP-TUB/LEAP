import asyncio
from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind

from leap.core import Action, Table
from leap.core.actions import REGISTRY
from leap.generation.sampling import SamplingConfig, SamplingLayer
from leap.generation.shuffle_invariant_sampling import ShuffleInvariantSamplingLayer
from leap.generation.strategies import ChainOfTableGenerationStrategy
from leap.inference.json_constraints import JsonActionCodec


@pytest.fixture(autouse=True)
def enabled_actions():
    previous = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["select_row", "select_column", "add_column", "group_by", "sort_by", "end"])
    yield
    REGISTRY._enabled_actions = previous


def run_step(action, *, enabled=True, constrained=False, output_format="function", layer_type=SamplingLayer, responses=None):
    table = Table(columns=["Name", "Score"], rows=[["Ada", "2"], ["Bob", "1"]])
    calls = []
    argument_tables = []
    action_text = '{"action":"' + action.name + '"}' if output_format == "json" else "f_" + action.name
    argument_text = (
        JsonActionCodec.dumps(action, include_action=False) if output_format == "json" else action.to_string().split("(", 1)[1][:-1]
    )

    def generate(prompt, params, request_id):
        calls.append((prompt, params, request_id))
        if "_action_step" in request_id:
            text = action_text
        else:
            index = int(request_id.rsplit("sample", 1)[1])
            text = responses[index] if responses is not None else argument_text

        async def results():
            yield SimpleNamespace(outputs=[SimpleNamespace(text=text)])

        return results()

    def build_arguments(**kwargs):
        argument_tables.append(kwargs["table"])
        return "arguments"

    worker = SimpleNamespace(
        engine=SimpleNamespace(generate=generate),
        tokenizer=SimpleNamespace(eos_token_id=0, encode=lambda prompt, add_special_tokens: [1]),
        max_model_len=32_000,
        use_constraints=constrained,
        use_global_constraints=False,
        constraint_backend="xgrammar",
        output_format=output_format,
        effective_temperature=lambda value: value,
    )
    # Deliberately conflicting raw settings must not override the CoT policy.
    layer = layer_type(
        SamplingConfig(
            enabled=enabled,
            n_samples=99,
            per_action_samples={"select_row": 2, "select_column": 3, "add_column": 99, "sort_by": 99},
            shuffle_invariant=layer_type is ShuffleInvariantSamplingLayer,
        )
    )
    strategy = ChainOfTableGenerationStrategy(
        prompt_builder=SimpleNamespace(build_cot_action_prompt=lambda **kwargs: "action", build_cot_arguments_prompt=build_arguments),
        sampling_layer=layer,
    )
    history = ["select_column('Name')"] if action.name == "end" else []
    result = asyncio.run(strategy.generate_action_step(worker, table, history, "request", {}, "Who?", 0))
    return result, calls, argument_tables, table


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("output_format", ["function", "json"])
@pytest.mark.parametrize("layer_type", [SamplingLayer, ShuffleInvariantSamplingLayer])
@pytest.mark.parametrize(
    "action",
    [
        Action("select_row", ["*"]),
        Action("select_column", ["Name"]),
        Action("add_column", ["Derived", ["a", "b"]]),
        Action("group_by", ["Name"]),
        Action("sort_by", ["Score", "desc"]),
        Action("end", []),
    ],
)
def test_cot_uses_action_specific_counts(enabled, constrained, output_format, layer_type, action):
    result, calls, argument_tables, table = run_step(
        action, enabled=enabled, constrained=constrained, output_format=output_format, layer_type=layer_type
    )
    count = 8 if enabled and action.name in {"select_row", "select_column"} else 1
    action_calls = [call for call in calls if "_action_step" in call[2]]
    argument_calls = [call for call in calls if "_args_step" in call[2]]
    assert len(action_calls) == 1
    assert len(argument_calls) == (0 if action.name == "end" else count)
    assert len({call[2] for call in calls}) == len(calls)
    assert all(call[1].n == 1 for call in calls)
    assert action_calls[0][1].temperature == 0.0
    assert all(call[1].temperature == 0.7 for call in argument_calls)
    assert all(call[1].output_kind == RequestOutputKind.FINAL_ONLY for call in calls)
    assert result.action == action
    assert (result.metadata.n_requested, result.metadata.n_generated, result.metadata.n_valid) == (count, count, count)
    assert (result.metadata.winner_votes, result.metadata.total_votes) == (count, count)
    assert result.metadata.fallback_reason is None
    if enabled and action.name == "select_row" and layer_type is ShuffleInvariantSamplingLayer:
        assert any(candidate.rows != table.rows for candidate in argument_tables)
    else:
        assert all(candidate.rows == table.rows for candidate in argument_tables)


def test_cot_preserves_selection_majority_voting():
    result, calls, _, _ = run_step(Action("select_column", ["Name"]), responses=['["Score"]'] + ['["Name"]'] * 7)
    assert len(calls) == 9
    assert result.action == Action("select_column", ["Name"])
    assert result.metadata.winner_votes == 7
    assert result.metadata.total_votes == 8


def test_cot_debug_disabled_does_not_dump_argument_responses(capsys):
    run_step(Action("add_column", ["Derived", ["a", "b"]]), responses=['{"column":"Derived","values":["a","b"]}'])
    assert "[PHASE 2 RESPONSE" not in capsys.readouterr().out


def test_cot_single_candidate_failure_keeps_diagnostics():
    result, calls, _, _ = run_step(Action("add_column", ["Derived", ["a", "b"]]), responses=["invalid"])
    assert len(calls) == 2
    assert result.action == Action("end", [])
    assert result.metadata.n_requested == 1
    assert result.metadata.n_generated == result.metadata.n_valid == 0
    assert result.metadata.fallback_reason == "no_valid_candidates"
    assert len(result.metadata.add_column_diagnostics) == 1
