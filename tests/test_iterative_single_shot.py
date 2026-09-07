import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.sampling_params import RequestOutputKind

from leap.core import Action, Table
from leap.core.actions import REGISTRY
from leap.generation.sampling import SamplingConfig, SamplingLayer
from leap.generation.shuffle_invariant_sampling import ShuffleInvariantSamplingLayer
from leap.generation.strategies import IterativeGenerationStrategy
from leap.inference.json_constraints import JsonActionCodec


@pytest.fixture(autouse=True)
def enabled_actions():
    previous = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["select_row", "select_column", "add_column", "group_by", "sort_by", "end"])
    yield
    REGISTRY._enabled_actions = previous


def run_step(monkeypatch, output, *, constrained=False, output_format="function", layer_type=SamplingLayer):
    table = Table(columns=["Name", "Score"], rows=[["Ada", "2"], ["Bob", "1"]])
    history = []
    calls = []

    def generate(prompt, params, request_id):
        calls.append((prompt, params, request_id))
        if isinstance(output, Exception):
            raise output

        async def results():
            yield SimpleNamespace(outputs=[] if output is None else [SimpleNamespace(text=output)])

        return results()

    worker = SimpleNamespace(
        engine=SimpleNamespace(generate=generate),
        tokenizer=SimpleNamespace(eos_token_id=0),
        use_constraints=constrained,
        use_global_constraints=False,
        constraint_backend="xgrammar",
        output_format=output_format,
        effective_temperature=lambda value: value,
    )
    layer = layer_type(SamplingConfig(enabled=True, n_samples=8, shuffle_invariant=True))
    no_vote = Mock(side_effect=AssertionError("Iterative must not vote"))
    no_transform = Mock(side_effect=AssertionError("Iterative must not transform context"))
    monkeypatch.setattr(layer, "aggregate_candidates", no_vote)
    monkeypatch.setattr(layer, "transform_context", no_transform)

    def build_prompt(**kwargs):
        assert kwargs["table"] is table
        assert kwargs["action_history"] is history
        return "select_row select_column add_column group_by sort_by end"

    strategy = IterativeGenerationStrategy(prompt_builder=SimpleNamespace(build_iterative_prompt=build_prompt), sampling_layer=layer)
    result = asyncio.run(strategy.generate_action_step(worker, table, history, "request", {}, "Who?", 0))
    no_vote.assert_not_called()
    no_transform.assert_not_called()
    return result, calls


@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("output_format", ["function", "json"])
@pytest.mark.parametrize("layer_type", [SamplingLayer, ShuffleInvariantSamplingLayer])
@pytest.mark.parametrize(
    "action",
    [
        Action("select_row", [0]),
        Action("select_column", ["Name"]),
        Action("add_column", ["Derived", ["a", "b"]]),
        Action("group_by", ["Name"]),
        Action("sort_by", ["Score", "desc"]),
        Action("end", []),
    ],
)
def test_iterative_generates_one_operation_without_voting_or_shuffle(monkeypatch, constrained, output_format, layer_type, action):
    output = JsonActionCodec.dumps(action) if output_format == "json" else action.to_string()
    result, calls = run_step(monkeypatch, output, constrained=constrained, output_format=output_format, layer_type=layer_type)

    assert len(calls) == 1
    assert calls[0][1].n == 1
    assert calls[0][1].temperature == 0.7
    assert calls[0][1].output_kind == RequestOutputKind.FINAL_ONLY
    assert result.action == action
    metadata = result.metadata
    assert (metadata.n_requested, metadata.n_generated, metadata.n_valid) == (1, 1, 1)
    assert (metadata.winner_votes, metadata.total_votes) == (1, 1)
    assert len(metadata.candidate_actions) == len(metadata.valid_actions) == 1
    assert metadata.fallback_reason is None


@pytest.mark.parametrize(
    "output, generated",
    [("not an operation", 0), (None, 0), (RuntimeError("engine failed"), 0), ('select_column("Missing")', 1)],
)
def test_iterative_preserves_single_attempt_failure_fallback(monkeypatch, output, generated):
    result, calls = run_step(monkeypatch, output)

    assert len(calls) == 1
    assert result.action == Action("end", [])
    assert result.metadata.fallback_reason == "no_valid_candidates"
    assert (result.metadata.n_requested, result.metadata.n_generated, result.metadata.n_valid) == (1, generated, 0)
    assert (result.metadata.winner_votes, result.metadata.total_votes) == (0, 0)


def test_iterative_debug_disabled_does_not_dump_prompt_or_response(monkeypatch, capsys):
    run_step(monkeypatch, "f_end()")
    output = capsys.readouterr().out
    assert "[PROMPT]" not in output
    assert "[RESPONSE]" not in output


def test_iterative_preserves_end_only_shortcut(monkeypatch):
    REGISTRY.set_enabled_actions(["end"])
    layer = SamplingLayer(SamplingConfig(enabled=True, n_samples=8))
    generate = Mock(side_effect=AssertionError("No inference needed for end"))
    monkeypatch.setattr(layer, "generate_candidates", generate)
    result = asyncio.run(
        layer.sample_action(
            SimpleNamespace(use_constraints=False, output_format="function"),
            Table(columns=["Name"], rows=[["Ada"]]),
            ["select_row([row 0])"],
            "request",
            {},
            None,
            "Who?",
            1,
        )
    )
    generate.assert_not_called()
    assert result.action == Action("end", [])
    assert result.n_requested == result.n_generated == result.n_valid == 1
