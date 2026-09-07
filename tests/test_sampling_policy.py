from dataclasses import asdict

import pytest

from leap.config.loader import _build_generation_config
from leap.generation.sampling import SamplingConfig, SamplingLayer
from leap.generation.shuffle_invariant_sampling import ShuffleInvariantSamplingLayer
from main import create_sampling_layer


@pytest.mark.parametrize("strategy", ["iterative", "cot"])
@pytest.mark.parametrize("enabled", [False, True])
def test_sampling_policy_resolves_counts_and_shuffle(strategy, enabled):
    requested = SamplingConfig(
        enabled=enabled,
        n_samples=99,
        per_action_samples={"add_column": 99, "select_row": 2},
        shuffle_invariant=True,
        debug=True,
    )

    resolved = requested.for_strategy(strategy)

    assert resolved.enabled is (strategy == "cot" and enabled)
    assert resolved.n_samples == 1
    assert resolved.shuffle_invariant is resolved.enabled
    assert resolved.debug is True
    for action in ("select_row", "select_column", "add_column", "group_by", "sort_by", "end"):
        expected = 8 if resolved.enabled and action in {"select_row", "select_column"} else 1
        assert resolved.get_n_samples(action) == expected
    assert resolved.get_n_samples("unknown") == 1
    assert resolved.for_strategy(strategy) == resolved
    assert requested.n_samples == 99
    assert requested.per_action_samples["add_column"] == 99


def test_direct_query_sampling_settings_are_unchanged():
    requested = SamplingConfig(enabled=True, n_samples=3, debug=True)
    assert requested.for_strategy("direct_query") is requested


@pytest.mark.parametrize("strategy", ["iterative", "cot"])
def test_loaded_config_serializes_effective_sampling_policy(strategy, capsys):
    generation = _build_generation_config(
        {"strategy": strategy},
        None,
        SamplingConfig(enabled=True, n_samples=8, shuffle_invariant=True, debug=True),
    )

    saved = asdict(generation)["sampling"]
    assert saved["n_samples"] == 1
    assert saved["per_action_samples"]["add_column"] == 1
    assert saved["per_action_samples"]["select_column"] == (8 if strategy == "cot" else 1)
    layer = create_sampling_layer(generation)
    assert layer.config == generation.sampling
    assert layer.config.debug is True
    if strategy == "iterative":
        assert type(layer) is SamplingLayer
        assert "one complete operation" in capsys.readouterr().out
    else:
        assert isinstance(layer, ShuffleInvariantSamplingLayer)
        assert "8 argument candidates for select_row/select_column, 1 otherwise" in capsys.readouterr().out
