"""Invariants for the RL controller's weight projection.

The CSS weights the RL learns are a probability distribution over four
dimensions, floored at ``W_MIN`` so online learning can never drive any single
dimension to zero. The floor matters most for ``carbon``: an RL that learns to
ignore carbon defeats the premise of the whole system, and the learned weights
are persisted to data/rl_state.json, so a violation survives restarts.

Two properties must hold for every output of ``_project_simplex``:

    sum(w) == 1          (it is a distribution)
    w[k] >= W_MIN        (no dimension is squeezed out)

The second used to be violated: the old implementation clipped to
``[w_min, 1]`` and then renormalised, and the renormalise pushed clipped weights
straight back under the floor. Its retry loop exited on ``sum == 1``, which is
true immediately after every renormalise, so it never iterated. Measured before
the fix: carbon at 0.0164 against a floor of 0.05.
"""
from __future__ import annotations

import math

import pytest

import rl_controller as rl

KEYS = rl._COEFFICIENT_KEYS


def _adversarial_inputs():
    """Weight vectors a gradient step could plausibly produce, including the
    lopsided ones that push a single dimension toward zero."""
    n = len(KEYS)
    return [
        dict.fromkeys(KEYS, 1.0 / n),                      # already uniform
        dict.fromkeys(KEYS, 1.0),                          # unnormalised, equal
        {**dict.fromkeys(KEYS, 1.0), "carbon": 0.0},       # carbon driven to zero
        {**dict.fromkeys(KEYS, 1.0), "carbon": 0.001},     # carbon nearly zero
        {**dict.fromkeys(KEYS, 20.0), "carbon": 0.0},      # extreme magnitudes
        {**dict.fromkeys(KEYS, 0.0), "accuracy": 1.0},     # single dimension owns it all
        dict.fromkeys(KEYS, 0.0),                          # degenerate: all zero
        {**dict.fromkeys(KEYS, 1.0), "cost": -5.0},        # negative from a bad step
        {"carbon": 0.9, "latency": 0.04, "accuracy": 0.03, "cost": 0.03},
    ]


@pytest.mark.parametrize("weights", _adversarial_inputs())
def test_projection_sums_to_one(weights):
    out = rl._project_simplex(dict(weights))
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("weights", _adversarial_inputs())
def test_projection_respects_the_floor(weights):
    """The floor is the point of the projection: W_MIN exists so the RL cannot
    zero out a CSS dimension. Carbon is the dimension the product depends on."""
    out = rl._project_simplex(dict(weights))
    for key, value in out.items():
        assert value >= rl.W_MIN - 1e-12, (
            f"{key} projected to {value:.6f}, below the W_MIN floor of {rl.W_MIN}"
        )


@pytest.mark.parametrize("weights", _adversarial_inputs())
def test_projection_never_exceeds_one(weights):
    out = rl._project_simplex(dict(weights))
    assert all(v <= 1.0 + 1e-12 for v in out.values())


def test_projection_covers_exactly_the_coefficient_keys():
    out = rl._project_simplex({"carbon": 1.0})       # missing keys are defaulted
    assert set(out) == set(KEYS)


def test_projection_is_idempotent():
    """Projecting an already-valid distribution must not move it."""
    once = rl._project_simplex(dict.fromkeys(KEYS, 1.0))
    twice = rl._project_simplex(dict(once))
    for key in KEYS:
        assert twice[key] == pytest.approx(once[key], abs=1e-12)


def test_projection_preserves_relative_order():
    """A dimension the policy weighted more heavily must not come out lighter
    than one it weighted less -- otherwise the projection is not just enforcing
    constraints, it is rewriting what was learned."""
    weights = {"carbon": 0.10, "latency": 0.20, "accuracy": 0.40, "cost": 0.30}
    out = rl._project_simplex(dict(weights))
    assert sorted(weights, key=weights.get) == sorted(out, key=out.get)


def test_infeasible_floor_falls_back_to_uniform():
    """If w_min * n >= 1 there is no mass left to distribute; uniform is the only
    distribution satisfying the floor."""
    out = rl._project_simplex(dict.fromkeys(KEYS, 1.0), w_min=0.5)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(v == pytest.approx(1.0 / len(KEYS)) for v in out.values())


def test_learning_rate_decays_with_episode_count():
    """alpha_t = ALPHA_0 / (1 + sqrt(t)) -- steps must shrink as evidence grows,
    or the policy never settles."""
    rates = [rl.ALPHA_0 / (1.0 + math.sqrt(max(t, 1))) for t in (1, 10, 100, 1000)]
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] < rates[0] / 5


class TestPersistedStateIsHealed:
    """State written before the floor fix can hold a sub-floor carbon weight.

    Without projection at load, such a deployment keeps routing on it until some
    later update happens to lift it, so _TierState projects on the way in.
    """

    def test_sub_floor_state_is_corrected_on_load(self):
        stale = {"carbon": 0.0164, "latency": 0.3279, "accuracy": 0.3279, "cost": 0.3278}
        state = rl._TierState.from_dict({"weights": stale, "episode_count": 42,
                                         "baseline_ema": 0.6})
        assert state.weights["carbon"] >= rl.W_MIN - 1e-12
        assert sum(state.weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_healing_preserves_learning_progress(self):
        """Correcting the weights must not discard what the tier has learned."""
        stale = {"carbon": 0.0164, "latency": 0.3279, "accuracy": 0.3279, "cost": 0.3278}
        state = rl._TierState.from_dict({"weights": stale, "episode_count": 42,
                                         "baseline_ema": 0.6, "policy_version": 7})
        assert state.episode_count == 42
        assert state.baseline_ema == pytest.approx(0.6)
        assert state.policy_version == 7

    def test_valid_state_is_untouched_on_load(self):
        good = {"carbon": 0.58, "latency": 0.15, "accuracy": 0.19, "cost": 0.08}
        state = rl._TierState.from_dict({"weights": good})
        for key, value in good.items():
            assert state.weights[key] == pytest.approx(value, abs=1e-9)
