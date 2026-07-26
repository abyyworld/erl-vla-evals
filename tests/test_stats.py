"""Tests for the statistics that the promotion decision rests on."""

from __future__ import annotations

import numpy as np
import pytest

from erl_vla_evals.stats import (
    bootstrap_mean_ci,
    mcnemar_exact,
    paired_bootstrap_diff,
    required_episodes,
)


def test_mcnemar_ignores_concordant_pairs():
    """Episodes both policies solved carry no information about which is better.

    A test that counted them would report the same p-value for 5-vs-0 out of 10
    as for 5-vs-0 out of 10,000, which is exactly backwards.
    """
    baseline = np.array([True] * 5 + [False] * 5)
    candidate = np.array([True] * 5 + [True] * 5)
    wins, losses, p_small = mcnemar_exact(baseline, candidate)
    assert (wins, losses) == (5, 0)

    padding = np.array([True] * 1000)
    _, _, p_padded = mcnemar_exact(
        np.concatenate([baseline, padding]), np.concatenate([candidate, padding])
    )
    assert p_padded == pytest.approx(p_small), "concordant pairs changed the p-value"


def test_mcnemar_identical_policies_are_not_significant():
    outcomes = np.array([True, False, True, True, False] * 20)
    wins, losses, p = mcnemar_exact(outcomes, outcomes)
    assert (wins, losses) == (0, 0)
    # No disagreement is not evidence of equality; 1.0 is the honest answer.
    assert p == 1.0


def test_mcnemar_matches_the_binomial_by_hand():
    """10 discordant pairs, all won by the candidate: p = 2 * (1/2)^10."""
    baseline = np.array([False] * 10)
    candidate = np.array([True] * 10)
    wins, losses, p = mcnemar_exact(baseline, candidate)
    assert (wins, losses) == (10, 0)
    assert p == pytest.approx(2 * 0.5**10)


def test_mcnemar_is_symmetric():
    rng = np.random.default_rng(0)
    a = rng.random(200) > 0.4
    b = rng.random(200) > 0.5
    wins_ab, losses_ab, p_ab = mcnemar_exact(a, b)
    wins_ba, losses_ba, p_ba = mcnemar_exact(b, a)
    assert (wins_ab, losses_ab) == (losses_ba, wins_ba)
    assert p_ab == pytest.approx(p_ba)


def test_pairing_narrows_the_interval():
    """The point of paired evaluation, demonstrated.

    Episodes differ wildly in difficulty. Pairing removes that variance from the
    comparison; treating the two runs as independent samples leaves it in, and
    the interval is correspondingly wider for the same data.
    """
    rng = np.random.default_rng(7)
    difficulty = rng.normal(0, 1.0, 300)  # shared across both policies
    baseline = difficulty + rng.normal(0, 0.1, 300)
    candidate = difficulty + 0.15 + rng.normal(0, 0.1, 300)

    paired = paired_bootstrap_diff(baseline, candidate, n_samples=2000, seed=1)
    paired_width = paired.high - paired.low

    # Unpaired: bootstrap each arm separately and add the variances.
    a = bootstrap_mean_ci(baseline, n_samples=2000, seed=1)
    b = bootstrap_mean_ci(candidate, n_samples=2000, seed=1)
    unpaired_width = (b.high - b.low) + (a.high - a.low)

    assert paired_width < unpaired_width / 3
    assert paired.excludes_zero(), "a real 0.15 effect should be detected"


def test_paired_bootstrap_centres_on_the_true_difference():
    baseline = np.zeros(200)
    candidate = np.ones(200) * 0.2
    interval = paired_bootstrap_diff(baseline, candidate, n_samples=1000, seed=3)
    assert interval.point == pytest.approx(0.2)
    assert interval.low == pytest.approx(0.2)


def test_paired_bootstrap_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        paired_bootstrap_diff(np.zeros(10), np.zeros(11))


def test_no_difference_interval_contains_zero():
    rng = np.random.default_rng(11)
    outcomes = rng.random(400) > 0.5
    interval = paired_bootstrap_diff(outcomes.astype(float), outcomes.astype(float), n_samples=1000)
    assert not interval.excludes_zero()


def test_required_episodes_grows_as_the_effect_shrinks():
    """Detecting a smaller difference needs more episodes — steeply."""
    big = required_episodes(0.60, 0.20)
    small = required_episodes(0.60, 0.05)
    assert small > big * 4
    # Sanity: resolving 5 points around a 60% baseline is a large study.
    assert small > 1000


def test_bootstrap_handles_degenerate_input():
    assert np.isnan(bootstrap_mean_ci(np.array([])).point)
    single = bootstrap_mean_ci(np.array([0.5]))
    assert single.point == single.low == single.high == 0.5
