"""Statistics for comparing two policies without fooling yourself.

The default way labs compare checkpoints is: run both, look at the two success
rates, ship the bigger one. With 50 episodes, a 6-point difference in success
rate is well inside noise, and that comparison will promote a worse model
roughly as often as it promotes a better one.

Three things here fix that:

**Paired evaluation.** Both policies face identical seeds, so each episode is a
matched pair. Pairing removes episode difficulty from the comparison entirely,
and typically shrinks the confidence interval on the difference by a large
factor versus treating the two runs as independent samples.

**Paired bootstrap over the difference**, not over each arm separately. The
question is "is B better than A", so the quantity to put an interval around is
B − A.

**McNemar's exact test** for the binary outcomes. With paired binary data, only
the *discordant* pairs — the ones where the policies disagree — carry
information. The episodes both solved and both failed tell you nothing about
which is better, and any test that counts them is throwing away power and
overstating certainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def __str__(self) -> str:
        return f"{self.point:+.4f} [{self.low:+.4f}, {self.high:+.4f}]"


def bootstrap_mean_ci(
    values: np.ndarray, n_samples: int = 10_000, seed: int = 17, alpha: float = 0.05
) -> Interval:
    """Percentile CI for a mean."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    if values.size == 1:
        v = float(values[0])
        return Interval(v, v, v)

    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_samples, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(values.mean()), float(lo), float(hi))


def paired_bootstrap_diff(
    baseline: np.ndarray,
    candidate: np.ndarray,
    n_samples: int = 10_000,
    seed: int = 17,
    alpha: float = 0.05,
) -> Interval:
    """CI for mean(candidate) − mean(baseline) over matched episodes.

    Resamples *pairs*, never the two arms independently. Resampling
    independently would discard the pairing and reintroduce the episode-difficulty
    variance that pairing exists to remove.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.shape != candidate.shape:
        raise ValueError(
            f"paired comparison needs equal lengths, got {baseline.shape} and {candidate.shape}"
        )
    return bootstrap_mean_ci(candidate - baseline, n_samples=n_samples, seed=seed, alpha=alpha)


def mcnemar_exact(baseline: np.ndarray, candidate: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact McNemar test on paired binary outcomes.

    Returns `(wins, losses, p_value)` where a *win* is an episode the candidate
    solved and the baseline did not.

    Exact binomial rather than the chi-squared approximation: benchmark suites
    routinely produce fewer than 25 discordant pairs, which is exactly where the
    approximation stops being trustworthy — and it errs toward reporting
    significance that is not there.
    """
    baseline = np.asarray(baseline).astype(bool)
    candidate = np.asarray(candidate).astype(bool)
    if baseline.shape != candidate.shape:
        raise ValueError("paired test needs equal lengths")

    wins = int(np.sum(candidate & ~baseline))
    losses = int(np.sum(~candidate & baseline))
    n = wins + losses
    if n == 0:
        # The policies never disagreed. That is not evidence they are equal, and
        # p = 1.0 is the honest answer rather than a small number.
        return wins, losses, 1.0

    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return wins, losses, float(min(1.0, 2.0 * tail))


def required_episodes(
    baseline_rate: float, detectable_difference: float, power: float = 0.8, alpha: float = 0.05
) -> int:
    """Roughly how many episodes are needed to detect a given improvement.

    A normal-approximation power calculation for two proportions. Approximate on
    purpose — its job is to answer "is 50 episodes anywhere near enough?" before
    a week is spent running a benchmark that could never have resolved the
    effect. It usually is not.
    """
    from math import sqrt

    # Inverse normal CDF via the Beasley-Springer-Moro style rational
    # approximation would be overkill; these two z values cover the standard
    # alpha/power choices and the function documents its own bluntness.
    z_alpha = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}.get(alpha, 1.9600)
    z_power = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}.get(power, 0.8416)

    p1 = float(np.clip(baseline_rate, 1e-6, 1 - 1e-6))
    p2 = float(np.clip(baseline_rate + detectable_difference, 1e-6, 1 - 1e-6))
    p_bar = 0.5 * (p1 + p2)

    numerator = z_alpha * sqrt(2 * p_bar * (1 - p_bar)) + z_power * sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )
    return int(np.ceil((numerator / max(abs(p2 - p1), 1e-9)) ** 2))
