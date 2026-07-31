"""Compare two benchmark runs and decide whether the candidate may ship.

The gate answers one question — *"is this checkpoint worse than the one it would
replace, by more than we are willing to tolerate?"* — and it answers it with a
confidence interval rather than a point estimate.

The distinction matters, because the naive comparison is what most labs actually
do. Candidate got 0.68, production got 0.62, ship it. With 50 episodes that
6-point gap is comfortably inside noise, and promoting on it is close to a coin
flip dressed up as a decision.

Three deliberate positions:

**Absence of evidence is not evidence of absence.** A comparison whose interval
is too wide to resolve the tolerance does not pass. It returns `INCONCLUSIVE`
and says how many episodes would have been needed. A gate that silently passes
underpowered comparisons is worse than no gate, because it manufactures
confidence.

**Per-task regressions are checked separately.** A candidate can hold its overall
success rate while collapsing on one task and improving on another. Aggregates
hide exactly the failure mode that matters on a robot.

**The gate is not a promotion.** It reports; promotion stays an explicit act.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from .benchmark import BenchmarkResult
from .stats import Interval, mcnemar_exact, paired_bootstrap_diff, required_episodes

Verdict = Literal["PASS", "FAIL", "INCONCLUSIVE"]


@dataclass
class TaskComparison:
    task_id: str
    baseline_rate: float
    candidate_rate: float
    difference: float
    ci_low: float
    ci_high: float
    wins: int
    losses: int
    p_value: float
    regressed: bool


@dataclass
class Comparison:
    benchmark: str
    baseline_policy: str
    candidate_policy: str
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)

    n_episodes: int = 0
    baseline_rate: float = 0.0
    candidate_rate: float = 0.0
    difference: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    wins: int = 0
    losses: int = 0
    p_value: float = 1.0
    significant: bool = False

    tolerance: float = 0.0
    tasks: list[TaskComparison] = field(default_factory=list)
    suggested_episodes: int | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tasks"] = [asdict(t) for t in self.tasks]
        return data

    def headline(self) -> str:
        return (
            f"{self.candidate_policy} vs {self.baseline_policy}: "
            f"{self.candidate_rate:.1%} vs {self.baseline_rate:.1%} "
            f"({self.difference:+.1%}, 95% CI [{self.ci_low:+.1%}, {self.ci_high:+.1%}])"
        )


def _aligned(baseline: BenchmarkResult, candidate: BenchmarkResult) -> None:
    """Refuse to compare runs that are not actually paired."""
    if baseline.benchmark != candidate.benchmark:
        raise ValueError(f"different benchmarks: {baseline.benchmark!r} vs {candidate.benchmark!r}")
    if baseline.keys() != candidate.keys():
        raise ValueError(
            "the two runs did not evaluate the same (task, seed) episodes, so a "
            "paired comparison is invalid. Re-run both against the same benchmark spec."
        )


def compare(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    *,
    tolerance: float = 0.02,
    alpha: float = 0.05,
    bootstrap_samples: int = 10_000,
    seed: int = 17,
    task_tolerance: float = 0.10,
) -> Comparison:
    """Paired comparison with a regression gate.

    `tolerance` is how much overall success rate may drop before the gate fails —
    not zero, because insisting on never regressing at all blocks every
    checkpoint once the benchmark is noisy enough.
    """
    _aligned(baseline, candidate)

    base_success = baseline.success_vector.astype(float)
    cand_success = candidate.success_vector.astype(float)
    n = base_success.size

    interval: Interval = paired_bootstrap_diff(
        base_success, cand_success, n_samples=bootstrap_samples, seed=seed, alpha=alpha
    )
    wins, losses, p_value = mcnemar_exact(baseline.success_vector, candidate.success_vector)

    result = Comparison(
        benchmark=baseline.benchmark,
        baseline_policy=baseline.policy,
        candidate_policy=candidate.policy,
        verdict="PASS",
        n_episodes=int(n),
        baseline_rate=float(base_success.mean()),
        candidate_rate=float(cand_success.mean()),
        difference=interval.point,
        ci_low=interval.low,
        ci_high=interval.high,
        wins=wins,
        losses=losses,
        p_value=p_value,
        significant=p_value < alpha,
        tolerance=tolerance,
    )

    # -- per-task ----------------------------------------------------------
    base_by_task = {t.task_id: t for t in baseline.tasks}
    for task in candidate.tasks:
        reference = base_by_task.get(task.task_id)
        if reference is None:
            continue
        b = np.array([e.success for e in reference.episodes], dtype=float)
        c = np.array([e.success for e in task.episodes], dtype=float)
        task_ci = paired_bootstrap_diff(b, c, n_samples=bootstrap_samples, seed=seed, alpha=alpha)
        t_wins, t_losses, t_p = mcnemar_exact(b.astype(bool), c.astype(bool))

        # A task regression counts only when the interval clears the tolerance —
        # a point estimate alone would fire constantly on small per-task samples.
        regressed = task_ci.high < -task_tolerance
        result.tasks.append(
            TaskComparison(
                task_id=task.task_id,
                baseline_rate=float(b.mean()),
                candidate_rate=float(c.mean()),
                difference=task_ci.point,
                ci_low=task_ci.low,
                ci_high=task_ci.high,
                wins=t_wins,
                losses=t_losses,
                p_value=t_p,
                regressed=regressed,
            )
        )

    # -- verdict -----------------------------------------------------------
    # Fail when the interval rules out "no worse than tolerance": even the
    # optimistic end of the range is below the acceptable drop.
    if interval.high < -tolerance:
        result.verdict = "FAIL"
        result.reasons.append(
            f"overall success rate regressed by {-interval.point:.1%} "
            f"(95% CI upper bound {interval.high:+.1%} is below the "
            f"−{tolerance:.0%} tolerance)"
        )

    for task in result.tasks:
        if task.regressed:
            result.verdict = "FAIL"
            result.reasons.append(
                f"task `{task.task_id}` regressed {-task.difference:.1%} "
                f"(CI upper bound {task.ci_high:+.1%}), beyond the "
                f"{task_tolerance:.0%} per-task tolerance"
            )

    if result.verdict == "PASS":
        ci_width = interval.high - interval.low
        # If the interval is wider than the decision it is supposed to inform,
        # this run cannot support a PASS. Say so instead of implying confidence.
        if ci_width > 2 * tolerance and not interval.excludes_zero():
            result.verdict = "INCONCLUSIVE"
            result.suggested_episodes = required_episodes(
                baseline_rate=result.baseline_rate,
                detectable_difference=max(tolerance, 0.05),
            ) * max(len(result.tasks), 1)
            result.reasons.append(
                f"the 95% CI spans {ci_width:.1%}, wider than the ±{tolerance:.0%} "
                f"tolerance being tested — this run cannot resolve the question. "
                f"About {result.suggested_episodes} episodes would be needed."
            )
        else:
            direction = "improved" if interval.point > 0 else "held"
            result.reasons.append(
                f"success rate {direction} ({interval.point:+.1%}, 95% CI "
                f"[{interval.low:+.1%}, {interval.high:+.1%}]); no task regressed "
                f"beyond tolerance"
            )

    return result
