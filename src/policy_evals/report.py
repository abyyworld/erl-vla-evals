"""Markdown reports for benchmark runs and comparisons.

Markdown so a comparison can be posted as a PR comment, which is where a
promotion decision should actually be argued.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .benchmark import BenchmarkResult
from .compare import Comparison

VERDICT_BADGE = {
    "PASS": "✅ **PASS**",
    "FAIL": "❌ **FAIL**",
    "INCONCLUSIVE": "⚠️ **INCONCLUSIVE**",
}


def render_run(result: BenchmarkResult) -> str:
    lines = [
        f"# {result.benchmark} — `{result.policy}`",
        "",
        f"_{result.ran_at}_",
        "",
        f"**Overall success rate: {result.overall_success_rate:.1%}** "
        f"across {len(result.episodes)} episodes",
        "",
        "| Task | Success | Mean steps (solved) | Path efficiency |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task in result.tasks:
        steps = task.mean_steps_on_success
        lines.append(
            f"| `{task.task_id}` | {task.success_rate:.1%} | "
            f"{f'{steps:.0f}' if steps is not None else '—'} | "
            f"{task.mean_path_efficiency:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_comparison(comparison: Comparison) -> str:
    lines = [
        f"# {VERDICT_BADGE[comparison.verdict]} — {comparison.benchmark}",
        "",
        f"`{comparison.candidate_policy}` (candidate) vs `{comparison.baseline_policy}` (baseline)",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Overall",
        "",
        "| | Baseline | Candidate | Difference (95% CI) |",
        "| --- | ---: | ---: | --- |",
        f"| Success rate | {comparison.baseline_rate:.1%} | {comparison.candidate_rate:.1%} | "
        f"{comparison.difference:+.1%} [{comparison.ci_low:+.1%}, {comparison.ci_high:+.1%}] |",
        "",
        f"- Paired over **{comparison.n_episodes}** episodes — identical seeds for both policies",
        f"- Discordant pairs: **{comparison.wins}** candidate wins, **{comparison.losses}** losses",
        f"- McNemar exact p = **{comparison.p_value:.4f}**"
        f"{' (significant)' if comparison.significant else ' (not significant)'}",
        f"- Regression tolerance: {comparison.tolerance:.0%}",
        "",
    ]

    lines += ["## Verdict", ""]
    for reason in comparison.reasons:
        lines.append(f"- {reason}")
    lines.append("")

    if comparison.tasks:
        lines += [
            "## Per task",
            "",
            "Checked separately: a candidate can hold its overall rate while "
            "collapsing on one task.",
            "",
            "| Task | Baseline | Candidate | Difference (95% CI) | W/L | p | |",
            "| --- | ---: | ---: | --- | ---: | ---: | --- |",
        ]
        for task in comparison.tasks:
            mark = "🔴" if task.regressed else ""
            lines.append(
                f"| `{task.task_id}` | {task.baseline_rate:.1%} | {task.candidate_rate:.1%} | "
                f"{task.difference:+.1%} [{task.ci_low:+.1%}, {task.ci_high:+.1%}] | "
                f"{task.wins}/{task.losses} | {task.p_value:.3f} | {mark} |"
            )
        lines.append("")

    if comparison.suggested_episodes:
        lines += [
            "## Statistical power",
            "",
            f"This run cannot resolve a {comparison.tolerance:.0%} difference. "
            f"Roughly **{comparison.suggested_episodes} episodes** "
            f"(vs {comparison.n_episodes} run) would be needed to detect one at "
            "80% power.",
            "",
        ]

    lines += [
        "---",
        "",
        "_Success rates come from a kinematic environment, not a physics "
        "simulator or hardware. They measure the harness; treat them as such._",
        "",
    ]
    return "\n".join(lines)


def write(path: Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
