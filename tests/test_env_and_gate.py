"""Environment invariants and the end-to-end regression gate.

The gate tests are the ones that matter: they inject a degradation of known
magnitude and assert the gate reaches the right verdict. Without them,
"the harness catches regressions" is an untested claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from erl_vla_evals.adapters import build_policy
from erl_vla_evals.benchmark import BenchmarkSpec, run_benchmark, run_episode
from erl_vla_evals.compare import compare
from erl_vla_evals.env import Environment, TaskSpec

BENCHMARK = Path(__file__).resolve().parents[1] / "conf" / "benchmarks" / "manipulation_v1.yaml"


@pytest.fixture(scope="module")
def spec():
    return BenchmarkSpec.load(BENCHMARK)


@pytest.fixture(scope="module")
def fast_spec(spec):
    """A cut-down suite so the gate tests stay quick without losing the point."""
    return BenchmarkSpec(
        name=spec.name,
        episodes_per_task=25,
        seed_base=spec.seed_base,
        tasks=spec.tasks,
    )


# -- environment ------------------------------------------------------------


def test_same_seed_gives_an_identical_episode():
    """Determinism is what makes paired comparison valid."""
    task = TaskSpec(task_id="t")
    a, b = Environment(task), Environment(task)
    first, second = a.reset(42), b.reset(42)

    for key in first:
        assert np.array_equal(first[key], second[key]), f"{key} differed"
    assert a.goal_xyz.tolist() == b.goal_xyz.tolist()


def test_different_seeds_give_different_episodes():
    env = Environment(TaskSpec(task_id="t"))
    assert not np.array_equal(env.reset(1)["q"], env.reset(2)["q"])


def test_goals_respect_the_distance_band(spec):
    """An episode must never begin already solved."""
    for task in spec.tasks:
        env = Environment(task)
        distances = [env.reset(s).__class__ and env.start_distance for s in spec.seeds_for(task)]
        distances = np.array(distances)
        assert distances.min() > task.position_tolerance * 2, (
            f"{task.task_id}: closest goal was {distances.min():.3f} m, within "
            f"reach of a policy that does nothing"
        )


def test_actions_are_clipped_to_the_rig_limit():
    """A policy must not be able to teleport to the goal."""
    task = TaskSpec(task_id="t", action_limit=0.08)
    env = Environment(task)
    env.reset(0)
    before = env.q.copy()
    env.step(np.concatenate([np.full(7, 100.0), [1.0]]))
    assert np.all(np.abs(env.q - before) <= 0.08 + 1e-9)


def test_success_requires_holding_the_goal():
    """Touching the tolerance ball for one step is not a success."""
    task = TaskSpec(task_id="t", settle_steps=5)
    env = Environment(task)
    env.reset(0)
    # Teleport-free: drive straight to the goal with an oracle, then confirm the
    # first in-tolerance step does not immediately end the episode.
    policy = build_policy("scripted")
    policy.reset()
    observation = env.observation()
    first_reached_step = None
    for _ in range(task.max_steps):
        result = env.step(policy.act(observation))
        observation = result.observation
        if result.info["reached"] and first_reached_step is None:
            first_reached_step = result.info["steps"]
            assert not result.success, "success latched on first touch"
        if result.done:
            break
    assert first_reached_step is not None
    assert result.success
    assert result.info["steps"] >= first_reached_step + task.settle_steps - 1


# -- baselines --------------------------------------------------------------


def test_doing_nothing_scores_zero(fast_spec):
    """If this is above zero, the benchmark is measuring the environment."""
    result = run_benchmark(fast_spec, build_policy("zero"))
    assert result.overall_success_rate == 0.0


def test_random_barely_scores(fast_spec):
    result = run_benchmark(fast_spec, build_policy("random"))
    assert result.overall_success_rate < 0.05


def test_the_oracle_solves_every_task(fast_spec):
    """A task the oracle cannot solve makes every other score on it meaningless."""
    result = run_benchmark(fast_spec, build_policy("scripted"))
    for task in result.tasks:
        assert task.success_rate == 1.0, f"{task.task_id} unsolvable at {task.success_rate:.0%}"


# -- pairing ----------------------------------------------------------------


def test_two_runs_evaluate_identical_episodes(fast_spec):
    a = run_benchmark(fast_spec, build_policy("scripted"))
    b = run_benchmark(fast_spec, build_policy("random"))
    assert a.keys() == b.keys()


def test_seeds_are_stable_across_processes(spec):
    """Seeds must not depend on Python's per-process hash randomisation.

    This was a real bug: `seeds_for` used the builtin `hash()`, which Python
    salts per process. Within one process everything looked fine and the tests
    passed; across two CLI invocations every seed differed, so no two runs were
    ever comparable. Invisible in-process, which is why this test shells out.
    """
    import subprocess
    import sys

    script = (
        "from erl_vla_evals.benchmark import BenchmarkSpec;"
        f"s=BenchmarkSpec.load({str(BENCHMARK)!r});"
        "print([s.seeds_for(t)[0] for t in s.tasks])"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(runs) == 1, f"seeds differed between processes: {runs}"


def test_distinct_tasks_get_distinct_seeds(spec):
    """Otherwise per-task scores are correlated and the breakdown misleads."""
    firsts = [spec.seeds_for(t)[0] for t in spec.tasks]
    assert len(set(firsts)) == len(firsts)


def test_comparison_refuses_unpaired_runs(fast_spec, spec):
    """Different episode counts mean the pairing is a lie. Refuse, don't guess."""
    a = run_benchmark(fast_spec, build_policy("scripted"))
    smaller = BenchmarkSpec(
        name=spec.name, episodes_per_task=5, seed_base=spec.seed_base, tasks=spec.tasks
    )
    b = run_benchmark(smaller, build_policy("scripted"))

    with pytest.raises(ValueError, match="same"):
        compare(a, b)


# -- the gate ---------------------------------------------------------------


def test_identical_policy_passes(fast_spec):
    baseline = run_benchmark(fast_spec, build_policy("scripted"))
    candidate = run_benchmark(fast_spec, build_policy("scripted"))
    result = compare(baseline, candidate, bootstrap_samples=2000)

    assert result.verdict == "PASS"
    assert result.difference == 0.0
    assert (result.wins, result.losses) == (0, 0)


def test_a_real_regression_is_caught(fast_spec):
    """Inject a known degradation; the gate must fail.

    `scripted+noise:0.12` drops overall success from 100% to roughly 55% on this
    suite, driven almost entirely by `reach_precise` and `grasp_at_target`.
    """
    baseline = run_benchmark(fast_spec, build_policy("scripted"))
    candidate = run_benchmark(fast_spec, build_policy("scripted+noise:0.12"))
    result = compare(baseline, candidate, tolerance=0.02, bootstrap_samples=2000)

    assert result.verdict == "FAIL"
    assert result.difference < -0.15
    assert result.ci_high < -0.02, "the CI should exclude the tolerance"
    assert result.significant
    assert any("regressed" in r for r in result.reasons)


def test_a_per_task_collapse_is_caught_behind_a_stable_average(fast_spec):
    """The failure mode aggregates hide.

    `reach_precise` is the only task sensitive to small noise, so a mild
    corruption collapses it while the overall rate barely moves. A gate that
    only watched the average would wave this through.
    """
    baseline = run_benchmark(fast_spec, build_policy("scripted"))
    candidate = run_benchmark(fast_spec, build_policy("scripted+noise:0.08"))
    result = compare(
        baseline, candidate, tolerance=0.20, task_tolerance=0.10, bootstrap_samples=2000
    )

    assert result.difference > -0.20, "overall drop should be inside the loose tolerance"
    assert result.verdict == "FAIL", "per-task collapse should still fail the gate"
    regressed = [t.task_id for t in result.tasks if t.regressed]
    assert "reach_precise" in regressed


def _synthetic_run(policy: str, outcomes: list[bool], task_id: str = "reach_near"):
    """Build a BenchmarkResult from explicit outcomes.

    The verdict logic is tested against constructed data rather than a policy
    run: the interesting cases are specific win/loss patterns, and reaching them
    by tuning a noise level would make the test both slow and flaky.
    """
    from erl_vla_evals.benchmark import BenchmarkResult, EpisodeResult, TaskResult

    episodes = [
        EpisodeResult(
            task_id=task_id,
            seed=1000 + i,
            success=bool(success),
            steps=50,
            final_distance=0.01 if success else 0.4,
            path_efficiency=0.9 if success else 0.0,
            timed_out=not success,
        )
        for i, success in enumerate(outcomes)
    ]
    return BenchmarkResult(
        benchmark="manipulation_v1",
        policy=policy,
        ran_at="2027-01-01T00:00:00Z",
        tasks=[TaskResult(task_id=task_id, episodes=episodes)],
    )


def test_underpowered_comparison_is_inconclusive_not_pass():
    """Too few episodes to resolve the tolerance must not silently pass.

    Eight episodes, one disagreement in each direction. The point estimate is
    exactly zero, but the interval is far too wide to support the claim that the
    candidate has not regressed by 2%.
    """
    baseline = _synthetic_run("baseline", [True, True, True, False, True, True, False, True])
    candidate = _synthetic_run("candidate", [True, True, False, True, True, True, False, True])

    result = compare(baseline, candidate, tolerance=0.02, bootstrap_samples=4000)

    assert result.verdict == "INCONCLUSIVE"
    assert result.ci_high - result.ci_low > 2 * result.tolerance
    assert result.suggested_episodes and result.suggested_episodes > result.n_episodes
    assert any("cannot resolve" in r for r in result.reasons)


def test_a_clear_improvement_passes_on_a_small_sample():
    """A large enough effect passes even when the sample is small.

    INCONCLUSIVE is for genuinely unresolvable comparisons, not a blanket
    penalty on short runs — otherwise the gate would block every early result.
    """
    baseline = _synthetic_run("baseline", [False] * 20 + [True] * 5)
    candidate = _synthetic_run("candidate", [True] * 25)

    result = compare(baseline, candidate, tolerance=0.02, bootstrap_samples=4000)
    assert result.verdict == "PASS"
    assert result.difference > 0.5
    assert result.significant


def test_a_regression_below_tolerance_still_passes():
    """The tolerance has to mean something, or nothing ever ships."""
    outcomes = [True] * 96 + [False] * 4
    candidate_outcomes = [True] * 95 + [False] * 5
    baseline = _synthetic_run("baseline", outcomes)
    candidate = _synthetic_run("candidate", candidate_outcomes)

    result = compare(baseline, candidate, tolerance=0.05, bootstrap_samples=4000)
    assert result.verdict == "PASS"


def test_results_round_trip_through_disk(tmp_path, fast_spec):
    from erl_vla_evals.benchmark import BenchmarkResult

    original = run_benchmark(fast_spec, build_policy("scripted"))
    path = original.save(tmp_path / "run.json")
    reloaded = BenchmarkResult.load(path)

    assert reloaded.keys() == original.keys()
    assert reloaded.overall_success_rate == original.overall_success_rate
    # Per-episode detail must survive, or a paired test cannot be run later.
    assert len(reloaded.episodes) == len(original.episodes)


def test_report_renders(fast_spec):
    from erl_vla_evals import report as report_mod

    baseline = run_benchmark(fast_spec, build_policy("scripted"))
    candidate = run_benchmark(fast_spec, build_policy("scripted+noise:0.12"))
    markdown = report_mod.render_comparison(compare(baseline, candidate, bootstrap_samples=500))
    assert "FAIL" in markdown
    assert "Per task" in markdown
    assert "McNemar" in markdown


def test_episode_result_records_enough_for_later_analysis(fast_spec):
    result = run_benchmark(fast_spec, build_policy("scripted"))
    episode = result.episodes[0]
    assert episode.task_id and episode.seed
    assert episode.steps > 0
    assert 0.0 <= episode.path_efficiency <= 1.0


def test_run_episode_terminates_on_a_stuck_policy():
    task = TaskSpec(task_id="t", max_steps=30)
    env = Environment(task)
    outcome = run_episode(env, build_policy("zero"), seed=0)
    assert outcome.steps == 30
    assert not outcome.success
    assert outcome.timed_out
