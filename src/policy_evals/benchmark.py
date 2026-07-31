"""Benchmark suite runner.

The one rule that everything else depends on: **every policy on a suite sees the
same seeds.** Episode `i` of task `t` is identical for every policy evaluated,
which turns the comparison into a matched-pairs design and lets `stats.py` remove
episode difficulty from the difference entirely. Evaluating two policies on
independently sampled episodes throws that away and needs several times as many
episodes to reach the same confidence.

Per-episode outcomes are recorded, not just aggregates. A suite that reports only
a success rate cannot support a paired test afterwards, and re-running to get the
detail costs another full evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .adapters.base import Policy
from .env import Environment, TaskSpec


@dataclass
class EpisodeResult:
    task_id: str
    seed: int
    success: bool
    steps: int
    final_distance: float
    path_efficiency: float
    timed_out: bool


@dataclass
class TaskResult:
    task_id: str
    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return float(np.mean([e.success for e in self.episodes])) if self.episodes else 0.0

    @property
    def mean_steps_on_success(self) -> float | None:
        solved = [e.steps for e in self.episodes if e.success]
        return float(np.mean(solved)) if solved else None

    @property
    def mean_path_efficiency(self) -> float:
        solved = [e.path_efficiency for e in self.episodes if e.success]
        return float(np.mean(solved)) if solved else 0.0


@dataclass
class BenchmarkResult:
    benchmark: str
    policy: str
    ran_at: str
    tasks: list[TaskResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def episodes(self) -> list[EpisodeResult]:
        """Every episode, in a stable order so two runs align pairwise."""
        return [e for task in self.tasks for e in task.episodes]

    @property
    def success_vector(self) -> np.ndarray:
        return np.array([e.success for e in self.episodes], dtype=bool)

    @property
    def overall_success_rate(self) -> float:
        episodes = self.episodes
        return float(np.mean([e.success for e in episodes])) if episodes else 0.0

    def keys(self) -> list[tuple[str, int]]:
        """(task, seed) identifiers — used to verify two runs really are paired."""
        return [(e.task_id, e.seed) for e in self.episodes]

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "policy": self.policy,
            "ran_at": self.ran_at,
            "metadata": self.metadata,
            "overall_success_rate": self.overall_success_rate,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "success_rate": t.success_rate,
                    "mean_steps_on_success": t.mean_steps_on_success,
                    "mean_path_efficiency": t.mean_path_efficiency,
                    "episodes": [asdict(e) for e in t.episodes],
                }
                for t in self.tasks
            ],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @staticmethod
    def load(path: Path) -> BenchmarkResult:
        data = json.loads(Path(path).read_text())
        return BenchmarkResult(
            benchmark=data["benchmark"],
            policy=data["policy"],
            ran_at=data["ran_at"],
            metadata=data.get("metadata", {}),
            tasks=[
                TaskResult(
                    task_id=t["task_id"],
                    episodes=[EpisodeResult(**e) for e in t["episodes"]],
                )
                for t in data["tasks"]
            ],
        )


@dataclass
class BenchmarkSpec:
    name: str
    episodes_per_task: int
    seed_base: int
    tasks: list[TaskSpec]

    @staticmethod
    def load(path: Path) -> BenchmarkSpec:
        data = yaml.safe_load(Path(path).read_text())
        return BenchmarkSpec(
            name=data["name"],
            episodes_per_task=int(data.get("episodes_per_task", 50)),
            seed_base=int(data.get("seed_base", 1000)),
            tasks=[TaskSpec.from_dict(t) for t in data["tasks"]],
        )

    def seeds_for(self, task: TaskSpec) -> list[int]:
        """Deterministic, task-specific seeds.

        Derived from the task id so two tasks do not evaluate on the identical
        set of goals — which would make their scores correlated and any per-task
        breakdown misleading.

        Uses SHA-256 rather than the builtin `hash()`. Python randomises string
        hashing per process, so `hash()` here silently gave every CLI invocation
        a different set of seeds: two runs of the same suite were not comparable,
        and `compare` rejected them as unpaired. The seeds must be stable across
        processes, machines and Python versions, because the entire paired design
        rests on both policies seeing identical episodes.
        """
        digest = hashlib.sha256(task.task_id.encode()).digest()
        offset = int.from_bytes(digest[:4], "big") % 9973
        return [self.seed_base + offset + i for i in range(self.episodes_per_task)]


def run_episode(env: Environment, policy: Policy, seed: int) -> EpisodeResult:
    observation = env.reset(seed)
    policy.reset()

    result = None
    while True:
        result = env.step(policy.act(observation))
        observation = result.observation
        if result.done:
            break

    return EpisodeResult(
        task_id=env.task.task_id,
        seed=seed,
        success=bool(result.success),
        steps=int(result.info["steps"]),
        final_distance=float(result.info["distance"]),
        path_efficiency=env.path_efficiency(),
        timed_out=not result.success,
    )


def run_benchmark(
    spec: BenchmarkSpec,
    policy: Policy,
    metadata: dict[str, Any] | None = None,
    progress=None,
) -> BenchmarkResult:
    """Run every task in the suite. Deterministic given the spec and the policy."""
    tasks: list[TaskResult] = []
    for task in spec.tasks:
        env = Environment(task)
        episodes = [run_episode(env, policy, seed) for seed in spec.seeds_for(task)]
        tasks.append(TaskResult(task_id=task.task_id, episodes=episodes))
        if progress:
            progress(task.task_id, episodes)

    return BenchmarkResult(
        benchmark=spec.name,
        policy=getattr(policy, "name", policy.__class__.__name__),
        ran_at=datetime.now(timezone.utc).isoformat(),
        tasks=tasks,
        metadata={
            "episodes_per_task": spec.episodes_per_task,
            "seed_base": spec.seed_base,
            "n_tasks": len(spec.tasks),
            **(metadata or {}),
        },
    )
