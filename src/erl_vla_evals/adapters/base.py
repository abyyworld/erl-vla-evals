"""The policy interface every evaluated model implements.

Two methods. Anything more and the harness starts encoding assumptions about
what kind of model it is evaluating, which is how an evaluation suite ends up
only able to evaluate the model it was written for.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    """Maps an observation to an action.

    `observation` is the named-channel dict from `env.Environment.observation`.
    `act` returns `[7 joint deltas, gripper target]`.
    """

    name: str

    def reset(self) -> None:
        """Called at the start of every episode. Clear any internal state here."""

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray: ...


class ZeroPolicy:
    """Commands nothing. The floor.

    A benchmark on which this scores above zero is measuring the environment,
    not the policy — worth checking before trusting any other number on the
    suite.
    """

    name = "zero"

    def reset(self) -> None:
        return None

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros(8)


class RandomPolicy:
    """Uniform random joint deltas. The noise floor for a suite."""

    name = "random"

    def __init__(self, action_limit: float = 0.08, seed: int = 0) -> None:
        self.action_limit = action_limit
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        # Re-seeded per episode so a random baseline is itself reproducible.
        self._rng = np.random.default_rng(self._seed)

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([self._rng.uniform(-self.action_limit, self.action_limit, 7), [1.0]])


class ScriptedPolicy:
    """Proportional controller straight to the goal configuration.

    An oracle: it reads `goal_q`, which a policy trained on demonstrations never
    sees. Its role is to establish the achievable ceiling on each task. If the
    oracle cannot solve a task, the task is broken or the step budget is too
    tight, and no learned policy's score on it means anything.
    """

    name = "scripted"

    def __init__(self, gain: float = 0.5, action_limit: float = 0.08) -> None:
        self.gain = gain
        self.action_limit = action_limit

    def reset(self) -> None:
        return None

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        error = observation["goal_q"] - observation["q"]
        delta = np.clip(self.gain * error, -self.action_limit, self.action_limit)
        # Close the gripper once essentially on target, so grasp tasks are
        # solvable by the oracle too.
        close = float(np.linalg.norm(error) < 0.15)
        return np.concatenate([delta, [0.0 if close else 1.0]])


class NoisyScriptedPolicy(ScriptedPolicy):
    """The oracle with its commands corrupted.

    The regression gate needs a *known* degradation to be tested against: a
    policy that is genuinely, measurably worse by a controllable amount. Without
    one, "the gate catches regressions" is an untested claim.
    """

    def __init__(
        self, noise: float = 0.02, gain: float = 0.5, action_limit: float = 0.08, seed: int = 0
    ) -> None:
        super().__init__(gain=gain, action_limit=action_limit)
        self.name = f"scripted+noise{noise}"
        self.noise = noise
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        action = super().act(observation)
        # Corrupt only the joint deltas; the gripper decision stays correct, so
        # the degradation is purely in control precision.
        action[:7] = np.clip(
            action[:7] + self._rng.normal(0, self.noise, 7),
            -self.action_limit,
            self.action_limit,
        )
        return action
