"""A minimal deterministic reach-and-grasp environment.

**What this is and is not.** It is a kinematic environment, not a physics
simulator: joints integrate commanded deltas subject to limits and rate caps,
and there are no contacts, no dynamics and no friction. It is genuinely
*closed-loop* — the policy's action changes the state, the next observation
depends on that action, and errors compound exactly as they would on hardware —
which is the property that offline action-error metrics fundamentally cannot
give you.

It exists so the harness around it (registry, paired seeding, statistics,
regression gating) can be built, tested and trusted before a MuJoCo or Isaac
scene is wired in. Everything above `Environment` is simulator-agnostic; swapping
in a real one means implementing `reset` and `step` and nothing else.

Do not report success rates from this environment as robot results. They are
harness results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

N_JOINTS = 7

# Franka Panda joint limits (rad). Shared with erl-teleop-pipeline so a policy
# trained there sees the same state space here.
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Smooth, deterministic joint-space -> Cartesian map.

    Deliberately the *same* function `erl-teleop-pipeline` uses to synthesise
    end-effector pose. It is not a real Panda FK chain, and pretending otherwise
    would be worse than an honest placeholder — but it is smooth, invertible
    enough to be reachable, and consistent between the two repositories, which
    is what matters for the harness.
    """
    return np.array(
        [
            0.30 + 0.25 * np.sin(q[0]) * np.cos(q[1]),
            0.25 * np.sin(q[1]) * np.sin(q[0]),
            0.40 + 0.20 * np.cos(q[1] + q[3]),
        ]
    )


@dataclass
class TaskSpec:
    """One benchmark task. Difficulty is expressed in the tolerances, not in code."""

    task_id: str
    position_tolerance: float = 0.05  # metres to count as reached
    max_steps: int = 150
    requires_grasp: bool = False  # must also close the gripper at the target
    grasp_threshold: float = 0.3  # gripper opening below this counts as closed
    action_limit: float = 0.08  # rad per step, matching the teleop rig

    # Difficulty is set by how far the goal is, in metres of end-effector travel.
    #
    # Specified in Cartesian space rather than as a joint-space offset because
    # the joint -> Cartesian map here is strongly compressive and saturating: a
    # 0.35 rad joint offset produces a median of only ~0.05 m of end-effector
    # motion, and three of the seven joints barely move the tip at all. Setting
    # difficulty in joint space therefore does not control difficulty, and it
    # was silently producing episodes that began already solved.
    goal_distance_min: float = 0.15
    goal_distance_max: float = 0.30

    # The policy must *hold* the goal for this many consecutive steps.
    #
    # Without it, success latches the instant the tolerance ball is touched, and
    # a policy emitting uniform noise wanders into it often enough to score 40%
    # on a suite it should score ~0% on. It also matches what a robot actually
    # needs to do: arriving at a grasp pose and immediately leaving is not a
    # successful grasp.
    settle_steps: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class StepResult:
    observation: dict[str, np.ndarray]
    success: bool
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class Environment:
    """Deterministic given (task, seed). Same seed ⇒ byte-identical episode.

    Determinism is not a nicety here: it is what makes paired comparison between
    two checkpoints valid. Both policies must face the *same* problems, or the
    difference between them is partly the difference between their luck.
    """

    def __init__(self, task: TaskSpec) -> None:
        self.task = task
        self._rng = np.random.default_rng(0)
        self.reset(0)

    # -- lifecycle ---------------------------------------------------------

    def reset(self, seed: int) -> dict[str, np.ndarray]:
        self._rng = np.random.default_rng(seed)
        centre = 0.5 * (JOINT_LOWER + JOINT_UPPER)
        spread = 0.15 * (JOINT_UPPER - JOINT_LOWER)

        self.q = np.clip(
            centre + self._rng.uniform(-1, 1, N_JOINTS) * spread, JOINT_LOWER, JOINT_UPPER
        )
        self.dq = np.zeros(N_JOINTS)
        self.grip = 1.0
        self.prev_action = np.zeros(N_JOINTS + 1)
        self.steps = 0
        self.succeeded = False
        self.consecutive_in_tolerance = 0

        # Rejection-sample a goal whose end-effector distance from the start
        # falls in the task's band. Goals are drawn as full joint configurations
        # so they are reachable by construction — sampling a Cartesian point and
        # inverting would need a real IK solver, and an unreachable goal makes a
        # benchmark that measures nothing.
        start_xyz = self.ee_position
        goal_q, goal_xyz, best_error = None, None, np.inf
        lo, hi = self.task.goal_distance_min, self.task.goal_distance_max

        for _ in range(2000):
            candidate_q = self._rng.uniform(JOINT_LOWER, JOINT_UPPER)
            candidate_xyz = forward_kinematics(candidate_q)
            distance = float(np.linalg.norm(start_xyz - candidate_xyz))
            if lo <= distance <= hi:
                goal_q, goal_xyz = candidate_q, candidate_xyz
                break
            # Keep the closest near-miss, so an over-tight band degrades to the
            # best available goal rather than to an arbitrary one.
            error = max(lo - distance, distance - hi)
            if error < best_error:
                best_error, goal_q, goal_xyz = error, candidate_q, candidate_xyz

        self.goal_q = goal_q
        self.goal_xyz = goal_xyz
        self.start_distance = float(np.linalg.norm(start_xyz - self.goal_xyz))
        self.path_length = 0.0
        return self.observation()

    def step(self, action: np.ndarray) -> StepResult:
        """`action` is [7 joint deltas, gripper target]."""
        action = np.asarray(action, dtype=np.float64).ravel()
        if action.size != N_JOINTS + 1:
            raise ValueError(f"expected {N_JOINTS + 1} action dims, got {action.size}")

        # The rig clips commands; so does the environment. A policy that emits
        # enormous actions must not be able to teleport to the goal.
        delta = np.clip(action[:N_JOINTS], -self.task.action_limit, self.task.action_limit)
        previous_xyz = self.ee_position

        new_q = np.clip(self.q + delta, JOINT_LOWER, JOINT_UPPER)
        self.dq = (new_q - self.q) * 20.0  # control runs at 20 Hz
        self.q = new_q
        self.grip = float(np.clip(action[N_JOINTS], 0.0, 1.0))
        self.prev_action = np.concatenate([delta, [self.grip]])
        self.steps += 1
        self.path_length += float(np.linalg.norm(self.ee_position - previous_xyz))

        distance = float(np.linalg.norm(self.ee_position - self.goal_xyz))
        reached = distance <= self.task.position_tolerance
        grasped = (not self.task.requires_grasp) or self.grip <= self.task.grasp_threshold

        # Holding the goal is the success condition, not touching it. The
        # counter resets the moment either condition lapses, so a policy that
        # oscillates through the tolerance ball never accumulates a success.
        if reached and grasped:
            self.consecutive_in_tolerance += 1
        else:
            self.consecutive_in_tolerance = 0
        self.succeeded = self.consecutive_in_tolerance >= self.task.settle_steps

        done = self.succeeded or self.steps >= self.task.max_steps
        return StepResult(
            observation=self.observation(),
            success=self.succeeded,
            done=done,
            info={
                "distance": distance,
                "steps": self.steps,
                "reached": reached,
                "grasped": grasped,
                "settled_for": self.consecutive_in_tolerance,
            },
        )

    # -- state -------------------------------------------------------------

    @property
    def ee_position(self) -> np.ndarray:
        return forward_kinematics(self.q)

    def observation(self) -> dict[str, np.ndarray]:
        """Named channels, matching erl-teleop-pipeline's canonical schema.

        Returned as a dict rather than a flat vector so an adapter can select
        exactly the columns its checkpoint was trained on. A flat vector would
        make the observation layout an implicit contract between two
        repositories, which is precisely the kind of coupling that breaks
        silently six months later.
        """
        half = 0.5 * self.q[5]
        quat = np.array([np.sin(half), 0.0, 0.0, np.cos(half)])
        quat = quat / np.linalg.norm(quat)
        return {
            "q": self.q.copy(),
            "dq": self.dq.copy(),
            "ee_pos": self.ee_position,
            "ee_quat": quat,
            "grip": np.array([self.grip]),
            "prev_act_q": self.prev_action[:N_JOINTS].copy(),
            "prev_act_grip": self.prev_action[N_JOINTS : N_JOINTS + 1].copy(),
            # Goal channels. Not part of the teleop schema — a policy trained on
            # demonstrations alone does not receive them, which is exactly why a
            # goal-blind policy scores near zero here and should.
            "goal_pos": self.goal_xyz.copy(),
            "goal_q": self.goal_q.copy(),
        }

    def path_efficiency(self) -> float:
        """Straight-line distance over distance actually travelled, in [0, 1].

        Distinguishes a policy that reaches the goal directly from one that
        wanders into it. Both score 1.0 on success rate.
        """
        if self.path_length <= 1e-9:
            return 0.0
        return float(min(self.start_distance / self.path_length, 1.0))
