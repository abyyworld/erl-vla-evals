# The 0.0078 rad policy that succeeds 0% of the time

This is the finding that justifies the existence of this repository.

## The two numbers

The behaviour-cloning policy trained by
[`teleop-data-pipeline`](https://github.com/abyyworld/teleop-data-pipeline) was
evaluated twice: once offline on held-out demonstrations, once closed-loop here.

**Offline** (`teleop-pipeline eval`, 55 held-out episodes across 10 held-out sessions):

| Metric | Value |
| --- | ---: |
| action MAE | 0.00779 rad (95% CI [0.00701, 0.00849]) |
| action RMSE | 0.03354 rad |
| gripper accuracy | 97.9% |
| rollout drift @1 step | 0.0046 |

An action error under a hundredth of a radian and 98% gripper agreement. Nothing
in that table says "this policy does not work".

**Closed-loop** (`policy-evals run`, 240 episodes across 4 tasks):

| Task | Success rate |
| --- | ---: |
| `reach_near` | 0.0% |
| `reach_far` | 0.0% |
| `reach_precise` | 0.0% |
| `grasp_at_target` | 0.0% |
| **Overall** | **0.0%** |

Zero. Not degraded — zero. The same checkpoint, on the same robot description,
in the same units.

## Why

The policy's observation is:

```
q_0..q_6, dq_0..dq_6, ee_x, ee_y, ee_z, ee_qx..ee_qw, grip,
prev_act_q_0..prev_act_q_6, prev_act_grip
```

There is no goal channel. The policy was trained to imitate teleoperation
trajectories, and those trajectories never carried a representation of *where
the operator was trying to go* — the goal lived in the operator's head. So the
policy learned the marginal distribution of plausible next actions given the
arm's current state, which is genuinely what the offline metric measures, and
which is genuinely useful for predicting the next action.

It cannot reach a specified target, because it was never told there was one.

The offline metric is not wrong. It is answering a different question:

- **Offline action error** asks *"given this state, would a demonstrator have
  moved like this?"* Averaged over states drawn from demonstrations.
- **Closed-loop success** asks *"starting here, does this policy accomplish the
  task?"* Over states the policy itself drives into.

A goal-blind policy scores well on the first and zero on the second, and no
amount of staring at the first will reveal the second.

## Why this is not a straw man

The failure is stark here because the benchmark is goal-conditioned and the
policy is not. On real data the same gap appears in subtler and more dangerous
forms:

- **Compounding error.** Offline metrics evaluate at states drawn from
  demonstrations. A policy runs at states it produced itself, which drift from
  that distribution — the classic covariate shift argument for DAgger. Offline
  error is measured exactly where the policy is strongest.
- **Causal confusion.** A policy fed its own previous action can learn to copy
  it and ignore the state entirely. Offline this looks near-perfect; on hardware
  it drifts. (`teleop-data-pipeline` feeds `prev_act_*` in, and documents the
  trade-off in its own `docs/BASELINES.md`.)
- **Multimodality.** When demonstrators solved a task two ways, the
  error-minimising prediction is the average of the two — which is often a
  trajectory that hits the obstacle between them. Low error, guaranteed failure.

In each case the offline number is fine and the robot does not work.

## What to do with this

Nothing here says the pipeline is broken. It says the evaluation was
incomplete, which was true and is now visible. Concretely:

1. **Goal-condition the policy.** Add a target channel to the observation and
   collect demonstrations labelled with what the operator was reaching for.
   That is a data-collection change, not a modelling one, and it is much cheaper
   to discover now than after the demonstrations are collected.
2. **Keep both metrics.** Offline error is a fast regression signal — it catches
   a broken checkpoint or a normalisation mismatch in seconds. It is a smoke
   test, not a result.
3. **Gate promotions on closed-loop numbers only.** `policy-evals gate` compares
   against the production checkpoint on identical seeds and refuses to pass a
   regression, or an underpowered comparison that cannot tell.

## Reproducing

```bash
# in teleop-data-pipeline
dvc repro
cat reports/eval_val.json

# in policy-eval-harness
policy-evals registry add ../teleop-data-pipeline/artifacts/policy.pt --name bc-teleop-v1
policy-evals run <checkpoint-id>
```

## The honest caveat

The environment here is kinematic, not physical — no contacts, no dynamics. A
0% here is not a claim about hardware. But the *mechanism* is not an artefact of
the simulator: a policy with no goal input cannot reach a specified goal in any
environment, and that is exactly the class of thing the offline metric could
never have told you.
