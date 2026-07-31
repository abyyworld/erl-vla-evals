# policy-eval-harness

**Checkpoint registry and a statistically honest evaluation harness for robot policies.**

Every checkpoint is content-addressed and traceable to the data and commit that
produced it. Every comparison is paired, interval-estimated, and refuses to
declare a winner it cannot actually resolve.

```bash
git clone https://github.com/abyyworld/policy-eval-harness
cd policy-eval-harness
make install
make baselines     # confirm the suite discriminates at all
make demo-gate     # watch the gate reject an injected regression
```

Downstream of [`teleop-data-pipeline`](https://github.com/abyyworld/teleop-data-pipeline),
which produces the checkpoints and the lineage records this consumes.

---

## The finding this repo exists for

A behaviour-cloning policy with an offline action error of **0.0078 rad** and
**97.9% gripper accuracy** achieves **0.0% closed-loop success** on this suite.

| | Offline (`teleop-pipeline eval`) | Closed-loop (`policy-evals run`) |
| --- | ---: | ---: |
| action MAE | 0.00779 rad | — |
| gripper accuracy | 97.9% | — |
| success rate | — | **0.0%** |

Same checkpoint, same robot description, same units. The cause is concrete: the
policy's observation contains no goal channel, because the teleoperation
demonstrations never carried one — the goal lived in the operator's head. The
offline metric is not wrong, it is answering a different question, and no amount
of examining it would have revealed this.

Full write-up: [`docs/OFFLINE_VS_CLOSED_LOOP.md`](docs/OFFLINE_VS_CLOSED_LOOP.md).

## What this does

### 1. A registry that refuses untraceable checkpoints

```bash
policy-evals registry add ../teleop-data-pipeline/artifacts/policy.pt
policy-evals registry list
policy-evals registry promote a4f894f8 --stage production
```

- **Content-addressed.** A checkpoint's identity is the SHA-256 of its bytes.
  `policy_final_v2_REAL.pt` is not an identifier. Registering identical bytes
  twice is detected and idempotent, so "have we already evaluated this?" is
  answerable.
- **Provenance is mandatory by default.** Registration reads the `lineage.json`
  written by the training pipeline and *refuses* a checkpoint that cannot say
  which dataset and commit produced it. `--no-require-lineage` exists, and
  records the gap on the entry.
- **A dirty working tree is a recorded provenance gap.** It means the commit
  does not describe what actually ran.
- **Promotion archives, never deletes.** "What was in production in March" stays
  answerable.

### 2. A benchmark suite that is checked for being able to discriminate

Four tasks, run closed-loop. Difficulty is set in metres of end-effector travel
and in tolerance, not in code.

| Task | Goal distance | Tolerance | What it isolates |
| --- | --- | ---: | --- |
| `reach_near` | 0.15–0.30 m | 5 cm | Basic competence |
| `reach_far` | 0.35–0.55 m | 5 cm | Sustained motion; where compounding error shows |
| `reach_precise` | 0.15–0.30 m | 1.5 cm | Settling vs merely approaching |
| `grasp_at_target` | 0.20–0.40 m | 5 cm | Gripper timing, invisible in joint-error metrics |

Reference policies, and what the suite says about them:

| Policy | Success |
| --- | ---: |
| `zero` (do nothing) | 0.0% |
| `random` | 0.4% |
| `scripted` (oracle, sees the goal) | 100.0% |
| `scripted+noise:0.05` | 97.9% |
| `scripted+noise:0.12` | 55.4% |

Those three anchors are **asserted in CI**. A benchmark on which a do-nothing
policy scores well is measuring the environment, not the policy — and every
number the repo produces would be worthless. Two real bugs were caught exactly
this way while building it:

- `zero` scored **38%**, because goals were sampled as joint-space offsets while
  success was measured in Cartesian space, and the map between them is
  compressive enough that episodes routinely began already solved.
- `random` scored **42%**, because success latched the instant the tolerance
  ball was *touched* — so flailing worked. Success now requires *holding* the
  goal for `settle_steps` consecutive steps, which is also what a robot actually
  has to do.

### 3. A gate that will not be rushed

```bash
policy-evals gate <candidate> --tolerance 0.02
```

Benchmarks the candidate and the production checkpoint on **identical seeds**,
then compares:

- **Paired**, so episode difficulty is removed from the comparison entirely.
  `test_pairing_narrows_the_interval` demonstrates this shrinks the interval by
  more than 3× on the same data.
- **Paired bootstrap CI on the difference**, not on each arm separately.
- **McNemar's exact test** on the binary outcomes. Only discordant pairs — the
  episodes where the policies disagreed — carry information; the ones both
  solved tell you nothing about which is better. Exact binomial rather than
  chi-squared, because suites routinely produce fewer than 25 discordant pairs,
  which is precisely where the approximation errs toward false confidence.
- **Per-task regressions checked separately.** A candidate can hold its overall
  rate while collapsing on one task. That is the failure aggregates hide, and it
  has its own test.

Three verdicts, not two:

| Verdict | Meaning |
| --- | --- |
| `PASS` | The candidate is demonstrably not a regression beyond tolerance |
| `FAIL` | The interval rules out "no worse than tolerance", or a task collapsed |
| `INCONCLUSIVE` | **The run cannot resolve the question.** Reports how many episodes would be needed |

`INCONCLUSIVE` is the one most harnesses lack. Absence of evidence is not
evidence of absence: a comparison whose interval is wider than the tolerance it
is testing does not pass, because a gate that silently waves through
underpowered comparisons manufactures confidence rather than providing it.

## Why not just compare the two success rates?

Because with 50 episodes a 6-point difference is comfortably inside noise, and
promoting on it is a coin flip wearing a decision's clothing.

`stats.required_episodes` will tell you how big the study needs to be before you
spend a week running it. Resolving a 5-point difference around a 60% baseline
takes over a thousand episodes per arm. Most published robot comparisons are not
close to that, and most are reported without an interval.

## Adding your own model

Two methods. Anything more and the harness starts encoding assumptions about
what it is evaluating.

```python
class MyVLAPolicy:
    name = "openvla-ft-v3"

    def reset(self) -> None:
        ...  # called at the start of every episode

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        ...  # returns [7 joint deltas, gripper target]
```

`CheckpointPolicy` (for `teleop-data-pipeline` checkpoints) is the worked example.
It validates the observation mapping by name at load time, so a checkpoint
expecting a column the environment does not supply fails loudly with the
column's name — rather than silently receiving the wrong number in that slot and
producing a bad success rate that looks like a modelling problem.

## About the environment

**It is kinematic, not physical.** Joints integrate commanded deltas subject to
limits and rate caps. No contacts, no dynamics, no friction. Success rates from
it are harness results, not robot results, and the reports say so on every page.

It is genuinely *closed-loop* — the policy's action changes the state, the next
observation depends on that action, and error compounds as it would on hardware.
That is the property offline action-error metrics cannot provide at any price,
and it is what the whole apparatus above is built to exploit.

Swapping in MuJoCo or Isaac means implementing `reset` and `step`. Nothing in
the registry, the pairing, the statistics or the gate knows what simulator it is
talking to.

## Layout

```
conf/benchmarks/manipulation_v1.yaml   the suite, versioned as a file
src/policy_evals/
  registry.py     content-addressed checkpoints, provenance, staging
  env.py          kinematic closed-loop environment
  benchmark.py    suite runner; identical seeds for every policy
  stats.py        paired bootstrap, McNemar exact, power calculation
  compare.py      the regression gate and its three verdicts
  report.py       Markdown reports for PR comments
  adapters/       policy interface, reference baselines, checkpoint loader
tests/            40 tests, incl. injected regressions the gate must catch
```

## Commands

```bash
make install-all   # venv + torch, for evaluating real checkpoints
make test          # 40 tests
make baselines     # confirm the suite still discriminates
make demo-gate     # inject a regression and watch the gate reject it

policy-evals registry add <checkpoint>       # register, with provenance
policy-evals registry promote <id> --stage production
policy-evals run <policy|id>                 # benchmark
policy-evals compare <baseline> <candidate>  # paired comparison + gate
policy-evals gate <candidate>                # both, against production
```

## Status

Working end to end. CI asserts the suite discriminates and that the gate
rejects a known regression on every push. The environment is a placeholder for a
real simulator; the registry, statistics and gate are not.

## Licence

MIT.
