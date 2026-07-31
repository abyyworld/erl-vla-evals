"""Policy adapters.

Adding a new model family — an OpenVLA fine-tune, a diffusion policy, an RT-2
style model — means adding one class with `reset` and `act`. Nothing else in the
harness changes.
"""

from __future__ import annotations

from pathlib import Path

from .base import NoisyScriptedPolicy, Policy, RandomPolicy, ScriptedPolicy, ZeroPolicy

__all__ = [
    "Policy",
    "ZeroPolicy",
    "RandomPolicy",
    "ScriptedPolicy",
    "NoisyScriptedPolicy",
    "build_policy",
]


def build_policy(spec: str, **kwargs) -> Policy:
    """Construct a policy from a short spec string.

    Built-ins by name (`zero`, `random`, `scripted`, `scripted+noise:0.02`), or a
    path to an teleop-data-pipeline checkpoint.
    """
    if spec == "zero":
        return ZeroPolicy()
    if spec == "random":
        return RandomPolicy(**kwargs)
    if spec == "scripted":
        return ScriptedPolicy(**kwargs)
    if spec.startswith("scripted+noise"):
        _, _, amount = spec.partition(":")
        return NoisyScriptedPolicy(noise=float(amount or 0.02), **kwargs)

    path = Path(spec)
    if path.exists():
        from .checkpoint import CheckpointPolicy

        return CheckpointPolicy(path, **kwargs)

    raise ValueError(
        f"unknown policy {spec!r}: expected one of zero, random, scripted, "
        f"scripted+noise:<sigma>, or a path to a checkpoint"
    )
