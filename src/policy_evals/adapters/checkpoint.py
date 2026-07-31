"""Adapter for checkpoints produced by teleop-data-pipeline.

The checkpoint format is self-describing — weights, normalisation statistics and
the exact observation column order travel with it — so this loads without
importing the training package or reading its config. That property is the whole
reason the two repositories can be versioned independently.

The observation mapping is explicit and validated rather than positional. If a
checkpoint expects a column this environment does not provide, it fails loudly
at load time with the name of the missing column, instead of silently feeding
the policy the wrong number in that slot and reporting a bad success rate that
looks like a modelling problem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Maps a canonical column prefix to the observation channel that supplies it.
CHANNEL_FOR_PREFIX = {
    "q_": "q",
    "dq_": "dq",
    "ee_x": "ee_pos",
    "ee_y": "ee_pos",
    "ee_z": "ee_pos",
    "ee_q": "ee_quat",
    "grip": "grip",
    "prev_act_q_": "prev_act_q",
    "prev_act_grip": "prev_act_grip",
}

CHANNEL_ORDER = {
    "q": [f"q_{i}" for i in range(7)],
    "dq": [f"dq_{i}" for i in range(7)],
    "ee_pos": ["ee_x", "ee_y", "ee_z"],
    "ee_quat": ["ee_qx", "ee_qy", "ee_qz", "ee_qw"],
    "grip": ["grip"],
    "prev_act_q": [f"prev_act_q_{i}" for i in range(7)],
    "prev_act_grip": ["prev_act_grip"],
}


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "evaluating a trained checkpoint needs PyTorch.\n  pip install -e '.[torch]'"
        ) from exc
    return torch


def build_index(obs_columns: list[str]) -> list[tuple[str, int]]:
    """Resolve each expected column to (channel, offset within that channel)."""
    lookup: dict[str, tuple[str, int]] = {}
    for channel, columns in CHANNEL_ORDER.items():
        for offset, column in enumerate(columns):
            lookup[column] = (channel, offset)

    index: list[tuple[str, int]] = []
    missing: list[str] = []
    for column in obs_columns:
        if column in lookup:
            index.append(lookup[column])
        else:
            missing.append(column)

    if missing:
        raise KeyError(
            f"checkpoint expects observation column(s) this environment does not "
            f"provide: {missing}. Available channels: {sorted(CHANNEL_ORDER)}"
        )
    return index


class CheckpointPolicy:
    """Runs an teleop-data-pipeline behaviour-cloning checkpoint in the loop."""

    def __init__(self, checkpoint_path: Path, name: str | None = None, device: str = "cpu") -> None:
        torch = _require_torch()
        from torch import nn

        ckpt = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        self.name = name or ckpt.get("run_name") or Path(checkpoint_path).stem
        self.obs_columns: list[str] = ckpt["obs_columns"]
        self.action_columns: list[str] = ckpt["action_columns"]
        self.horizon = int(ckpt["action_horizon"])
        self.dataset_hash = ckpt.get("dataset_hash")
        self._index = build_index(self.obs_columns)

        norm = ckpt["norm"]
        self._obs_mean = np.array([norm["obs"][c]["mean"] for c in self.obs_columns], np.float32)
        self._obs_std = np.array([norm["obs"][c]["std"] for c in self.obs_columns], np.float32)
        self._act_mean = np.array(
            [norm["action"][c]["mean"] for c in self.action_columns], np.float32
        )
        self._act_std = np.array(
            [norm["action"][c]["std"] for c in self.action_columns], np.float32
        )

        hidden = ckpt["arch"]["hidden_sizes"]
        dropout = float(ckpt["arch"]["dropout"])
        layers: list[nn.Module] = []
        prev = len(self.obs_columns)
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.LayerNorm(width), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, self.horizon * len(self.action_columns)))

        self._torch = torch
        self._device = torch.device(device)
        self._net = nn.Sequential(*layers)
        self._net.load_state_dict(
            {k.removeprefix("net."): v for k, v in ckpt["state_dict"].items()}
        )
        self._net.eval().to(self._device)

        self._queue: list[np.ndarray] = []

    def reset(self) -> None:
        # Any buffered chunk belongs to the previous episode.
        self._queue = []

    def _flatten(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.array(
            [float(observation[channel][offset]) for channel, offset in self._index],
            dtype=np.float32,
        )

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        # Re-query only when the previous chunk is exhausted — the same open-loop
        # execution the policy was trained for, and the same one it would run on
        # hardware. Re-querying every step would flatter a chunked policy.
        if not self._queue:
            obs = (self._flatten(observation) - self._obs_mean) / self._obs_std
            with self._torch.no_grad():
                tensor = self._torch.from_numpy(obs[None, :]).float().to(self._device)
                raw = self._net(tensor).cpu().numpy()
            chunk = raw.reshape(self.horizon, len(self.action_columns))
            chunk = chunk * self._act_std + self._act_mean
            self._queue = list(chunk)

        step = self._queue.pop(0)
        action = np.zeros(8, dtype=np.float64)
        for i, column in enumerate(self.action_columns):
            if column.startswith("act_q_"):
                action[int(column.rsplit("_", 1)[1])] = step[i]
            elif column == "act_grip":
                action[7] = step[i]
        return action
