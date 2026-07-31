"""Content-addressed checkpoint registry.

A checkpoint you cannot trace to a data version and a commit is not a result,
it is a file. This registry makes the trace mandatory rather than optional:
registration reads the `lineage.json` written by `teleop-data-pipeline` and
refuses anything that cannot say where it came from.

Design choices worth stating:

**Content addressing.** A checkpoint's identity is the SHA-256 of its bytes, not
its filename. `policy_final_v2_REAL.pt` is not an identifier. Registering the
same bytes twice is idempotent and detected, which is what makes "did we already
evaluate this?" answerable.

**JSON index, not a database.** The index is a single human-readable file that
diffs cleanly in git and needs no server. A lab that has to run Postgres to see
its own model list will stop looking at its model list. If this outgrows JSON,
the interface is narrow enough to swap.

**Stages, not deletion.** Checkpoints move between `staging`, `production` and
`archived`. Nothing is ever removed by promotion, so "what was in production in
March" stays answerable.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Stage = Literal["staging", "production", "archived"]
DEFAULT_REGISTRY = "registry"


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class RegistryError(RuntimeError):
    pass


class Checkpoint(BaseModel):
    """One registered model version."""

    checkpoint_id: str  # first 16 hex chars of the content hash
    sha256: str
    name: str
    stage: Stage = "staging"
    registered_at: str
    bytes: int
    path: str

    # Provenance, carried forward from the training pipeline. `dataset_hash` is
    # the join key between this registry and teleop-data-pipeline.
    dataset_hash: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    train_run_name: str | None = None
    train_metrics: dict[str, float] = Field(default_factory=dict)

    notes: str = ""
    tags: dict[str, str] = Field(default_factory=dict)

    @property
    def short_commit(self) -> str:
        return (self.git_commit or "")[:12] or "—"

    def provenance_gaps(self) -> list[str]:
        """What is missing before this could be trusted in production."""
        gaps = []
        if not self.dataset_hash:
            gaps.append("no dataset_hash")
        if not self.git_commit:
            gaps.append("no git commit")
        if self.git_dirty:
            gaps.append("trained from a dirty working tree")
        return gaps


class RegistryIndex(BaseModel):
    version: int = 1
    checkpoints: list[Checkpoint] = Field(default_factory=list)


class Registry:
    """File-backed checkpoint registry rooted at a directory."""

    def __init__(self, root: Path | str = DEFAULT_REGISTRY) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        self.blobs = self.root / "checkpoints"

    # -- persistence -------------------------------------------------------

    def load(self) -> RegistryIndex:
        if not self.index_path.exists():
            return RegistryIndex()
        return RegistryIndex.model_validate_json(self.index_path.read_text())

    def save(self, index: RegistryIndex) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(index.model_dump_json(indent=2))

    # -- queries -----------------------------------------------------------

    def list(self, stage: Stage | None = None) -> list[Checkpoint]:
        items = self.load().checkpoints
        if stage:
            items = [c for c in items if c.stage == stage]
        return sorted(items, key=lambda c: c.registered_at, reverse=True)

    def get(self, checkpoint_id: str) -> Checkpoint:
        """Look up by id, accepting any unambiguous prefix."""
        matches = [
            c
            for c in self.load().checkpoints
            if c.checkpoint_id == checkpoint_id or c.checkpoint_id.startswith(checkpoint_id)
        ]
        if not matches:
            raise RegistryError(f"no checkpoint matching {checkpoint_id!r}")
        if len(matches) > 1:
            ids = ", ".join(c.checkpoint_id for c in matches)
            raise RegistryError(f"{checkpoint_id!r} is ambiguous: {ids}")
        return matches[0]

    def production(self) -> Checkpoint | None:
        """The current production checkpoint — the baseline to beat."""
        items = self.list(stage="production")
        return items[0] if items else None

    def resolve(self, ref: str | None) -> Checkpoint:
        """Resolve `production`, `latest`, or a checkpoint id."""
        if ref in (None, "production"):
            current = self.production()
            if current is None:
                raise RegistryError("no checkpoint is in production")
            return current
        if ref == "latest":
            items = self.list()
            if not items:
                raise RegistryError("registry is empty")
            return items[0]
        return self.get(ref)

    def blob_path(self, checkpoint: Checkpoint) -> Path:
        return self.root / checkpoint.path

    # -- mutation ----------------------------------------------------------

    def register(
        self,
        checkpoint_path: Path,
        *,
        name: str | None = None,
        lineage_path: Path | None = None,
        stage: Stage = "staging",
        notes: str = "",
        tags: dict[str, str] | None = None,
        copy: bool = True,
        require_lineage: bool = True,
    ) -> tuple[Checkpoint, bool]:
        """Register a checkpoint. Returns (record, was_newly_created).

        Registering identical bytes twice is a no-op rather than an error —
        re-running a pipeline should not fail, and it should not create a
        duplicate entry that quietly splits the evaluation history in two.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise RegistryError(f"no such checkpoint: {checkpoint_path}")

        sha = file_sha256(checkpoint_path)
        checkpoint_id = sha[:16]

        index = self.load()
        for existing in index.checkpoints:
            if existing.sha256 == sha:
                return existing, False

        lineage = self._read_lineage(checkpoint_path, lineage_path, require_lineage)

        self.blobs.mkdir(parents=True, exist_ok=True)
        rel = Path("checkpoints") / f"{checkpoint_id}.pt"
        dest = self.root / rel
        if copy:
            shutil.copy2(checkpoint_path, dest)
        else:
            # Reference in place. Cheaper for large checkpoints, but the
            # registry no longer controls the bytes it is addressing.
            rel = Path(checkpoint_path).resolve()

        git = (lineage or {}).get("git") or {}
        record = Checkpoint(
            checkpoint_id=checkpoint_id,
            sha256=sha,
            name=name or (lineage or {}).get("run_name") or checkpoint_path.stem,
            stage=stage,
            registered_at=datetime.now(timezone.utc).isoformat(),
            bytes=checkpoint_path.stat().st_size,
            path=str(rel),
            dataset_hash=(lineage or {}).get("dataset_hash"),
            git_commit=git.get("commit"),
            git_dirty=git.get("dirty"),
            train_run_name=(lineage or {}).get("run_name"),
            train_metrics={
                k: float(v)
                for k, v in ((lineage or {}).get("metrics") or {}).items()
                if isinstance(v, (int, float))
            },
            notes=notes,
            tags=tags or {},
        )

        index.checkpoints.append(record)
        self.save(index)
        return record, True

    @staticmethod
    def _read_lineage(
        checkpoint_path: Path, lineage_path: Path | None, require: bool
    ) -> dict | None:
        candidates = [lineage_path] if lineage_path else []
        candidates += [
            checkpoint_path.with_suffix(".lineage.json"),
            checkpoint_path.parent / "lineage.json",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return json.loads(Path(candidate).read_text())

        if require:
            raise RegistryError(
                "no lineage.json found next to the checkpoint.\n"
                "A checkpoint without provenance cannot be traced to the data or "
                "the commit that produced it, so it is refused by default.\n"
                "Pass --lineage explicitly, or --no-require-lineage to register "
                "an untraceable checkpoint anyway."
            )
        return None

    def promote(self, checkpoint_id: str, stage: Stage) -> Checkpoint:
        """Move a checkpoint to a stage.

        Promoting to production archives whatever was there — one production
        checkpoint at a time, and the previous one stays in the index so the
        history remains answerable.
        """
        index = self.load()
        target = self.get(checkpoint_id)

        for record in index.checkpoints:
            if stage == "production" and record.stage == "production":
                record.stage = "archived"
            if record.checkpoint_id == target.checkpoint_id:
                record.stage = stage
                target = record

        self.save(index)
        return target

    def set_eval(self, checkpoint_id: str, benchmark: str, summary: dict) -> None:
        """Attach an evaluation summary to a checkpoint's tags for quick listing."""
        index = self.load()
        for record in index.checkpoints:
            if record.checkpoint_id == self.get(checkpoint_id).checkpoint_id:
                record.tags[f"eval:{benchmark}"] = json.dumps(summary, sort_keys=True)
        self.save(index)
