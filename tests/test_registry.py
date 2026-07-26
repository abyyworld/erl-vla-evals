from __future__ import annotations

import json

import pytest

from erl_vla_evals.registry import Registry, RegistryError


@pytest.fixture
def checkpoint(tmp_path):
    """A fake checkpoint with the lineage erl-teleop-pipeline would have written."""

    def _make(name: str = "policy", content: bytes = b"weights-v1", **lineage_overrides):
        directory = tmp_path / name
        directory.mkdir(exist_ok=True)
        path = directory / "policy.pt"
        path.write_bytes(content)

        lineage = {
            "run_name": f"bc-{name}",
            "dataset_hash": "4ddbf4ffc8a625b6",
            "git": {"commit": "a" * 40, "branch": "main", "dirty": False},
            "metrics": {"best_val_loss": 0.301},
        }
        lineage.update(lineage_overrides)
        (directory / "lineage.json").write_text(json.dumps(lineage))
        return path

    return _make


def test_identity_is_content_not_filename(tmp_path, checkpoint):
    """Two files with different names but identical bytes are one checkpoint."""
    reg = Registry(tmp_path / "registry")
    first, created_a = reg.register(checkpoint("run_a", b"same-bytes"))
    second, created_b = reg.register(checkpoint("run_b", b"same-bytes"))

    assert created_a is True
    assert created_b is False, "identical bytes were registered twice"
    assert first.checkpoint_id == second.checkpoint_id
    assert len(reg.list()) == 1


def test_different_bytes_are_different_checkpoints(tmp_path, checkpoint):
    reg = Registry(tmp_path / "registry")
    a, _ = reg.register(checkpoint("a", b"weights-v1"))
    b, _ = reg.register(checkpoint("b", b"weights-v2"))
    assert a.checkpoint_id != b.checkpoint_id
    assert len(reg.list()) == 2


def test_provenance_is_captured(tmp_path, checkpoint):
    reg = Registry(tmp_path / "registry")
    record, _ = reg.register(checkpoint())
    assert record.dataset_hash == "4ddbf4ffc8a625b6"
    assert record.git_commit == "a" * 40
    assert record.train_metrics["best_val_loss"] == pytest.approx(0.301)
    assert record.provenance_gaps() == []


def test_checkpoint_without_lineage_is_refused(tmp_path):
    """The default must be to refuse an untraceable checkpoint."""
    orphan = tmp_path / "orphan.pt"
    orphan.write_bytes(b"who-made-me")

    reg = Registry(tmp_path / "registry")
    with pytest.raises(RegistryError, match="lineage"):
        reg.register(orphan)

    # Explicit opt-out still works, and the gap is recorded.
    record, _ = reg.register(orphan, require_lineage=False)
    assert "no dataset_hash" in record.provenance_gaps()
    assert "no git commit" in record.provenance_gaps()


def test_dirty_tree_is_flagged_as_a_provenance_gap(tmp_path, checkpoint):
    reg = Registry(tmp_path / "registry")
    record, _ = reg.register(
        checkpoint("dirty", git={"commit": "b" * 40, "branch": "main", "dirty": True})
    )
    assert "trained from a dirty working tree" in record.provenance_gaps()


def test_promotion_archives_the_incumbent(tmp_path, checkpoint):
    """Exactly one production checkpoint, and history is never destroyed."""
    reg = Registry(tmp_path / "registry")
    first, _ = reg.register(checkpoint("v1", b"v1"))
    second, _ = reg.register(checkpoint("v2", b"v2"))

    reg.promote(first.checkpoint_id, "production")
    assert reg.production().checkpoint_id == first.checkpoint_id

    reg.promote(second.checkpoint_id, "production")
    assert reg.production().checkpoint_id == second.checkpoint_id
    assert reg.get(first.checkpoint_id).stage == "archived"
    # Nothing was removed — "what was in production in March" stays answerable.
    assert len(reg.list()) == 2


def test_lookup_accepts_an_unambiguous_prefix(tmp_path, checkpoint):
    reg = Registry(tmp_path / "registry")
    record, _ = reg.register(checkpoint())
    assert reg.get(record.checkpoint_id[:8]).checkpoint_id == record.checkpoint_id

    with pytest.raises(RegistryError, match="no checkpoint"):
        reg.get("ffffffff")


def test_resolve_handles_production_and_latest(tmp_path, checkpoint):
    reg = Registry(tmp_path / "registry")
    first, _ = reg.register(checkpoint("v1", b"v1"))
    second, _ = reg.register(checkpoint("v2", b"v2"))

    with pytest.raises(RegistryError, match="production"):
        reg.resolve("production")

    reg.promote(first.checkpoint_id, "production")
    assert reg.resolve("production").checkpoint_id == first.checkpoint_id
    assert reg.resolve("latest").checkpoint_id == second.checkpoint_id


def test_registry_survives_a_reload(tmp_path, checkpoint):
    root = tmp_path / "registry"
    record, _ = Registry(root).register(checkpoint())
    assert Registry(root).get(record.checkpoint_id).name == record.name


def test_blob_is_copied_into_the_registry(tmp_path, checkpoint):
    reg = Registry(tmp_path / "registry")
    source = checkpoint("v1", b"payload")
    record, _ = reg.register(source)

    source.unlink()  # the original goes away; the registry must not care
    assert reg.blob_path(record).read_bytes() == b"payload"
