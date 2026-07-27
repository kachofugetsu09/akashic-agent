# pyright: reportPrivateUsage=false

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from migrations.workspace_veda.migration import (
    MigrationContext,
    _apply,
    _assess,
    _revert,
    _verify,
)


def _context(tmp_path: Path) -> MigrationContext:
    return MigrationContext(
        config_path=tmp_path / "config.toml",
        workspace=tmp_path / "workspace",
        migration_commit="a" * 40,
        backup_dir=tmp_path / "backups" / "run",
    )


def test_missing_veda_is_created_and_verified(tmp_path: Path) -> None:
    context = _context(tmp_path)

    assert _assess(context) == {"status": "needed"}
    _apply(context)
    _verify(context)

    target = context.workspace / "memory/veda.md"
    assert "你是 Akashic" in target.read_text(encoding="utf-8")
    assert _assess(context) == {"status": "satisfied"}
    manifest = context.backup_dir / "manifest.json"
    assert manifest.is_file()
    assert stat.S_IMODE(context.backup_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_existing_valid_veda_is_preserved(tmp_path: Path) -> None:
    context = _context(tmp_path)
    target = context.workspace / "memory/veda.md"
    target.parent.mkdir(parents=True)
    original = b"custom veda\n"
    target.write_bytes(original)

    assert _assess(context) == {"status": "satisfied"}
    _verify(context)

    assert target.read_bytes() == original
    assert not context.backup_dir.exists()


@pytest.mark.parametrize("payload", [b"", b" \n", b"\xff"])
def test_invalid_existing_veda_blocks_migration(
    tmp_path: Path,
    payload: bytes,
) -> None:
    context = _context(tmp_path)
    target = context.workspace / "memory/veda.md"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    assessment = _assess(context)

    assert assessment["status"] == "blocked"
    assert "Veda" in assessment["reason"]


def test_revert_removes_only_unchanged_migration_file(tmp_path: Path) -> None:
    context = _context(tmp_path)
    target = context.workspace / "memory/veda.md"
    _apply(context)

    _revert(context)

    assert not target.exists()


def test_revert_rejects_veda_modified_after_migration(tmp_path: Path) -> None:
    context = _context(tmp_path)
    target = context.workspace / "memory/veda.md"
    _apply(context)
    target.write_text("customized later\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="拒绝删除"):
        _revert(context)

    assert target.read_text(encoding="utf-8") == "customized later\n"
