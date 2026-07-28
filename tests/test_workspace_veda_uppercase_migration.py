# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import pytest

from migrations.workspace_veda_uppercase.migration import (
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
        migration_commit="b" * 40,
        backup_dir=tmp_path / "backups" / "run",
    )


def test_lowercase_veda_is_moved_and_revertible(tmp_path: Path) -> None:
    context = _context(tmp_path)
    legacy = context.workspace / "memory/veda.md"
    target = context.workspace / "memory/VEDA.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("custom veda\n", encoding="utf-8")

    assert _assess(context) == {"status": "needed"}
    _apply(context)
    _verify(context)

    assert not legacy.exists()
    assert target.read_text(encoding="utf-8") == "custom veda\n"

    _revert(context)
    assert legacy.read_text(encoding="utf-8") == "custom veda\n"
    assert not target.exists()


def test_matching_paths_remove_only_legacy_copy(tmp_path: Path) -> None:
    context = _context(tmp_path)
    legacy = context.workspace / "memory/veda.md"
    target = context.workspace / "memory/VEDA.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("same\n", encoding="utf-8")
    target.write_text("same\n", encoding="utf-8")

    _apply(context)
    _verify(context)
    assert not legacy.exists()
    assert target.read_text(encoding="utf-8") == "same\n"

    _revert(context)
    assert legacy.read_text(encoding="utf-8") == "same\n"
    assert target.read_text(encoding="utf-8") == "same\n"


def test_uppercase_only_is_already_satisfied(tmp_path: Path) -> None:
    context = _context(tmp_path)
    target = context.workspace / "memory/VEDA.md"
    target.parent.mkdir(parents=True)
    target.write_text("current\n", encoding="utf-8")

    assert _assess(context) == {"status": "satisfied"}
    _verify(context)


@pytest.mark.parametrize(
    "setup",
    ("neither", "conflict", "invalid-lower", "invalid-upper"),
)
def test_ambiguous_or_invalid_state_blocks(tmp_path: Path, setup: str) -> None:
    context = _context(tmp_path)
    legacy = context.workspace / "memory/veda.md"
    target = context.workspace / "memory/VEDA.md"
    legacy.parent.mkdir(parents=True)
    if setup == "conflict":
        legacy.write_text("old\n", encoding="utf-8")
        target.write_text("new\n", encoding="utf-8")
    elif setup == "invalid-lower":
        legacy.write_bytes(b"\xff")
    elif setup == "invalid-upper":
        target.write_bytes(b" \n")

    assert _assess(context)["status"] == "blocked"


def test_revert_rejects_target_modified_after_migration(tmp_path: Path) -> None:
    context = _context(tmp_path)
    legacy = context.workspace / "memory/veda.md"
    target = context.workspace / "memory/VEDA.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("before\n", encoding="utf-8")
    _apply(context)
    target.write_text("after\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="拒绝回滚"):
        _revert(context)
