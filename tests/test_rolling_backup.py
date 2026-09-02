from __future__ import annotations

import json
import os
import sqlite3
import stat
from contextlib import closing
from pathlib import Path

import pytest

import scripts.rolling_backup as rolling_backup
from scripts.rolling_backup import (
    BackupSource,
    _load_config,
    create_snapshot,
    restore_snapshot,
    verify_snapshot,
)


def _create_database(path: Path, value: str) -> None:
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE records (value TEXT NOT NULL)")
        db.execute("INSERT INTO records VALUES (?)", (value,))
        db.commit()


def test_create_snapshot_copies_files_and_sqlite_consistently(tmp_path: Path) -> None:
    source_file = tmp_path / "memory.md"
    source_file.write_text("memory\n", encoding="utf-8")
    source_db = tmp_path / "source.db"
    _create_database(source_db, "session")

    snapshot = create_snapshot(
        sources=[
            BackupSource("docs/memory.md", source_file, "file"),
            BackupSource("data/sessions.db", source_db, "sqlite"),
        ],
        destination=tmp_path / "backups",
    )

    assert (snapshot / "docs/memory.md").read_text(encoding="utf-8") == "memory\n"
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["data/sessions.db"]["kind"] == "sqlite"
    with closing(sqlite3.connect(snapshot / "data/sessions.db")) as db:
        assert db.execute("SELECT value FROM records").fetchone() == ("session",)


def test_create_snapshot_prunes_only_after_a_complete_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("content\n", encoding="utf-8")
    sources = [BackupSource("source.txt", source, "file")]
    destination = tmp_path / "backups"

    snapshots = [
        create_snapshot(sources=sources, destination=destination, retention=10)
        for _ in range(3)
    ]
    latest = create_snapshot(sources=sources, destination=destination, retention=2)

    assert not snapshots[0].exists()
    assert latest.exists()
    assert len(list(destination.glob("snapshot-*"))) == 2
    assert not list(destination.glob(".snapshot-*.tmp"))


def test_source_names_must_not_escape_snapshot_root(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("content\n", encoding="utf-8")

    try:
        create_snapshot(
            sources=[BackupSource("../outside.txt", source, "file")],
            destination=tmp_path / "backups",
        )
    except ValueError as exc:
        assert "相对安全路径" in str(exc)
    else:
        raise AssertionError("不安全的快照内路径应被拒绝")


def test_directory_source_manifest_and_restore_are_hash_verified(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plugin-data"
    nested = source / "nested"
    nested.mkdir(parents=True)
    empty = source / "empty"
    empty.mkdir()
    payload = nested / "config.local.toml"
    payload.write_bytes(b"token = 'redacted-in-test'\n")
    os.chmod(source, 0o750)
    os.chmod(nested, 0o710)
    os.chmod(payload, 0o640)

    snapshot = create_snapshot(
        sources=[BackupSource("plugin-data", source, "directory")],
        destination=tmp_path / "backups",
    )
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    directory = manifest["directories"]["plugin-data"]
    assert directory["mode"] == 0o750
    assert {
        (entry["relative_path"], entry["kind"]) for entry in directory["entries"]
    } == {
        ("empty", "directory"),
        ("nested", "directory"),
        ("nested/config.local.toml", "file"),
    }
    file_entry = next(
        entry
        for entry in directory["entries"]
        if entry["relative_path"] == "nested/config.local.toml"
    )
    assert file_entry["mode"] == 0o640
    assert file_entry["size"] == payload.stat().st_size
    assert len(file_entry["sha256"]) == 64
    verify_snapshot(snapshot)
    snapshot_payload = snapshot / "plugin-data/nested/config.local.toml"
    snapshot_payload.write_bytes(b"x" * payload.stat().st_size)
    with pytest.raises(ValueError, match="摘要校验失败"):
        verify_snapshot(snapshot)
    snapshot_payload.write_bytes(payload.read_bytes())
    os.chmod(snapshot_payload, 0o640)

    restored = restore_snapshot(snapshot, tmp_path / "restored")
    restored_payload = restored / "plugin-data/nested/config.local.toml"
    assert restored_payload.read_bytes() == payload.read_bytes()
    assert stat.S_IMODE(restored_payload.stat().st_mode) == 0o640
    assert stat.S_IMODE((restored / "plugin-data").stat().st_mode) == 0o750
    assert (restored / "plugin-data/empty").is_dir()
    verify_snapshot(snapshot)

    restored_payload.write_bytes(b"changed")
    with pytest.raises(FileExistsError):
        restore_snapshot(snapshot, tmp_path / "restored")


@pytest.mark.parametrize("mutation", ["extra", "missing-empty", "directory-mode"])
def test_directory_manifest_rejects_exact_tree_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = tmp_path / "plugin-data"
    (source / "empty").mkdir(parents=True)
    (source / "nested").mkdir()
    (source / "nested/state.json").write_text("{}\n", encoding="utf-8")
    os.chmod(source / "nested", 0o710)
    snapshot = create_snapshot(
        sources=[BackupSource("plugin-data", source, "directory")],
        destination=tmp_path / "backups",
    )

    if mutation == "extra":
        (snapshot / "plugin-data/extra").write_text("extra", encoding="utf-8")
    elif mutation == "missing-empty":
        (snapshot / "plugin-data/empty").rmdir()
    else:
        os.chmod(snapshot / "plugin-data/nested", 0o700)

    with pytest.raises(ValueError, match="路径集合|权限校验"):
        verify_snapshot(snapshot)


def test_restore_rejects_symlinked_parent_without_outside_write(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    snapshot = create_snapshot(
        sources=[BackupSource("source", source, "directory")],
        destination=tmp_path / "backups",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="符号链接"):
        restore_snapshot(snapshot, linked / "restored")

    assert not (outside / "restored").exists()


def test_manifest_reserved_name_rejects_descendants_at_admission(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.json"):
        create_snapshot(
            sources=[BackupSource("manifest.json/child", source, "file")],
            destination=tmp_path / "backups",
        )


def test_read_only_directory_modes_are_applied_after_copy_and_restore(
    tmp_path: Path,
) -> None:
    source = tmp_path / "read-only"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (nested / "readme").write_text("immutable\n", encoding="utf-8")
    os.chmod(nested, 0o555)
    os.chmod(source, 0o555)

    snapshot = create_snapshot(
        sources=[BackupSource("data", source, "directory")],
        destination=tmp_path / "backups",
    )
    restored = restore_snapshot(snapshot, tmp_path / "restored-read-only")

    assert (restored / "data/nested/readme").read_text(encoding="utf-8") == (
        "immutable\n"
    )
    assert stat.S_IMODE((snapshot / "data").stat().st_mode) == 0o555
    assert stat.S_IMODE((restored / "data/nested").stat().st_mode) == 0o555


@pytest.mark.parametrize("name", ["foo/./bar", "foo//bar"])
def test_programmatic_source_names_must_already_be_canonical(
    tmp_path: Path,
    name: str,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")

    with pytest.raises(ValueError, match="规范路径"):
        create_snapshot(
            sources=[BackupSource(name, source, "file")],
            destination=tmp_path / "backups",
        )


def test_manifest_rejects_invalid_directory_mode_before_restore(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    snapshot = create_snapshot(
        sources=[BackupSource("data", source, "directory")],
        destination=tmp_path / "backups",
    )
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["directories"]["data"]["mode"] = "bad"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="mode 无效"):
        verify_snapshot(snapshot)


def test_directory_source_rejects_symlinks_and_destination_inside_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plugin-data"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (source / "escape.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="符号链接"):
        create_snapshot(
            sources=[BackupSource("plugin-data", source, "directory")],
            destination=tmp_path / "backups",
        )

    (source / "escape.txt").unlink()
    linked_root = tmp_path / "linked-plugin-data"
    linked_root.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        create_snapshot(
            sources=[BackupSource("plugin-data", linked_root, "directory")],
            destination=tmp_path / "backups",
        )

    with pytest.raises(ValueError, match="位于目录源内部"):
        create_snapshot(
            sources=[BackupSource("plugin-data", source, "directory")],
            destination=source / "backups",
        )


def test_failed_directory_snapshot_keeps_previous_successful_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "plugin-data"
    source.mkdir()
    (source / "state.json").write_text('{"ok": true}\n', encoding="utf-8")
    destination = tmp_path / "backups"
    previous = create_snapshot(
        sources=[BackupSource("plugin-data", source, "directory")],
        destination=destination,
    )

    def fail_copy(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected directory copy failure")

    monkeypatch.setattr(rolling_backup, "_copy_directory", fail_copy)
    with pytest.raises(RuntimeError, match="injected directory copy failure"):
        create_snapshot(
            sources=[BackupSource("plugin-data", source, "directory")],
            destination=destination,
        )

    assert previous.is_dir()
    verify_snapshot(previous)
    assert not list(destination.glob(".snapshot-*.tmp"))


def test_load_config_defines_paths_and_backup_kinds(tmp_path: Path) -> None:
    config = tmp_path / "backup.toml"
    config.write_text(
        f"""destination = "{tmp_path / "backups"}"
retention = 3

[[sources]]
name = "notes.md"
kind = "file"
path = "{tmp_path / "notes.md"}"
""",
        encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text("notes\n", encoding="utf-8")

    destination, retention, sources = _load_config(config)

    assert destination == tmp_path / "backups"
    assert retention == 3
    assert sources == [BackupSource("notes.md", tmp_path / "notes.md", "file")]


def test_load_config_accepts_directory_source(tmp_path: Path) -> None:
    config = tmp_path / "backup.toml"
    config.write_text(
        f"""destination = "{tmp_path / "backups"}"

[[sources]]
name = "plugin-data"
kind = "directory"
path = "{tmp_path / "plugin-data"}"
""",
        encoding="utf-8",
    )

    destination, retention, sources = _load_config(config)

    assert destination == tmp_path / "backups"
    assert retention == 14
    assert sources == [
        BackupSource("plugin-data", tmp_path / "plugin-data", "directory")
    ]
