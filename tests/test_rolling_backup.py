from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from scripts.rolling_backup import BackupSource, _load_config, create_snapshot


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


def test_load_config_defines_paths_and_backup_kinds(tmp_path: Path) -> None:
    config = tmp_path / "backup.toml"
    config.write_text(
        f'''destination = "{tmp_path / "backups"}"
retention = 3

[[sources]]
name = "notes.md"
kind = "file"
path = "{tmp_path / "notes.md"}"
''',
        encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text("notes\n", encoding="utf-8")

    destination, retention, sources = _load_config(config)

    assert destination == tmp_path / "backups"
    assert retention == 3
    assert sources == [BackupSource("notes.md", tmp_path / "notes.md", "file")]
