from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import agent.migrations.session_db_backup as session_db_backup


def test_sqlite_backup_failure_keeps_published_evidence_and_cleans_temps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sessions.db"
    connection = sqlite3.connect(source)
    try:
        connection.execute("CREATE TABLE sessions (key TEXT PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()
    backup_root = tmp_path / "backups"

    def fail_after_publish(_path: Path) -> None:
        raise RuntimeError("forced directory fsync failure")

    monkeypatch.setattr(session_db_backup, "_fsync_directory", fail_after_publish)
    with pytest.raises(RuntimeError, match="forced directory fsync failure"):
        session_db_backup.backup_sqlite_database(
            source,
            backup_root,
            migration="test-backup",
        )

    assert (backup_root / "sessions.db").is_file()
    assert (backup_root / "manifest.json").is_file()
    assert not list(backup_root.glob("*.tmp"))
    assert not list(backup_root.glob(".*.tmp"))
