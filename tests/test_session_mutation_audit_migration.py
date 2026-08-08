from __future__ import annotations

import json
import sqlite3
import stat
from pathlib import Path

import pytest

import agent.migrations.session_db_backup as session_db_backup
from agent.migrations.runner import MigrationRunner


_PROJECT_ROOT = Path(__file__).parents[1]


def _create_sessions(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE sessions ("
            "key TEXT PRIMARY KEY, last_consolidated INTEGER NOT NULL)"
        )
        connection.commit()
    finally:
        connection.close()


_DELETE_AUDIT_TABLE = """
CREATE TABLE session_delete_audits (
    audit_id TEXT,
    targets_json TEXT NOT NULL,
    message_ids_json TEXT NOT NULL,
    compactions_json TEXT NOT NULL,
    action_source TEXT NOT NULL,
    cascade INTEGER NOT NULL CHECK (cascade IN (0, 1)),
    backup_path TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    result TEXT NOT NULL,
    deleted_count INTEGER NOT NULL,
    error TEXT
)
"""


def _create_table(path: Path, sql: str, index_sql: str | None = None) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(sql)
        if index_sql is not None:
            connection.execute(index_sql)
        connection.commit()
    finally:
        connection.close()


def test_audit_migration_publishes_manifest_schema_and_indexes(tmp_path: Path) -> None:
    root = tmp_path / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    _create_sessions(workspace / "sessions.db")

    outcome = MigrationRunner(
        repo_root=_PROJECT_ROOT,
        config_path=root / "config.toml",
        workspace=workspace,
    ).run()

    assert "20260808_01_session_mutation_audits" in outcome.migrations
    connection = sqlite3.connect(workspace / "sessions.db")
    try:
        expected = {
            "session_delete_audits": {
                "audit_id",
                "targets_json",
                "message_ids_json",
                "compactions_json",
                "action_source",
                "cascade",
                "backup_path",
                "started_at",
                "completed_at",
                "result",
                "deleted_count",
                "error",
            },
            "session_source_mutation_audits": {
                "audit_id",
                "operation",
                "session_key",
                "message_ids_json",
                "action_source",
                "backup_path",
                "completed_at",
            },
        }
        for table, columns in expected.items():
            actual = {
                row[1]
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            assert actual == columns
        indexes = {
            row[1]
            for row in connection.execute(
                "SELECT type, name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert {
            "idx_session_delete_audits_time",
            "idx_source_mutation_audits_lookup",
        } <= indexes
    finally:
        connection.close()
    backups = sorted(
        (workspace / "backups/session-mutation-audits").iterdir()
    )
    assert len(backups) == 1
    manifest = json.loads((backups[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sqlite_integrity"] == "ok"
    assert stat.S_IMODE((backups[0] / "sessions.db").stat().st_mode) == 0o600
    assert stat.S_IMODE((backups[0] / "manifest.json").stat().st_mode) == 0o600
    archived = sqlite3.connect(backups[0] / "sessions.db")
    try:
        assert archived.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        archived.close()


def test_audit_migration_rejects_incompatible_existing_table(tmp_path: Path) -> None:
    root = tmp_path / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    sessions = workspace / "sessions.db"
    _create_sessions(sessions)
    connection = sqlite3.connect(sessions)
    try:
        connection.execute(
            "CREATE TABLE session_source_mutation_audits (audit_id TEXT PRIMARY KEY)"
        )
        connection.commit()
    finally:
        connection.close()

    runner = MigrationRunner(
        repo_root=_PROJECT_ROOT,
        config_path=root / "config.toml",
        workspace=workspace,
    )
    with pytest.raises(RuntimeError, match="schema lineage"):
        runner.run()
    restored = sqlite3.connect(sessions)
    try:
        assert restored.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'session_delete_audits'"
        ).fetchone() is None
    finally:
        restored.close()
    backups = sorted((workspace / "backups/session-mutation-audits").iterdir())
    assert len(backups) == 1
    if runner.ledger_path.exists():
        ledger = sqlite3.connect(runner.ledger_path)
        try:
            applied = {
                row[0]
                for row in ledger.execute(
                    "SELECT migration_id FROM _yoyo_migration"
                )
            }
        finally:
            ledger.close()
        assert "20260808_01_session_mutation_audits" not in applied


def test_audit_migration_rejects_wrong_primary_key(tmp_path: Path) -> None:
    root = tmp_path / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    sessions = workspace / "sessions.db"
    _create_sessions(sessions)
    _create_table(sessions, _DELETE_AUDIT_TABLE)

    with pytest.raises(RuntimeError, match="schema lineage"):
        MigrationRunner(
            repo_root=_PROJECT_ROOT,
            config_path=root / "config.toml",
            workspace=workspace,
        ).run()


def test_audit_migration_rejects_wrong_named_index(tmp_path: Path) -> None:
    root = tmp_path / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    sessions = workspace / "sessions.db"
    _create_sessions(sessions)
    _create_table(
        sessions,
        _DELETE_AUDIT_TABLE.replace("audit_id TEXT,", "audit_id TEXT PRIMARY KEY,"),
        "CREATE INDEX idx_session_delete_audits_time "
        "ON session_delete_audits(audit_id, completed_at)",
    )

    with pytest.raises(RuntimeError, match="schema lineage"):
        MigrationRunner(
            repo_root=_PROJECT_ROOT,
            config_path=root / "config.toml",
            workspace=workspace,
        ).run()


def test_audit_migration_rejects_unknown_extra_index(tmp_path: Path) -> None:
    root = tmp_path / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    sessions = workspace / "sessions.db"
    _create_sessions(sessions)
    _create_table(
        sessions,
        _DELETE_AUDIT_TABLE.replace("audit_id TEXT,", "audit_id TEXT PRIMARY KEY,"),
        "CREATE INDEX extra_delete_audit_index "
        "ON session_delete_audits(audit_id)",
    )

    with pytest.raises(RuntimeError, match="schema lineage"):
        MigrationRunner(
            repo_root=_PROJECT_ROOT,
            config_path=root / "config.toml",
            workspace=workspace,
        ).run()


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
