from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

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

    assert outcome.migrations[-1] == "20260808_01_session_mutation_audits"
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
