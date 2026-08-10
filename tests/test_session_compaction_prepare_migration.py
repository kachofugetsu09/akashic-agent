from __future__ import annotations

import json
import sqlite3
import stat
from pathlib import Path

import pytest

from agent.migrations.runner import MigrationRunner


_PROJECT_ROOT = Path(__file__).parents[1]
_PREPARE_ID = "20260808_02_session_compaction_prepares"


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


_PREPARE_TABLE_WITHOUT_UNIQUE = """
CREATE TABLE session_compaction_prepares (
    session_key TEXT NOT NULL,
    session_created_at TEXT NOT NULL,
    generation INTEGER NOT NULL,
    parent_generation INTEGER NOT NULL,
    source_ref TEXT NOT NULL,
    source_from_seq INTEGER NOT NULL,
    consolidated_through_seq INTEGER NOT NULL,
    source_message_ids_json TEXT NOT NULL,
    retained_tail_json TEXT NOT NULL,
    prepared_at TEXT NOT NULL,
    PRIMARY KEY (session_key, generation)
)
"""


def _runner(root: Path) -> MigrationRunner:
    return MigrationRunner(
        repo_root=_PROJECT_ROOT,
        config_path=root / "config.toml",
        workspace=root / "workspace",
    )


def test_prepare_migration_publishes_exact_schema_and_indexes(tmp_path: Path) -> None:
    root = tmp_path / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    _create_sessions(workspace / "sessions.db")

    outcome = _runner(root).run()

    assert _PREPARE_ID in outcome.migrations
    connection = sqlite3.connect(workspace / "sessions.db")
    try:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(session_compaction_prepares)"
            )
        }
        assert columns == {
            "session_key",
            "session_created_at",
            "generation",
            "parent_generation",
            "source_ref",
            "source_from_seq",
            "consolidated_through_seq",
            "source_message_ids_json",
            "retained_tail_json",
            "prepared_at",
        }
        indexes = {
            row[1]
            for row in connection.execute(
                "SELECT type, name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert "idx_session_compaction_prepares_ref" in indexes
        assert {
            row[1]
            for row in connection.execute(
                "PRAGMA index_list(session_compaction_prepares)"
            )
        } >= {
            "idx_session_compaction_prepares_ref",
        }
    finally:
        connection.close()
    backups = sorted(
        (workspace / "backups/session-compaction-prepares").iterdir()
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


def test_prepare_migration_rejects_incompatible_existing_table(tmp_path: Path) -> None:
    root = tmp_path / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    sessions = workspace / "sessions.db"
    _create_sessions(sessions)
    connection = sqlite3.connect(sessions)
    try:
        connection.execute(
            "CREATE TABLE session_compaction_prepares (session_key TEXT PRIMARY KEY)"
        )
        connection.commit()
    finally:
        connection.close()

    runner = _runner(root)
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
        assert _PREPARE_ID not in applied


def test_prepare_migration_rejects_missing_source_ref_unique_constraint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    sessions = workspace / "sessions.db"
    _create_sessions(sessions)
    connection = sqlite3.connect(sessions)
    try:
        connection.execute(_PREPARE_TABLE_WITHOUT_UNIQUE)
        connection.commit()
    finally:
        connection.close()

    runner = _runner(root)
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
        assert _PREPARE_ID not in applied
