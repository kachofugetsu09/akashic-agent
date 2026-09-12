import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from agent.migrations.context import bind_migration_context
from session.store import SessionStore
from tests.legacy_migration_loader import load_migration_namespace

_PROJECT_ROOT = Path(__file__).parents[1]
_MIGRATION_PATH = (
    _PROJECT_ROOT / "plugins/legacy_upgrade/legacy_upgrade_migrations/20260827_01_normalize_session_timestamps.py"
)


def _load_migration():
    """Load the migration callback without wrapping it in Yoyo."""
    return load_migration_namespace("20260827_01_normalize_session_timestamps")


def _database(path: Path) -> None:
    SessionStore(path).close()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "INSERT INTO sessions(key, created_at, updated_at, last_consolidated, "
            "metadata, last_user_at, last_proactive_at, next_seq) "
            "VALUES ('akashic:test', ?, ?, 0, '{}', ?, ?, 1)",
            (
                "2026-07-14T01:16:02.059227",
                "2026-07-14T01:37:51.488915",
                "2026-07-14T01:30:00+08:00",
                None,
            ),
        )
        connection.execute(
            "INSERT INTO messages(id, session_key, seq, role, content, ts) "
            "VALUES ('akashic:test:0', 'akashic:test', 0, 'user', 'hello', ?)",
            ("2026-07-14T01:37:51.488887",),
        )


def test_normalizes_legacy_shanghai_times_and_keeps_aware_values(tmp_path: Path) -> None:
    migration = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = workspace / "sessions.db"
    _database(database)
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")

    with bind_migration_context(config_path=config, workspace=workspace):
        migration.normalize_session_timestamps(object())

    with closing(sqlite3.connect(database)) as connection:
        session = connection.execute(
            "SELECT created_at, updated_at, last_user_at FROM sessions "
            "WHERE key = 'akashic:test'"
        ).fetchone()
        message = connection.execute(
            "SELECT ts FROM messages WHERE id = 'akashic:test:0'"
        ).fetchone()
    assert session == (
        "2026-07-13T17:16:02.059227+00:00",
        "2026-07-13T17:37:51.488915+00:00",
        "2026-07-14T01:30:00+08:00",
    )
    assert message == ("2026-07-13T17:37:51.488887+00:00",)
    backups = list((workspace / "backups" / migration._MIGRATION).glob("*/sessions.db"))
    assert len(backups) == 1
    with closing(sqlite3.connect(backups[0])) as connection:
        assert connection.execute(
            "SELECT updated_at FROM sessions WHERE key = 'akashic:test'"
        ).fetchone() == ("2026-07-14T01:37:51.488915",)

    with bind_migration_context(config_path=config, workspace=workspace):
        migration.normalize_session_timestamps(object())
    assert list((workspace / "backups" / migration._MIGRATION).glob("*/sessions.db")) == backups


def test_rejects_malformed_timestamp_before_backup_or_write(tmp_path: Path) -> None:
    migration = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = workspace / "sessions.db"
    _database(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "UPDATE sessions SET updated_at = 'not-a-time' WHERE key = 'akashic:test'"
        )
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")

    with (
        bind_migration_context(config_path=config, workspace=workspace),
        pytest.raises(RuntimeError, match="不是有效 ISO 时间"),
    ):
        migration.normalize_session_timestamps(object())

    assert not (workspace / "backups" / migration._MIGRATION).exists()
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute(
            "SELECT created_at FROM sessions WHERE key = 'akashic:test'"
        ).fetchone() == ("2026-07-14T01:16:02.059227",)


def test_rejects_session_creation_time_used_by_compaction_identity(
    tmp_path: Path,
) -> None:
    migration = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = workspace / "sessions.db"
    _database(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "INSERT INTO session_compaction_prepares("
            "session_key, session_created_at, generation, parent_generation, "
            "source_ref, source_from_seq, consolidated_through_seq, "
            "source_message_ids_json, retained_tail_json, prepared_at"
            ") VALUES ('akashic:test', '2026-07-14T01:16:02.059227', 1, 0, "
            "'context-compaction:test', 0, 0, '[\"akashic:test:0\"]', '[]', "
            "'2026-07-14T01:38:00+08:00')"
        )
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")

    with (
        bind_migration_context(config_path=config, workspace=workspace),
        pytest.raises(RuntimeError, match="durable identity"),
    ):
        migration.normalize_session_timestamps(object())

    assert not (workspace / "backups" / migration._MIGRATION).exists()
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute(
            "SELECT created_at FROM sessions WHERE key = 'akashic:test'"
        ).fetchone() == ("2026-07-14T01:16:02.059227",)
