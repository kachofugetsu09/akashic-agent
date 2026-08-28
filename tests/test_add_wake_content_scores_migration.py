from __future__ import annotations

import importlib.util
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest
import yoyo

from agent.migrations.context import bind_migration_context
from plugins.wake.state import WakeState

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "migrations/yoyo/20260828_02_add_wake_content_scores.py"


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("wake_score_migration_test", MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_step = yoyo.step
    yoyo.step = lambda callback: callback  # type: ignore[assignment]
    try:
        spec.loader.exec_module(module)
    finally:
        yoyo.step = original_step
    return module


def _wake_v7(workspace: Path) -> Path:
    path = workspace / "plugin-data/wake-builtin/wake.sqlite3"
    state = WakeState(path)
    state.initialize()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DROP TABLE content_scores")
        connection.execute("ALTER TABLE admission_state RENAME TO admission_state_v8")
        connection.execute(
            """
            CREATE TABLE admission_state(
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                content_high_watermark INTEGER NOT NULL,
                last_content_attempt_at TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO admission_state VALUES(1, 17, '2026-08-28T08:00:00+00:00')"
        )
        connection.execute("DROP TABLE admission_state_v8")
        connection.execute("PRAGMA user_version = 7")
    return path


def test_migration_backs_up_v7_then_adds_empty_score_ledger(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = _wake_v7(workspace)
    migration = _load_migration()

    with bind_migration_context(
        config_path=tmp_path / "config.toml", workspace=workspace
    ):
        migration.add_wake_content_scores(object())

    WakeState(path).initialize()
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (8,)
        assert connection.execute("SELECT count(*) FROM content_scores").fetchone() == (
            0,
        )
        assert connection.execute("SELECT * FROM admission_state").fetchone() == (1, 17)
        assert [
            str(row[1])
            for row in connection.execute("PRAGMA table_info(admission_state)")
        ] == ["singleton", "content_high_watermark"]
    backups = list((workspace / "backups/add-wake-content-scores").glob("*"))
    assert len(backups) == 1
    backup_root = backups[0] / "wake-db"
    manifest = json.loads((backup_root / "manifest.json").read_text())
    assert manifest["migration"] == "add-wake-content-scores"
    with closing(sqlite3.connect(backup_root / "wake.sqlite3")) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (7,)
        assert connection.execute("SELECT * FROM admission_state").fetchone() == (
            1,
            17,
            "2026-08-28T08:00:00+00:00",
        )
        assert "content_scores" not in {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    with bind_migration_context(
        config_path=tmp_path / "config.toml", workspace=workspace
    ):
        migration.add_wake_content_scores(object())
    assert list((workspace / "backups/add-wake-content-scores").glob("*")) == backups


def test_migration_rejects_unknown_v7_tables_before_backup(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = _wake_v7(workspace)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE unknown_state(value TEXT)")
    migration = _load_migration()

    with (
        bind_migration_context(
            config_path=tmp_path / "config.toml", workspace=workspace
        ),
        pytest.raises(RuntimeError, match="不支持 schema version 7"),
    ):
        migration.add_wake_content_scores(object())

    assert not (workspace / "backups").exists()
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (7,)


def test_migration_rejects_same_version_v7_schema_mutation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = _wake_v7(workspace)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("ALTER TABLE seen_content RENAME TO old_seen_content")
        connection.execute("CREATE TABLE seen_content(item_identity TEXT)")
        connection.execute("DROP TABLE old_seen_content")
    migration = _load_migration()

    with (
        bind_migration_context(
            config_path=tmp_path / "config.toml", workspace=workspace
        ),
        pytest.raises(RuntimeError, match="v7 schema identity 不匹配"),
    ):
        migration.add_wake_content_scores(object())

    assert not (workspace / "backups").exists()
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (7,)
