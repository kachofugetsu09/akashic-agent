from __future__ import annotations

import importlib.util
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest
import yoyo

from agent.migrations.context import bind_migration_context
from plugins.eventmail.store import EventMailStore

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "migrations/yoyo/20260828_01_migrate_eventmail_state.py"
NOW = datetime(2026, 8, 28, 8, tzinfo=UTC)


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eventmail_migration_test", MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_step = yoyo.step
    yoyo.step = lambda callback: callback  # type: ignore[assignment]
    try:
        spec.loader.exec_module(module)
    finally:
        yoyo.step = original_step
    return module


def test_yoyo_discovery_does_not_import_repository_eventmail_plugin(
    monkeypatch,
) -> None:
    class RejectEventMailPluginImport:
        def find_spec(self, fullname, path=None, target=None):
            _ = path, target
            if fullname == "plugins.eventmail" or fullname.startswith(
                "plugins.eventmail."
            ):
                raise AssertionError("Yoyo imported repository EventMail plugin")
            return None

    blocker = RejectEventMailPluginImport()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

    migration = _load_migration()

    assert migration.EventMailV3MigrationStore.__module__ == (
        "agent.migrations.payloads.eventmail_v3"
    )


def _legacy_content(workspace: Path, version: int = 3) -> None:
    root = workspace / "plugin-data/content-builtin"
    root.mkdir(parents=True)
    store = EventMailStore(root / "content.sqlite3")
    store.submit(
        "feed",
        "batch:one",
        (
            {
                "item_id": "story:one",
                "revision": "1",
                "payload": {"title": "Legacy story"},
                "not_before": NOW,
                "requires_ack": True,
            },
        ),
    )
    if version not in {1, 2, 3}:
        raise ValueError(f"unsupported Content fixture version: {version}")
    if version < 3:
        with closing(sqlite3.connect(store.path)) as connection, connection:
            connection.executescript("""
                DROP TABLE context_projection;
                DROP TABLE alert_projection;
                DROP TABLE mail_transitions;
                DROP TABLE mail_envelopes;
                """)
            if version == 1:
                connection.executescript("""
                    DROP TABLE content_selection_members;
                    DROP TABLE content_selections;
                    """)
            connection.execute(f"PRAGMA user_version={version}")
    (root / "config.local.toml").write_text("# preserved\n", encoding="utf-8")


def _legacy_wake(workspace: Path) -> Path:
    path = workspace / "plugin-data/wake-builtin/wake.sqlite3"
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript("""
            CREATE TABLE admission_state(
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                content_high_watermark INTEGER NOT NULL,
                last_content_attempt_at TEXT
            );
            INSERT INTO admission_state VALUES(1, 9, NULL);
            CREATE TABLE seen_content(item_identity TEXT PRIMARY KEY);
            INSERT INTO seen_content VALUES('legacy-seen');
            CREATE TABLE alert_events(
                source_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                not_before TEXT NOT NULL,
                status TEXT NOT NULL,
                accepted_session TEXT,
                accepted_turn TEXT,
                PRIMARY KEY(source_id, event_id)
            );
            CREATE TABLE alert_expiry(
                source_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                expires_at TEXT,
                PRIMARY KEY(source_id, event_id)
            );
            CREATE TABLE context_events(
                source_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                expires_at TEXT,
                PRIMARY KEY(source_id, event_id)
            );
            CREATE TABLE wake_runs(
                run_id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                started_at TEXT NOT NULL,
                candidates_seen INTEGER NOT NULL,
                candidates_selected INTEGER NOT NULL,
                screening_json TEXT NOT NULL,
                decision TEXT,
                decision_detail TEXT,
                completed_at TEXT
            );
            INSERT INTO wake_runs VALUES(
                'run:old', 'content', '2026-08-28T08:00:00+00:00',
                1, 0, '[]', 'skip', 'legacy', '2026-08-28T08:00:01+00:00'
            );
            CREATE TABLE wake_attempts(
                attempt_id TEXT PRIMARY KEY,
                timer_id TEXT NOT NULL,
                scheduled_for TEXT NOT NULL,
                fired_at TEXT NOT NULL,
                mail_watermark INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                owner TEXT,
                detail TEXT,
                completed_at TEXT
            );
            PRAGMA user_version=5;
            """)
        connection.execute(
            "INSERT INTO wake_attempts VALUES(?,?,?,?,?,'completed','content',?,?)",
            (
                "attempt:legacy",
                "timer:legacy",
                NOW.isoformat(),
                NOW.isoformat(),
                9,
                "legacy completed",
                NOW.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO alert_events VALUES(?,?,?,?,?,'selected',?,?)",
            (
                "calendar",
                "meeting:one",
                '{"title":"Meeting"}',
                NOW.isoformat(),
                NOW.isoformat(),
                "wake:default",
                "turn:one",
            ),
        )
        connection.execute(
            "INSERT INTO alert_expiry VALUES(?,?,?)",
            (
                "calendar",
                "meeting:one",
                (NOW + timedelta(minutes=10)).isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO context_events VALUES(?,?,?,?,?)",
            (
                "steam",
                "current",
                '{"presence":"in_game"}',
                NOW.isoformat(),
                (NOW + timedelta(minutes=30)).isoformat(),
            ),
        )
    return path


@pytest.mark.parametrize("content_version", (1, 2, 3))
def test_migration_moves_all_mail_and_retires_both_old_sources(
    tmp_path: Path, content_version: int
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _legacy_content(workspace, content_version)
    wake_db = _legacy_wake(workspace)
    migration = _load_migration()

    with bind_migration_context(
        config_path=tmp_path / "config.toml",
        workspace=workspace,
    ):
        migration.migrate_eventmail_state(object())

    target_root = workspace / "plugin-data/eventmail-builtin"
    target = EventMailStore(target_root / "eventmail.sqlite3")
    assert target.snapshot(NOW)["items"][0]["payload"] == {"title": "Legacy story"}
    assert target.alert_status("calendar", "meeting:one") == "selected"
    selected = target.selected_alert(
        {"session_id": "wake:default", "turn_id": "turn:one"}
    )
    assert selected is not None and selected["payload"] == {"title": "Meeting"}
    assert target.active_context(NOW)[0]["payload"] == {"presence": "in_game"}
    assert (target_root / "config.local.toml").read_text() == "# preserved\n"
    assert not (workspace / "plugin-data/content-builtin").exists()

    backups = list((workspace / "backups/migrate-eventmail-state").iterdir())
    assert len(backups) == 1
    assert (backups[0] / "retired-content-builtin/content.sqlite3").is_file()
    assert (backups[0] / "wake-db/wake.sqlite3").is_file()
    with closing(sqlite3.connect(wake_db)) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert tables == {
            "admission_state",
            "seen_content",
            "wake_runs",
            "wake_attempts",
        }
        assert connection.execute("PRAGMA user_version").fetchone() == (7,)
        assert connection.execute(
            "SELECT decision FROM wake_runs WHERE run_id='run:old'"
        ).fetchone() == ("skip",)
        assert connection.execute(
            "SELECT outcome FROM wake_attempts WHERE attempt_id='attempt:legacy'"
        ).fetchone() == ("delivery_unknown",)

    with bind_migration_context(
        config_path=tmp_path / "config.toml",
        workspace=workspace,
    ):
        migration.migrate_eventmail_state(object())
    assert list((workspace / "backups/migrate-eventmail-state").iterdir()) == backups


def test_existing_unverified_target_fails_before_old_source_changes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _legacy_content(workspace)
    target = workspace / "plugin-data/eventmail-builtin"
    target.mkdir(parents=True)
    migration = _load_migration()

    with (
        bind_migration_context(
            config_path=tmp_path / "config.toml",
            workspace=workspace,
        ),
        pytest.raises(RuntimeError, match="缺少迁移 receipt"),
    ):
        migration.migrate_eventmail_state(object())

    assert (workspace / "plugin-data/content-builtin/content.sqlite3").is_file()
    assert not (workspace / "backups").exists()


def test_verified_target_digest_rejects_changed_state_on_crash_resume(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _legacy_content(workspace)
    _legacy_wake(workspace)
    migration = _load_migration()
    context = bind_migration_context(
        config_path=tmp_path / "config.toml",
        workspace=workspace,
    )
    with context:
        migration.migrate_eventmail_state(object())
    target = workspace / "plugin-data/eventmail-builtin/eventmail.sqlite3"
    with closing(sqlite3.connect(target)) as connection, connection:
        connection.execute("UPDATE items SET payload_json='{}' WHERE source_id='feed'")

    with (
        bind_migration_context(
            config_path=tmp_path / "config.toml",
            workspace=workspace,
        ),
        pytest.raises(RuntimeError, match="receipt digest"),
    ):
        migration.migrate_eventmail_state(object())
