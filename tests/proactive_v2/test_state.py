from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from proactive_v2.state import ProactiveStateStore


def _record_finish(store: ProactiveStateStore, tick_id: str) -> None:
    store.record_tick_log_finish(
        tick_id=tick_id,
        session_key="telegram:42",
        started_at="2026-07-13T00:00:00+08:00",
        finished_at="2026-07-13T00:00:01+08:00",
        gate_exit=None,
        terminal_action="skip",
        skip_reason="test",
        steps_taken=0,
        alert_count=0,
        content_count=0,
        context_count=0,
        interesting_ids=[],
        discarded_ids=[],
        cited_ids=[],
        drift_entered=False,
        final_message="",
    )


def test_store_migrates_legacy_tick_log_during_initialization(tmp_path: Path) -> None:
    db_path = tmp_path / "proactive.db"
    with closing(sqlite3.connect(db_path)) as db:
        _ = db.execute(
            """
            CREATE TABLE tick_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick_id TEXT NOT NULL UNIQUE,
                session_key TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                gate_exit TEXT,
                terminal_action TEXT,
                skip_reason TEXT,
                steps_taken INTEGER,
                alert_count INTEGER,
                content_count INTEGER,
                context_count INTEGER,
                interesting_ids TEXT,
                discarded_ids TEXT,
                cited_ids TEXT,
                drift_entered INTEGER DEFAULT 0,
                final_message TEXT
            )
            """
        )
        db.commit()

    store = ProactiveStateStore(db_path)
    try:
        columns = {
            str(row["name"])
            for row in store._db.execute("PRAGMA table_info(tick_log)").fetchall()
        }
        assert "proactive_effects_json" in columns
        _record_finish(store, "legacy")
    finally:
        store.close()


def test_tick_finish_does_not_repeat_schema_inspection(tmp_path: Path) -> None:
    store = ProactiveStateStore(tmp_path / "proactive.db")
    statements: list[str] = []
    store._db.set_trace_callback(statements.append)
    try:
        for index in range(10):
            _record_finish(store, f"tick-{index}")
    finally:
        store.close()

    assert not any(
        "PRAGMA table_info(tick_log)" in statement
        for statement in statements
    )
