"""Frozen legacy Wake SQLite fixture from Core exact bd5db8c2."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

LEGACY_WAKE_SCHEMA = """
CREATE TABLE wake_runs(
    wake_id TEXT PRIMARY KEY, session_key TEXT NOT NULL, now_utc TEXT NOT NULL,
    scratchpad_json TEXT NOT NULL, investigations_json TEXT NOT NULL,
    final_message TEXT NOT NULL, cited_ids_json TEXT NOT NULL,
    display_event_map_json TEXT NOT NULL, source_refs_json TEXT NOT NULL,
    investigation_completed INTEGER NOT NULL DEFAULT 0, terminal_action TEXT
);
CREATE TABLE wake_observations(
    id INTEGER PRIMARY KEY AUTOINCREMENT, wake_id TEXT NOT NULL,
    session_key TEXT NOT NULL, kind TEXT NOT NULL, now_utc TEXT NOT NULL,
    trigger_json TEXT NOT NULL, candidates_json TEXT NOT NULL,
    llm_input_json TEXT NOT NULL
);
CREATE TABLE reservoir_events(
    item_id TEXT PRIMARY KEY, kind TEXT NOT NULL, source_id TEXT NOT NULL,
    original_source_id TEXT NOT NULL, ack_source_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL, published_at TEXT NOT NULL,
    first_seen_at TEXT NOT NULL, preprocess_score REAL NOT NULL,
    payload_json TEXT NOT NULL, embedding_json TEXT,
    status TEXT NOT NULL DEFAULT 'unread', consumed_at TEXT
);
CREATE TABLE reservoir_quarantine(
    identity TEXT PRIMARY KEY, source_id TEXT NOT NULL, item_id TEXT NOT NULL,
    reason TEXT NOT NULL, payload_json TEXT NOT NULL,
    first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
    occurrences INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE reservoir_tombstones(
    identity TEXT PRIMARY KEY, source_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL, acknowledged_at TEXT NOT NULL
);
CREATE TABLE hazard_state(
    session_key TEXT PRIMARY KEY, hazard REAL NOT NULL,
    threshold REAL NOT NULL, updated_at TEXT NOT NULL, last_wake_at TEXT
);
CREATE TABLE hazard_monitor(
    session_key TEXT PRIMARY KEY, hazard_before REAL NOT NULL,
    hazard_after REAL NOT NULL, preference_pressure REAL NOT NULL,
    threshold REAL NOT NULL, evidence REAL NOT NULL, rate REAL NOT NULL,
    driver_item_id TEXT NOT NULL, candidate_count INTEGER NOT NULL,
    should_wake INTEGER NOT NULL, evaluated_at TEXT NOT NULL
);
CREATE TABLE context_state(
    source_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL,
    presence TEXT NOT NULL, interruptibility REAL NOT NULL,
    confidence REAL NOT NULL, transition_name TEXT NOT NULL,
    observed_at TEXT, expires_at TEXT, updated_at TEXT NOT NULL
);
CREATE TABLE context_reevaluate_state(
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    last_signaled_at TEXT, last_candidate_at TEXT,
    suppressed_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE drift_state(
    session_key TEXT PRIMARY KEY, hazard REAL NOT NULL,
    threshold REAL NOT NULL, updated_at TEXT NOT NULL, last_drift_at TEXT,
    last_fingerprint TEXT NOT NULL DEFAULT '',
    repeat_count INTEGER NOT NULL DEFAULT 0, timer_anchor TEXT,
    next_attempt_at TEXT
);
CREATE TABLE pending_acknowledgements(
    source_id TEXT NOT NULL, source_event_id TEXT NOT NULL,
    item_id TEXT NOT NULL DEFAULT '', action TEXT NOT NULL DEFAULT 'consume',
    queued_at TEXT NOT NULL, PRIMARY KEY(source_id, source_event_id, item_id)
);
"""


def create_legacy_wake_database(path: Path) -> None:
    """Create the exact eleven-table legacy Wake schema without importing runtime code."""

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(LEGACY_WAKE_SCHEMA)
    connection.commit()
    connection.close()


def populate_continuity_table(path: Path, table: str) -> None:
    """Insert one structurally real opaque row into a continuity table."""

    # 1. Preserve the removed writer's persisted shapes, not its business behavior.
    values: dict[str, tuple[object, ...]] = {
        "reservoir_quarantine": (
            "feed:bad-item",
            "feed",
            "bad-item",
            "fixture",
            json.dumps({"bad": True}),
            "2026-08-23T00:00:00+00:00",
            "2026-08-23T00:00:00+00:00",
            1,
        ),
        "reservoir_tombstones": (
            "feed:expired-item",
            "feed",
            "expired-item",
            "2026-08-23T00:00:00+00:00",
        ),
        "hazard_state": (
            "wake:default",
            0.2,
            0.5,
            "2026-08-23T00:00:00+00:00",
            None,
        ),
        "context_state": (
            "presence",
            json.dumps({"presence": "active"}),
            "active",
            1.0,
            1.0,
            "fixture",
            None,
            None,
            "2026-08-23T00:00:00+00:00",
        ),
        "context_reevaluate_state": (1, None, None, 0),
        "drift_state": (
            "wake:default",
            0.3,
            0.8,
            "2026-08-23T00:00:00+00:00",
            None,
            "",
            0,
            None,
            None,
        ),
    }
    columns = {
        "reservoir_quarantine": 8,
        "reservoir_tombstones": 4,
        "hazard_state": 5,
        "context_state": 9,
        "context_reevaluate_state": 4,
        "drift_state": 9,
    }

    # 2. Keep the helper deliberately schema-only; unknown fixture tables fail loudly.
    row = values[table]
    connection = sqlite3.connect(path)
    connection.execute(
        f"INSERT INTO {table} VALUES({','.join('?' for _ in range(columns[table]))})",
        row,
    )
    connection.commit()
    connection.close()
