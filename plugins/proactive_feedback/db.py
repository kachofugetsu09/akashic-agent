from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeedbackEvent:
    session_key: str
    user_message_id: str
    assistant_message_id: str
    proactive_message_id: str | None
    feedback_type: str
    confidence: str
    pa_score: float | None
    pua_score: float | None
    lag_seconds: int | None
    candidate_count: int
    matched_by: str
    reason: str


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _ = conn.execute("PRAGMA journal_mode = WAL")
    _ = conn.execute("PRAGMA synchronous = NORMAL")
    _ = conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS proactive_feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            session_key TEXT NOT NULL,
            user_message_id TEXT NOT NULL,
            assistant_message_id TEXT NOT NULL,
            proactive_message_id TEXT,
            feedback_type TEXT NOT NULL,
            confidence TEXT NOT NULL,
            pa_score REAL,
            pua_score REAL,
            lag_seconds INTEGER,
            candidate_count INTEGER NOT NULL,
            matched_by TEXT NOT NULL,
            reason TEXT NOT NULL,
            UNIQUE(user_message_id, proactive_message_id)
        );

        CREATE INDEX IF NOT EXISTS idx_pfe_session_created
        ON proactive_feedback_events(session_key, created_at);

        CREATE INDEX IF NOT EXISTS idx_pfe_proactive
        ON proactive_feedback_events(proactive_message_id);

        CREATE UNIQUE INDEX IF NOT EXISTS idx_pfe_one_user_per_proactive
        ON proactive_feedback_events(proactive_message_id)
        WHERE proactive_message_id IS NOT NULL;
        """
    )
    conn.commit()
    return conn


def insert_feedback(conn: sqlite3.Connection, event: FeedbackEvent) -> int | None:
    if event.proactive_message_id is not None:
        existing = conn.execute(
            """
            SELECT id
            FROM proactive_feedback_events
            WHERE proactive_message_id = ?
              AND user_message_id <> ?
            LIMIT 1
            """,
            (event.proactive_message_id, event.user_message_id),
        ).fetchone()
        if existing is not None:
            return None
    _ = conn.execute(
        "DELETE FROM proactive_feedback_events WHERE user_message_id = ?",
        (event.user_message_id,),
    )
    cursor = conn.execute(
        """
        INSERT INTO proactive_feedback_events (
            session_key,
            user_message_id,
            assistant_message_id,
            proactive_message_id,
            feedback_type,
            confidence,
            pa_score,
            pua_score,
            lag_seconds,
            candidate_count,
            matched_by,
            reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.session_key,
            event.user_message_id,
            event.assistant_message_id,
            event.proactive_message_id,
            event.feedback_type,
            event.confidence,
            event.pa_score,
            event.pua_score,
            event.lag_seconds,
            event.candidate_count,
            event.matched_by,
            event.reason,
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    if row_id is None:
        raise RuntimeError("feedback insert failed")
    return int(row_id)
