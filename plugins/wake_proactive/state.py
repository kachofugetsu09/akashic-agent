from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from plugins.wake_proactive.context import WakeContext
from plugins.wake_proactive.context_drive import (
    ContextDriveResult,
    NormalizedContext,
    Presence,
    evaluate_context,
)


class WakeStateStore:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        _ = self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wake_runs (
                wake_id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                now_utc TEXT NOT NULL,
                scratchpad_json TEXT NOT NULL,
                investigations_json TEXT NOT NULL,
                final_message TEXT NOT NULL,
                cited_ids_json TEXT NOT NULL,
                display_event_map_json TEXT NOT NULL,
                source_refs_json TEXT NOT NULL,
                investigation_completed INTEGER NOT NULL DEFAULT 0,
                terminal_action TEXT
            )
            """
        )
        _ = self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reservoir_events (
                item_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                published_at TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                preprocess_score REAL NOT NULL,
                payload_json TEXT NOT NULL,
                embedding_json TEXT,
                status TEXT NOT NULL DEFAULT 'unread',
                consumed_at TEXT
            )
            """
        )
        _ = self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reservoir_unread
            ON reservoir_events(kind, status, source_id, published_at DESC)
            """
        )
        _ = self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hazard_state (
                session_key TEXT PRIMARY KEY,
                hazard REAL NOT NULL,
                threshold REAL NOT NULL,
                updated_at TEXT NOT NULL,
                last_wake_at TEXT
            )
            """
        )
        _ = self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_state (
                source_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                presence TEXT NOT NULL,
                interruptibility REAL NOT NULL,
                confidence REAL NOT NULL,
                transition_name TEXT NOT NULL,
                observed_at TEXT,
                expires_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        _ = self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_state (
                session_key TEXT PRIMARY KEY,
                hazard REAL NOT NULL,
                threshold REAL NOT NULL,
                updated_at TEXT NOT NULL,
                last_drift_at TEXT,
                last_fingerprint TEXT NOT NULL DEFAULT '',
                repeat_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        _ = self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_acknowledgements (
                source_id TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                queued_at TEXT NOT NULL,
                PRIMARY KEY(source_id, source_event_id)
            )
            """
        )
        self._conn.commit()

    def save(self, ctx: WakeContext) -> None:
        scratchpad = {
            item_id: {
                "initial_interest": item.initial_interest,
                "investigate": item.investigate,
                "question": item.question,
                "recall_query": item.recall_query,
            }
            for item_id, item in ctx.scratchpad.items()
        }
        _ = self._conn.execute(
            """
            INSERT INTO wake_runs (
                wake_id, session_key, now_utc, scratchpad_json,
                investigations_json, final_message, cited_ids_json,
                display_event_map_json, source_refs_json,
                investigation_completed, terminal_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(wake_id) DO UPDATE SET
                scratchpad_json=excluded.scratchpad_json,
                investigations_json=excluded.investigations_json,
                final_message=excluded.final_message,
                cited_ids_json=excluded.cited_ids_json,
                display_event_map_json=excluded.display_event_map_json,
                source_refs_json=excluded.source_refs_json,
                investigation_completed=excluded.investigation_completed,
                terminal_action=excluded.terminal_action
            """,
            (
                ctx.wake_id,
                ctx.session_key,
                ctx.now_utc.isoformat(),
                json.dumps(scratchpad, ensure_ascii=False),
                json.dumps(ctx.investigation_results, ensure_ascii=False),
                ctx.final_message,
                json.dumps(ctx.cited_item_ids, ensure_ascii=False),
                json.dumps(ctx.display_event_map, ensure_ascii=False),
                json.dumps(ctx.source_refs, ensure_ascii=False),
                int(ctx.investigation_completed),
                ctx.terminal_action,
            ),
        )
        self._conn.commit()

    def get(self, wake_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM wake_runs WHERE wake_id = ?", (wake_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def ingest(self, kind: str, events: list[dict[str, Any]], now: datetime) -> int:
        inserted = 0
        for event in events:
            source_id = str(event.get("ack_server") or event.get("_source") or "").strip()
            source_event_id = str(event.get("event_id") or event.get("id") or "").strip()
            if not source_id or not source_event_id:
                continue
            item_id = f"{source_id}:{source_event_id}"
            payload = dict(event)
            payload["id"] = item_id
            payload["item_id"] = item_id
            published_at = str(
                event.get("published_at") or event.get("triggered_at") or now.isoformat()
            )
            first_seen_at = str(event.get("first_seen_at") or now.isoformat())
            score = float(event.get("preprocess_score") or event.get("rank_score") or 0.0)
            existing = self._conn.execute(
                "SELECT 1 FROM reservoir_events WHERE item_id = ?",
                (item_id,),
            ).fetchone()
            _ = self._conn.execute(
                """
                INSERT INTO reservoir_events(
                    item_id, kind, source_id, source_event_id, published_at,
                    first_seen_at, preprocess_score, payload_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'unread')
                ON CONFLICT(item_id) DO UPDATE SET
                    published_at=excluded.published_at,
                    preprocess_score=excluded.preprocess_score,
                    payload_json=excluded.payload_json
                """,
                (
                    item_id,
                    kind,
                    source_id,
                    source_event_id,
                    published_at,
                    first_seen_at,
                    score,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            inserted += int(existing is None)
        self._conn.commit()
        return inserted

    def unread(self, kind: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM reservoir_events
            WHERE kind = ? AND status = 'unread'
            ORDER BY source_id ASC, published_at DESC, first_seen_at DESC
            """,
            (kind,),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["id"] = row["item_id"]
            payload["item_id"] = row["item_id"]
            payload["_reservoir_source_id"] = row["source_id"]
            payload["_reservoir_source_event_id"] = row["source_event_id"]
            payload["preprocess_score"] = row["preprocess_score"]
            if row["embedding_json"]:
                payload["_event_embedding"] = json.loads(row["embedding_json"])
            result.append(payload)
        return result

    def consume(self, item_ids: list[str], now: datetime) -> None:
        if not item_ids:
            return
        placeholders = ",".join("?" for _ in item_ids)
        _ = self._conn.execute(
            f"""
            UPDATE reservoir_events
            SET status = 'consumed', consumed_at = ?
            WHERE item_id IN ({placeholders})
            """,
            (now.isoformat(), *item_ids),
        )
        self._conn.commit()

    def consume_and_queue_ack(
        self,
        *,
        item_ids: list[str],
        acknowledgements: dict[str, list[str]],
        now: datetime,
    ) -> None:
        if item_ids:
            placeholders = ",".join("?" for _ in item_ids)
            _ = self._conn.execute(
                f"""
                UPDATE reservoir_events
                SET status = 'consumed', consumed_at = ?
                WHERE item_id IN ({placeholders})
                """,
                (now.isoformat(), *item_ids),
            )
        for source_id, event_ids in acknowledgements.items():
            for event_id in event_ids:
                _ = self._conn.execute(
                    """
                    INSERT INTO pending_acknowledgements(
                        source_id, source_event_id, queued_at
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(source_id, source_event_id) DO NOTHING
                    """,
                    (source_id, event_id, now.isoformat()),
                )
        self._conn.commit()

    def pending_acknowledgements(self) -> dict[str, list[str]]:
        rows = self._conn.execute(
            """
            SELECT source_id, source_event_id
            FROM pending_acknowledgements
            ORDER BY queued_at, source_id, source_event_id
            """
        ).fetchall()
        grouped: dict[str, list[str]] = {}
        for row in rows:
            grouped.setdefault(str(row["source_id"]), []).append(
                str(row["source_event_id"])
            )
        return grouped

    def mark_acknowledged(
        self,
        source_id: str,
        event_ids: list[str],
    ) -> None:
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        _ = self._conn.execute(
            f"""
            DELETE FROM pending_acknowledgements
            WHERE source_id = ? AND source_event_id IN ({placeholders})
            """,
            (source_id, *event_ids),
        )
        self._conn.commit()

    def unembedded(self, limit: int = 64) -> list[dict[str, str]]:
        rows = self._conn.execute(
            """
            SELECT item_id, payload_json FROM reservoir_events
            WHERE kind = 'content' AND status = 'unread' AND embedding_json IS NULL
            ORDER BY first_seen_at ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        result: list[dict[str, str]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            text = "\n".join(
                part
                for part in (
                    str(payload.get("title") or "").strip(),
                    str(payload.get("content") or payload.get("body") or "").strip(),
                )
                if part
            )
            if text:
                result.append({"item_id": str(row["item_id"]), "text": text})
        return result

    def save_event_embeddings(
        self, item_ids: list[str], embeddings: list[list[float]]
    ) -> None:
        for item_id, embedding in zip(item_ids, embeddings, strict=False):
            _ = self._conn.execute(
                "UPDATE reservoir_events SET embedding_json = ? WHERE item_id = ?",
                (json.dumps(embedding), item_id),
            )
        self._conn.commit()

    def load_hazard(self, session_key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM hazard_state WHERE session_key = ?", (session_key,)
        ).fetchone()
        return dict(row) if row is not None else None

    def save_hazard(
        self,
        *,
        session_key: str,
        hazard: float,
        threshold: float,
        updated_at: datetime,
        last_wake_at: datetime | None,
    ) -> None:
        _ = self._conn.execute(
            """
            INSERT INTO hazard_state(session_key, hazard, threshold, updated_at, last_wake_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_key) DO UPDATE SET
                hazard=excluded.hazard,
                threshold=excluded.threshold,
                updated_at=excluded.updated_at,
                last_wake_at=excluded.last_wake_at
            """,
            (
                session_key,
                hazard,
                threshold,
                updated_at.isoformat(),
                last_wake_at.isoformat() if last_wake_at is not None else None,
            ),
        )
        self._conn.commit()

    def ingest_context(
        self,
        snapshots: list[dict[str, Any]],
        now: datetime,
    ) -> list[ContextDriveResult]:
        results: list[ContextDriveResult] = []
        for snapshot in snapshots:
            source_id = str(snapshot.get("_source") or snapshot.get("source_id") or "").strip()
            if not source_id:
                continue
            previous = self.load_context(source_id)
            result = evaluate_context(snapshot, previous=previous)
            context = result.context
            _ = self._conn.execute(
                """
                INSERT INTO context_state(
                    source_id, payload_json, presence, interruptibility,
                    confidence, transition_name, observed_at, expires_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    presence=excluded.presence,
                    interruptibility=excluded.interruptibility,
                    confidence=excluded.confidence,
                    transition_name=excluded.transition_name,
                    observed_at=excluded.observed_at,
                    expires_at=excluded.expires_at,
                    updated_at=excluded.updated_at
                """,
                (
                    source_id,
                    json.dumps(snapshot, ensure_ascii=False),
                    context.presence,
                    context.interruptibility,
                    context.confidence,
                    context.transition,
                    context.observed_at.isoformat() if context.observed_at else None,
                    context.expires_at.isoformat() if context.expires_at else None,
                    now.isoformat(),
                ),
            )
            results.append(result)
        self._conn.commit()
        return results

    def load_context(self, source_id: str) -> NormalizedContext | None:
        row = self._conn.execute(
            "SELECT * FROM context_state WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if row is None:
            return None
        return NormalizedContext(
            presence=cast(Presence, row["presence"]),
            interruptibility=float(row["interruptibility"]),
            confidence=float(row["confidence"]),
            transition=str(row["transition_name"]),
            observed_at=_parse_optional_time(row["observed_at"]),
            expires_at=_parse_optional_time(row["expires_at"]),
        )

    def list_contexts(self) -> list[NormalizedContext]:
        rows = self._conn.execute(
            "SELECT source_id FROM context_state ORDER BY source_id"
        ).fetchall()
        return [
            context
            for row in rows
            if (context := self.load_context(str(row["source_id"]))) is not None
        ]

    def load_drift(self, session_key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM drift_state WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        return dict(row) if row is not None else None

    def save_drift_progress(
        self,
        *,
        session_key: str,
        hazard: float,
        threshold: float,
        updated_at: datetime,
    ) -> None:
        _ = self._conn.execute(
            """
            INSERT INTO drift_state(session_key, hazard, threshold, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_key) DO UPDATE SET
                hazard=excluded.hazard,
                threshold=excluded.threshold,
                updated_at=excluded.updated_at
            """,
            (session_key, hazard, threshold, updated_at.isoformat()),
        )
        self._conn.commit()

    def record_drift_success(
        self,
        *,
        session_key: str,
        now: datetime,
        fingerprint: str,
    ) -> None:
        previous = self.load_drift(session_key) or {}
        repeat_count = (
            int(previous.get("repeat_count") or 0) + 1
            if fingerprint and fingerprint == str(previous.get("last_fingerprint") or "")
            else 0
        )
        _ = self._conn.execute(
            """
            INSERT INTO drift_state(
                session_key, hazard, threshold, updated_at, last_drift_at,
                last_fingerprint, repeat_count
            ) VALUES (?, 0, ?, ?, ?, ?, ?)
            ON CONFLICT(session_key) DO UPDATE SET
                hazard=0,
                threshold=excluded.threshold,
                updated_at=excluded.updated_at,
                last_drift_at=excluded.last_drift_at,
                last_fingerprint=excluded.last_fingerprint,
                repeat_count=excluded.repeat_count
            """,
            (
                session_key,
                float(previous.get("threshold") or 0.8),
                now.isoformat(),
                now.isoformat(),
                fingerprint,
                repeat_count,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _parse_optional_time(value: object) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    return datetime.fromisoformat(str(value))
