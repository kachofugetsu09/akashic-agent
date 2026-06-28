from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI


class EmotionDashboardReader:
    def __init__(self, workspace: Path) -> None:
        self.db_path = workspace / "emotion" / "emotion.db"
        self._lock = threading.RLock()

    def get_overview(self) -> dict[str, Any]:
        if not self.db_path.exists():
            return _empty_overview()
        with self._lock:
            with _connect(self.db_path) as db:
                state = db.execute(
                    "SELECT valence, arousal, dominance, updated_at FROM emotion_state WHERE id = 1"
                ).fetchone()
                event_count = _scalar_int(db, "SELECT count(*) FROM emotion_events")
                effect_count = _scalar_int(db, "SELECT count(*) FROM emotion_effects")
                last_effect = db.execute(
                    """
                    SELECT created_at
                    FROM emotion_effects
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                ).fetchone()
        return {
            "state": dict(state) if state is not None else None,
            "event_count": event_count,
            "effect_count": effect_count,
            "last_effect_at": last_effect["created_at"] if last_effect else None,
        }

    def list_effects(
        self,
        *,
        page: int = 1,
        page_size: int = 25,
    ) -> tuple[list[dict[str, Any]], int]:
        if not self.db_path.exists():
            return [], 0
        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 100))
        offset = (safe_page - 1) * safe_size
        with self._lock:
            with _connect(self.db_path) as db:
                total = _scalar_int(db, "SELECT count(*) FROM emotion_effects")
                rows = db.execute(
                    """
                    SELECT *
                    FROM emotion_effects
                    ORDER BY created_at DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (safe_size, offset),
                ).fetchall()
        return [_decode_effect(row) for row in rows], total

    def get_effect(self, effect_id: int) -> dict[str, Any] | None:
        if not self.db_path.exists():
            return None
        with self._lock:
            with _connect(self.db_path) as db:
                row = db.execute(
                    "SELECT * FROM emotion_effects WHERE id = ?",
                    (effect_id,),
                ).fetchone()
        return _decode_effect(row) if row is not None else None


def register(app: FastAPI, plugin_dir: Path, workspace: Path) -> None:
    _ = plugin_dir
    reader = EmotionDashboardReader(workspace)

    @app.get("/api/dashboard/emotion/overview")
    def get_emotion_overview() -> dict[str, Any]:
        return reader.get_overview()

    @app.get("/api/dashboard/emotion/effects")
    def list_emotion_effects(
        page: int = 1,
        page_size: int = 25,
    ) -> dict[str, Any]:
        items, total = reader.list_effects(page=page, page_size=page_size)
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 100)),
        }

    @app.get("/api/dashboard/emotion/effects/{effect_id}")
    def get_emotion_effect(effect_id: int) -> dict[str, Any]:
        return reader.get_effect(effect_id) or {}


def _empty_overview() -> dict[str, Any]:
    return {
        "state": None,
        "event_count": 0,
        "effect_count": 0,
        "last_effect_at": None,
    }


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _scalar_int(db: sqlite3.Connection, sql: str) -> int:
    row = db.execute(sql).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _decode_effect(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    try:
        payload["metadata"] = json.loads(str(payload.pop("metadata_json") or "{}"))
    except Exception:
        payload["metadata"] = {}
    return payload
