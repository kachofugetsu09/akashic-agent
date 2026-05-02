from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI


class KVCacheDashboardReader:
    def __init__(self, workspace: Path) -> None:
        self.db_path = workspace / "observe" / "observe.db"
        self._lock = threading.RLock()

    def get_summary(self) -> dict[str, Any]:
        if not self.db_path.exists():
            return _summary_from_row(None)
        with self._lock:
            with _connect(self.db_path) as db:
                row = db.execute(
                    """
                    SELECT
                        COUNT(*) AS turn_count,
                        SUM(CASE WHEN react_cache_prompt_tokens IS NOT NULL THEN 1 ELSE 0 END) AS tracked_turn_count,
                        COALESCE(SUM(react_cache_prompt_tokens), 0) AS prompt_tokens,
                        COALESCE(SUM(react_cache_hit_tokens), 0) AS hit_tokens,
                        MAX(CASE WHEN react_cache_prompt_tokens IS NOT NULL THEN ts ELSE NULL END) AS last_tracked_at
                    FROM turns
                    """
                ).fetchone()
        return _summary_from_row(row)

    def list_turns(
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
                total_row = db.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM turns
                    WHERE react_cache_prompt_tokens IS NOT NULL
                    """
                ).fetchone()
                rows = db.execute(
                    """
                    SELECT
                        id,
                        ts,
                        source,
                        session_key,
                        user_msg,
                        react_cache_prompt_tokens AS prompt_tokens,
                        react_cache_hit_tokens AS hit_tokens
                    FROM turns
                    WHERE react_cache_prompt_tokens IS NOT NULL
                    ORDER BY ts DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (safe_size, offset),
                ).fetchall()
        total = int(total_row["total"] or 0) if total_row is not None else 0
        return [_row_to_cache_turn(row) for row in rows], total

    def get_turn(self, turn_id: int) -> dict[str, Any] | None:
        if not self.db_path.exists():
            return None
        with self._lock:
            with _connect(self.db_path) as db:
                row = db.execute(
                    """
                    SELECT
                        id,
                        ts,
                        source,
                        session_key,
                        user_msg,
                        react_cache_prompt_tokens AS prompt_tokens,
                        react_cache_hit_tokens AS hit_tokens
                    FROM turns
                    WHERE id = ? AND react_cache_prompt_tokens IS NOT NULL
                    """,
                    (turn_id,),
                ).fetchone()
        return _row_to_cache_turn(row) if row is not None else None


def register(app: FastAPI, plugin_dir: Path, workspace: Path) -> None:
    reader = KVCacheDashboardReader(workspace)

    @app.get("/api/dashboard/status-commands/kvcache/overview")
    def get_kvcache_overview() -> dict[str, Any]:
        return reader.get_summary()

    @app.get("/api/dashboard/status-commands/kvcache/turns")
    def list_kvcache_turns(
        page: int = 1,
        page_size: int = 25,
    ) -> dict[str, Any]:
        items, total = reader.list_turns(page=page, page_size=page_size)
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 100)),
        }

    @app.get("/api/dashboard/status-commands/kvcache/turns/{turn_id}")
    def get_kvcache_turn(turn_id: int) -> dict[str, Any]:
        item = reader.get_turn(turn_id)
        if item is None:
            return {}
        return {**item, "summary": reader.get_summary()}


def _summary_from_row(row: sqlite3.Row | None) -> dict[str, Any]:
    prompt_tokens = int(row["prompt_tokens"] or 0) if row is not None else 0
    hit_tokens = int(row["hit_tokens"] or 0) if row is not None else 0
    miss_tokens = max(0, prompt_tokens - hit_tokens)
    return {
        "turn_count": int(row["turn_count"] or 0) if row is not None else 0,
        "tracked_turn_count": (
            int(row["tracked_turn_count"] or 0) if row is not None else 0
        ),
        "prompt_tokens": prompt_tokens,
        "hit_tokens": hit_tokens,
        "miss_tokens": miss_tokens,
        "hit_rate": (hit_tokens / prompt_tokens) if prompt_tokens > 0 else None,
        "last_tracked_at": row["last_tracked_at"] if row is not None else None,
    }


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_cache_turn(row: sqlite3.Row) -> dict[str, Any]:
    prompt_tokens = int(row["prompt_tokens"] or 0)
    hit_tokens = int(row["hit_tokens"] or 0)
    miss_tokens = max(0, prompt_tokens - hit_tokens)
    return {
        "id": int(row["id"]),
        "ts": row["ts"],
        "source": row["source"],
        "session_key": row["session_key"],
        "user_preview": _preview_text(row["user_msg"], 90),
        "prompt_tokens": prompt_tokens,
        "hit_tokens": hit_tokens,
        "miss_tokens": miss_tokens,
        "hit_rate": (hit_tokens / prompt_tokens) if prompt_tokens > 0 else None,
    }


def _preview_text(value: Any, limit: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
