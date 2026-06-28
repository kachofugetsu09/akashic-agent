from __future__ import annotations

from contextlib import contextmanager
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI


class ProactiveFeedbackDashboardReader:
    def __init__(self, workspace: Path) -> None:
        self.db_path = workspace / "proactive_feedback" / "proactive_feedback.db"
        self.sessions_db_path = workspace / "sessions.db"
        self._lock = threading.RLock()

    def get_overview(self) -> dict[str, Any]:
        if not self.db_path.exists():
            return _empty_overview()
        with self._lock:
            with _connect(self.db_path) as db:
                total = _scalar_int(db, "SELECT count(*) FROM proactive_feedback_events")
                topic_follow = _scalar_int(
                    db,
                    """
                    SELECT count(*) FROM proactive_feedback_events
                    WHERE feedback_type IN ('topic_follow', 'explicit_quote')
                    """,
                )
                explicit_quote = _scalar_int(
                    db,
                    """
                    SELECT count(*) FROM proactive_feedback_events
                    WHERE feedback_type = 'explicit_quote'
                    """,
                )
                high_confidence = _scalar_int(
                    db,
                    """
                    SELECT count(*) FROM proactive_feedback_events
                    WHERE confidence IN ('gold', 'high')
                    """,
                )
                last_row = db.execute(
                    """
                    SELECT created_at
                    FROM proactive_feedback_events
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                ).fetchone()
                by_type = _group_rows(db, "feedback_type")
                by_confidence = _group_rows(db, "confidence")
        return {
            "total": total,
            "topic_follow": topic_follow,
            "explicit_quote": explicit_quote,
            "high_confidence": high_confidence,
            "follow_rate": topic_follow / total if total else None,
            "last_created_at": last_row["created_at"] if last_row else None,
            "by_type": by_type,
            "by_confidence": by_confidence,
        }

    def list_events(
        self,
        *,
        page: int = 1,
        page_size: int = 25,
        feedback_type: str = "",
    ) -> tuple[list[dict[str, Any]], int]:
        if not self.db_path.exists():
            return [], 0
        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 100))
        offset = (safe_page - 1) * safe_size
        where = ""
        params: list[Any] = []
        if feedback_type:
            where = "WHERE feedback_type = ?"
            params.append(feedback_type)
        with self._lock:
            with _connect(self.db_path) as db:
                total = _scalar_int(
                    db,
                    f"SELECT count(*) FROM proactive_feedback_events {where}",
                    params,
                )
                rows = db.execute(
                    f"""
                    SELECT
                        id,
                        created_at,
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
                    FROM proactive_feedback_events
                    {where}
                    ORDER BY created_at DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (*params, safe_size, offset),
                ).fetchall()
            previews = self._load_previews(rows)
        return [_event_row(row, previews, preview_limit=360) for row in rows], total

    def get_event(self, event_id: int) -> dict[str, Any] | None:
        if not self.db_path.exists():
            return None
        with self._lock:
            with _connect(self.db_path) as db:
                row = db.execute(
                    """
                    SELECT *
                    FROM proactive_feedback_events
                    WHERE id = ?
                    """,
                    (event_id,),
                ).fetchone()
            if row is None:
                return None
            previews = self._load_previews([row])
        return _event_row(row, previews, preview_limit=2400)

    def _load_previews(self, rows: list[sqlite3.Row]) -> dict[str, str]:
        ids: list[str] = []
        for row in rows:
            ids.extend(
                str(value)
                for value in (
                    row["user_message_id"],
                    row["assistant_message_id"],
                    row["proactive_message_id"],
                )
                if value
            )
        if not ids or not self.sessions_db_path.exists():
            return {}
        unique_ids = list(dict.fromkeys(ids))
        placeholders = ",".join("?" for _ in unique_ids)
        with _connect(self.sessions_db_path) as db:
            msg_rows = db.execute(
                f"""
                SELECT id, content
                FROM messages
                WHERE id IN ({placeholders})
                """,
                unique_ids,
            ).fetchall()
        return {str(row["id"]): str(row["content"] or "") for row in msg_rows}


def register(app: FastAPI, plugin_dir: Path, workspace: Path) -> None:
    _ = plugin_dir
    reader = ProactiveFeedbackDashboardReader(workspace)

    @app.get("/api/dashboard/proactive-feedback/overview")
    def get_proactive_feedback_overview() -> dict[str, Any]:
        return reader.get_overview()

    @app.get("/api/dashboard/proactive-feedback/events")
    def list_proactive_feedback_events(
        page: int = 1,
        page_size: int = 25,
        feedback_type: str = "",
    ) -> dict[str, Any]:
        items, total = reader.list_events(
            page=page,
            page_size=page_size,
            feedback_type=feedback_type,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 100)),
        }

    @app.get("/api/dashboard/proactive-feedback/events/{event_id}")
    def get_proactive_feedback_event(event_id: int) -> dict[str, Any]:
        item = reader.get_event(event_id)
        return item or {}


def _empty_overview() -> dict[str, Any]:
    return {
        "total": 0,
        "topic_follow": 0,
        "explicit_quote": 0,
        "high_confidence": 0,
        "follow_rate": None,
        "last_created_at": None,
        "by_type": [],
        "by_confidence": [],
    }


def _group_rows(db: sqlite3.Connection, column: str) -> list[dict[str, Any]]:
    rows = db.execute(
        f"""
        SELECT {column} AS key, count(*) AS count
        FROM proactive_feedback_events
        GROUP BY {column}
        ORDER BY count DESC
        """
    ).fetchall()
    return [{"key": row["key"], "count": int(row["count"] or 0)} for row in rows]


def _scalar_int(
    db: sqlite3.Connection,
    sql: str,
    params: list[Any] | None = None,
) -> int:
    row = db.execute(sql, params or []).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _event_row(
    row: sqlite3.Row,
    previews: dict[str, str],
    *,
    preview_limit: int = 120,
) -> dict[str, Any]:
    user_id = str(row["user_message_id"])
    assistant_id = str(row["assistant_message_id"])
    proactive_id = str(row["proactive_message_id"] or "")
    user_text = previews.get(user_id)
    return {
        "id": int(row["id"]),
        "created_at": row["created_at"],
        "session_key": row["session_key"],
        "user_message_id": user_id,
        "assistant_message_id": assistant_id,
        "proactive_message_id": proactive_id,
        "feedback_type": row["feedback_type"],
        "confidence": row["confidence"],
        "pa_score": row["pa_score"],
        "pua_score": row["pua_score"],
        "lag_seconds": row["lag_seconds"],
        "candidate_count": int(row["candidate_count"] or 0),
        "matched_by": row["matched_by"],
        "reason": row["reason"],
        "user_preview": _preview(user_text, preview_limit),
        "user_reply_preview": _preview(_current_reply(user_text), preview_limit),
        "quoted_preview": _preview(_quoted_reply(user_text), preview_limit),
        "assistant_preview": _preview(previews.get(assistant_id), preview_limit),
        "proactive_preview": _preview(previews.get(proactive_id), preview_limit),
    }


def _preview(value: str | None, limit: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _current_reply(value: str | None) -> str:
    text = str(value or "")
    marker = "【你当前新消息】"
    if marker not in text:
        return text
    return text.split(marker, 1)[1].strip()


def _quoted_reply(value: str | None) -> str:
    text = str(value or "")
    start = "被回复消息"
    end = "【你当前新消息】"
    if start not in text or end not in text:
        return ""
    quoted = text.split(start, 1)[1].split(end, 1)[0]
    if "：" in quoted:
        quoted = quoted.split("：", 1)[1]
    return quoted.strip()


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
