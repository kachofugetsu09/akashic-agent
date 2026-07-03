from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, cast

SKILL_NAME = "emotion:feedback-preference-context"
PROACTIVE_TEXT_LIMIT = 100
QUESTION_MARKERS = ("吗", "么", "为什么", "怎么", "谁", "哪")


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _load_cursor(drift_dir: Path) -> dict[str, Any]:
    db_path = drift_dir / "drift.db"
    if not db_path.exists():
        return {}
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT cursor_json
            FROM skill_continuum
            WHERE skill_name = ?
            """,
            (SKILL_NAME,),
        ).fetchone()
    if row is None:
        return {}
    try:
        data = json.loads(str(row["cursor_json"] or "{}"))
    except json.JSONDecodeError:
        return {}
    return cast(dict[str, Any], data) if isinstance(data, dict) else {}


def _message_previews(workspace: Path, ids: list[str]) -> dict[str, str]:
    db_path = workspace / "sessions.db"
    if not ids or not db_path.exists():
        return {}
    unique_ids = list(dict.fromkeys(text for text in ids if text))
    placeholders = ",".join("?" for _ in unique_ids)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, content
            FROM messages
            WHERE id IN ({placeholders})
            """,
            unique_ids,
        ).fetchall()
    return {str(row["id"]): str(row["content"] or "") for row in rows}


def _clip_text(text: str, limit: int) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _signal_hints(user: str, feedback_type: str) -> list[str]:
    text = _clean_text(user)
    hints: list[str] = []
    if feedback_type == "explicit_quote":
        hints.append("explicit_quote")
    if any(marker in text for marker in QUESTION_MARKERS):
        hints.append("question")
    if len(text) >= 8:
        hints.append("substantive_reply")
    return hints


def sample(
    drift_dir: Path,
    limit: int,
    chunk_size: int,
    chunk_index: int,
) -> dict[str, Any]:
    workspace = drift_dir.parent
    feedback_db = workspace / "proactive_feedback" / "proactive_feedback.db"
    if not feedback_db.exists():
        return {"found": False, "reason": "feedback_db_missing"}

    cursor = _load_cursor(drift_dir)
    last_feedback_id = int(
        cursor.get("latest_processed_feedback_id")
        or cursor.get("last_feedback_id")
        or 0
    )
    safe_limit = max(1, min(int(limit), 50))
    with _connect(feedback_db) as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                created_at,
                session_key,
                user_message_id,
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
            WHERE id > ?
              AND feedback_type IN ('topic_follow', 'explicit_quote')
            ORDER BY id DESC
            LIMIT ?
            """,
            (last_feedback_id, safe_limit),
        ).fetchall()

    if not rows:
        return {
            "found": False,
            "last_feedback_id": last_feedback_id,
            "latest_processed_feedback_id": last_feedback_id,
            "cursor_tail_feedback_id": last_feedback_id,
            "count": 0,
            "chunk_index": max(0, int(chunk_index)),
            "chunk_size": max(1, min(int(chunk_size), 10)),
            "has_more": False,
            "next_chunk_index": None,
        }

    safe_chunk_size = max(1, min(int(chunk_size), 10))
    safe_chunk_index = max(0, int(chunk_index))
    chunk_start = safe_chunk_index * safe_chunk_size
    chunk_end = chunk_start + safe_chunk_size
    chunk_rows = rows[chunk_start:chunk_end]
    has_more = chunk_end < len(rows)

    message_ids: list[str] = []
    for row in chunk_rows:
        message_ids.extend(
            str(row[key] or "")
            for key in (
                "proactive_message_id",
                "user_message_id",
            )
            if row[key]
        )
    previews = _message_previews(workspace, message_ids)
    events: list[dict[str, Any]] = []
    for row in chunk_rows:
        proactive_id = str(row["proactive_message_id"] or "")
        user_id = str(row["user_message_id"] or "")
        user_text = _clean_text(previews.get(user_id, ""))
        events.append(
            {
                "id": int(row["id"]),
                "created_at": str(row["created_at"] or ""),
                "session_key": str(row["session_key"] or ""),
                "feedback_type": str(row["feedback_type"] or ""),
                "confidence": str(row["confidence"] or ""),
                "pa_score": row["pa_score"],
                "pua_score": row["pua_score"],
                "lag_seconds": row["lag_seconds"],
                "candidate_count": int(row["candidate_count"] or 0),
                "matched_by": str(row["matched_by"] or ""),
                "reason": str(row["reason"] or ""),
                "message_ids": {
                    "proactive": proactive_id,
                    "user": user_id,
                },
                "signal_hints": _signal_hints(
                    user_text,
                    str(row["feedback_type"] or ""),
                ),
                "texts": {
                    "proactive": _clip_text(
                        previews.get(proactive_id, ""),
                        PROACTIVE_TEXT_LIMIT,
                    ),
                    "user": user_text,
                },
            }
        )

    cursor_tail = max(int(row["id"]) for row in rows)
    return {
        "found": True,
        "last_feedback_id": last_feedback_id,
        "latest_processed_feedback_id": last_feedback_id,
        "count": len(rows),
        "cursor_tail_feedback_id": cursor_tail,
        "feedback_ids": [int(row["id"]) for row in rows],
        "chunk_index": safe_chunk_index,
        "chunk_size": safe_chunk_size,
        "chunk_count": len(events),
        "chunk_feedback_ids": [item["id"] for item in events],
        "has_more": has_more,
        "next_chunk_index": safe_chunk_index + 1 if has_more else None,
        "text_limits": {
            "proactive": PROACTIVE_TEXT_LIMIT,
            "user": None,
        },
        "events": events,
    }


def evidence_bundle(
    drift_dir: Path,
    limit: int,
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    chunk_index = 0
    while True:
        payload = sample(drift_dir, limit, 10, chunk_index)
        chunks.append(payload)
        if not payload.get("found"):
            return payload
        events.extend(payload.get("events", []))
        if not payload.get("has_more"):
            break
        chunk_index = int(payload.get("next_chunk_index") or chunk_index + 1)

    first = chunks[0]
    compact_events: list[dict[str, Any]] = []
    for event in events:
        texts = event.get("texts", {})
        message_ids = event.get("message_ids", {})
        compact_events.append(
            {
                "fid": int(event["id"]),
                "type": str(event.get("feedback_type") or ""),
                "conf": str(event.get("confidence") or ""),
                "uid": str(message_ids.get("user") or ""),
                "pid": str(message_ids.get("proactive") or ""),
                "hints": list(event.get("signal_hints") or []),
                "p": str(texts.get("proactive") or ""),
                "u": str(texts.get("user") or ""),
            }
        )

    return {
        "found": True,
        "last_feedback_id": first["last_feedback_id"],
        "count": first["count"],
        "cursor_tail_feedback_id": first["cursor_tail_feedback_id"],
        "feedback_ids": first["feedback_ids"],
        "events": compact_events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sample_cmd = sub.add_parser("sample")
    _ = sample_cmd.add_argument("--drift-dir", default=".")
    _ = sample_cmd.add_argument("--limit", type=int, default=50)
    _ = sample_cmd.add_argument("--chunk-size", type=int, default=10)
    _ = sample_cmd.add_argument("--chunk-index", type=int, default=0)
    bundle_cmd = sub.add_parser("evidence-bundle")
    _ = bundle_cmd.add_argument("--drift-dir", default=".")
    _ = bundle_cmd.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    if args.command == "sample":
        payload = sample(
            Path(args.drift_dir).expanduser().resolve(),
            args.limit,
            args.chunk_size,
            args.chunk_index,
        )
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    elif args.command == "evidence-bundle":
        payload = evidence_bundle(
            Path(args.drift_dir).expanduser().resolve(),
            args.limit,
        )
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
