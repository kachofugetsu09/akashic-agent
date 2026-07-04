from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class MessageRow:
    id: str
    seq: int
    role: str
    content: str
    extra: str | None
    ts: str


@dataclass(frozen=True)
class QuoteParts:
    quoted_text: str | None
    current_text: str


@dataclass(frozen=True)
class FeedbackScore:
    proactive: MessageRow
    pa_score: float
    pua_score: float
    matched_by: str
    feedback_type: str
    confidence: str
    reason: str
    candidate_count: int
    lag_seconds: int | None


class EmbedBatch(Protocol):
    async def __call__(self, texts: list[str]) -> list[list[float]]: ...


def clean_text(text: str, max_chars: int = 1200) -> str:
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def normalize_quote_text(text: str, max_chars: int = 1200) -> str:
    cleaned = clean_text(text, max_chars=max_chars).lower()
    cleaned = re.sub(r"[*_`#>\[\]()]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_quote_parts(content: str) -> QuoteParts:
    marker = "【你当前新消息】"
    if marker not in content:
        return QuoteParts(quoted_text=None, current_text=content.strip())

    before, after = content.split(marker, 1)
    quoted = before
    quote_prefix = "被回复消息"
    if quote_prefix in quoted:
        quoted = quoted.split(quote_prefix, 1)[1]
    if "：" in quoted:
        quoted = quoted.split("：", 1)[1]
    return QuoteParts(
        quoted_text=clean_text(quoted, max_chars=300) or None,
        current_text=after.strip(),
    )


def is_proactive(extra: str | None) -> bool:
    if not extra:
        return False
    try:
        payload = json.loads(extra)
    except json.JSONDecodeError:
        return "proactive" in extra and "true" in extra.lower()
    return bool(payload.get("proactive"))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def classify_pua(score: float) -> tuple[str, str, str]:
    if score >= 0.62:
        return "topic_follow", "high", "pua_high"
    if score >= 0.54:
        return "topic_follow", "medium", "pua_medium"
    return "no_topic_follow", "low", "pua_low"


def latest_turn_messages(
    conn: sqlite3.Connection,
    *,
    session_key: str,
    user_content: str,
    assistant_content: str,
) -> tuple[MessageRow, MessageRow] | None:
    user = conn.execute(
        """
        SELECT id, seq, role, content, extra, ts
        FROM messages
        WHERE session_key = ? AND role = 'user' AND content = ?
        ORDER BY seq DESC
        LIMIT 1
        """,
        (session_key, user_content),
    ).fetchone()
    assistant = conn.execute(
        """
        SELECT id, seq, role, content, extra, ts
        FROM messages
        WHERE session_key = ? AND role = 'assistant' AND content = ?
        ORDER BY seq DESC
        LIMIT 1
        """,
        (session_key, assistant_content),
    ).fetchone()
    if user is None or assistant is None:
        return None
    return _row(user), _row(assistant)


def iter_user_assistant_turns(
    conn: sqlite3.Connection,
) -> list[tuple[str, MessageRow, MessageRow]]:
    rows = conn.execute(
        """
        SELECT u.session_key,
               u.id AS user_id,
               u.seq AS user_seq,
               u.role AS user_role,
               u.content AS user_content,
               u.extra AS user_extra,
               u.ts AS user_ts,
               a.id AS assistant_id,
               a.seq AS assistant_seq,
               a.role AS assistant_role,
               a.content AS assistant_content,
               a.extra AS assistant_extra,
               a.ts AS assistant_ts
        FROM messages u
        JOIN messages a
          ON a.id = (
              SELECT m.id
              FROM messages m
              WHERE m.session_key = u.session_key
                AND m.seq > u.seq
                AND m.role = 'assistant'
                AND m.content IS NOT NULL
              ORDER BY m.seq ASC
              LIMIT 1
          )
        WHERE u.role = 'user'
          AND u.content IS NOT NULL
        ORDER BY u.session_key ASC, u.seq ASC
        """
    ).fetchall()
    turns: list[tuple[str, MessageRow, MessageRow]] = []
    for row in rows:
        turns.append((
            str(row["session_key"]),
            MessageRow(
                id=str(row["user_id"]),
                seq=int(row["user_seq"]),
                role=str(row["user_role"]),
                content=str(row["user_content"] or ""),
                extra=row["user_extra"],
                ts=str(row["user_ts"]),
            ),
            MessageRow(
                id=str(row["assistant_id"]),
                seq=int(row["assistant_seq"]),
                role=str(row["assistant_role"]),
                content=str(row["assistant_content"] or ""),
                extra=row["assistant_extra"],
                ts=str(row["assistant_ts"]),
            ),
        ))
    return turns


def recent_proactive_messages(
    conn: sqlite3.Connection,
    *,
    session_key: str,
    before_seq: int,
    limit: int,
) -> list[MessageRow]:
    rows = conn.execute(
        """
        SELECT id, seq, role, content, extra, ts
        FROM messages
        WHERE session_key = ?
          AND role = 'assistant'
          AND seq < ?
          AND content IS NOT NULL
        ORDER BY seq DESC
        LIMIT ?
        """,
        (session_key, before_seq, limit * 4),
    ).fetchall()
    proactive = [_row(row) for row in rows if is_proactive(row["extra"])]
    return proactive[:limit]


def proactive_since_previous_user(
    conn: sqlite3.Connection,
    *,
    session_key: str,
    before_seq: int,
    limit: int | None = None,
) -> list[MessageRow]:
    previous = conn.execute(
        """
        SELECT seq
        FROM messages
        WHERE session_key = ?
          AND role = 'user'
          AND seq < ?
        ORDER BY seq DESC
        LIMIT 1
        """,
        (session_key, before_seq),
    ).fetchone()
    after_seq = int(previous["seq"]) if previous is not None else -1
    rows = conn.execute(
        """
        SELECT id, seq, role, content, extra, ts
        FROM messages
        WHERE session_key = ?
          AND role = 'assistant'
          AND seq > ?
          AND seq < ?
          AND content IS NOT NULL
        ORDER BY seq DESC
        """,
        (session_key, after_seq, before_seq),
    ).fetchall()
    proactive = [_row(row) for row in rows if is_proactive(row["extra"])]
    if limit is None:
        return proactive
    return proactive[:limit]


async def score_followup(
    *,
    embed_batch: EmbedBatch,
    user: MessageRow,
    assistant: MessageRow,
    candidates: list[MessageRow],
    allow_pua: bool = True,
) -> FeedbackScore | None:
    if not candidates:
        return None

    quote = parse_quote_parts(user.content)
    quoted_match = _match_quoted(candidates, quote.quoted_text)
    if quoted_match is not None:
        return FeedbackScore(
            proactive=quoted_match,
            pa_score=1.0,
            pua_score=1.0,
            matched_by="explicit_quote",
            feedback_type="explicit_quote",
            confidence="gold",
            reason="explicit_quote",
            candidate_count=len(candidates),
            lag_seconds=_lag_seconds(quoted_match.ts, user.ts),
        )
    if not allow_pua:
        return None

    target = candidates[0]
    p_text = clean_text(target.content)
    a_text = clean_text(assistant.content)
    ua_text = clean_text(f"{quote.current_text}\n\n{assistant.content}")
    vectors = await embed_batch([p_text, a_text, ua_text])
    pa_score = cosine(vectors[0], vectors[1])
    pua_score = cosine(vectors[0], vectors[2])
    feedback_type, confidence, reason = classify_pua(pua_score)
    return FeedbackScore(
        proactive=target,
        pa_score=pa_score,
        pua_score=pua_score,
        matched_by="recent_pua",
        feedback_type=feedback_type,
        confidence=confidence,
        reason=reason,
        candidate_count=len(candidates),
        lag_seconds=_lag_seconds(target.ts, user.ts),
    )


def _match_quoted(candidates: list[MessageRow], quoted_text: str | None) -> MessageRow | None:
    if not quoted_text:
        return None
    needle = normalize_quote_text(quoted_text, max_chars=220)
    if len(needle) < 12:
        return None
    for candidate in candidates:
        haystack = normalize_quote_text(candidate.content, max_chars=3000)
        if needle in haystack:
            return candidate
    return None


def _lag_seconds(start: str, end: str) -> int | None:
    try:
        return int(datetime.fromisoformat(end).timestamp() - datetime.fromisoformat(start).timestamp())
    except ValueError:
        return None


def _row(row: sqlite3.Row) -> MessageRow:
    return MessageRow(
        id=str(row["id"]),
        seq=int(row["seq"]),
        role=str(row["role"]),
        content=str(row["content"] or ""),
        extra=row["extra"],
        ts=str(row["ts"]),
    )
