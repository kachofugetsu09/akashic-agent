from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from proactive_v2.energy import compute_energy


@dataclass(frozen=True)
class EmotionState:
    valence: float
    arousal: float
    dominance: float
    updated_at: str


@dataclass(frozen=True)
class FeedbackDelta:
    valence: float
    dominance: float
    reason: str


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _ = conn.execute("PRAGMA journal_mode = WAL")
    _ = conn.execute("PRAGMA synchronous = NORMAL")
    _ = conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS emotion_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            valence REAL NOT NULL,
            arousal REAL NOT NULL,
            dominance REAL NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS emotion_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            source_plugin TEXT NOT NULL,
            source_event_id TEXT NOT NULL UNIQUE,
            source_type TEXT NOT NULL,
            session_key TEXT NOT NULL,
            valence_before REAL NOT NULL,
            arousal_before REAL NOT NULL,
            dominance_before REAL NOT NULL,
            valence_delta REAL NOT NULL,
            arousal_delta REAL NOT NULL,
            dominance_delta REAL NOT NULL,
            valence_after REAL NOT NULL,
            arousal_after REAL NOT NULL,
            dominance_after REAL NOT NULL,
            reason TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS emotion_effects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            tick_id TEXT NOT NULL UNIQUE,
            session_key TEXT NOT NULL,
            valence REAL NOT NULL,
            arousal REAL NOT NULL,
            dominance REAL NOT NULL,
            base_threshold REAL NOT NULL,
            final_threshold REAL NOT NULL,
            threshold_delta REAL NOT NULL,
            tone_label TEXT NOT NULL,
            expected_effect TEXT NOT NULL,
            prompt_section TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    _ = conn.execute(
        """
        INSERT OR IGNORE INTO emotion_state(id, valence, arousal, dominance, updated_at)
        VALUES(1, 0.0, 0.0, 0.0, ?)
        """,
        (now,),
    )
    conn.commit()
    return conn


def classify_feedback_delta(feedback_type: str, confidence: str) -> FeedbackDelta:
    if feedback_type == "explicit_quote":
        return FeedbackDelta(0.03, 0.08, "explicit_quote")
    if feedback_type == "topic_follow":
        if confidence in {"gold", "high"}:
            return FeedbackDelta(0.02, 0.05, "topic_follow_high")
        return FeedbackDelta(0.01, 0.03, "topic_follow_medium")
    if feedback_type == "no_topic_follow":
        return FeedbackDelta(0.0, -0.015, "no_topic_follow")
    return FeedbackDelta(0.0, 0.0, "neutral_feedback")


def apply_feedback(
    conn: sqlite3.Connection,
    *,
    source_event_id: str,
    session_key: str,
    feedback_type: str,
    confidence: str,
    payload: dict[str, Any],
) -> EmotionState:
    before = get_state(conn)
    delta = classify_feedback_delta(feedback_type, confidence)
    now = datetime.now(timezone.utc).isoformat()
    decayed = _decay(before, now)
    after = EmotionState(
        valence=_clamp(decayed.valence + delta.valence),
        arousal=decayed.arousal,
        dominance=_clamp(decayed.dominance + delta.dominance),
        updated_at=now,
    )
    try:
        _ = conn.execute(
            """
            INSERT INTO emotion_events (
                source_plugin, source_event_id, source_type, session_key,
                valence_before, arousal_before, dominance_before,
                valence_delta, arousal_delta, dominance_delta,
                valence_after, arousal_after, dominance_after,
                reason, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "proactive_feedback",
                source_event_id,
                feedback_type,
                session_key,
                before.valence,
                before.arousal,
                before.dominance,
                delta.valence,
                0.0,
                delta.dominance,
                after.valence,
                after.arousal,
                after.dominance,
                delta.reason,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
    except sqlite3.IntegrityError:
        return before
    _save_state(conn, after)
    conn.commit()
    return after


def build_effect(
    conn: sqlite3.Connection,
    *,
    tick_id: str,
    session_key: str,
    now_utc: datetime,
    last_user_at: datetime | None,
    base_threshold: float,
) -> dict[str, Any]:
    stored = _decay(get_state(conn), now_utc.isoformat())
    energy = compute_energy(last_user_at, now_utc)
    arousal = _clamp((1.0 - energy) * 2.0 - 1.0)
    state = EmotionState(
        valence=stored.valence,
        arousal=arousal,
        dominance=stored.dominance,
        updated_at=now_utc.isoformat(),
    )
    _save_state(conn, state)
    tone_label, tone_instruction = _tone(state)
    threshold_delta = _threshold_delta(state.dominance)
    final_threshold = _clamp_threshold(base_threshold + threshold_delta)
    expected_effect = _expected_effect(threshold_delta)
    prompt_section = (
        f"当前 VAD: valence={state.valence:.2f}, arousal={state.arousal:.2f}, dominance={state.dominance:.2f}。\n"
        f"语气约束: {tone_instruction}\n"
        f"发送克制程度: base_threshold={base_threshold:.2f}, effective_threshold={final_threshold:.2f}。"
        " effective_threshold 越高，越需要确认内容确实值得打扰用户。"
    )
    metadata: dict[str, object] = {
        "valence": round(state.valence, 4),
        "arousal": round(state.arousal, 4),
        "dominance": round(state.dominance, 4),
        "base_threshold": round(base_threshold, 4),
        "final_threshold": round(final_threshold, 4),
        "tone_label": tone_label,
        "expected_effect": expected_effect,
    }
    _ = conn.execute(
        """
        INSERT INTO emotion_effects (
            tick_id, session_key, valence, arousal, dominance,
            base_threshold, final_threshold, threshold_delta,
            tone_label, expected_effect, prompt_section, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tick_id) DO UPDATE SET
            session_key = excluded.session_key,
            valence = excluded.valence,
            arousal = excluded.arousal,
            dominance = excluded.dominance,
            base_threshold = excluded.base_threshold,
            final_threshold = excluded.final_threshold,
            threshold_delta = excluded.threshold_delta,
            tone_label = excluded.tone_label,
            expected_effect = excluded.expected_effect,
            prompt_section = excluded.prompt_section,
            metadata_json = excluded.metadata_json
        """,
        (
            tick_id,
            session_key,
            state.valence,
            state.arousal,
            state.dominance,
            base_threshold,
            final_threshold,
            threshold_delta,
            tone_label,
            expected_effect,
            prompt_section,
            json.dumps(metadata, ensure_ascii=False),
        ),
    )
    conn.commit()
    return {
        "provider_name": "emotion",
        "prompt_section": prompt_section,
        "threshold_delta": threshold_delta,
        "metadata": metadata,
    }


def get_state(conn: sqlite3.Connection) -> EmotionState:
    row = conn.execute(
        """
        SELECT valence, arousal, dominance, updated_at
        FROM emotion_state
        WHERE id = 1
        """
    ).fetchone()
    if row is None:
        now = datetime.now(timezone.utc).isoformat()
        return EmotionState(0.0, 0.0, 0.0, now)
    return EmotionState(
        valence=float(row["valence"]),
        arousal=float(row["arousal"]),
        dominance=float(row["dominance"]),
        updated_at=str(row["updated_at"]),
    )


def _save_state(conn: sqlite3.Connection, state: EmotionState) -> None:
    _ = conn.execute(
        """
        INSERT INTO emotion_state(id, valence, arousal, dominance, updated_at)
        VALUES(1, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            valence = excluded.valence,
            arousal = excluded.arousal,
            dominance = excluded.dominance,
            updated_at = excluded.updated_at
        """,
        (state.valence, state.arousal, state.dominance, state.updated_at),
    )


def _decay(state: EmotionState, now_iso: str) -> EmotionState:
    try:
        before = datetime.fromisoformat(state.updated_at)
        now = datetime.fromisoformat(now_iso)
        hours = max(0.0, (now - before).total_seconds() / 3600.0)
    except Exception:
        hours = 0.0
    factor = math.exp(-hours / 72.0)
    return EmotionState(
        valence=_clamp(state.valence * factor),
        arousal=state.arousal,
        dominance=_clamp(state.dominance * factor),
        updated_at=now_iso,
    )


def _threshold_delta(dominance: float) -> float:
    if dominance >= 0.60:
        return -0.04
    if dominance >= 0.25:
        return -0.02
    if dominance <= -0.60:
        return 0.08
    if dominance <= -0.25:
        return 0.04
    return 0.0


def _tone(state: EmotionState) -> tuple[str, str]:
    if state.valence >= 0.2 and state.arousal >= 0.2 and state.dominance >= 0.1:
        return "bright_confident", "带着轻快的分享欲，但不要夸张，不要自我表演。"
    if state.valence < -0.2 and state.arousal >= 0.2:
        return "careful_tentative", "语气更谨慎克制，先确认价值，不要显得急着打扰。"
    if state.dominance <= -0.25:
        return "low_confidence", "语气保持简短、试探和低打扰感，除非内容明显重要。"
    if state.arousal <= -0.2:
        return "calm", "语气平稳放松，少用强烈表达。"
    return "neutral", "保持自然、简洁、贴近上下文的语气。"


def _expected_effect(threshold_delta: float) -> str:
    if threshold_delta > 0:
        return "raise_send_bar"
    if threshold_delta < 0:
        return "lower_send_bar"
    return "tone_only"


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _clamp_threshold(value: float) -> float:
    return max(0.54, min(0.78, value))
