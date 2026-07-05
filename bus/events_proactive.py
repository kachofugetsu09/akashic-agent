from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProactiveFeedbackRecorded:
    event_id: int
    session_key: str
    user_message_id: str
    assistant_message_id: str
    proactive_message_id: str
    feedback_type: str
    confidence: str
    pua_score: float | None
    lag_seconds: int | None
    matched_by: str
