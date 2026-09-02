from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from agent.turn_events.proactive_feedback import (
    PROACTIVE_FEEDBACK_COMMITTED,
    PROACTIVE_FEEDBACK_PREVIEW_MAX_CHARS,
    ProactiveFeedbackCommitted,
)


def _event(**changes: object) -> ProactiveFeedbackCommitted:
    values: dict[str, object] = {
        "event_id": "proactive_feedback:1",
        "session_key": "telegram:1",
        "user_message_id": "user-1",
        "assistant_message_id": "assistant-1",
        "proactive_message_id": "proactive-1",
        "feedback_type": "topic_follow",
        "confidence": "high",
        "pa_score": 0.8,
        "pua_score": 0.9,
        "lag_seconds": 12,
        "candidate_count": 2,
        "matched_by": "recent_pua",
        "reason": "pua_high",
        "user_content_preview": "用户回应",
        "assistant_content_preview": "助手继续",
        "proactive_content_preview": "主动消息",
    }
    values.update(changes)
    return ProactiveFeedbackCommitted(**cast(Any, values))


def test_feedback_payload_is_frozen_and_preview_bounded() -> None:
    event = _event(
        user_content_preview="x" * PROACTIVE_FEEDBACK_PREVIEW_MAX_CHARS,
        pa_score=1,
    )

    assert event.pa_score == 1.0
    assert event.user_content_preview == "x" * PROACTIVE_FEEDBACK_PREVIEW_MAX_CHARS
    assert PROACTIVE_FEEDBACK_COMMITTED.name == "proactive.feedback.committed"
    with pytest.raises(FrozenInstanceError):
        event.reason = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    (
        {"event_id": ""},
        {"assistant_message_id": " assistant-1"},
        {"pa_score": float("nan")},
        {"pua_score": float("inf")},
        {"lag_seconds": -1},
        {"candidate_count": True},
        {
            "proactive_content_preview": "x"
            * (PROACTIVE_FEEDBACK_PREVIEW_MAX_CHARS + 1)
        },
    ),
)
def test_feedback_payload_rejects_unbounded_or_malformed_values(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _event(**changes)
