from __future__ import annotations

from dataclasses import dataclass

from .db import FeedbackEvent


@dataclass(frozen=True)
class ProactiveFeedbackRecorded:
    event_id: int
    feedback: FeedbackEvent
