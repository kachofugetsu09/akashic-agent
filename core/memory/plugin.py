from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActiveRecallRecord:
    """Save one bounded recall item suitable for plugin UI projection."""

    user_text: str
    assistant_preview: str
    started_at: str
    score: float


@dataclass(frozen=True, slots=True)
class ActiveRecallView:
    """Save the bounded active recall view for one Turn."""

    query_id: str
    dense: tuple[ActiveRecallRecord, ...]
    completion: tuple[ActiveRecallRecord, ...]
