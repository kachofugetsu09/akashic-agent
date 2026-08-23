from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

from plugins.wake.hazard import rank_events


SelectionDecision = Literal["select", "decline"]


@dataclass(frozen=True, slots=True)
class DutyProposal:
    """Describe one frozen duty candidate without changing domain state."""

    ref: Mapping[str, object]
    payload: Mapping[str, object]
    decision: SelectionDecision


def propose_content(
    items: Sequence[Mapping[str, object]], *, now: datetime
) -> DutyProposal | None:
    """Choose the highest-ranked due Content revision from one snapshot."""

    due = [item for item in items if item.get("due") is True]
    if not due:
        return None
    ranked = rank_events(
        [_content_event(item) for item in due],
        now=now,
    )
    selected = ranked[0]["_wake_item"]
    if not isinstance(selected, Mapping):
        raise TypeError("Wake ranked Content item 必须是 Mapping")
    selected_map = cast(Mapping[str, object], selected)
    payload = selected_map.get("payload")
    ref = selected_map.get("ref")
    if not isinstance(payload, Mapping) or not isinstance(ref, Mapping):
        raise ValueError("Wake Content snapshot 缺少 payload/ref")
    decision = "decline" if payload.get("wake_action") == "decline" else "select"
    return DutyProposal(
        ref=cast(Mapping[str, object], ref),
        payload=cast(Mapping[str, object], payload),
        decision=decision,
    )


def propose_drift(proposals: Sequence[Mapping[str, object]]) -> DutyProposal | None:
    """Choose the first durable due Drift proposal."""

    selected = next((item for item in proposals if item.get("due") is True), None)
    if selected is None:
        return None
    payload = selected.get("payload")
    ref = selected.get("ref")
    if not isinstance(payload, Mapping) or not isinstance(ref, Mapping):
        raise ValueError("Wake Drift snapshot 缺少 payload/ref")
    decision = "decline" if payload.get("wake_action") == "decline" else "select"
    return DutyProposal(
        ref=cast(Mapping[str, object], ref),
        payload=cast(Mapping[str, object], payload),
        decision=decision,
    )


def _content_event(item: Mapping[str, object]) -> dict[str, object]:
    payload = item.get("payload")
    ref = item.get("ref")
    if not isinstance(payload, Mapping) or not isinstance(ref, Mapping):
        raise ValueError("Wake Content snapshot 缺少 payload/ref")
    event = dict(cast(Mapping[str, object], payload))
    event["id"] = ref.get("item_id", "")
    event["source_id"] = ref.get("source_id", "")
    event["_wake_item"] = item
    if "_wake_interest_score" not in event:
        event["_wake_interest_score"] = payload.get("preprocess_score", 1.0)
    return event
