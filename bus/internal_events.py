from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agent.delegation_policy import SpawnDecision, SpawnDecisionMeta
from bus.events import InboundMessage

SPAWN_COMPLETED = "spawn_completed"
_EVENT_KEY = "internal_event"
_SPAWN_KEY = "spawn"
_SPAWN_DECISION_KEY = "spawn_decision"
_SPAWN_COMPLETED_CONTENT = "[internal spawn completed]"


@dataclass(frozen=True)
class SpawnCompletionEvent:
    job_id: str
    label: str
    task: str
    status: str
    exit_reason: str
    result: str


def make_spawn_completion_message(
    *,
    channel: str,
    chat_id: str,
    event: SpawnCompletionEvent,
    decision: SpawnDecision | None = None,
) -> InboundMessage:
    metadata = {
        _EVENT_KEY: SPAWN_COMPLETED,
        _SPAWN_KEY: asdict(event),
    }
    if decision is not None:
        metadata[_SPAWN_DECISION_KEY] = asdict(decision)
    return InboundMessage(
        channel=channel,
        sender="spawn",
        chat_id=chat_id,
        content=_SPAWN_COMPLETED_CONTENT,
        metadata=metadata,
    )


def is_spawn_completion_message(msg: InboundMessage) -> bool:
    md = msg.metadata if isinstance(msg.metadata, dict) else {}
    return md.get(_EVENT_KEY) == SPAWN_COMPLETED


def parse_spawn_completion(msg: InboundMessage) -> SpawnCompletionEvent:
    md = msg.metadata if isinstance(msg.metadata, dict) else {}
    raw = md.get(_SPAWN_KEY, {}) if isinstance(md, dict) else {}
    payload: dict[str, Any] = raw if isinstance(raw, dict) else {}
    return SpawnCompletionEvent(
        job_id=str(payload.get("job_id", "") or ""),
        label=str(payload.get("label", "") or ""),
        task=str(payload.get("task", "") or ""),
        status=str(payload.get("status", "") or ""),
        exit_reason=str(payload.get("exit_reason", "") or ""),
        result=str(payload.get("result", "") or ""),
    )


def parse_spawn_decision(msg: InboundMessage) -> SpawnDecision | None:
    md = msg.metadata if isinstance(msg.metadata, dict) else {}
    raw = md.get(_SPAWN_DECISION_KEY, {}) if isinstance(md, dict) else {}
    payload: dict[str, Any] = raw if isinstance(raw, dict) else {}
    meta_raw = payload.get("meta", {})
    meta_payload: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    source = str(meta_payload.get("source", "") or "")
    confidence = str(meta_payload.get("confidence", "") or "")
    reason_code = str(meta_payload.get("reason_code", "") or "")
    if not source or not confidence or not reason_code:
        return None
    return SpawnDecision(
        should_spawn=bool(payload.get("should_spawn", False)),
        label=str(payload.get("label", "") or ""),
        meta=SpawnDecisionMeta(
            source=source,  # type: ignore[arg-type]
            confidence=confidence,  # type: ignore[arg-type]
            reason_code=reason_code,  # type: ignore[arg-type]
        ),
    )
