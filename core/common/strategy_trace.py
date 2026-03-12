from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal

StrategyTraceType = Literal[
    "route", "proactive_stage", "proactive_config", "proactive_rate", "spawn"
]
StrategyTraceSubjectKind = Literal["session", "job", "action", "global"]


@dataclass(frozen=True)
class StrategyTraceSubject:
    kind: StrategyTraceSubjectKind
    id: str


@dataclass(frozen=True)
class StrategyTraceEnvelope:
    trace_type: StrategyTraceType
    source: str
    subject: StrategyTraceSubject
    timestamp: str
    payload: dict[str, Any]


def build_strategy_trace_envelope(
    *,
    trace_type: StrategyTraceType,
    source: str,
    subject_kind: StrategyTraceSubjectKind,
    subject_id: str,
    payload: dict[str, Any],
    timestamp: str | None = None,
) -> dict[str, Any]:
    envelope = StrategyTraceEnvelope(
        trace_type=trace_type,
        source=source,
        subject=StrategyTraceSubject(kind=subject_kind, id=subject_id),
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        payload=payload,
    )
    return {
        "trace_type": envelope.trace_type,
        "source": envelope.source,
        "subject": asdict(envelope.subject),
        "ts": envelope.timestamp,
        "payload": envelope.payload,
    }
