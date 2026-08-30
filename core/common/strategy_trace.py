from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

StrategyTraceType = Literal[
    "route", "proactive_stage", "proactive_config", "proactive_rate", "spawn"
]
StrategyTraceSubjectKind = Literal["session", "job", "action", "global"]


def build_strategy_trace_envelope(
    *,
    trace_type: StrategyTraceType,
    source: str,
    subject_kind: StrategyTraceSubjectKind,
    subject_id: str,
    payload: dict[str, Any],
    timestamp: str | None = None,
) -> dict[str, Any]:
    return {
        "trace_type": trace_type,
        "source": source,
        "subject": {"kind": subject_kind, "id": subject_id},
        "ts": timestamp or datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
