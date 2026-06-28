from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _empty_slots() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class ProactiveTickInput:
    session_key: str
    started_at: datetime


@dataclass
class ProactiveTickResult:
    base_score: float | None = None


@dataclass
class ProactiveFrame:
    input: ProactiveTickInput
    slots: dict[str, Any] = field(default_factory=_empty_slots)
    output: ProactiveTickResult | None = None


def new_proactive_frame(session_key: str) -> ProactiveFrame:
    return ProactiveFrame(
        input=ProactiveTickInput(
            session_key=session_key,
            started_at=datetime.now(UTC),
        )
    )
