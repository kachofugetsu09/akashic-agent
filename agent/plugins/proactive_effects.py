from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


def _empty_metadata() -> dict[str, object]:
    return {}


@dataclass(frozen=True)
class ProactiveEffectContext:
    session_key: str
    tick_id: str
    now_utc: datetime
    base_judge_send_threshold: float
    last_user_at: datetime | None


@dataclass(frozen=True)
class ProactiveEffect:
    provider_name: str
    prompt_section: str = ""
    threshold_delta: float = 0.0
    metadata: dict[str, object] = field(default_factory=_empty_metadata)


class ProactiveEffectProvider(Protocol):
    def build_proactive_effect(
        self,
        ctx: ProactiveEffectContext,
    ) -> ProactiveEffect | None: ...
