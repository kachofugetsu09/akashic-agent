from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SpawnDecisionSource = Literal["heuristic", "llm", "manual_rule"]
SpawnDecisionConfidence = Literal["high", "medium", "low"]
SpawnDecisionReasonCode = Literal[
    "long_running",
    "context_isolation_needed",
    "tool_chain_heavy",
    "stay_inline",
    "fallback_inline",
]


@dataclass(frozen=True)
class SpawnDecisionMeta:
    source: SpawnDecisionSource
    confidence: SpawnDecisionConfidence
    reason_code: SpawnDecisionReasonCode


@dataclass(frozen=True)
class SpawnDecision:
    should_spawn: bool
    label: str
    meta: SpawnDecisionMeta


class DelegationPolicy:
    """Traceability-only policy: the LLM decides when to spawn, this just records metadata."""

    def decide(self, *, task: str, label: str | None = None) -> SpawnDecision:
        normalized_label = (label or (task or "")[:30] or "").strip()
        return SpawnDecision(
            should_spawn=True,
            label=normalized_label,
            meta=SpawnDecisionMeta(
                source="llm",
                confidence="high",
                reason_code="tool_chain_heavy",
            ),
        )
