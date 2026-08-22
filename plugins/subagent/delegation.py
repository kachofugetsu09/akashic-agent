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

_MAX_CONCURRENT_SPAWNS = 3


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
    block_reason: str = ""


class DelegationPolicy:
    """限制并发委派数量，其余决策交由模型指引。"""

    def decide(
        self, *, task: str, label: str | None = None, running_count: int = 0
    ) -> SpawnDecision:
        normalized_label = (label or (task or "")[:30] or "").strip()
        if running_count >= _MAX_CONCURRENT_SPAWNS:
            return SpawnDecision(
                should_spawn=False,
                label=normalized_label,
                block_reason=(
                    f"已有 {running_count} 个并发子任务在运行，上限 {_MAX_CONCURRENT_SPAWNS}，"
                    "请等待当前任务完成后再试"
                ),
                meta=SpawnDecisionMeta(
                    source="heuristic",
                    confidence="high",
                    reason_code="stay_inline",
                ),
            )
        return SpawnDecision(
            should_spawn=True,
            label=normalized_label,
            meta=SpawnDecisionMeta(
                source="llm",
                confidence="high",
                reason_code="tool_chain_heavy",
            ),
        )
