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

_LONG_RUNNING_HINTS = (
    "长任务",
    "长时间",
    "后台",
    "异步",
    "持续",
    "遍历",
    "扫描",
    "全仓",
    "全项目",
    "批量",
    "递归",
    "crawl",
    "background",
)
_ISOLATION_HINTS = (
    "调研",
    "研究",
    "整理",
    "归纳",
    "总结",
    "分析",
    "审查",
    "review",
    "research",
    "summarize",
)
_TOOL_CHAIN_HINTS = (
    "搜索",
    "网页",
    "web",
    "search",
    "fetch",
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "shell",
    "命令",
    "终端",
    "文件",
    "rss",
    "feed",
)


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
    """Heuristic-only spawn policy used for traceability, not automatic routing."""

    def decide(self, *, task: str, label: str | None = None) -> SpawnDecision:
        normalized_task = (task or "").strip()
        normalized_label = (label or normalized_task[:30] or "").strip()
        task_lower = normalized_task.lower()

        if not normalized_task:
            return SpawnDecision(
                should_spawn=False,
                label=normalized_label,
                meta=SpawnDecisionMeta(
                    source="heuristic",
                    confidence="low",
                    reason_code="fallback_inline",
                ),
            )

        if any(hint.lower() in task_lower for hint in _LONG_RUNNING_HINTS):
            return SpawnDecision(
                should_spawn=True,
                label=normalized_label,
                meta=SpawnDecisionMeta(
                    source="heuristic",
                    confidence="high",
                    reason_code="long_running",
                ),
            )

        tool_chain_hits = sum(
            1 for hint in _TOOL_CHAIN_HINTS if hint.lower() in task_lower
        )
        if tool_chain_hits >= 2:
            return SpawnDecision(
                should_spawn=True,
                label=normalized_label,
                meta=SpawnDecisionMeta(
                    source="heuristic",
                    confidence="medium",
                    reason_code="tool_chain_heavy",
                ),
            )

        if len(normalized_task) >= 120 or any(
            hint.lower() in task_lower for hint in _ISOLATION_HINTS
        ):
            return SpawnDecision(
                should_spawn=True,
                label=normalized_label,
                meta=SpawnDecisionMeta(
                    source="heuristic",
                    confidence="medium",
                    reason_code="context_isolation_needed",
                ),
            )

        return SpawnDecision(
            should_spawn=False,
            label=normalized_label,
            meta=SpawnDecisionMeta(
                source="heuristic",
                confidence="medium",
                reason_code="stay_inline",
            ),
        )
