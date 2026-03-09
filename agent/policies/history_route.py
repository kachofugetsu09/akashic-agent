from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from agent.llm_json import load_json_object_loose
from agent.looping.constants import _FLOW_SEQUENCE_PATTERN, _FLOW_TRIGGER_WORDS

RouteDecisionSource = Literal["heuristic", "llm", "fallback"]
RouteDecisionConfidence = Literal["high", "medium", "low"]
RouteDecisionReasonCode = Literal[
    "route_disabled",
    "flow_execution_state",
    "llm_no_retrieve",
    "llm_retrieve",
    "llm_low_confidence_fail_open",
    "llm_exception_fail_open",
    "empty_query_fallback",
]


@dataclass(frozen=True)
class DecisionMeta:
    source: RouteDecisionSource
    confidence: RouteDecisionConfidence
    reason_code: RouteDecisionReasonCode


@dataclass(frozen=True)
class RouteDecision:
    needs_history: bool
    rewritten_query: str
    fail_open: bool
    latency_ms: int
    meta: DecisionMeta


class HistoryRoutePolicy:
    def __init__(
        self,
        *,
        light_provider: Any,
        light_model: str,
        enabled: bool,
        llm_timeout_ms: int,
        max_tokens: int,
    ) -> None:
        self._light_provider = light_provider
        self._light_model = light_model
        self._enabled = bool(enabled)
        self._llm_timeout_ms = int(llm_timeout_ms)
        self._max_tokens = int(max_tokens)

    @staticmethod
    def is_flow_execution_state(user_msg: str, metadata: dict[str, object]) -> bool:
        text = user_msg or ""
        if any(word in text for word in _FLOW_TRIGGER_WORDS):
            return True
        if _FLOW_SEQUENCE_PATTERN.search(text):
            return True
        if bool(metadata.get("last_turn_had_task_tool", False)):
            return True
        recent_task_tools = metadata.get("recent_task_tools")
        if isinstance(recent_task_tools, list) and any(
            isinstance(tool, str) and tool for tool in recent_task_tools
        ):
            return True
        return False

    async def decide(
        self,
        *,
        user_msg: str,
        metadata: dict[str, object],
        recent_history: str = "",
    ) -> RouteDecision:
        start = datetime.now()
        default_query = user_msg

        if not self._enabled:
            return self._build_decision(
                needs_history=True,
                rewritten_query=default_query,
                fail_open=False,
                meta=DecisionMeta(
                    source="heuristic",
                    confidence="high",
                    reason_code="route_disabled",
                ),
                start=start,
            )

        if self.is_flow_execution_state(user_msg, metadata):
            return self._build_decision(
                needs_history=True,
                rewritten_query=default_query,
                fail_open=False,
                meta=DecisionMeta(
                    source="heuristic",
                    confidence="high",
                    reason_code="flow_execution_state",
                ),
                start=start,
            )

        prompt = self._build_prompt(user_msg=user_msg, recent_history=recent_history)
        try:
            timeout_s = max(0.1, self._llm_timeout_ms / 1000.0)
            resp = await asyncio.wait_for(
                self._light_provider.chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[],
                    model=self._light_model,
                    max_tokens=self._max_tokens,
                ),
                timeout=timeout_s,
            )
            payload = load_json_object_loose((resp.content or "").strip())
        except Exception:
            return self._build_decision(
                needs_history=True,
                rewritten_query=default_query,
                fail_open=True,
                meta=DecisionMeta(
                    source="fallback",
                    confidence="low",
                    reason_code="llm_exception_fail_open",
                ),
                start=start,
            )

        decision = (
            str(payload.get("decision", "")).upper() if payload is not None else ""
        )
        rewritten = (
            str(payload.get("rewritten_query", "")).strip()
            if payload is not None
            else ""
        )
        confidence = self._normalize_confidence(
            str(payload.get("confidence", "medium")).lower()
            if payload is not None
            else "low"
        )

        if confidence == "low":
            return self._build_decision(
                needs_history=True,
                rewritten_query=rewritten or default_query,
                fail_open=True,
                meta=DecisionMeta(
                    source="llm",
                    confidence=confidence,
                    reason_code="llm_low_confidence_fail_open",
                ),
                start=start,
            )

        needs_history = decision != "NO_RETRIEVE"
        if rewritten:
            reason_code: RouteDecisionReasonCode = (
                "llm_retrieve" if needs_history else "llm_no_retrieve"
            )
            return self._build_decision(
                needs_history=needs_history,
                rewritten_query=rewritten,
                fail_open=False,
                meta=DecisionMeta(
                    source="llm",
                    confidence=confidence,
                    reason_code=reason_code,
                ),
                start=start,
            )

        return self._build_decision(
            needs_history=needs_history,
            rewritten_query=default_query,
            fail_open=False,
            meta=DecisionMeta(
                source="llm",
                confidence=confidence,
                reason_code="empty_query_fallback",
            ),
            start=start,
        )

    @staticmethod
    def _normalize_confidence(value: str) -> RouteDecisionConfidence:
        if value in {"high", "medium", "low"}:
            return value
        return "low"

    @staticmethod
    def _build_prompt(*, user_msg: str, recent_history: str) -> str:
        history_section = f"\n近期对话摘要：\n{recent_history}\n" if recent_history else ""
        return f"""判断当前用户消息是否需要检索历史事件记忆。
{history_section}
当前消息：{user_msg}

规则：
- 闲聊、通识问答、无需历史上下文 -> NO_RETRIEVE
- 涉及历史偏好、过往对话、用户特征 -> RETRIEVE

若 RETRIEVE：rewritten_query 只保留检索主题关键词（如"仁王 游戏偏好"），
去掉"我之前/之前说过/聊过"等 meta 表述，方便向量检索命中记忆。
若 NO_RETRIEVE：rewritten_query 返回原文不变。

只返回 JSON：{{"decision":"RETRIEVE|NO_RETRIEVE","rewritten_query":"...","confidence":"high|medium|low"}}"""

    @staticmethod
    def _build_decision(
        *,
        needs_history: bool,
        rewritten_query: str,
        fail_open: bool,
        meta: DecisionMeta,
        start: datetime,
    ) -> RouteDecision:
        latency = int((datetime.now() - start).total_seconds() * 1000)
        return RouteDecision(
            needs_history=needs_history,
            rewritten_query=rewritten_query,
            fail_open=fail_open,
            latency_ms=latency,
            meta=meta,
        )
