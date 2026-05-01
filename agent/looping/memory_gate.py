from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agent.policies.history_route import HistoryRoutePolicy, RouteDecision
from agent.prompting import is_context_frame
from core.common.strategy_trace import build_strategy_trace_envelope

if TYPE_CHECKING:
    from agent.provider import LLMProvider

logger = logging.getLogger("agent.loop")


# ── Module-level functions (moved from AgentLoopMemoryGateMixin in Phase 2) ──


def _is_flow_execution_state(user_msg: str, metadata: dict) -> bool:
    return HistoryRoutePolicy.is_flow_execution_state(user_msg, metadata)


def _format_gate_history(
    history: list[dict],
    max_turns: int = 3,
    max_content_len: int | None = 100,
) -> str:
    turns = []
    for msg in reversed(history):
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        content = str(content).strip()
        if is_context_frame(content):
            continue
        if max_content_len is not None:
            content = content[:max_content_len]
        if content:
            turns.append(f"[{role}] {content}")
        if len(turns) >= max_turns * 2:
            break
    return "\n".join(reversed(turns))


def _trace_route_reason(decision: RouteDecision) -> str:
    reason_code = decision.meta.reason_code
    if reason_code == "route_disabled":
        return "disabled"
    if reason_code == "flow_execution_state":
        return "flow_execution_state"
    if reason_code == "llm_exception_fail_open":
        return "route_gate_exception"
    return "ok"


async def _decide_history_route(
    *,
    user_msg: str,
    metadata: dict,
    recent_history: str = "",
    light_provider: LLMProvider,
    light_model: str,
    route_intention_enabled: bool,
    gate_llm_timeout_ms: int,
    gate_max_tokens: int,
) -> RouteDecision:
    policy = HistoryRoutePolicy(
        light_provider=light_provider,
        light_model=light_model,
        enabled=route_intention_enabled,
        llm_timeout_ms=gate_llm_timeout_ms,
        max_tokens=gate_max_tokens,
    )
    return await policy.decide(
        user_msg=user_msg,
        metadata=metadata,
        recent_history=recent_history,
    )
