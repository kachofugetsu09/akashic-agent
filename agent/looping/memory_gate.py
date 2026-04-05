from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agent.looping.constants import _RETRIEVE_TRACE_SUMMARY_MAX
from agent.policies.history_route import HistoryRoutePolicy, RouteDecision
from core.common.strategy_trace import build_strategy_trace_envelope

if TYPE_CHECKING:
    from agent.provider import LLMProvider

logger = logging.getLogger("agent.loop")


# ── Module-level functions (moved from AgentLoopMemoryGateMixin in Phase 2) ──


def _trace_memory_retrieve(
    workspace: Path,
    *,
    session_key: str,
    channel: str,
    chat_id: str,
    user_msg: str,
    items: list[dict],
    injected_block: str,
    gate_type: str = "history_route",
    route_decision: str = "RETRIEVE",
    rewritten_query: str = "",
    fallback_reason: str = "",
    procedure_guard_applied: bool = False,
    procedure_hits: int = 0,
    history_hits: int = 0,
    injected_item_ids: list[str] | None = None,
    gate_latency_ms: dict | None = None,
    sufficiency_check: dict | None = None,
    error: str = "",
) -> None:
    try:
        memory_dir = workspace / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        trace_file = memory_dir / "memory2_retrieve_trace.jsonl"
        payload = {
            "session_key": session_key,
            "channel": channel,
            "chat_id": chat_id,
            "user_msg": user_msg,
            "hit_count": len(items),
            "procedure_hits": procedure_hits,
            "history_hits": history_hits,
            "injected_chars": len(injected_block or ""),
            "gate_type": gate_type,
            "route_decision": route_decision,
            "rewritten_query": rewritten_query,
            "fallback_reason": fallback_reason,
            "procedure_guard_applied": procedure_guard_applied,
            "injected_item_ids": injected_item_ids or [],
            "gate_latency_ms": gate_latency_ms or {},
            "sufficiency_check": sufficiency_check or {},
            "error": error,
            "top_items": [
                {
                    "id": item.get("id", ""),
                    "memory_type": item.get("memory_type", ""),
                    "score": round(float(item.get("score", 0.0)), 4),
                    "summary": (item.get("summary", "") or "")[
                        :_RETRIEVE_TRACE_SUMMARY_MAX
                    ],
                }
                for item in items
            ],
        }
        line = {
            **build_strategy_trace_envelope(
                trace_type="route",
                source="agent.memory_route",
                subject_kind="session",
                subject_id=session_key,
                payload=payload,
                timestamp=datetime.now().astimezone().isoformat(),
            ),
            **payload,
        }
        with trace_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("memory2 retrieve trace write failed: %s", e)


# Canonical implementations moved to agent/core/context_store.py.
# Lazy-import wrappers here so existing callers continue to work without
# creating a circular import at module load time.


def _extract_task_tools(tools_used: list[str]) -> list[str]:
    from agent.core.context_store import _extract_task_tools as _impl
    return _impl(tools_used)


def _update_session_runtime_metadata(
    session: object,
    *,
    tools_used: list[str],
    tool_chain: list[dict],
) -> None:
    from agent.core.context_store import _update_session_runtime_metadata as _impl
    return _impl(session, tools_used=tools_used, tool_chain=tool_chain)


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
