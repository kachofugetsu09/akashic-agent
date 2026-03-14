import json
import logging
from datetime import datetime

from agent.looping.constants import _RETRIEVE_TRACE_SUMMARY_MAX
from agent.policies.history_route import HistoryRoutePolicy, RouteDecision
from core.common.strategy_trace import build_strategy_trace_envelope

logger = logging.getLogger("agent.loop")


class AgentLoopMemoryGateMixin:
    def _trace_memory_retrieve(
        self,
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
        sop_guard_applied: bool = False,
        procedure_hits: int = 0,
        history_hits: int = 0,
        injected_item_ids: list[str] | None = None,
        gate_latency_ms: dict | None = None,
        error: str = "",
    ) -> None:
        try:
            memory_dir = self.workspace / "memory"
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
                "sop_guard_applied": sop_guard_applied,
                "injected_item_ids": injected_item_ids or [],
                "gate_latency_ms": gate_latency_ms or {},
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

    @staticmethod
    def _extract_task_tools(tools_used: list[str]) -> list[str]:
        task_tools = []
        for name in tools_used:
            if name.startswith("skill_action_") or name in {"task_note", "update_now"}:
                task_tools.append(name)
        return task_tools

    def _update_session_runtime_metadata(
        self,
        session,
        *,
        tools_used: list[str],
        tool_chain: list[dict],
    ) -> None:
        md = session.metadata if isinstance(session.metadata, dict) else {}
        call_count = 0
        for group in tool_chain:
            if not isinstance(group, dict):
                continue
            calls = group.get("calls") or []
            if isinstance(calls, list):
                call_count += len(calls)

        turn_task_tools = self._extract_task_tools(tools_used)
        turns = md.get("_task_tools_turns")
        if not isinstance(turns, list):
            turns = []
        turns.append(turn_task_tools)
        turns = turns[-2:]

        flat_recent = []
        seen = set()
        for turn in turns:
            if not isinstance(turn, list):
                continue
            for name in turn:
                if isinstance(name, str) and name not in seen:
                    seen.add(name)
                    flat_recent.append(name)

        md["last_turn_tool_calls_count"] = call_count
        md["recent_task_tools"] = flat_recent
        md["last_turn_had_task_tool"] = bool(turn_task_tools)
        md["last_turn_ts"] = datetime.now().astimezone().isoformat()
        md["_task_tools_turns"] = turns
        session.metadata = md

    @staticmethod
    def _is_flow_execution_state(user_msg: str, metadata: dict[str, object]) -> bool:
        return HistoryRoutePolicy.is_flow_execution_state(user_msg, metadata)

    @staticmethod
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

    async def _decide_history_route(
        self,
        *,
        user_msg: str,
        metadata: dict[str, object],
        recent_history: str = "",
    ) -> RouteDecision:
        policy = HistoryRoutePolicy(
            light_provider=self.light_provider,
            light_model=self.light_model,
            enabled=self._memory_route_intention_enabled,
            llm_timeout_ms=self._memory_gate_llm_timeout_ms,
            max_tokens=self._memory_gate_max_tokens,
        )
        return await policy.decide(
            user_msg=user_msg,
            metadata=metadata,
            recent_history=recent_history,
        )

    @staticmethod
    def _trace_route_reason(decision: RouteDecision) -> str:
        reason_code = decision.meta.reason_code
        if reason_code == "route_disabled":
            return "disabled"
        if reason_code == "flow_execution_state":
            return "flow_execution_state"
        if reason_code == "llm_exception_fail_open":
            return "route_gate_exception"
        return "ok"
