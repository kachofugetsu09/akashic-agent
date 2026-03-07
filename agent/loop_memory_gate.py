import asyncio
import json
import logging
from datetime import datetime

import json_repair

from agent.loop_constants import (
    _FLOW_SEQUENCE_PATTERN,
    _FLOW_TRIGGER_WORDS,
    _RETRIEVE_TRACE_SUMMARY_MAX,
)

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
                "ts": datetime.now().astimezone().isoformat(),
                "session_key": session_key,
                "channel": channel,
                "chat_id": chat_id,
                "user_msg": user_msg,
                "hit_count": len(items),
                "procedure_hits": procedure_hits,
                "history_hits": history_hits,
                "injected_chars": len(injected_block or ""),
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
            with trace_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
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
        text = user_msg or ""
        if any(w in text for w in _FLOW_TRIGGER_WORDS):
            return True
        if _FLOW_SEQUENCE_PATTERN.search(text):
            return True
        if bool(metadata.get("last_turn_had_task_tool", False)):
            return True
        recent_task_tools = metadata.get("recent_task_tools")
        if isinstance(recent_task_tools, list) and any(
            isinstance(t, str) and t for t in recent_task_tools
        ):
            return True
        return False

    @staticmethod
    def _format_gate_history(history: list[dict], max_turns: int = 3) -> str:
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
            content = str(content).strip()[:100]
            if content:
                turns.append(f"[{role}] {content}")
            if len(turns) >= max_turns * 2:
                break
        return "\n".join(reversed(turns))

    async def _decide_history_retrieval(
        self,
        *,
        user_msg: str,
        metadata: dict[str, object],
        recent_history: str = "",
    ) -> tuple[bool, str, str, int]:
        start = datetime.now()

        if not self._memory_route_intention_enabled:
            latency = int((datetime.now() - start).total_seconds() * 1000)
            return True, user_msg, "disabled", latency

        if self._is_flow_execution_state(user_msg, metadata):
            latency = int((datetime.now() - start).total_seconds() * 1000)
            return True, user_msg, "flow_execution_state", latency

        history_section = f"\n近期对话摘要：\n{recent_history}\n" if recent_history else ""
        prompt = f"""判断当前用户消息是否需要检索历史事件记忆。
{history_section}
当前消息：{user_msg}

规则：
- 闲聊、通识问答、无需历史上下文 -> NO_RETRIEVE
- 涉及历史偏好、过往对话、用户特征 -> RETRIEVE

若 RETRIEVE：rewritten_query 只保留检索主题关键词（如"仁王 游戏偏好"），
去掉"我之前/之前说过/聊过"等 meta 表述，方便向量检索命中记忆。
若 NO_RETRIEVE：rewritten_query 返回原文不变。

只返回 JSON：{{"decision":"RETRIEVE|NO_RETRIEVE","rewritten_query":"...","confidence":"high|medium|low"}}"""

        try:
            timeout_s = max(0.1, self._memory_gate_llm_timeout_ms / 1000.0)
            resp = await asyncio.wait_for(
                self.light_provider.chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[],
                    model=self.light_model,
                    max_tokens=self._memory_gate_max_tokens,
                ),
                timeout=timeout_s,
            )
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            payload = json_repair.loads(text)
            decision = (
                str(payload.get("decision", "")).upper()
                if isinstance(payload, dict)
                else ""
            )
            rewritten = (
                str(payload.get("rewritten_query", "")).strip()
                if isinstance(payload, dict)
                else ""
            )
            confidence = (
                str(payload.get("confidence", "medium")).lower()
                if isinstance(payload, dict)
                else "low"
            )
            if confidence not in {"high", "medium", "low"}:
                confidence = "low"
            if confidence == "low":
                decision = "RETRIEVE"
            needs_history = decision != "NO_RETRIEVE"
            latency = int((datetime.now() - start).total_seconds() * 1000)
            return needs_history, (rewritten or user_msg), "ok", latency
        except Exception:
            latency = int((datetime.now() - start).total_seconds() * 1000)
            return True, user_msg, "route_gate_exception", latency
