from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

_WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

from bus.events import InboundMessage, OutboundMessage
from bus.internal_events import parse_spawn_completion
from memory2.injection_planner import (
    retrieve_episodic,
    retrieve_procedure_items,
)
from memory2.query_rewriter import GateDecision

if TYPE_CHECKING:
    from agent.looping.core import AgentLoop
    from core.observe.events import RagItemTrace, RagTrace, TurnTrace

logger = logging.getLogger("agent.loop_handlers")


class ConversationTurnHandler:
    def __init__(self, loop: "AgentLoop") -> None:
        self._loop = loop

    async def process(self, msg: InboundMessage, key: str) -> OutboundMessage:
        """处理一次普通用户消息，把“检索 -> 执行 -> 持久化 -> 回包”串成主路径。"""
        loop = self._loop
        session = loop.session_manager.get_or_create(key)
        # 1. 先解析 skill 提及和主会话历史，准备这轮上下文。
        skill_mentions = self._collect_skill_mentions(msg)
        main_history = session.get_history(max_messages=loop.memory_window)
        # 2. 再独立跑 memory 注入，尽量让主执行链只拿最终 block。
        retrieved_block, rag_trace = await self._retrieve_memory_block(
            msg=msg,
            key=key,
            session=session,
            main_history=main_history,
        )
        # 3. 用会话历史、skill 和 memory block 执行真正的 conversation turn。
        final_content, tools_used, tool_chain, thinking = await self._run_conversation_turn(
            msg=msg,
            session=session,
            skill_mentions=skill_mentions,
            main_history=main_history,
            retrieved_block=retrieved_block,
        )
        # 4. 最后统一做 session append / memory worker / outbound 组装。
        await self._persist_turn(
            msg=msg,
            key=key,
            session=session,
            final_content=final_content,
            tools_used=tools_used,
            tool_chain=tool_chain,
            thinking=thinking,
        )
        # 5. 写入 observe trace（非阻塞）。
        self._emit_observe_traces(
            loop=loop,
            key=key,
            msg=msg,
            final_content=final_content,
            tool_chain=tool_chain,
            rag_trace=rag_trace,
        )
        return self._build_outbound_message(
            msg=msg,
            final_content=final_content,
            tools_used=tools_used,
            tool_chain=tool_chain,
            thinking=thinking,
        )

    def _collect_skill_mentions(self, msg: InboundMessage) -> list[str]:
        loop = self._loop
        skill_mentions = loop._collect_skill_mentions(msg.content)
        if skill_mentions:
            logger.info(f"检测到 $skill 提及，直接注入完整内容: {skill_mentions}")
        return skill_mentions

    async def _retrieve_memory_block(
        self,
        *,
        msg: InboundMessage,
        key: str,
        session,
        main_history: list[dict],
    ) -> tuple[str, "RagTrace | None"]:
        """为主对话路径准备 memory block，并把 route / injection 细节写入 trace。"""
        loop = self._loop
        retrieved_block = ""
        rag_trace: RagTrace | None = None
        gate_type = "history_route"
        sufficiency_trace: dict[str, object] = self._empty_sufficiency_state()
        try:
            # 1. 先整理 gate 和 HyDE 需要的最近上下文，作为“要不要检索历史”的输入。
            recent_turns = loop._format_gate_history(main_history, max_turns=3)
            hyde_context = self._build_hyde_context(main_history)

            # 2. 再做 memory gate：
            #    - 一路取 procedure/preference 规则类记忆
            #    - 一路判断这轮是否需要检索 event/profile 历史类记忆
            gate_result, p_items = await self._resolve_memory_gate(
                msg=msg,
                session=session,
                recent_turns=recent_turns,
            )
            gate_type = str(gate_result["gate_type"])
            rewritten_query = str(gate_result["episodic_query"])
            route_decision = str(gate_result["route_decision"])
            route_ms = int(gate_result["route_latency_ms"])
            fallback_reason = str(gate_result["fallback_reason"])
            history_memory_types = list(gate_result["history_memory_types"])
            gate_latency_ms = {"route": route_ms}

            # 3. 若 gate 判定需要历史检索，就按改写后的 query 去取 episodic items；
            #    然后把规则类记忆和历史类记忆合并，生成最终注入主模型的 memory block。
            h_items, h_scope_mode, hyde_hypothesis = await self._retrieve_episodic_items(
                route_decision=route_decision,
                rewritten_query=rewritten_query,
                history_memory_types=history_memory_types,
                hyde_context=hyde_context,
            )
            selected_items, retrieved_block, injected_item_ids = self._build_injection_payload(
                procedure_items=p_items,
                history_items=h_items,
            )

            # 4. 若第一次历史召回结果不够支撑回答，再做一次 sufficiency retry；
            #    最后把 route / hits / injected ids 等细节写入 trace，返回 memory block 给主链路。
            h_items, h_scope_mode, selected_items, retrieved_block, injected_item_ids = (
                await self._retry_empty_episodic_block(
                    msg=msg,
                    recent_turns=recent_turns,
                    route_decision=route_decision,
                    rewritten_query=rewritten_query,
                    history_memory_types=history_memory_types,
                    procedure_items=p_items,
                    history_items=h_items,
                    history_scope_mode=h_scope_mode,
                    selected_items=selected_items,
                    retrieved_block=retrieved_block,
                    injected_item_ids=injected_item_ids,
                    sufficiency_trace=sufficiency_trace,
                )
            )
            rag_trace = self._finalize_memory_retrieval(
                key=key,
                msg=msg,
                gate_type=gate_type,
                route_decision=route_decision,
                rewritten_query=rewritten_query,
                route_ms=route_ms,
                fallback_reason=fallback_reason,
                gate_latency_ms=gate_latency_ms,
                p_items=p_items,
                h_items=h_items,
                h_scope_mode=h_scope_mode,
                hyde_hypothesis=hyde_hypothesis,
                selected_items=selected_items,
                retrieved_block=retrieved_block,
                injected_item_ids=injected_item_ids,
                sufficiency_trace=sufficiency_trace,
            )
        except Exception as e:
            logger.warning(f"memory2 retrieve 失败，跳过: {e}")
            loop._trace_memory_retrieve(
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                user_msg=msg.content,
                items=[],
                injected_block="",
                gate_type=gate_type,
                fallback_reason="retrieve_exception",
                sufficiency_check=sufficiency_trace,
                error=str(e),
            )
            rag_trace = _build_agent_rag_trace(
                session_key=key,
                user_msg=msg.content,
                rewritten_query=msg.content,
                gate_type=gate_type,
                route_decision=None,
                route_latency_ms=None,
                h_scope_mode=None,
                p_items=[],
                h_items=[],
                hyde_hypothesis=None,
                injected_id_set=set(),
                injected_block="",
                sufficiency_check=sufficiency_trace,
                fallback_reason="retrieve_exception",
                error=str(e),
            )
        return retrieved_block, rag_trace

    def _empty_sufficiency_state(self) -> dict[str, object]:
        return {
            "triggered": False,
            "result": "",
            "refined_query": "",
            "retry_count": 0,
        }

    def _build_hyde_context(self, main_history: list[dict]) -> str:
        now = datetime.now()
        date_str = now.strftime(f"%Y-%m-%d {_WEEKDAY_CN[now.weekday()]} %H:%M")
        hyde_turns = self._loop._format_gate_history(
            main_history,
            max_turns=3,
            max_content_len=None,
        )
        return f"当前时间：{date_str}\n{hyde_turns}" if hyde_turns else f"当前时间：{date_str}"

    async def _resolve_memory_gate(
        self,
        *,
        msg: InboundMessage,
        session,
        recent_turns: str,
    ) -> tuple[dict[str, object], list[dict]]:
        loop = self._loop
        if loop._query_rewriter is not None:
            decision: GateDecision = await loop._query_rewriter.decide(
                user_msg=msg.content,
                recent_history=recent_turns,
            )
            p_items = await retrieve_procedure_items(
                loop._memory_port,
                query=msg.content,
                top_k=loop._memory_top_k_procedure,
            )
            return {
                "gate_type": "query_rewriter",
                "episodic_query": decision.episodic_query,
                "route_decision": "RETRIEVE" if decision.needs_episodic else "NO_RETRIEVE",
                "route_latency_ms": decision.latency_ms,
                "fallback_reason": "",
                "history_memory_types": ["event", "profile"],
            }, p_items
        return await self._resolve_fallback_memory_gate(
            msg=msg,
            session=session,
            recent_turns=recent_turns,
        )

    async def _resolve_fallback_memory_gate(
        self,
        *,
        msg: InboundMessage,
        session,
        recent_turns: str,
    ) -> tuple[dict[str, object], list[dict]]:
        loop = self._loop
        runtime_md = session.metadata if isinstance(session.metadata, dict) else {}
        p_task = asyncio.create_task(
            retrieve_procedure_items(
                loop._memory_port,
                query=msg.content,
                top_k=loop._memory_top_k_procedure,
            )
        )
        route_task = asyncio.create_task(
            loop._decide_history_route(
                user_msg=msg.content,
                metadata=runtime_md,
                recent_history=recent_turns,
            )
        )
        p_items, route_decision_obj = await asyncio.gather(p_task, route_task)
        route_reason = loop._trace_route_reason(route_decision_obj)
        return {
            "gate_type": "history_route",
            "episodic_query": route_decision_obj.rewritten_query,
            "route_decision": "RETRIEVE" if route_decision_obj.needs_history else "NO_RETRIEVE",
            "route_latency_ms": route_decision_obj.latency_ms,
            "fallback_reason": "" if route_reason == "ok" else route_reason,
            "history_memory_types": ["event", "profile"],
        }, p_items

    async def _retrieve_episodic_items(
        self,
        *,
        route_decision: str,
        rewritten_query: str,
        history_memory_types: list[str],
        hyde_context: str,
    ) -> tuple[list[dict], str, str | None]:
        if route_decision != "RETRIEVE" or not history_memory_types:
            return [], "disabled", None
        return await retrieve_episodic(
            self._loop._memory_port,
            rewritten_query,
            memory_types=history_memory_types,
            top_k=self._loop._memory_top_k_history,
            context=hyde_context,
            hyde_enhancer=self._loop._hyde_enhancer,
        )

    def _build_injection_payload(
        self,
        *,
        procedure_items: list[dict],
        history_items: list[dict],
    ) -> tuple[list[dict], str, list[str]]:
        memory = self._loop._memory_port
        merged = self._merge_memory_items(procedure_items + history_items)
        selected_items = memory.select_for_injection(merged)
        block, item_ids = memory.build_injection_block(merged)
        return selected_items, block, item_ids

    async def _retry_empty_episodic_block(
        self,
        *,
        msg: InboundMessage,
        recent_turns: str,
        route_decision: str,
        rewritten_query: str,
        history_memory_types: list[str],
        procedure_items: list[dict],
        history_items: list[dict],
        history_scope_mode: str,
        selected_items: list[dict],
        retrieved_block: str,
        injected_item_ids: list[str],
        sufficiency_trace: dict[str, object],
    ) -> tuple[list[dict], str, list[dict], str, list[str]]:
        checker = getattr(self._loop, "_sufficiency_checker", None)
        if route_decision != "RETRIEVE" or checker is None or retrieved_block:
            return (
                history_items,
                history_scope_mode,
                selected_items,
                retrieved_block,
                injected_item_ids,
            )
        sufficiency_trace["triggered"] = True
        result = await checker.check(
            query=rewritten_query or msg.content,
            items=selected_items,
            context=recent_turns,
        )
        sufficiency_trace["result"] = result.reason
        sufficiency_trace["refined_query"] = result.refined_query or ""
        if result.is_sufficient or not result.refined_query or not history_memory_types:
            if not history_memory_types and not result.is_sufficient and result.refined_query:
                logger.debug("sufficiency check: no history_memory_types, skip retry")
            return (
                history_items,
                history_scope_mode,
                selected_items,
                retrieved_block,
                injected_item_ids,
            )
        extra_h_items, extra_scope_mode, _retry_hypothesis = await retrieve_episodic(
            self._loop._memory_port,
            result.refined_query,
            memory_types=history_memory_types,
            top_k=self._loop._memory_top_k_history,
            context=recent_turns,
            hyde_enhancer=self._loop._hyde_enhancer,
        )
        sufficiency_trace["retry_count"] = 1
        history_items = history_items + extra_h_items
        history_scope_mode = extra_scope_mode or history_scope_mode
        selected_items, retrieved_block, injected_item_ids = self._build_injection_payload(
            procedure_items=procedure_items,
            history_items=history_items,
        )
        return history_items, history_scope_mode, selected_items, retrieved_block, injected_item_ids

    def _merge_memory_items(self, items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        merged: list[dict] = []
        for item in items:
            item_id = str(item.get("id", "") or "")
            if item_id and item_id in seen:
                continue
            if item_id:
                seen.add(item_id)
            merged.append(item)
        return merged

    def _finalize_memory_retrieval(
        self,
        *,
        key: str,
        msg: InboundMessage,
        gate_type: str,
        route_decision: str,
        rewritten_query: str,
        route_ms: int,
        fallback_reason: str,
        gate_latency_ms: dict[str, int],
        p_items: list[dict],
        h_items: list[dict],
        h_scope_mode: str,
        hyde_hypothesis: str | None,
        selected_items: list[dict],
        retrieved_block: str,
        injected_item_ids: list[str],
        sufficiency_trace: dict[str, object],
    ) -> "RagTrace":
        # 1. 先打本轮命中日志，便于排查注入结果。
        logger.info(
            "memory2 retrieve: route=%s scope=%s query=%r p=%d h=%d 命中，选出 %d 条，注入 %d 条%s",
            route_decision,
            h_scope_mode,
            rewritten_query[:50],
            len(p_items),
            len(h_items),
            len(selected_items),
            len(injected_item_ids),
            "" if retrieved_block else "（无内容注入）",
        )
        self._log_memory_injection(selected_items)
        # 2. 再把检索细节写入 jsonl trace。
        sop_guard_applied = self._has_sop_guard_hit(p_items, injected_item_ids)
        self._loop._trace_memory_retrieve(
            session_key=key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            user_msg=msg.content,
            items=selected_items,
            injected_block=retrieved_block,
            gate_type=gate_type,
            route_decision=route_decision,
            rewritten_query=rewritten_query,
            fallback_reason=fallback_reason,
            sop_guard_applied=sop_guard_applied,
            procedure_hits=len(p_items),
            history_hits=len(h_items),
            injected_item_ids=injected_item_ids,
            gate_latency_ms=gate_latency_ms,
            sufficiency_check=sufficiency_trace,
        )
        # 3. 最后构造 observe 用的 RagTrace。
        return _build_agent_rag_trace(
            session_key=key,
            user_msg=msg.content,
            rewritten_query=rewritten_query,
            gate_type=gate_type,
            route_decision=route_decision,
            route_latency_ms=route_ms,
            h_scope_mode=h_scope_mode,
            p_items=p_items,
            h_items=h_items,
            hyde_hypothesis=hyde_hypothesis,
            injected_id_set=set(injected_item_ids),
            injected_block=retrieved_block,
            sufficiency_check=sufficiency_trace,
            fallback_reason=fallback_reason,
        )

    def _log_memory_injection(self, selected_items: list[dict]) -> None:
        if selected_items:
            injected_preview = " | ".join(
                f"{str(item.get('memory_type', ''))}:{str(item.get('summary', ''))[:40]}"
                for item in selected_items[:4]
                if isinstance(item, dict)
            )
            logger.info("memory2 injected_summary: %s", injected_preview)
        for item in selected_items:
            logger.debug(
                "memory2 injected: id=%s score=%.3f type=%s summary=%s",
                item.get("id", ""),
                float(item.get("score", 0.0)),
                item.get("memory_type", ""),
                str(item.get("summary", ""))[:60],
            )

    def _has_sop_guard_hit(
        self,
        procedure_items: list[dict],
        injected_item_ids: list[str],
    ) -> bool:
        protected_ids = {
            str(item.get("id", ""))
            for item in procedure_items
            if isinstance(item, dict)
            and item.get("memory_type") == "procedure"
            and (item.get("extra_json") or {}).get("tool_requirement")
            and item.get("id")
        }
        return bool(
            self._loop._memory_sop_guard_enabled
            and any(item_id in protected_ids for item_id in injected_item_ids)
        )

    async def _run_conversation_turn(
        self,
        *,
        msg: InboundMessage,
        session,
        skill_mentions: list[str],
        main_history: list[dict],
        retrieved_block: str,
    ) -> tuple[str, list[str], list[dict], str | None]:
        """真正执行一轮 agent 对话，并返回内容、工具使用、tool_chain 和 thinking。"""
        loop = self._loop
        # 1. 先把 channel/chat_id 写进 tool context，保证工具知道当前会话来源。
        loop._set_tool_context(msg.channel, msg.chat_id)
        # 2. 再统一走 safety retry 包装，避免模型输出异常直接打断主流程。
        final_content, tools_used, tool_chain, thinking = await loop._run_with_safety_retry(
            msg,
            session,
            skill_names=skill_mentions or None,
            base_history=main_history,
            retrieved_memory_block=retrieved_block,
        )

        # 3. 最后保证 content 至少有一个兜底字符串，避免 assistant 回复为空。
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        return final_content, tools_used, tool_chain, thinking

    async def _persist_turn(
        self,
        *,
        msg: InboundMessage,
        key: str,
        session,
        final_content: str,
        tools_used: list[str],
        tool_chain: list[dict],
        thinking: str | None = None,
    ) -> None:
        """把本轮结果统一写回 session、presence 和 post-response worker。"""
        loop = self._loop
        preview = (
            final_content[:120] + "..." if len(final_content) > 120 else final_content
        )
        logger.info(f"Response to {msg.channel}:{msg.sender}: {preview}")

        # 1. 先把 user/assistant 两条消息落到 session 内存对象中。
        # final_content 已由 provider 层剥离 thinking，直接存入 session。
        if loop._presence:
            loop._presence.record_user_message(key)
        session.add_message("user", msg.content, media=msg.media if msg.media else None)
        session.add_message(
            "assistant",
            final_content,
            tools_used=tools_used if tools_used else None,
            tool_chain=tool_chain if tool_chain else None,
        )
        # 2. 再补 runtime metadata，让后续 memory gate / proactive 能读到这轮工具状态。
        loop._update_session_runtime_metadata(
            session,
            tools_used=tools_used,
            tool_chain=tool_chain,
        )
        # 3. 最后做持久化 append，并异步调度 consolidation / post-response memory。
        await loop.session_manager.append_messages(session, session.messages[-2:])
        loop._schedule_consolidation_if_needed(session, key)
        loop._schedule_post_response_memory(
            msg=msg,
            key=key,
            final_content=final_content,
            tool_chain=tool_chain,
        )

    @staticmethod
    def _build_outbound_message(
        *,
        msg: InboundMessage,
        final_content: str,
        tools_used: list[str],
        tool_chain: list[dict],
        thinking: str | None = None,
    ) -> OutboundMessage:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            thinking=thinking,
            metadata={
                **(msg.metadata or {}),
                "tools_used": tools_used,
                "tool_chain": tool_chain,
            },
        )

    @staticmethod
    def _emit_observe_traces(
        *,
        loop: "AgentLoop",
        key: str,
        msg: InboundMessage,
        final_content: str,
        tool_chain: list[dict],
        rag_trace: "RagTrace | None",
    ) -> None:
        writer = getattr(loop, "_observe_writer", None)
        if writer is None:
            return
        import json as _json
        from core.observe.events import TurnTrace

        tool_calls = [
            {
                "name": call.get("name", ""),
                "args": str(call.get("arguments", ""))[:300],
                "result": str(call.get("result", ""))[:500],
            }
            for group in tool_chain
            for call in (group.get("calls") or [])
        ]
        # 完整迭代链路：包含每轮 LLM 推理文本 + 工具调用（args/result 分别限 800/1200 字）
        def _slim_chain(chain: list[dict]) -> list[dict]:
            out = []
            for group in chain:
                text = str(group.get("text") or "")
                calls = [
                    {
                        "name": c.get("name", ""),
                        "args": str(c.get("arguments", ""))[:800],
                        "result": str(c.get("result", ""))[:1200],
                    }
                    for c in (group.get("calls") or [])
                ]
                out.append({"text": text, "calls": calls})
            return out

        tool_chain_json = _json.dumps(_slim_chain(tool_chain), ensure_ascii=False) if tool_chain else None

        writer.emit(
            TurnTrace(
                source="agent",
                session_key=key,
                user_msg=msg.content,
                llm_output=final_content,
                tool_calls=tool_calls,
                tool_chain_json=tool_chain_json,
            )
        )
        if rag_trace is not None:
            writer.emit(rag_trace)


def _build_agent_rag_trace(
    *,
    session_key: str,
    user_msg: str,
    rewritten_query: str,
    gate_type: str | None,
    route_decision: str | None,
    route_latency_ms: int | None,
    h_scope_mode: str | None,
    p_items: list[dict],
    h_items: list[dict],
    hyde_hypothesis: str | None,
    injected_id_set: set[str],
    injected_block: str,
    sufficiency_check: dict[str, object],
    fallback_reason: str = "",
    error: str | None = None,
) -> "RagTrace":
    """把本次 agent memory 检索的原始数据组装成 RagTrace。"""
    from core.observe.events import RagItemTrace, RagTrace
    import json as _json

    def _item_to_trace(item: dict, path: str) -> "RagItemTrace":
        raw_extra = item.get("extra_json")
        extra_str = _json.dumps(raw_extra, ensure_ascii=False) if raw_extra else None
        return RagItemTrace(
            item_id=str(item.get("id", "")),
            memory_type=str(item.get("memory_type", "")),
            score=float(item.get("score", 0.0)),
            summary=str(item.get("summary", "")),
            happened_at=item.get("happened_at"),
            extra_json=extra_str,
            retrieval_path=path,
            injected=str(item.get("id", "")) in injected_id_set,
        )

    trace_items: list[RagItemTrace] = []

    # procedure items
    for item in p_items:
        trace_items.append(_item_to_trace(item, "procedure"))

    # history items：retrieve_episodic 已直接标好 raw / hyde 路径。
    for item in h_items:
        path = str(item.get("_retrieval_path", "history_raw") or "history_raw")
        trace_items.append(_item_to_trace(item, path))

    return RagTrace(
        source="agent",
        session_key=session_key,
        original_query=user_msg,
        query=rewritten_query,
        gate_type=gate_type,
        route_decision=route_decision,
        route_latency_ms=route_latency_ms,
        hyde_hypothesis=hyde_hypothesis,
        history_scope_mode=h_scope_mode,
        history_gate_reason=None,
        items=trace_items,
        injected_block=injected_block,
        sufficiency_check_json=json.dumps(sufficiency_check, ensure_ascii=False),
        fallback_reason=fallback_reason,
        error=error,
    )


class InternalEventHandler:
    def __init__(self, loop: "AgentLoop") -> None:
        self._loop = loop

    async def process_spawn_completion(
        self, msg: InboundMessage, key: str
    ) -> OutboundMessage:
        loop = self._loop
        session = loop.session_manager.get_or_create(key)
        event = parse_spawn_completion(msg)
        label = event.label or "后台任务"
        task = event.task.strip()
        status = (event.status or "incomplete").strip()
        result = event.result.strip()
        exit_reason = event.exit_reason.strip()

        if status == "completed":
            header = "[后台任务已完成]"
        elif status == "incomplete":
            header = "[后台任务未全部完成]"
        else:
            header = "[后台任务出错]"

        current_message = (
            f"{header}\n"
            f"任务标签: {label}\n"
            f"原始任务: {task or '（未提供）'}\n"
            f"状态: {status}\n"
            f"执行结果:\n{result or '（无结果）'}\n\n"
            "请基于当前会话上下文，用自然中文向用户汇报这次后台任务的结果。\n"
            "不要提及 subagent、spawn、内部事件、job_id。\n"
            "如果状态是 incomplete，明确告诉用户任务尚未完成，说明目前做到哪里、接下来还差什么。\n"
            "如果状态是 error，只说明用户需要知道的失败结论，不暴露内部技术细节。\n"
            "必要时你可以读取结果里提到的文件来补充说明。"
        )

        loop._set_tool_context(msg.channel, msg.chat_id)
        initial_messages = loop.context.build_messages(
            history=session.get_history(max_messages=loop.memory_window),
            current_message=current_message,
            channel=msg.channel,
            chat_id=msg.chat_id,
            message_timestamp=msg.timestamp,
        )
        final_content, tools_used, tool_chain, _, _thinking = await loop._run_agent_loop(
            initial_messages,
            request_time=msg.timestamp,
            preloaded_tools=None,
        )
        if final_content is None:
            if status == "completed":
                final_content = "后台任务已完成。"
            elif status == "incomplete":
                final_content = "后台任务未全部完成，部分工作尚未收尾。"
            else:
                final_content = "后台任务执行出错。"

        marker = f"[后台任务完成] {label} ({status})"
        if exit_reason:
            marker += f" [{exit_reason}]"
        session.add_message("user", marker)
        session.add_message(
            "assistant",
            final_content,
            tools_used=tools_used if tools_used else None,
            tool_chain=tool_chain if tool_chain else None,
        )
        loop._update_session_runtime_metadata(
            session,
            tools_used=tools_used,
            tool_chain=tool_chain,
        )
        await loop.session_manager.append_messages(session, session.messages[-2:])

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata={
                **(msg.metadata or {}),
                "tools_used": tools_used,
                "tool_chain": tool_chain,
            },
        )
