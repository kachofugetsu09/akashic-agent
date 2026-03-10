from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from bus.events import InboundMessage, OutboundMessage
from bus.internal_events import parse_spawn_completion
from memory2.injection_planner import (
    build_memory_injection_result,
    retrieve_history_items,
    retrieve_procedure_items,
)

if TYPE_CHECKING:
    from agent.looping.core import AgentLoop

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
        retrieved_block = await self._retrieve_memory_block(
            msg=msg,
            key=key,
            session=session,
            main_history=main_history,
        )
        # 3. 用会话历史、skill 和 memory block 执行真正的 conversation turn。
        final_content, tools_used, tool_chain = await self._run_conversation_turn(
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
        )
        return self._build_outbound_message(
            msg=msg,
            final_content=final_content,
            tools_used=tools_used,
            tool_chain=tool_chain,
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
    ) -> str:
        """为主对话路径准备 memory block，并把 route / injection 细节写入 trace。"""
        loop = self._loop
        retrieved_block = ""
        try:
            route_decision = "RETRIEVE"
            rewritten_query = msg.content
            fallback_reason = ""
            gate_latency_ms: dict[str, int] = {}
            runtime_md = session.metadata if isinstance(session.metadata, dict) else {}

            # 1. 先并行获取 procedure items 和 history route decision，压缩门控延迟。
            p_query = f"{msg.content} 操作规范"
            recent_turns = loop._format_gate_history(main_history, max_turns=3)
            p_task = asyncio.create_task(
                retrieve_procedure_items(
                    loop._memory_port,
                    p_query,
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
            needs_history = route_decision_obj.needs_history
            rewritten_query = route_decision_obj.rewritten_query
            route_reason = loop._trace_route_reason(route_decision_obj)
            route_ms = route_decision_obj.latency_ms

            # 2. 再按 route decision 决定是否继续检索 history items。
            gate_latency_ms["route"] = route_ms
            if route_reason != "ok":
                fallback_reason = route_reason
            route_decision = "RETRIEVE" if needs_history else "NO_RETRIEVE"

            h_items: list[dict] = []
            if needs_history:
                h_items, _ = await retrieve_history_items(
                    loop._memory_port,
                    rewritten_query,
                    memory_types=["event", "profile"],
                    top_k=loop._memory_top_k_history,
                    allow_global=True,
                )

            # 3. 把 procedure/history 合并成最终 injection block，并做 SOP guard 记录。
            injection = build_memory_injection_result(
                loop._memory_port,
                procedure_items=p_items,
                history_items=h_items,
            )
            selected_items = injection.selected_items
            retrieved_block = injection.block
            injected_item_ids = injection.item_ids
            total_hits = len(p_items) + len(h_items)
            logger.info(
                "memory2 retrieve: route=%s query=%r p=%d h=%d 命中，筛选后 %d 条注入%s",
                route_decision,
                rewritten_query[:50],
                len(p_items),
                len(h_items),
                len(selected_items),
                "" if retrieved_block else "（无内容注入）",
            )
            for _item in selected_items:
                logger.info(
                    "memory2 injected: id=%s score=%.3f type=%s summary=%s",
                    _item.get("id", ""),
                    float(_item.get("score", 0.0)),
                    _item.get("memory_type", ""),
                    str(_item.get("summary", ""))[:60],
                )

            protected_ids = {
                str(i.get("id", ""))
                for i in p_items
                if isinstance(i, dict)
                and i.get("memory_type") == "procedure"
                and (i.get("extra_json") or {}).get("tool_requirement")
                and i.get("id")
            }
            sop_guard_applied = bool(
                loop._memory_sop_guard_enabled
                and any(item_id in protected_ids for item_id in injected_item_ids)
            )

            # 4. 无论命中还是失败，最后都把 route / retrieve 信息写入 trace。
            loop._trace_memory_retrieve(
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                user_msg=msg.content,
                items=selected_items,
                injected_block=retrieved_block,
                route_decision=route_decision,
                rewritten_query=rewritten_query,
                fallback_reason=fallback_reason,
                sop_guard_applied=sop_guard_applied,
                procedure_hits=len(p_items),
                history_hits=len(h_items),
                injected_item_ids=injected_item_ids,
                gate_latency_ms=gate_latency_ms,
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
                fallback_reason="retrieve_exception",
                error=str(e),
            )
        return retrieved_block

    async def _run_conversation_turn(
        self,
        *,
        msg: InboundMessage,
        session,
        skill_mentions: list[str],
        main_history: list[dict],
        retrieved_block: str,
    ) -> tuple[str, list[str], list[dict]]:
        """真正执行一轮 agent 对话，并返回内容、工具使用和 tool_chain。"""
        loop = self._loop
        # 1. 先把 channel/chat_id 写进 tool context，保证工具知道当前会话来源。
        loop._set_tool_context(msg.channel, msg.chat_id)
        # 2. 再统一走 safety retry 包装，避免模型输出异常直接打断主流程。
        final_content, tools_used, tool_chain = await loop._run_with_safety_retry(
            msg,
            session,
            skill_names=skill_mentions or None,
            base_history=main_history,
            retrieved_memory_block=retrieved_block,
        )

        # 3. 最后保证 content 至少有一个兜底字符串，避免 assistant 回复为空。
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        return final_content, tools_used, tool_chain

    async def _persist_turn(
        self,
        *,
        msg: InboundMessage,
        key: str,
        session,
        final_content: str,
        tools_used: list[str],
        tool_chain: list[dict],
    ) -> None:
        """把本轮结果统一写回 session、presence 和 post-response worker。"""
        loop = self._loop
        preview = (
            final_content[:120] + "..." if len(final_content) > 120 else final_content
        )
        logger.info(f"Response to {msg.channel}:{msg.sender}: {preview}")

        # 1. 先把 user/assistant 两条消息落到 session 内存对象中。
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
    ) -> OutboundMessage:
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
        final_content, tools_used, tool_chain, _ = await loop._run_agent_loop(
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
