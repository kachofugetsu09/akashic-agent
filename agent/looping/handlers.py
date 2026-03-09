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
        loop = self._loop
        session = loop.session_manager.get_or_create(key)
        skill_mentions = loop._collect_skill_mentions(msg.content)
        if skill_mentions:
            logger.info(f"检测到 $skill 提及，直接注入完整内容: {skill_mentions}")

        main_history = session.get_history(max_messages=loop.memory_window)
        retrieved_block = ""
        try:
            route_decision = "RETRIEVE"
            rewritten_query = msg.content
            fallback_reason = ""
            gate_latency_ms: dict[str, int] = {}
            runtime_md = session.metadata if isinstance(session.metadata, dict) else {}

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
                loop._decide_history_retrieval(
                    user_msg=msg.content,
                    metadata=runtime_md,
                    recent_history=recent_turns,
                )
            )
            p_items, (
                needs_history,
                rewritten_query,
                route_reason,
                route_ms,
            ) = await asyncio.gather(p_task, route_task)

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

            injection = build_memory_injection_result(
                loop._memory_port,
                procedure_items=p_items,
                history_items=h_items,
            )
            selected_items = injection.selected_items
            retrieved_block = injection.block
            injected_item_ids = injection.item_ids
            if retrieved_block:
                logger.info(
                    "memory2 retrieve: %d 条命中，筛选后 %d 条注入",
                    len(p_items) + len(h_items),
                    len(selected_items),
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

        loop._set_tool_context(msg.channel, msg.chat_id)
        final_content, tools_used, tool_chain = await loop._run_with_safety_retry(
            msg,
            session,
            skill_names=skill_mentions or None,
            base_history=main_history,
            retrieved_memory_block=retrieved_block,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = (
            final_content[:120] + "..." if len(final_content) > 120 else final_content
        )
        logger.info(f"Response to {msg.channel}:{msg.sender}: {preview}")

        if loop._presence:
            loop._presence.record_user_message(key)
        session.add_message("user", msg.content, media=msg.media if msg.media else None)
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
        loop._schedule_consolidation_if_needed(session, key)
        loop._schedule_post_response_memory(
            msg=msg,
            key=key,
            final_content=final_content,
            tool_chain=tool_chain,
        )

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

        current_message = (
            "[后台任务已完成]\n"
            f"任务标签: {label}\n"
            f"原始任务: {task or '（未提供）'}\n"
            f"状态: {status}\n"
            f"执行结果:\n{result or '（无结果）'}\n\n"
            "请基于当前会话上下文，用自然中文向用户汇报这次后台任务的结果。\n"
            "不要提及 subagent、spawn、内部事件、job_id。\n"
            "如果状态是 incomplete，说明目前做到哪里、接下来还差什么。\n"
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
            final_content = "后台任务已完成。"

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
