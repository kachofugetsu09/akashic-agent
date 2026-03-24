from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from agent.looping.ports import (
    AgentLoopRunner,
    ConversationTurnDeps,
    SessionLike,
    SessionServices,
)
from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from agent.looping.turn_types import HistoryMessage, ToolCall, ToolCallGroup
from agent.retrieval.protocol import RetrievalRequest
from bus.events import InboundMessage, OutboundMessage
from bus.internal_events import parse_spawn_completion

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.tools.registry import ToolRegistry

logger = logging.getLogger("agent.loop_handlers")


class ConversationTurnHandler:
    """Phase 2 过渡形态：持有显式 ports，不再持有 AgentLoop 引用。

    10 个参数超过 ≤3 规则的硬性限制，属于设计文档明确说明的过渡例外；
    Phase 4 完成后子组件各自满足该规则，本类也随之收敛。
    """

    def __init__(self, deps: ConversationTurnDeps) -> None:
        self._llm = deps.llm
        self._llm_config = deps.llm_config
        self._turn_runner = deps.turn_runner
        self._retrieval = deps.retrieval
        self._orchestrator = deps.orchestrator
        self._session = deps.session
        self._tools = deps.tools
        self._context = deps.context

    async def process(self, msg: InboundMessage, key: str) -> OutboundMessage:
        """处理一次普通用户消息，把"检索 -> 执行 -> 持久化 -> 回包"串成主路径。"""
        session = self._session.session_manager.get_or_create(key)
        retrieval_history = session.get_history()
        history_messages = _to_history_messages(retrieval_history)
        retrieval_result = await self._retrieval.retrieve(
            RetrievalRequest(
                message=msg.content,
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                history=history_messages,
                session_metadata=(
                    session.metadata if isinstance(session.metadata, dict) else {}
                ),
                timestamp=msg.timestamp,
            )
        )
        skill_mentions = self._collect_skill_mentions(msg)
        final_content, tools_used, tool_chain, thinking = await self._run_conversation_turn(
            msg=msg,
            session=session,
            skill_mentions=skill_mentions,
            retrieved_block=retrieval_result.block,
        )
        result = TurnResult(
            decision="reply",
            outbound=TurnOutbound(session_key=key, content=final_content),
            trace=TurnTrace(
                source="passive",
                retrieval={
                    "raw": retrieval_result.trace.raw
                    if retrieval_result.trace is not None
                    else None
                },
                extra={
                    "tools_used": tools_used,
                    "tool_chain": tool_chain,
                    "thinking": thinking,
                },
            ),
        )
        return await self._orchestrator.handle_turn(
            msg=msg,
            result=result,
        )

    def _collect_skill_mentions(self, msg: InboundMessage) -> list[str]:
        raw_names = re.findall(r"\$([a-zA-Z0-9_-]+)", msg.content)
        if not raw_names:
            return []
        available = {
            s["name"] for s in self._context.skills.list_skills(filter_unavailable=False)
        }
        seen: set[str] = set()
        result: list[str] = []
        for name in raw_names:
            if name in available and name not in seen:
                seen.add(name)
                result.append(name)
        if result:
            logger.info(f"检测到 $skill 提及，直接注入完整内容: {result}")
        return result

    async def _run_conversation_turn(
        self,
        *,
        msg: InboundMessage,
        session: SessionLike,
        skill_mentions: list[str],
        retrieved_block: str,
    ) -> tuple[str, list[str], list[dict], str | None]:
        """真正执行一轮 agent 对话，并返回内容、工具使用、tool_chain 和 thinking。"""
        # 1. 先把 channel/chat_id 写进 tool context，保证工具知道当前会话来源。
        self._tools.set_context(channel=msg.channel, chat_id=msg.chat_id)
        # 2. 再统一走 safety retry 包装，避免模型输出异常直接打断主流程。
        final_content, tools_used, tool_chain, thinking = await self._turn_runner.run(
            msg,
            session,
            skill_names=skill_mentions or None,
            base_history=None,
            retrieved_memory_block=retrieved_block,
        )

        # 3. 最后保证 content 至少有一个兜底字符串，避免 assistant 回复为空。
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        return final_content, tools_used, tool_chain, thinking


def _to_history_messages(messages: list[dict]) -> list[HistoryMessage]:
    out: list[HistoryMessage] = []
    for msg in messages:
        role = str(msg.get("role", "") or "")
        content = str(msg.get("content", "") or "")
        tools_used = [
            str(tool_name)
            for tool_name in (msg.get("tools_used") or [])
            if isinstance(tool_name, str)
        ]
        out.append(
            HistoryMessage(
                role=role,
                content=content,
                tools_used=tools_used,
                tool_chain=_to_tool_call_groups(msg.get("tool_chain") or []),
            )
        )
    return out


def _to_tool_call_groups(raw_chain: list[dict]) -> list[ToolCallGroup]:
    groups: list[ToolCallGroup] = []
    for group in raw_chain:
        text = str(group.get("text", "") or "")
        calls: list[ToolCall] = []
        for call in (group.get("calls") or []):
            args = call.get("arguments")
            calls.append(
                ToolCall(
                    call_id=str(call.get("call_id", "") or ""),
                    name=str(call.get("name", "") or ""),
                    arguments=args if isinstance(args, dict) else {},
                    result=str(call.get("result", "") or ""),
                )
            )
        groups.append(ToolCallGroup(text=text, calls=calls))
    return groups


class InternalEventHandler:
    def __init__(
        self,
        session_svc: SessionServices,
        context: ContextBuilder,
        tools: ToolRegistry,
        memory_window: int,
        run_agent_loop_fn: AgentLoopRunner,
    ) -> None:
        self._session = session_svc
        self._context = context
        self._tools = tools
        self._memory_window = memory_window
        self._run_agent_loop = run_agent_loop_fn

    async def process_spawn_completion(
        self, msg: InboundMessage, key: str
    ) -> OutboundMessage:
        session = self._session.session_manager.get_or_create(key)
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

        self._tools.set_context(channel=msg.channel, chat_id=msg.chat_id)
        initial_messages = self._context.build_messages(
            history=session.get_history(max_messages=self._memory_window),
            current_message=current_message,
            channel=msg.channel,
            chat_id=msg.chat_id,
            message_timestamp=msg.timestamp,
        )
        final_content, tools_used, tool_chain, _, _thinking = await self._run_agent_loop(
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
        _update_session_runtime_metadata(
            session,
            tools_used=tools_used,
            tool_chain=tool_chain,
        )
        await self._session.session_manager.append_messages(session, session.messages[-2:])

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
