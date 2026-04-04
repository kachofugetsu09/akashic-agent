from __future__ import annotations

from typing import TYPE_CHECKING

from agent.core.context_store import DefaultContextStore
from agent.looping.ports import (
    AgentLoopRunner,
    ConversationTurnDeps,
    SessionLike,
    SessionServices,
)
from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from bus.events import InboundMessage, OutboundMessage
from bus.internal_events import parse_spawn_completion

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.tools.registry import ToolRegistry
    from agent.turns.orchestrator import TurnOrchestrator


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
        self._context_store = DefaultContextStore(
            retrieval=deps.retrieval,
            context=deps.context,
        )

    async def process(
        self,
        msg: InboundMessage,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        """处理一次普通用户消息，把"检索 -> 执行 -> 持久化 -> 回包"串成主路径。"""
        session = self._session.session_manager.get_or_create(key)
        context_bundle = await self._context_store.prepare(
            msg=msg,
            session_key=key,
            session=session,
        )
        final_content, tools_used, tool_chain, thinking = await self._run_conversation_turn(
            msg=msg,
            session=session,
            skill_mentions=list(context_bundle.metadata.get("skill_mentions") or []),
            retrieved_block=str(
                context_bundle.metadata.get("retrieved_memory_block") or ""
            ),
        )
        retry_trace = getattr(self._turn_runner, "last_retry_trace", {})
        result = TurnResult(
            decision="reply",
            outbound=TurnOutbound(session_key=key, content=final_content),
            trace=TurnTrace(
                source="passive",
                retrieval={
                    "raw": context_bundle.metadata.get("retrieval_trace_raw")
                },
                extra={
                    "tools_used": tools_used,
                    "tool_chain": tool_chain,
                    "thinking": thinking,
                    "context_retry": retry_trace,
                },
            ),
        )
        return await self._orchestrator.handle_turn(
            msg=msg,
            result=result,
            dispatch_outbound=dispatch_outbound,
        )

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


class InternalEventHandler:
    def __init__(
        self,
        session_svc: SessionServices,
        context: ContextBuilder,
        tools: ToolRegistry,
        memory_window: int,
        run_agent_loop_fn: AgentLoopRunner,
        orchestrator: "TurnOrchestrator",
    ) -> None:
        self._session = session_svc
        self._context = context
        self._tools = tools
        self._memory_window = memory_window
        self._run_agent_loop = run_agent_loop_fn
        self._orchestrator = orchestrator

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
        retry_count = event.retry_count

        _EXIT_LABELS: dict[str, str] = {
            "completed": "正常完成",
            "max_iterations": "迭代预算耗尽（任务可能不完整）",
            "tool_loop": "工具调用循环截断（任务可能不完整）",
            "error": "执行出错",
            "forced_summary": "强制汇总（任务可能不完整）",
        }
        exit_label = _EXIT_LABELS.get(exit_reason, exit_reason or "未知")

        if retry_count >= 1:
            guidance = (
                "⚠️ 已重试一次，不再重试。\n"
                "请直接将已获得的结果汇报给用户，说明已完成的部分和未完成的部分。"
            )
        else:
            guidance = (
                "**处理指引（按顺序判断，选其一执行）**\n"
                "1. 结果完整回答了原始任务 → 直接向用户汇报，不提及内部机制\n"
                "2. 退出原因是【迭代预算耗尽】或【工具调用循环截断】，且核心信息明显不足 → "
                "调用 spawn 重试；task 中说明上次卡在哪、这次从哪继续；"
                "run_in_background=true；同时简短告知用户正在补充\n"
                "3. 结果为空或明显出错 → 直接告知用户失败，询问是否需要重试\n"
                "重试只允许一次。"
            )

        current_message = (
            f"[后台任务回传]\n"
            f"任务标签: {label}\n"
            f"原始任务: {task or '（未提供）'}\n"
            f"退出原因: {exit_label}\n"
            f"执行结果:\n{result or '（无结果）'}\n\n"
            f"{guidance}\n\n"
            "禁止在回复中提及 subagent、spawn、job_id、内部事件等内部概念。\n"
            "必要时可读取结果里提到的文件来补充说明。"
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
        pseudo_msg = InboundMessage(
            channel=msg.channel,
            sender=msg.sender,
            chat_id=msg.chat_id,
            content=marker,
            timestamp=msg.timestamp,
            media=[],
            metadata={**(msg.metadata or {}), "skip_post_memory": True},
        )
        result_obj = TurnResult(
            decision="reply",
            outbound=TurnOutbound(session_key=key, content=final_content),
            trace=TurnTrace(
                source="passive",
                extra={
                    "tools_used": tools_used,
                    "tool_chain": tool_chain,
                },
            ),
        )
        return await self._orchestrator.handle_turn(
            msg=pseudo_msg,
            result=result_obj,
            dispatch_outbound=True,
        )
