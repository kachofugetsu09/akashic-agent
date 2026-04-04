from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.core.context_store import ContextStore
    from agent.looping.ports import SessionServices, TurnRunner
    from agent.tools.registry import ToolRegistry
    from agent.turns.orchestrator import TurnOrchestrator


@dataclass
class AgentCoreDeps:
    session: "SessionServices"
    context_store: "ContextStore"
    context: "ContextBuilder"
    tools: "ToolRegistry"
    turn_runner: "TurnRunner"
    orchestrator: "TurnOrchestrator"


class AgentCore:
    """
    ┌──────────────────────────────────────┐
    │ AgentCore                            │
    ├──────────────────────────────────────┤
    │ 1. prepare context                   │
    │ 2. render prompt preview             │
    │ 3. run turn runner                   │
    │ 4. commit via orchestrator           │
    │ 5. return outbound                   │
    └──────────────────────────────────────┘
    """

    def __init__(self, deps: AgentCoreDeps) -> None:
        self._session = deps.session
        self._context_store = deps.context_store
        self._context = deps.context
        self._tools = deps.tools
        self._turn_runner = deps.turn_runner
        self._orchestrator = deps.orchestrator

    async def process(
        self,
        msg: InboundMessage,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        # 1. 先读取真实 session，并准备本轮上下文。
        session = self._session.session_manager.get_or_create(key)
        context_bundle = await self._context_store.prepare(
            msg=msg,
            session_key=key,
            session=session,
        )

        # 2. 再渲染 system prompt 预览，先把 prompt 主编排收进 AgentCore。
        skill_mentions = list(context_bundle.metadata.get("skill_mentions") or [])
        retrieved_block = str(context_bundle.metadata.get("retrieved_memory_block") or "")
        self._context.build_system_prompt(
            skill_names=skill_mentions,
            message_timestamp=msg.timestamp,
            retrieved_memory_block=retrieved_block,
        )

        # 3. 先同步 tool context，再执行旧 turn runner。
        self._tools.set_context(channel=msg.channel, chat_id=msg.chat_id)
        final_content, tools_used, tool_chain, thinking = await self._turn_runner.run(
            msg,
            session,
            skill_names=skill_mentions or None,
            base_history=None,
            retrieved_memory_block=retrieved_block,
        )
        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # 4. 继续委托旧 orchestrator 做 commit / outbound。
        retry_trace = getattr(self._turn_runner, "last_retry_trace", {})
        result = TurnResult(
            decision="reply",
            outbound=TurnOutbound(session_key=key, content=final_content),
            trace=TurnTrace(
                source="passive",
                retrieval={"raw": context_bundle.metadata.get("retrieval_trace_raw")},
                extra={
                    "tools_used": tools_used,
                    "tool_chain": tool_chain,
                    "thinking": thinking,
                    "context_retry": retry_trace,
                },
            ),
        )

        # 5. 返回最终出站消息。
        return await self._orchestrator.handle_turn(
            msg=msg,
            result=result,
            dispatch_outbound=dispatch_outbound,
        )
