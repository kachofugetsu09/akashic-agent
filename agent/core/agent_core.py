from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.core.types import ContextRequest
from bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.core.context_store import ContextStore
    from agent.core.reasoner import Reasoner
    from agent.looping.ports import SessionServices
    from agent.tools.registry import ToolRegistry


@dataclass
class AgentCoreDeps:
    session: "SessionServices"
    context_store: "ContextStore"
    context: "ContextBuilder"
    tools: "ToolRegistry"
    reasoner: "Reasoner"


class AgentCore:
    """
    ┌──────────────────────────────────────┐
    │ AgentCore                            │
    ├──────────────────────────────────────┤
    │ 1. prepare context                   │
    │ 2. render prompt preview             │
    │ 3. run reasoner                      │
    │ 4. commit via ContextStore           │
    │ 5. return outbound                   │
    └──────────────────────────────────────┘
    """

    def __init__(self, deps: AgentCoreDeps) -> None:
        self._session = deps.session
        self._context_store = deps.context_store
        self._context = deps.context
        self._tools = deps.tools
        self._reasoner = deps.reasoner

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

        # 2. 再通过 Context 主接口渲染 prompt 预览，提前热身 prompt cache。
        skill_mentions = list(context_bundle.skill_mentions)
        retrieved_block = context_bundle.retrieved_memory_block
        self._context.render(
            ContextRequest(
                history=[],
                current_message="",
                skill_names=skill_mentions,
                channel=msg.channel,
                chat_id=msg.chat_id,
                message_timestamp=msg.timestamp,
                retrieved_memory_block=retrieved_block,
            )
        )

        # 3. 先同步 tool context，再执行被动链 reasoner。
        self._tools.set_context(channel=msg.channel, chat_id=msg.chat_id)
        turn_result = await self._reasoner.run_turn(
            msg=msg,
            skill_names=skill_mentions or None,
            session=session,
            base_history=None,
            retrieved_memory_block=retrieved_block,
        )
        final_content = turn_result.reply
        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # 4. 继续走新的 ContextStore.commit 做被动 turn 提交。
        return await self._context_store.commit(
            msg=msg,
            session_key=key,
            reply=final_content,
            tools_used=turn_result.tools_used,
            tool_chain=turn_result.tool_chain,
            thinking=turn_result.thinking,
            retrieval_raw=context_bundle.retrieval_trace_raw,
            context_retry=turn_result.context_retry,
            dispatch_outbound=dispatch_outbound,
        )
