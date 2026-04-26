from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from agent.core.response_parser import parse_response
from agent.core.types import ContextRequest
from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import BeforeReasoning

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.core.context_store import ContextStore
    from agent.core.reasoner import Reasoner
    from agent.looping.ports import SessionServices
    from agent.tools.registry import ToolRegistry
    from bus.event_bus import EventBus


@dataclass
class AgentCoreDeps:
    session: "SessionServices"
    context_store: "ContextStore"
    context: "ContextBuilder"
    tools: "ToolRegistry"
    reasoner: "Reasoner"
    event_bus: "EventBus | None" = None


class AgentCore:
    """
    ┌──────────────────────────────────────┐
    │ AgentCore                            │
    ├──────────────────────────────────────┤
    │ 1. 准备上下文                        │
    │ 2. 触发 BeforeReasoning              │
    │ 3. 渲染 prompt 预览                  │
    │ 4. 执行 reasoner                     │
    │ 5. 提交 ContextStore                 │
    │ 6. 返回出站消息                      │
    └──────────────────────────────────────┘
    """

    def __init__(self, deps: AgentCoreDeps) -> None:
        self._session = deps.session
        self._context_store = deps.context_store
        self._context = deps.context
        self._tools = deps.tools
        self._reasoner = deps.reasoner
        self._event_bus = deps.event_bus

    # 处理一条普通被动消息，并提交最终出站结果。
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

        # 2. 再允许 BeforeReasoning 干预技能列表和检索块。
        skill_mentions = list(context_bundle.skill_mentions)
        retrieved_block = context_bundle.retrieved_memory_block
        if self._event_bus is not None:
            before_reasoning = await self._event_bus.emit(
                BeforeReasoning(
                    session_key=key,
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=msg.content,
                    skill_names=skill_mentions,
                    retrieved_memory_block=retrieved_block,
                )
            )
            skill_mentions = list(before_reasoning.skill_names)
            retrieved_block = before_reasoning.retrieved_memory_block

        # 3. 然后通过 Context 主接口渲染 prompt 预览，提前热身 prompt cache。
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

        # 4. 先同步 tool context，再执行被动链 reasoner。
        self._tools.set_context(
            channel=msg.channel,
            chat_id=msg.chat_id,
            current_user_source_ref=_predict_current_user_source_ref(
                session_manager=self._session.session_manager,
                session=session,
            ),
        )
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
        tool_chain = cast(list[dict[str, object]], turn_result.tool_chain)
        parsed_response = parse_response(final_content, tool_chain=tool_chain)

        # 5. 最后走 ContextStore.commit 做被动 turn 提交。
        return await self._context_store.commit(
            msg=msg,
            session_key=key,
            reply=parsed_response.clean_text,
            response_metadata=parsed_response.metadata,
            tools_used=turn_result.tools_used,
            tool_chain=tool_chain,
            thinking=turn_result.thinking,
            streamed_reply=turn_result.streamed,
            retrieval_raw=context_bundle.retrieval_trace_raw,
            context_retry=turn_result.context_retry,
            dispatch_outbound=dispatch_outbound,
        )


def _predict_current_user_source_ref(*, session_manager, session) -> str:
    peek = getattr(session_manager, "peek_next_message_id", None)
    if callable(peek):
        return str(peek(session.key))
    if session.messages:
        last_id = str(session.messages[-1].get("id", "") or "").strip()
        if last_id:
            return last_id
    return ""
