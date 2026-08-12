from __future__ import annotations

from typing import TYPE_CHECKING

from agent.looping.handlers import process_spawn_completion_event
from bus.events import (
    InboundItem,
    InboundMessage,
    OutboundMessage,
    SpawnCompletionItem,
)

if TYPE_CHECKING:
    from agent.core.passive_turn import AgentCore


class CoreRunner:
    """
    ┌──────────────────────────────────────┐
    │ CoreRunner                           │
    ├──────────────────────────────────────┤
    │ 1. 判断是否内部事件                  │
    │ 2. spawn completion 走 helper        │
    │ 3. 普通被动消息走 AgentCore          │
    └──────────────────────────────────────┘
    """

    def __init__(self, agent_core: "AgentCore") -> None:
        self._agent_core = agent_core

    async def process(
        self,
        msg: InboundItem,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        # 1. 先处理 typed 内部工作项，统一走默认 helper 链。
        match msg:
            case SpawnCompletionItem():
                return await process_spawn_completion_event(
                    item=msg,
                    key=key,
                    pipeline=self._agent_core.pipeline,
                    dispatch_outbound=dispatch_outbound,
                )
            case InboundMessage():
                # 2. 默认普通被动消息统一走 AgentCore。
                return await self._agent_core.process(
                    msg,
                    key,
                    dispatch_outbound=dispatch_outbound,
                )
        raise TypeError(f"unsupported inbound item: {type(msg).__name__}")
