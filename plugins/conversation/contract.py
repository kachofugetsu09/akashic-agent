"""对话完成、命令与输入来源检查的公共合同。"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from agent.plugin_composition import ServiceKey
from plugins.ledger.contract import MessageReader
from agent.plugin_composition.tasks import Task
from plugins.channels.contract import (
    InputPartCheck as InputPartCheck,
    input_part_key as input_part_key,
)
from plugins.ledger.contract import ContentPart, ContentReferences, Message
from plugins.sources.contract import CompletionProgram


class ConversationCompleteV2(Protocol):
    async def __call__(
        self, session_id: str, program: CompletionProgram,
    ) -> Message: ...


CONVERSATION_COMPLETE_V2 = ServiceKey[ConversationCompleteV2]("conversation.complete.v2")
CONVERSATION_COMMANDS = ServiceKey[
    Callable[[Task, MessageReader, str], Awaitable[Message | None]]
]("conversation.commands.v1")


@dataclass(frozen=True, slots=True)
class SteerBinding:
    # 每一步模型请求之前调用的注入函数。
    before_step: Callable[[], Awaitable[None]]
    # 有插话待写入时置位，注入时清除。
    arrived: asyncio.Event


class ConversationSteer(Protocol):
    """回复程序一侧的插话：只有登记过的任务接纳插话，渠道一侧见 CHANNEL_STEER。"""

    # 登记本任务可接纳插话。
    def bind(self, task: Task, session_id: str) -> SteerBinding: ...


CONVERSATION_STEER = ServiceKey[ConversationSteer]("conversation.steer.v1")


class OriginCheck(Protocol):
    def __call__(self, part: ContentPart) -> ContentReferences: ...


CHECK_ORIGIN = ServiceKey[OriginCheck]("conversation.check_origin.v1")
