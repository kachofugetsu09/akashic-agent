"""对话完成、命令与输入来源检查的公共合同。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from agent.plugin_composition import ServiceKey
from plugins.ledger.contract import MessageReader
from agent.plugin_composition.tasks import Task
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


class ConversationSteer(Protocol):
    """回复程序一侧的插话：只有登记过的任务接纳插话，渠道一侧见 CHANNEL_STEER。"""

    # 登记本任务可接纳插话，返回每一步模型请求之前调用的注入函数。
    def bind(self, task: Task, session_id: str) -> Callable[[], Awaitable[None]]: ...


CONVERSATION_STEER = ServiceKey[ConversationSteer]("conversation.steer.v1")


class OriginCheck(Protocol):
    def __call__(self, part: ContentPart) -> ContentReferences: ...


CHECK_ORIGIN = ServiceKey[OriginCheck]("conversation.check_origin.v1")


class InputPartCheck(Protocol):
    def __call__(self, part: ContentPart, reader: MessageReader) -> ContentReferences: ...


# 客户端随输入附带的片段（例如模型选择）由片段 owner 声明并在目标会话里校验；没有 owner 的 kind 一律拒收。
# 值不合格时 owner 抛 plugins.channels.contract.InputPartRejected，渠道把它作为该请求的错误回给发送者；其他异常按程序错误处理。
def input_part_key(kind: str) -> ServiceKey[InputPartCheck]:
    return ServiceKey[InputPartCheck](f"conversation.input_part:{kind}")
