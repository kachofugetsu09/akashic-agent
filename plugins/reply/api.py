from collections.abc import Awaitable, Callable, Sequence, Mapping

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.tasks import Task
from agent.plugin_composition.messages import MessageReader
from agent.plugin_contracts import Message


# 原子材料入口；来源不改写用户 Input，也不复制主回复的配置与工具策略。
REPLY_PROGRAM = ServiceKey[
    Callable[[Task, MessageReader, str, Sequence[Mapping[str, object]]], Awaitable[Message]]
]("reply.program.v2")
