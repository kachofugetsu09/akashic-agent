"""会话接纳能力的公开结构合同。

`conversation.v1` 是「按来源打开一个已获授权会话」的公开名字；`check_origin`
是来源原始传输事实的纯校验函数。消费者（subagent、programmatic、delivery_policy
等）不应 import 会话实现模块，因此 key、Protocol 与纯校验由合同层拥有。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from agent.plugin_composition.model import ServiceKey
from agent.plugin_contracts.message import (
    ContentPart,
    ContentReferences,
    Control,
    Input,
    Message,
    Output,
)

if TYPE_CHECKING:
    from agent.plugin_composition.messages import MessageReader


@runtime_checkable
class ConversationPort(Protocol):
    """一个已获授权来源的接纳与控制；活动任务短命，重启只重读日志。"""

    async def accept(self, message_id: str, body: Input) -> Message:
        """接纳一次输入。"""
        ...

    async def pause(self, message_id: str) -> Message:
        """暂停一次输入。"""
        ...

    async def resume(self, message_id: str, input_id: str) -> Message:
        """恢复一次输入。"""
        ...

    async def complete(
        self,
        program: Callable[[object, "MessageReader"], Awaitable[Message]],
    ) -> Message:
        """用一个程序完成本次会话工作。"""
        ...


Changed = Callable[["MessageReader", str], None]


def check_origin(part: ContentPart) -> ContentReferences:
    """来源保存原始传输事实；metadata 不获得 source、角色或路由覆盖权。"""
    raw_value = part.value
    if not isinstance(raw_value, Mapping):
        raise ValueError("channel.origin 必须是对象")
    value = cast("Mapping[str, object]", raw_value)
    if set(value) != {"channel", "chat_id", "sender"} or any(
        not isinstance(item, str) or not item for item in value.values()
    ):
        raise ValueError("channel.origin 身份无效")
    return ContentReferences()


CONVERSATION = ServiceKey[Callable[[str], ConversationPort]]("conversation.v1")
