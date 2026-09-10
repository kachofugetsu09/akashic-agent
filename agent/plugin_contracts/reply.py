"""回复状态的公开结构合同。

`reply.status.v1` 是「只读当前回复与预览」的公开名字。Core 的
`bootstrap/reply_status.py` 需要订阅它，插件需要提供它；双方都不应 import
对方的实现模块，因此 key、值模型与 Protocol 由合同层拥有。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agent.plugin_composition.model import ServiceKey


@dataclass(frozen=True, slots=True)
class ReplyPreview:
    """一次回复的展示预览；不持有 provider continuation。"""

    message_id: str
    text: str = ""
    thinking: str = ""
    call_record_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReplyActivity:
    """一个会话当前正在进行的回复活动。"""

    session_id: str
    source: str
    handle: str
    active: bool
    preview: ReplyPreview | None = None


@runtime_checkable
class ReplyRead(Protocol):
    """只读当前回复与预览，不持有取消、写消息或执行模型的能力。"""

    def snapshot(self, session_id: str) -> tuple[ReplyActivity, ...]:
        """读取当前快照。"""
        ...

    def follow(self, session_id: str) -> AsyncGenerator[tuple[ReplyActivity, ...], None]:
        """订阅当前快照；重连不重放旧 token。"""
        ...


REPLY_STATUS = ServiceKey[ReplyRead]("reply.status.v1")
