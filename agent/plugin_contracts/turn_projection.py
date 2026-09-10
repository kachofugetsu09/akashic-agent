"""Turn 投影的公开结构合同。

`turn.projection.v1` 是「从消息读取逻辑 Turn」的公开名字。Core 的
`session`/`infra` 层与多个业务插件都要读 Turn，但不应 import 实现模块，
因此合同层拥有值模型、Protocol 与 key，实现留在 `plugins/turn_projection`。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from agent.plugin_composition.model import ServiceKey
from agent.plugin_contracts.message import CallRef, Message


@dataclass(frozen=True, slots=True)
class Turn:
    """一个日志区间的消息引用；不代表运行任务或持久化行。"""

    source: str
    after_seq: int
    through_seq: int
    ending_message_id: str | None
    status: Literal["open", "complete", "quiet", "abandoned"]
    message_ids: tuple[str, ...]
    observations: tuple[tuple[CallRef, str], ...]


@runtime_checkable
class TurnProjectionPort(Protocol):
    """只读 Turn 投影；不保存内容或消费进度。"""

    def project(self, messages: Sequence[Message], source: str) -> tuple[Turn, ...]:
        """从消息序列投影出逻辑 Turn。"""
        ...


TURN_PROJECTION = ServiceKey[TurnProjectionPort]("turn.projection.v1")
