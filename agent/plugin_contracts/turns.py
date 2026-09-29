"""由消息投影的 Turn 引用，不创建第二份执行状态。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition import ServiceKey
from agent.plugin_contracts import CallRef, Message


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


class TurnProjection(Protocol):
    def project(
        self, messages: Iterable[Message], source: str, *,
        after_seq: int = -1,
    ) -> tuple[Turn, ...]:
        """输入完整前缀或闭合 Turn 后的尾部；只返回引用，不保留正文。"""
        ...


TURN_PROJECTION = ServiceKey[TurnProjection]("turn.projection.v1")
