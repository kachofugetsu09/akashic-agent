from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.policies.delegation import SpawnDecision
    from bus.internal_events import SpawnCompletionEvent


class TurnDisposition(StrEnum):
    """标识无需进入完整提交阶段的合法 turn 结果。"""

    SHORT_CIRCUITED = "short_circuited"


@dataclass
class InboundMessage:
    """从 channel 传入的消息"""

    channel: str  # 来源渠道（如 "cli"、"slack"）
    sender: str  # 发送者标识
    chat_id: str  # 会话 ID（用于路由回复）
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list[str])
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    session_admission_id: str | None = field(default=None, repr=False, compare=False)

    @property
    def session_key(self) -> str:
        """唯一会话标识，用于维护对话历史"""
        override = str(self.metadata.get("session_key_override") or "").strip()
        if override:
            return override
        return f"{self.channel}:{self.chat_id}"

    @property
    def context_channel(self) -> str:
        return str(self.metadata.get("context_channel") or self.channel).strip()

    @property
    def context_chat_id(self) -> str:
        return str(self.metadata.get("context_chat_id") or self.chat_id).strip()


@dataclass
class OutboundMessage:
    """agent 发出的消息"""

    channel: str  # 目标渠道
    chat_id: str  # 目标会话 ID
    content: str
    thinking: str | None = None
    reply_to: str | None = None
    media: list[str] = field(default_factory=list[str])
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    control_turn_id: str | None = field(default=None, repr=False, compare=False)
    session_message_id: str | None = field(default=None, repr=False, compare=False)
    turn_disposition: TurnDisposition | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass
class SpawnCompletionItem:
    """Typed internal work item，替代 metadata 编解码。"""

    channel: str
    chat_id: str
    event: "SpawnCompletionEvent"
    decision: "SpawnDecision | None" = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def session_key(self) -> str:
        return f"{self.channel}:{self.chat_id}"


InboundItem = InboundMessage | SpawnCompletionItem
