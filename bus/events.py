from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InboundMessage:
    """从 channel 传入的消息"""

    channel: str  # 来源渠道（如 "cli"、"slack"）
    sender: str  # 发送者标识
    chat_id: str  # 会话 ID（用于路由回复）
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def session_key(self) -> str:
        """唯一会话标识，用于维护对话历史"""
        return f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """agent 发出的消息"""

    channel: str  # 目标渠道
    chat_id: str  # 目标会话 ID
    content: str
    thinking: str | None = None
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
