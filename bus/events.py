from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent.plugin_composition.channels import AttachmentKind


@dataclass(frozen=True, slots=True)
class ChannelAttachment:
    """渠道边界中带明确类型的单个附件。"""

    kind: AttachmentKind
    source: str
    filename: str | None = None


@dataclass
class InboundMessage:
    """从 channel 传入的消息"""

    channel: str  # 来源渠道（如 "cli"、"slack"）
    sender: str  # 发送者标识
    chat_id: str  # 会话 ID（用于路由回复）
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    media: list[str] = field(default_factory=list[str])
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    session_admission_id: str | None = field(default=None, repr=False, compare=False)
    handoff_id: str | None = field(default=None, repr=False, compare=False)

    @property
    def session_key(self) -> str:
        """唯一会话标识，用于维护对话历史"""
        override = str(self.metadata.get("session_key_override") or "").strip()
        if override:
            return override
        return f"{self.channel}:{self.chat_id}"
