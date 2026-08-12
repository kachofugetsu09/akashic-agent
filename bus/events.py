from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.policies.delegation import SpawnDecision
    from bus.internal_events import SpawnCompletionEvent


class TurnDisposition(StrEnum):
    """标识无需进入完整提交阶段的合法 turn 结果。"""

    SHORT_CIRCUITED = "short_circuited"


class DeliveryStatus(StrEnum):
    """表示一次完整逻辑消息的渠道提交终态。"""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class TurnTerminalStatus(StrEnum):
    """标识 OutboundMessage 投影的权威 turn 终态。"""

    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class AttachmentKind(StrEnum):
    FILE = "file"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class ChannelAttachment:
    """渠道边界中带明确类型的单个附件。"""

    kind: AttachmentKind
    source: str
    filename: str | None = None


@dataclass(frozen=True, slots=True)
class ChannelMessage:
    """提交给渠道 adapter 的完整逻辑消息。"""

    channel: str
    chat_id: str
    content: str
    attachments: tuple[ChannelAttachment, ...] = ()
    thinking: str | None = None
    metadata: dict[str, object] = field(default_factory=dict[str, object])
    session_message_id: str | None = None
    control_turn_id: str | None = None
    terminal_status: TurnTerminalStatus | None = None


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    """记录渠道对完整逻辑消息的结构化提交结果。"""

    status: DeliveryStatus
    canonical_media: tuple[str, ...] = ()
    detail: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is DeliveryStatus.SUCCESS


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
    terminal_status: TurnTerminalStatus | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    turn_disposition: TurnDisposition | None = field(
        default=None,
        repr=False,
        compare=False,
    )


def channel_message_from_outbound(
    message: OutboundMessage,
    *,
    media_kind: AttachmentKind = AttachmentKind.IMAGE,
) -> ChannelMessage:
    """把已提交 Turn 的字符串媒体投影转换为渠道边界类型。"""

    return ChannelMessage(
        channel=message.channel,
        chat_id=message.chat_id,
        content=message.content,
        attachments=tuple(
            ChannelAttachment(media_kind, source) for source in message.media
        ),
        thinking=message.thinking,
        metadata=dict(message.metadata),
        session_message_id=message.session_message_id,
        control_turn_id=message.control_turn_id,
        terminal_status=message.terminal_status,
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
