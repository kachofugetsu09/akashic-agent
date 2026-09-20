"""客户端插件内部的旧主动投影值类型。

这些类型只服务于 Web/Mobile 的被动投影路径。正式 v3 通道使用
``ProviderDeliveryRequest``，不会把 Core 的 bus 事件模型带进插件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agent.plugin_composition.channels import AttachmentRef


class AttachmentKind(StrEnum):
    FILE = "file"
    IMAGE = "image"


class DeliveryStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class TurnTerminalStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ChannelAttachment:
    kind: AttachmentKind
    source: str
    filename: str | None = None


@dataclass(frozen=True, slots=True)
class ChannelMessage:
    channel: str
    chat_id: str
    content: str
    attachments: tuple[ChannelAttachment, ...] = ()
    attachment_refs: tuple[AttachmentRef, ...] = ()
    thinking: str | None = None
    reply_to: str | None = None
    metadata: dict[str, object] = field(default_factory=dict[str, object])
    session_message_id: str | None = None
    control_turn_id: str | None = None
    execution_attempt_id: str | None = None
    terminal_status: TurnTerminalStatus | None = None


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    status: DeliveryStatus
    canonical_media: tuple[str, ...] = ()
    detail: str | None = None


__all__ = [
    "AttachmentKind",
    "ChannelAttachment",
    "ChannelMessage",
    "DeliveryReceipt",
    "DeliveryStatus",
    "TurnTerminalStatus",
]
