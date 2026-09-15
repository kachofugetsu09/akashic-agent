"""Channel 使用的窄传输、身份和附件端口，不暴露完整 Bus 或 store。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.channels import (
    AttachmentRef, InboundEnvelope, RawInbound,
    ChannelAttachmentImportPort, ChannelAttachmentReadPort,
)


class PendingInputs(Protocol):
    def __call__(self, *, channel: str, session_key: str, provider_message_id: str) -> bool: ...


class PendingAttachments(Protocol):
    def __call__(self, *, channel: str, session_key: str, provider_message_id: str) -> tuple[AttachmentRef, ...] | None: ...


class RejectInput(Protocol):
    def __call__(self, *, channel: str, session_key: str, provider_message_id: str) -> Awaitable[None]: ...


@dataclass(frozen=True, slots=True)
class InputCustody:
    prepare_channel_input: Callable[[InboundEnvelope], Awaitable[None]]
    complete_channel_input: Callable[[InboundEnvelope], Awaitable[None]]
    retain_channel_input: Callable[[InboundEnvelope], Awaitable[None]]
    reserve_durable_inbound: Callable[[RawInbound], Awaitable[bool]]
    defer_durable_inbound: Callable[[str], Awaitable[bool]]
    settle_rejected_inbound: RejectInput
    has_pending_durable_inbound: PendingInputs
    pending_durable_attachment_refs: PendingAttachments
    recover_durable_inbounds: Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ChannelIdentity:
    resolve: Callable[[str, str], str | None]
    remember: Callable[[str, str, str], Awaitable[object]]
    rollback: Callable[[object], Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class ChannelAttachmentImport:
    import_bytes: Callable[..., Awaitable[AttachmentRef]]


@dataclass(frozen=True, slots=True)
class ChannelAttachmentRead:
    resolve_refs: Callable[[tuple[str, ...]], tuple[AttachmentRef, ...]]
    acquire: Callable[..., object]


INPUT_CUSTODY = ServiceKey[InputCustody]("core.input_custody")
CHANNEL_IDENTITY = ServiceKey[ChannelIdentity]("core.channel_identity")
CHANNEL_ATTACHMENT_IMPORT = ServiceKey[ChannelAttachmentImportPort]("core.channel_attachment_import")
CHANNEL_ATTACHMENT_READ = ServiceKey[ChannelAttachmentReadPort]("core.channel_attachment_read")


def unavailable(*args: object, **kwargs: object):
    """隔离装配或缺少宿主绑定时显式拒绝 I/O，不借用正式数据。"""
    raise PermissionError("Channel I/O 未在当前宿主授权；需要独立绑定传输与数据端口")


def unavailable_input_custody() -> InputCustody:
    return InputCustody(
        unavailable, unavailable, unavailable, unavailable, unavailable,
        unavailable, unavailable, unavailable, unavailable,
    )
