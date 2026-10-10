from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias

from plugins.ledger.contract import Message
from plugins.ledger.contract import (
    AttachmentKind, AttachmentRef, AttachmentReadLease,
    AttachmentReadPort as ChannelAttachmentReadPort,
)

from agent.plugin_composition.context import Context
from agent.plugin_composition.requests import RequestContext
from agent.plugin_composition.credentials import CredentialRef, ProviderClient, ProviderClientFactory
from agent.plugin_composition.model import ServiceKey


_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def channel_config_revision(projection: Mapping[str, object]) -> str:
    """Hash a redacted config projection without exposing credential bytes."""

    payload = _canonical_channel_config_value(projection)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_channel_config_value(value: object) -> object:
    """Convert TOML values and CredentialRef into deterministic JSON."""

    if isinstance(value, CredentialRef):
        return {"$credential_ref": list(value.path)}
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("channel config projection key 必须是字符串")
            result[key] = _canonical_channel_config_value(value[key])
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_channel_config_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"$float": "nan"}
        if math.isinf(value):
            return {"$float": "inf" if value > 0 else "-inf"}
        return {"$float": value.hex()}
    if isinstance(value, (datetime, date, time)):
        return {
            "$toml_type": type(value).__name__,
            "value": value.isoformat(),
        }
    raise TypeError(
        "channel config projection 包含不受支持的值: "
        f"{type(value).__name__}"
    )


class ChannelCapability(StrEnum):
    INBOUND = "inbound"
    DURABLE_INBOUND = "durable_inbound"
    OUTBOUND = "outbound"
    CONTROL = "control"
    TURN_STREAM = "turn_stream"


class InboundIdentity(StrEnum):
    PROVIDER_MESSAGE_ID = "provider_message_id"


class DeliveryStatus(StrEnum):
    DELIVERED = "delivered"
    REJECTED = "rejected"
    FAILED = "failed"


class ChannelCommitRole(StrEnum):
    DIRECT = "direct"
    PASSIVE = "passive"


class ChannelTerminalStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


JsonValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["JsonValue", ...]
    | Mapping[str, "JsonValue"]
)

# These keys describe the transport handoff itself.  A provider message ID
# remains the identity source; metadata only carries the durable reservation.
DURABLE_INBOUND_MARKER = "durable_inbound"
DURABLE_HANDOFF_ID = "durable_handoff_id"
DURABLE_PROVIDER_MESSAGE_ID = "provider_message_id"
DURABLE_ATTACHMENT_REFS = "durable_attachment_refs"


@dataclass(frozen=True, slots=True)
class ChannelInboundMessage:
    """Represent one provider text message without retaining mutable input state."""

    channel: str
    sender: str
    chat_id: str
    content: str
    timestamp: datetime
    metadata: Mapping[str, JsonValue]
    attachments: tuple[AttachmentRef, ...] = ()

    def __post_init__(self) -> None:
        _text(self.channel, "channel")
        _text(self.sender, "sender")
        _text(self.chat_id, "chat_id")
        _content_string(self.content, "content")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp 必须是 datetime")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp 必须是 timezone-aware datetime")
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))
        object.__setattr__(
            self,
            "attachments",
            _attachment_refs(self.attachments, "attachments"),
        )


# 来源插件接纳已验证的传输输入；不获得 Channel lease 或发送权。

CHANNEL_INPUT_V2 = ServiceKey[
    Callable[[str, str, ChannelInboundMessage], Awaitable[Message]]
]("channel.input.v2")


class ChannelBindingLease(Protocol):
    @property
    def snapshot_id(self) -> str: ...

    @property
    def generation_id(self) -> str: ...

    @property
    def channel_name(self) -> str: ...

    @property
    def binding_token(self) -> str: ...

    @property
    def active(self) -> bool: ...

    async def deliver(self, envelope: OutboundEnvelope) -> ChannelDeliveryReceipt: ...

    async def aclose(self) -> None: ...


class ChannelIngressPort(Protocol):
    async def admit(self, raw: RawInbound) -> bool: ...


class ChannelDurableInboundPort(Protocol):
    """Expose the one durable handoff owner to a declared channel binding."""

    async def reserve(self, raw: RawInbound) -> bool: ...

    async def defer(self, handoff_id: str) -> None: ...

    async def settle_rejected(
        self,
        *,
        session_key: str,
        provider_message_id: str,
    ) -> None: ...

    def has_pending(
        self,
        *,
        session_key: str,
        provider_message_id: str,
    ) -> bool: ...

    def pending_attachment_refs(
        self,
        *,
        session_key: str,
        provider_message_id: str,
    ) -> tuple[AttachmentRef, ...] | None: ...

    async def recover(self, raw: RawInbound) -> bool: ...


class ChannelIdentityPort(Protocol):
    def resolve(self, provider_identity: str) -> str | None: ...


class ChannelAttachmentImportPort(Protocol):
    async def import_bytes(
        self,
        data: bytes,
        *,
        kind: AttachmentKind,
        filename: str | None,
        media_type: str | None,
    ) -> AttachmentRef: ...


class InboundEnvelope(Protocol):
    """借用一次已接纳输入；Channels 独占实际 lease 和关闭状态。"""

    @property
    def message_id(self) -> str: ...

    @property
    def session_key(self) -> str: ...

    @property
    def message(self) -> ChannelInboundMessage: ...

    @property
    def closed(self) -> bool: ...

    @property
    def metadata(self) -> Mapping[str, JsonValue]: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RawInbound:
    """Carry provider identity and a frozen text projection to Core admission."""

    message_id: str
    message: ChannelInboundMessage
    provider_identity: str | None = None
    recipient: str | None = None

    def __post_init__(self) -> None:
        _message_id(self.message_id)
        if not isinstance(self.message, ChannelInboundMessage):
            raise TypeError("message 必须是 ChannelInboundMessage")
        if self.provider_identity is not None:
            _text(self.provider_identity, "provider_identity")
        if self.recipient is not None:
            _text(self.recipient, "recipient")
        if (self.provider_identity is None) != (self.recipient is None):
            raise ValueError("provider_identity 与 recipient 必须同时提供")


@dataclass(frozen=True, slots=True)
class OutboundEnvelope:
    """Identify one exact channel delivery attempt and its immutable payload."""

    logical_delivery_id: str
    delivery_id: str
    attempt_sequence: int
    snapshot_id: str
    generation_id: str
    binding_token: str
    channel: str
    recipient: str
    body: str
    metadata: Mapping[str, JsonValue]
    attachments: tuple[AttachmentRef, ...] = ()
    commit_role: ChannelCommitRole = ChannelCommitRole.DIRECT
    thinking: str | None = None
    reply_to: str | None = None
    session_message_id: str | None = None
    control_turn_id: str | None = None
    execution_attempt_id: str | None = None
    terminal_status: ChannelTerminalStatus | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "logical_delivery_id",
            "delivery_id",
            "snapshot_id",
            "generation_id",
            "binding_token",
            "channel",
            "recipient",
        ):
            _text(getattr(self, field_name), field_name)
        _content_string(self.body, "body")
        if isinstance(self.attempt_sequence, bool) or not isinstance(
            self.attempt_sequence, int
        ) or self.attempt_sequence < 1:
            raise ValueError("attempt_sequence 必须是正整数")
        if self.attempt_sequence == 1 and self.logical_delivery_id != self.delivery_id:
            raise ValueError("首次 delivery 的 logical_delivery_id 必须等于 delivery_id")
        if self.attempt_sequence > 1 and self.logical_delivery_id == self.delivery_id:
            raise ValueError("重试 attempt 必须生成新的 delivery_id")
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))
        object.__setattr__(
            self,
            "attachments",
            _attachment_refs(self.attachments, "attachments"),
        )
        if not isinstance(self.commit_role, ChannelCommitRole):
            raise TypeError("commit_role 必须是 ChannelCommitRole")
        if self.thinking is not None:
            _content_string(self.thinking, "thinking")
        for field_name in (
            "reply_to",
            "session_message_id",
            "control_turn_id",
            "execution_attempt_id",
        ):
            _optional_string(getattr(self, field_name), field_name)
        if self.terminal_status is not None and not isinstance(
            self.terminal_status,
            ChannelTerminalStatus,
        ):
            raise TypeError("terminal_status 必须是 ChannelTerminalStatus 或 None")


@dataclass(frozen=True, slots=True)
class ChannelDeliveryReceipt:
    """Report a settled provider delivery attempt without encoding retry policy."""

    delivery_id: str
    status: DeliveryStatus
    provider_ids: tuple[str, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        _text(self.delivery_id, "delivery_id")
        if not isinstance(self.status, DeliveryStatus):
            raise TypeError("status 必须是 DeliveryStatus")
        object.__setattr__(self, "provider_ids", _text_tuple(self.provider_ids, "provider_ids"))
        if self.error is not None:
            _text(self.error, "error")


@dataclass(frozen=True, slots=True)
class ControlReceipt:
    """Report one deduplicated interrupt and its independent response delivery."""

    accepted: bool
    reason: Literal["interrupted", "idle", "duplicate", "binding_closed"]
    response: ChannelDeliveryReceipt | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted 必须是 bool")
        if self.reason not in {"interrupted", "idle", "duplicate", "binding_closed"}:
            raise ValueError(f"control reason 无效: {self.reason}")
        if self.reason == "interrupted" and not self.accepted:
            raise ValueError("interrupted control 必须 accepted=True")
        if self.reason != "interrupted" and self.accepted:
            raise ValueError("非 interrupted control 不得 accepted=True")
        if self.reason in {"duplicate", "binding_closed"} and self.response is not None:
            raise ValueError("duplicate/binding_closed control 不得携带 response")
        if self.response is not None and not isinstance(
            self.response, ChannelDeliveryReceipt
        ):
            raise TypeError("response 必须是 ChannelDeliveryReceipt 或 None")


@dataclass(frozen=True, slots=True)
class ControlResponseBodies:
    """Carry provider-localized response bodies for accepted and idle control."""

    interrupted: str
    idle: str

    def __post_init__(self) -> None:
        _content_string(self.interrupted, "interrupted")
        _content_string(self.idle, "idle")


class ChannelControlPort(Protocol):
    async def interrupt(
        self,
        raw: RawInbound,
        *,
        response_bodies: ControlResponseBodies,
    ) -> ControlReceipt: ...


class TurnStreamEventKind(StrEnum):
    TURN_STARTED = "turn.started"
    STREAM_DELTA = "stream.delta"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TURN_OUTPUT_COMPLETED = "turn.output.completed"


@dataclass(frozen=True, slots=True)
class TurnStartedPresentation:
    turn_id: str
    client_message_id: str

    def __post_init__(self) -> None:
        _text(self.turn_id, "turn_id")
        _message_id(self.client_message_id)


@dataclass(frozen=True, slots=True)
class StreamDeltaPresentation:
    turn_id: str
    sequence: int
    text_delta: str
    reasoning_delta: str

    def __post_init__(self) -> None:
        _text(self.turn_id, "turn_id")
        _positive_sequence(self.sequence)
        _content_string(self.text_delta, "text_delta")
        _content_string(self.reasoning_delta, "reasoning_delta")


@dataclass(frozen=True, slots=True)
class ToolPresentation:
    turn_id: str
    sequence: int
    tool_call_id: str
    tool_name: str

    def __post_init__(self) -> None:
        _text(self.turn_id, "turn_id")
        _positive_sequence(self.sequence)
        _text(self.tool_call_id, "tool_call_id")
        _text(self.tool_name, "tool_name")


@dataclass(frozen=True, slots=True)
class TurnOutputCompletedPresentation:
    turn_id: str
    sequence: int

    def __post_init__(self) -> None:
        _text(self.turn_id, "turn_id")
        _positive_sequence(self.sequence)


TurnStreamPayload: TypeAlias = (
    TurnStartedPresentation
    | StreamDeltaPresentation
    | ToolPresentation
    | TurnOutputCompletedPresentation
)


@dataclass(frozen=True, slots=True)
class TurnStreamEvent:
    """Freeze one typed turn presentation event before provider callbacks."""

    presentation_id: str
    kind: TurnStreamEventKind
    payload: TurnStreamPayload

    def __post_init__(self) -> None:
        _text(self.presentation_id, "presentation_id")
        if not isinstance(self.kind, TurnStreamEventKind):
            raise TypeError("kind 必须是 TurnStreamEventKind")
        expected: type[object]
        if self.kind is TurnStreamEventKind.TURN_STARTED:
            expected = TurnStartedPresentation
        elif self.kind is TurnStreamEventKind.STREAM_DELTA:
            expected = StreamDeltaPresentation
        elif self.kind in {
            TurnStreamEventKind.TOOL_STARTED,
            TurnStreamEventKind.TOOL_COMPLETED,
        }:
            expected = ToolPresentation
        else:
            expected = TurnOutputCompletedPresentation
        if not isinstance(self.payload, expected):
            raise TypeError(
                f"{self.kind.value} payload 必须是 {expected.__name__}"
            )


@dataclass(frozen=True, slots=True)
class PresentationReceipt:
    """Report one settled remote preview callback."""

    presentation_id: str
    status: DeliveryStatus
    provider_ids: tuple[str, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        _text(self.presentation_id, "presentation_id")
        if not isinstance(self.status, DeliveryStatus):
            raise TypeError("status 必须是 DeliveryStatus")
        object.__setattr__(
            self,
            "provider_ids",
            _text_tuple(self.provider_ids, "provider_ids"),
        )
        if self.error is not None:
            _text(self.error, "error")


TurnStreamCallback: TypeAlias = Callable[
    [TurnStreamEvent], Awaitable[PresentationReceipt]
]


class StreamSubscription(Protocol):
    def close_admission(self) -> None: ...

    async def await_quiescence(self) -> None: ...

    async def close(self) -> None: ...


class TurnStreamPort(Protocol):
    def subscribe(self, callback: TurnStreamCallback) -> StreamSubscription: ...


@dataclass(frozen=True, slots=True)
class ChannelPresentationPorts:
    """Expose only the presentation capabilities declared by one binding."""

    control: ChannelControlPort | None
    turn_stream: TurnStreamPort | None


@dataclass(frozen=True, slots=True)
class ChannelRuntimePorts:
    """Expose exact formal inbound ports without exposing the factory context."""

    snapshot_id: str
    generation_id: str
    binding_token: str
    ingress: ChannelIngressPort | None
    identity: ChannelIdentityPort | None
    attachment_import: ChannelAttachmentImportPort | None
    durable_inbound: ChannelDurableInboundPort | None = None

    def __post_init__(self) -> None:
        _text(self.snapshot_id, "snapshot_id")
        _text(self.generation_id, "generation_id")
        _text(self.binding_token, "binding_token")
        for name, value, method in (
            ("ingress", self.ingress, "admit"),
            ("identity", self.identity, "resolve"),
            ("attachment_import", self.attachment_import, "import_bytes"),
            ("durable_inbound", self.durable_inbound, "recover"),
        ):
            if value is not None and not callable(getattr(value, method, None)):
                raise TypeError(f"channel runtime {name} 必须提供 {method}(...)")
        if self.durable_inbound is not None:
            for method in (
                "reserve",
                "defer",
                "settle_rejected",
                "has_pending",
                "pending_attachment_refs",
                "recover",
            ):
                if not callable(getattr(self.durable_inbound, method, None)):
                    raise TypeError(
                        f"channel runtime durable_inbound 必须提供 {method}(...)"
                    )


@dataclass(frozen=True, slots=True)
class QueuedReceipt:
    """Represent queue admission separately from a settled delivery receipt."""

    delivery_id: str
    queued: bool

    def __post_init__(self) -> None:
        _text(self.delivery_id, "delivery_id")
        if not isinstance(self.queued, bool):
            raise TypeError("queued 必须是 bool")


@dataclass(frozen=True, slots=True)
class PushToolRequest:
    """Carry a direct push request before it is converted to an outbound envelope."""

    channel: str
    recipient: str
    body: str
    metadata: Mapping[str, JsonValue]
    attachments: tuple[AttachmentRef, ...] = ()

    def __post_init__(self) -> None:
        _text(self.channel, "channel")
        _text(self.recipient, "recipient")
        _content_string(self.body, "body")
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))
        object.__setattr__(
            self,
            "attachments",
            _attachment_refs(self.attachments, "attachments"),
        )


class ChannelTaskSpawner(Protocol):
    """仅在 adapter.start 内登记所属 Fiber 的后台任务。"""

    async def __call__[T](self, coroutine: Coroutine[Any, Any, T], *, name: str) -> asyncio.Task[T]: ...


@dataclass(frozen=True, slots=True)
class ChannelFactoryContext:
    snapshot_id: str
    generation_id: str
    boot_id: str
    binding_token: str
    config: Mapping[str, object]
    ingress: ChannelIngressPort | None
    identity: ChannelIdentityPort | None
    attachment_import: ChannelAttachmentImportPort | None = None
    attachment_read: ChannelAttachmentReadPort | None = None
    control: ChannelControlPort | None = None
    turn_stream: TurnStreamPort | None = None
    data_root: Path | None = None
    open_scope: Callable[[], AbstractAsyncContextManager[RequestContext]] | None = None
    spawn_owned: ChannelTaskSpawner | None = None

    def __post_init__(self) -> None:
        _text(self.snapshot_id, "snapshot_id")
        _text(self.generation_id, "generation_id")
        _text(self.boot_id, "boot_id")
        _text(self.binding_token, "binding_token")
        config = _freeze_channel_config(self.config)
        if not isinstance(config, Mapping):
            raise TypeError("channel factory config 必须是 mapping")
        if self.ingress is not None and not callable(
            getattr(self.ingress, "admit", None)
        ):
            raise TypeError("channel factory ingress 必须提供 admit(raw)")
        if self.identity is not None and not callable(
            getattr(self.identity, "resolve", None)
        ):
            raise TypeError("channel factory identity 必须提供 resolve(identity)")
        if self.attachment_import is not None and not callable(
            getattr(self.attachment_import, "import_bytes", None)
        ):
            raise TypeError(
                "channel factory attachment_import 必须提供 import_bytes(data, ...)"
            )
        if self.attachment_read is not None and not callable(
            getattr(self.attachment_read, "acquire", None)
        ):
            raise TypeError(
                "channel factory attachment_read 必须提供 acquire(ref)"
            )
        if self.control is not None and not callable(
            getattr(self.control, "interrupt", None)
        ):
            raise TypeError("channel factory control 必须提供 interrupt(raw, ...)")
        if self.turn_stream is not None and not callable(
            getattr(self.turn_stream, "subscribe", None)
        ):
            raise TypeError("channel factory turn_stream 必须提供 subscribe(callback)")
        object.__setattr__(self, "config", config)


@dataclass(frozen=True, slots=True)
class ChannelReady:
    binding_token: str
    subscriptions: tuple[str, ...] = ()
    admission_open: bool = False

    def __post_init__(self) -> None:
        _text(self.binding_token, "binding_token")
        object.__setattr__(self, "subscriptions", _text_tuple(self.subscriptions, "subscriptions"))
        if not isinstance(self.admission_open, bool):
            raise TypeError("admission_open 必须是 bool")


@dataclass(frozen=True, slots=True)
class ChannelCleanupFailure:
    stage: str
    plugin_id: str
    generation_id: str
    binding_token: str
    resource: str
    error_type: str
    message: str
    retry_action: str

    def __post_init__(self) -> None:
        for field_name in (
            "stage",
            "plugin_id",
            "generation_id",
            "binding_token",
            "resource",
            "error_type",
            "message",
            "retry_action",
        ):
            _text(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class StopReceipt:
    binding_token: str
    resources_closed: bool
    failures: tuple[ChannelCleanupFailure, ...] = ()

    def __post_init__(self) -> None:
        _text(self.binding_token, "binding_token")
        if not isinstance(self.resources_closed, bool):
            raise TypeError("resources_closed 必须是 bool")
        if not isinstance(self.failures, tuple) or any(
            not isinstance(item, ChannelCleanupFailure) for item in self.failures
        ):
            raise TypeError("failures 必须是 ChannelCleanupFailure tuple")


@dataclass(frozen=True, slots=True)
class ProviderDeliveryRequest:
    binding_token: str
    delivery_id: str
    recipient: str
    body: str
    attachments: tuple[AttachmentRef, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    commit_role: ChannelCommitRole = ChannelCommitRole.DIRECT
    thinking: str | None = None
    reply_to: str | None = None
    session_message_id: str | None = None
    control_turn_id: str | None = None
    execution_attempt_id: str | None = None
    terminal_status: ChannelTerminalStatus | None = None

    def __post_init__(self) -> None:
        _text(self.binding_token, "binding_token")
        _text(self.delivery_id, "delivery_id")
        _text(self.recipient, "recipient")
        _content_string(self.body, "body")
        object.__setattr__(
            self,
            "attachments",
            _attachment_refs(self.attachments, "attachments"),
        )
        metadata = _freeze_json_mapping(self.metadata)
        object.__setattr__(self, "metadata", metadata)
        if not isinstance(self.commit_role, ChannelCommitRole):
            raise TypeError("commit_role 必须是 ChannelCommitRole")
        if self.thinking is not None:
            _content_string(self.thinking, "thinking")
        for field_name in (
            "reply_to",
            "session_message_id",
            "control_turn_id",
            "execution_attempt_id",
        ):
            _optional_string(getattr(self, field_name), field_name)
        if self.terminal_status is not None and not isinstance(
            self.terminal_status,
            ChannelTerminalStatus,
        ):
            raise TypeError("terminal_status 必须是 ChannelTerminalStatus 或 None")


@dataclass(frozen=True, slots=True)
class ProviderDeliveryReceipt:
    delivery_id: str
    status: DeliveryStatus
    provider_ids: tuple[str, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        _text(self.delivery_id, "delivery_id")
        if not isinstance(self.status, DeliveryStatus):
            raise TypeError("status 必须是 DeliveryStatus")
        object.__setattr__(self, "provider_ids", _text_tuple(self.provider_ids, "provider_ids"))
        if self.error is not None:
            _text(self.error, "error")


class ChannelAdapter(Protocol):
    async def start(self) -> ChannelReady: ...

    async def deliver(self, request: ProviderDeliveryRequest) -> ProviderDeliveryReceipt: ...

    async def stop(self) -> StopReceipt: ...


class ChannelRuntimeAdapter(ChannelAdapter, Protocol):
    """Optional lifecycle seam for provider callbacks owned by a formal binding."""

    def attach_runtime(self, ports: ChannelRuntimePorts) -> None: ...

    def open_admission(self) -> None: ...

    def close_admission(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ChannelDefinition:
    """Describe one plugin-owned channel factory without opening it."""

    name: str
    capabilities: frozenset[ChannelCapability]
    factory: Callable[[ChannelFactoryContext], ChannelAdapter]
    inbound_identity: InboundIdentity | None
    config: Mapping[str, object] = field(default_factory=dict)
    interrupt: Callable[[RawInbound], Awaitable[bool]] | None = None
    optional_services: frozenset[ServiceKey[Any]] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError(f"channel name 无效: {self.name}")
        if not isinstance(self.capabilities, frozenset) or not self.capabilities:
            raise ValueError("channel capabilities 必须是非空 frozenset")
        if any(not isinstance(item, ChannelCapability) for item in self.capabilities):
            raise ValueError("channel capabilities 必须只包含 ChannelCapability")
        if not callable(self.factory):
            raise TypeError("channel factory 必须可调用")
        if not isinstance(self.optional_services, frozenset) or any(
            not isinstance(key, ServiceKey) for key in self.optional_services
        ):
            raise TypeError("optional_services 必须是 ServiceKey frozenset")
        object.__setattr__(self, "config", _freeze_channel_config(self.config))
        if ChannelCapability.CONTROL in self.capabilities and not callable(self.interrupt):
            raise TypeError("CONTROL channel 必须提供自己的 interrupt 回调")
        has_inbound = ChannelCapability.INBOUND in self.capabilities
        if has_inbound and not isinstance(self.inbound_identity, InboundIdentity):
            raise ValueError("inbound channel 必须声明 inbound_identity")
        if not has_inbound and self.inbound_identity is not None:
            raise ValueError("非 inbound channel 不得声明 inbound_identity")
        if ChannelCapability.DURABLE_INBOUND in self.capabilities and (
            not has_inbound
            or self.inbound_identity is not InboundIdentity.PROVIDER_MESSAGE_ID
        ):
            raise ValueError(
                "durable inbound channel 必须同时声明 INBOUND/PROVIDER_MESSAGE_ID"
            )


class Channels(Protocol):
    """普通 provider 的贡献与原连接租约；目录不复制到 Snapshot。"""

    async def register(self, ctx: Context, definition: ChannelDefinition) -> None: ...

    def acquire_binding(self, channel_name: str) -> ChannelBindingLease: ...

    async def dispatch_outbound(
        self, envelope: OutboundEnvelope, binding: ChannelBindingLease,
    ) -> ChannelDeliveryReceipt: ...

    async def recover_inbound(self, raw: RawInbound) -> bool: ...


CHANNELS = ServiceKey[Channels]("plugin.channels")


def _freeze_channel_config(value: object, *, seen: frozenset[int] = frozenset()) -> object:
    if value is None or isinstance(value, (bool, int, str, CredentialRef)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("channel factory config 不接受非有限 float")
        return value
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen:
            raise ValueError("channel factory config 不接受 cycle")
        next_seen = seen | {marker}
        result: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("channel factory config mapping key 必须是 str")
            result[key] = _freeze_channel_config(value[key], seen=next_seen)
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in seen:
            raise ValueError("channel factory config 不接受 cycle")
        next_seen = seen | {marker}
        return tuple(_freeze_channel_config(item, seen=next_seen) for item in value)
    raise TypeError(f"channel factory config 值类型无效: {type(value).__name__}")


def _freeze_json_mapping(value: object) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata 必须是 mapping")
    frozen = _freeze_json_value(value)
    assert isinstance(frozen, Mapping)
    return frozen


def _freeze_json_value(
    value: object,
    *,
    seen: frozenset[int] = frozenset(),
) -> JsonValue:
    """Freeze JSON-shaped metadata and reject mutable or non-finite values."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata 不接受非有限 float")
        return value
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen:
            raise ValueError("metadata 不接受 cycle")
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("metadata mapping key 必须是 str")
        next_seen = seen | {marker}
        result = {
            key: _freeze_json_value(value[key], seen=next_seen)
            for key in sorted(keys)
        }
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in seen:
            raise ValueError("metadata 不接受 cycle")
        next_seen = seen | {marker}
        return tuple(_freeze_json_value(item, seen=next_seen) for item in value)
    raise TypeError(f"metadata 值类型无效: {type(value).__name__}")


def _attachment_refs(value: object, field_name: str) -> tuple[AttachmentRef, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} 必须是 tuple")
    result = tuple(value)
    if any(not isinstance(item, AttachmentRef) for item in result):
        raise TypeError(f"{field_name} 必须只包含 AttachmentRef")
    return result


def _text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} 必须是 tuple")
    result = tuple(_text(item, field_name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} 不能重复")
    return result


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} 必须是非空且无首尾空白的字符串")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field_name} 不能包含控制字符")
    return value


def _message_id(value: object) -> str:
    result = _text(value, "message_id")
    if len(result) > 256:
        raise ValueError("message_id 长度必须在 1～256 字符")
    return result


def _positive_sequence(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("sequence 必须是正整数")
    return value


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是 str")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field_name} 不能包含控制字符")
    return value


def _content_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是 str")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
        raise ValueError(f"{field_name} 不能包含控制字符")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _string(value, field_name)


__all__ = [
    "CHANNELS",
    "Channels",
    "ChannelAdapter",
    "ChannelCapability",
    "ChannelCommitRole",
    "ChannelAttachmentImportPort",
    "ChannelAttachmentReadPort",
    "ChannelCleanupFailure",
    "ChannelControlPort",
    "ChannelDeliveryReceipt",
    "ChannelFactoryContext",
    "ChannelDurableInboundPort",
    "ChannelIngressPort",
    "DURABLE_ATTACHMENT_REFS",
    "DURABLE_HANDOFF_ID",
    "DURABLE_INBOUND_MARKER",
    "DURABLE_PROVIDER_MESSAGE_ID",
    "ChannelIdentityPort",
    "ChannelReady",
    "ChannelTerminalStatus",
    "ChannelPresentationPorts",
    "ChannelInboundMessage",
    "AttachmentKind",
    "AttachmentReadLease",
    "AttachmentRef",
    "DeliveryStatus",
    "ControlReceipt",
    "ControlResponseBodies",
    "ChannelDefinition",
    "InboundEnvelope",
    "InboundIdentity",
    "JsonValue",
    "OutboundEnvelope",
    "ProviderDeliveryReceipt",
    "ProviderDeliveryRequest",
    "PushToolRequest",
    "QueuedReceipt",
    "RawInbound",
    "PresentationReceipt",
    "StreamDeltaPresentation",
    "StreamSubscription",
    "StopReceipt",
    "ToolPresentation",
    "TurnOutputCompletedPresentation",
    "TurnStartedPresentation",
    "TurnStreamCallback",
    "TurnStreamEvent",
    "TurnStreamEventKind",
    "TurnStreamPayload",
    "TurnStreamPort",
]
