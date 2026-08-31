from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias

from agent.plugin_composition.context import Context, FiberHandle, HealthHandle
from agent.plugin_composition.model import CompositionError, IncidentView, ServiceKey


_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_FACTORY_EXPORT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:]*$")
_ATTACHMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_MEDIA_TYPE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
    OUTBOUND = "outbound"
    CONTROL = "control"
    TURN_STREAM = "turn_stream"


class InboundIdentity(StrEnum):
    PROVIDER_MESSAGE_ID = "provider_message_id"


class DeliveryStatus(StrEnum):
    DELIVERED = "delivered"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class ChannelCommitRole(StrEnum):
    DIRECT = "direct"
    PASSIVE = "passive"


class ChannelTerminalStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class AttachmentKind(StrEnum):
    FILE = "file"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class AttachmentRef:
    """Identify one immutable Core-owned attachment without exposing its path."""

    artifact_id: str
    kind: AttachmentKind
    filename: str | None
    media_type: str | None
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _attachment_id(self.artifact_id)
        if not isinstance(self.kind, AttachmentKind):
            raise TypeError("kind 必须是 AttachmentKind")
        _attachment_filename(self.filename)
        _attachment_media_type(self.media_type)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes 必须是 int")
        if self.size_bytes < 0:
            raise ValueError("size_bytes 不能是负数")
        if not isinstance(self.sha256, str):
            raise TypeError("sha256 必须是 str")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("sha256 必须是 64 位小写十六进制字符串")


JsonValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["JsonValue", ...]
    | Mapping[str, "JsonValue"]
)


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


class InboundOwner(StrEnum):
    INGRESS = "ingress"
    BUS = "bus"
    LANE = "lane"
    LOOP = "loop"
    CLOSED = "closed"


class InboundState(StrEnum):
    ADMITTED = "admitted"
    BUS_QUEUED = "bus_queued"
    LANE_QUEUED = "lane_queued"
    RUNNING = "running"
    TERMINAL = "terminal"


class ChannelBindingLease(Protocol):
    @property
    def snapshot_lease(self) -> object: ...

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

    async def aclose(self) -> None: ...


class ChannelIngressPort(Protocol):
    async def admit(self, raw: RawInbound) -> bool: ...


class ChannelRecoveryIngressPort(Protocol):
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


class AttachmentReadLease(Protocol):
    @property
    def ref(self) -> AttachmentRef: ...

    async def read_bytes(self, *, max_bytes: int) -> bytes: ...

    async def aclose(self) -> None: ...


class ChannelAttachmentReadPort(Protocol):
    async def acquire(self, ref: AttachmentRef) -> AttachmentReadLease: ...


@dataclass(slots=True, kw_only=True)
class InboundEnvelope:
    """Own one inbound message lease through its fixed Core processing path."""

    message_id: str
    snapshot_id: str
    generation_id: str
    binding_token: str
    message: ChannelInboundMessage
    lease: ChannelBindingLease
    state: InboundState = InboundState.ADMITTED
    owner: InboundOwner = InboundOwner.INGRESS
    _closed_by: InboundOwner | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _message_id(self.message_id)
        _text(self.snapshot_id, "snapshot_id")
        _text(self.generation_id, "generation_id")
        _text(self.binding_token, "binding_token")
        if not isinstance(self.message, ChannelInboundMessage):
            raise TypeError("message 必须是 ChannelInboundMessage")
        _validate_binding_lease(self.lease)
        if not isinstance(self.state, InboundState):
            raise TypeError("state 必须是 InboundState")
        if not isinstance(self.owner, InboundOwner):
            raise TypeError("owner 必须是 InboundOwner")
        if (self.owner, self.state) not in _INBOUND_STATES:
            raise ValueError("InboundEnvelope owner/state 组合无效")
        if self.state is InboundState.TERMINAL or self.owner is InboundOwner.CLOSED:
            raise ValueError("terminal envelope 只能由 close() 产生")
        if self.lease.snapshot_id != self.snapshot_id:
            raise ValueError("InboundEnvelope snapshot_id 与 lease 不一致")
        if self.lease.generation_id != self.generation_id:
            raise ValueError("InboundEnvelope generation_id 与 lease 不一致")
        if self.lease.binding_token != self.binding_token:
            raise ValueError("InboundEnvelope binding_token 与 lease 不一致")
        if self.lease.channel_name != self.message.channel:
            raise ValueError("InboundEnvelope channel 与 lease 不一致")

    @property
    def channel(self) -> str:
        return self.message.channel

    @property
    def sender(self) -> str:
        return self.message.sender

    @property
    def chat_id(self) -> str:
        return self.message.chat_id

    @property
    def content(self) -> str:
        return self.message.content

    @property
    def timestamp(self) -> datetime:
        return self.message.timestamp

    @property
    def metadata(self) -> Mapping[str, JsonValue]:
        return self.message.metadata

    @property
    def session_key(self) -> str:
        override = self.message.metadata.get("session_key_override")
        if isinstance(override, str) and override.strip():
            return override.strip()
        return f"{self.channel}:{self.chat_id}"

    def handoff(
        self,
        expected_owner: InboundOwner,
        next_owner: InboundOwner,
    ) -> InboundEnvelope:
        """Transfer the sole close owner along the fixed inbound path."""

        if not isinstance(expected_owner, InboundOwner):
            raise TypeError("expected_owner 必须是 InboundOwner")
        if not isinstance(next_owner, InboundOwner):
            raise TypeError("next_owner 必须是 InboundOwner")
        if self._closed_by is not None or self.state is InboundState.TERMINAL:
            raise CompositionError(
                "INBOUND_ENVELOPE_TERMINAL",
                "terminal inbound envelope 不能 handoff",
            )
        if self.owner is not expected_owner:
            raise CompositionError(
                "INBOUND_ENVELOPE_OWNER_MISMATCH",
                f"inbound envelope 当前 owner 是 {self.owner.value}，不是 {expected_owner.value}",
            )
        transition = _INBOUND_TRANSITIONS.get((expected_owner, self.state))
        if transition is not next_owner:
            raise CompositionError(
                "INBOUND_ENVELOPE_INVALID_HANDOFF",
                f"不能从 {expected_owner.value}/{self.state.value} 转移到 {next_owner.value}",
            )
        self.owner = next_owner
        self.state = _INBOUND_STATE_BY_OWNER[next_owner]
        return self

    async def close(self, expected_owner: InboundOwner) -> None:
        """Release the exact lease before publishing terminal state."""

        if not isinstance(expected_owner, InboundOwner):
            raise TypeError("expected_owner 必须是 InboundOwner")
        if self._closed_by is not None:
            if expected_owner is not self._closed_by:
                raise CompositionError(
                    "INBOUND_ENVELOPE_CLOSE_OWNER_MISMATCH",
                    "inbound envelope 已由另一 owner close",
                )
            return
        if self.owner is not expected_owner:
            raise CompositionError(
                "INBOUND_ENVELOPE_OWNER_MISMATCH",
                f"inbound envelope 当前 owner 是 {self.owner.value}，不是 {expected_owner.value}",
            )
        cancelled = await _close_lease_critically(self.lease)
        self._closed_by = expected_owner
        self.state = InboundState.TERMINAL
        self.owner = InboundOwner.CLOSED
        if cancelled:
            raise asyncio.CancelledError


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
    recovery_ingress: ChannelRecoveryIngressPort | None = None

    def __post_init__(self) -> None:
        _text(self.snapshot_id, "snapshot_id")
        _text(self.generation_id, "generation_id")
        _text(self.binding_token, "binding_token")
        for name, value, method in (
            ("ingress", self.ingress, "admit"),
            ("identity", self.identity, "resolve"),
            ("attachment_import", self.attachment_import, "import_bytes"),
            ("recovery_ingress", self.recovery_ingress, "recover"),
        ):
            if value is not None and not callable(getattr(value, method, None)):
                raise TypeError(f"channel runtime {name} 必须提供 {method}(...)")


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


@dataclass(frozen=True, slots=True)
class CredentialRef:
    """Opaque credential path; it never contains or resolves secret bytes."""

    path: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.path, tuple) or not self.path:
            raise ValueError("CredentialRef.path 必须是非空 tuple")
        for segment in self.path:
            if (
                not isinstance(segment, str)
                or not segment
                or segment.strip() != segment
                or segment in {".", ".."}
                or "/" in segment
                or "\\" in segment
                or "\x00" in segment
            ):
                raise ValueError("CredentialRef.path 包含非法段")


class ProviderClient(Protocol):
    def credential(self, ref: CredentialRef) -> str: ...

    async def aclose(self) -> None: ...


class ProviderClientFactory(Protocol):
    async def create(
        self,
        credentials: Mapping[str, CredentialRef],
    ) -> ProviderClient: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ChannelFactoryContext:
    snapshot_id: str
    generation_id: str
    binding_token: str
    config: Mapping[str, object]
    credentials: Mapping[str, CredentialRef]
    provider_client_factory: ProviderClientFactory
    ingress: ChannelIngressPort | None
    identity: ChannelIdentityPort | None
    attachment_import: ChannelAttachmentImportPort | None = None
    attachment_read: ChannelAttachmentReadPort | None = None
    control: ChannelControlPort | None = None
    turn_stream: TurnStreamPort | None = None

    def __post_init__(self) -> None:
        _text(self.snapshot_id, "snapshot_id")
        _text(self.generation_id, "generation_id")
        _text(self.binding_token, "binding_token")
        config = _freeze_channel_config(self.config)
        if not isinstance(config, Mapping):
            raise TypeError("channel factory config 必须是 mapping")
        credentials = _credential_refs(self.credentials)
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
        object.__setattr__(self, "credentials", credentials)


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
    factory_export: str
    inbound_identity: InboundIdentity | None
    credential_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError(f"channel name 无效: {self.name}")
        if not isinstance(self.capabilities, frozenset) or not self.capabilities:
            raise ValueError("channel capabilities 必须是非空 frozenset")
        if any(not isinstance(item, ChannelCapability) for item in self.capabilities):
            raise ValueError("channel capabilities 必须只包含 ChannelCapability")
        if (
            not isinstance(self.factory_export, str)
            or _FACTORY_EXPORT.fullmatch(self.factory_export) is None
            or ".." in self.factory_export
            or self.factory_export.endswith((".", ":"))
        ):
            raise ValueError(f"channel factory_export 无效: {self.factory_export}")
        has_inbound = ChannelCapability.INBOUND in self.capabilities
        if has_inbound and not isinstance(self.inbound_identity, InboundIdentity):
            raise ValueError("inbound channel 必须声明 inbound_identity")
        if not has_inbound and self.inbound_identity is not None:
            raise ValueError("非 inbound channel 不得声明 inbound_identity")
        object.__setattr__(self, "credential_paths", _credential_paths(self.credential_paths))


@dataclass(frozen=True, slots=True)
class CoreChannelDefinition:
    """Describe one Core-owned channel projection without opening provider state."""

    name: str
    capabilities: frozenset[ChannelCapability]
    factory: Callable[[ChannelFactoryContext], ChannelAdapter]
    inbound_identity: InboundIdentity | None
    source_revision: str
    config_revision: str
    generation_id: str
    credential_paths: tuple[str, ...] = ()
    factory_export: str = ""
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError(f"core channel name 无效: {self.name}")
        if not isinstance(self.capabilities, frozenset) or not self.capabilities:
            raise ValueError("core channel capabilities 必须是非空 frozenset")
        if any(not isinstance(item, ChannelCapability) for item in self.capabilities):
            raise ValueError("core channel capabilities 必须只包含 ChannelCapability")
        if not callable(self.factory):
            raise TypeError("core channel factory 必须可调用")
        has_inbound = ChannelCapability.INBOUND in self.capabilities
        if has_inbound and not isinstance(self.inbound_identity, InboundIdentity):
            raise ValueError("inbound core channel 必须声明 inbound_identity")
        if not has_inbound and self.inbound_identity is not None:
            raise ValueError("非 inbound core channel 不得声明 inbound_identity")
        factory_export = self.factory_export or (
            "core."
            + self.name.replace("-", "_")
            + ".factory"
        )
        if (
            not isinstance(factory_export, str)
            or _FACTORY_EXPORT.fullmatch(factory_export) is None
            or ".." in factory_export
            or factory_export.endswith((".", ":"))
        ):
            raise ValueError(f"core channel factory_export 无效: {factory_export}")
        for field_name in ("source_revision", "config_revision", "generation_id"):
            if not isinstance(getattr(self, field_name), str) or not getattr(
                self, field_name
            ):
                raise ValueError(f"core channel {field_name} 必须是非空字符串")
        if not isinstance(self.config, Mapping):
            raise TypeError("core channel config 必须是 mapping")
        object.__setattr__(
            self,
            "credential_paths",
            _credential_paths(self.credential_paths, allow_empty=True),
        )
        frozen_config = _freeze_channel_config(self.config)
        if not isinstance(frozen_config, Mapping):
            raise TypeError("core channel config 必须是 mapping")
        object.__setattr__(self, "config", frozen_config)
        object.__setattr__(self, "factory_export", factory_export)

    @property
    def descriptor(self) -> "ChannelDescriptor":
        """Project this Core definition into the common immutable descriptor."""

        return ChannelDescriptor(
            owner="core",
            name=self.name,
            capabilities=tuple(
                sorted(self.capabilities, key=lambda item: item.value)
            ),
            factory_export=self.factory_export,
            inbound_identity=self.inbound_identity,
            credential_paths=self.credential_paths,
        )

    @property
    def provenance(self) -> "ChannelFactoryProvenance":
        """Return the stable provenance identity used by a committed catalog."""

        return ChannelFactoryProvenance(
            plugin_id="core",
            generation_id=self.generation_id,
            channel_name=self.name,
            source_revision=self.source_revision,
            config_revision=(
                f"{self.config_revision}:{channel_config_revision(self.config)}"
            ),
            factory_export=self.factory_export,
        )


@dataclass(frozen=True, slots=True)
class ChannelDescriptor:
    owner: str
    name: str
    capabilities: tuple[ChannelCapability, ...]
    factory_export: str
    inbound_identity: InboundIdentity | None
    credential_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.owner, "owner")
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError(f"channel descriptor name 无效: {self.name}")
        if not self.capabilities or any(
            not isinstance(item, ChannelCapability) for item in self.capabilities
        ):
            raise ValueError("channel descriptor capabilities 类型无效")
        if tuple(sorted(self.capabilities, key=lambda item: item.value)) != self.capabilities:
            raise ValueError("channel descriptor capabilities 顺序必须 canonical")
        if (
            not isinstance(self.factory_export, str)
            or _FACTORY_EXPORT.fullmatch(self.factory_export) is None
            or ".." in self.factory_export
            or self.factory_export.endswith((".", ":"))
        ):
            raise ValueError("channel descriptor factory_export 无效")
        has_inbound = ChannelCapability.INBOUND in self.capabilities
        if has_inbound and not isinstance(self.inbound_identity, InboundIdentity):
            raise ValueError("inbound channel descriptor 必须声明 inbound_identity")
        if not has_inbound and self.inbound_identity is not None:
            raise ValueError("非 inbound channel descriptor 不得声明 inbound_identity")
        object.__setattr__(
            self,
            "credential_paths",
            _credential_paths(
                self.credential_paths,
                allow_empty=self.owner == "core",
            ),
        )


@dataclass(frozen=True, slots=True)
class ChannelFactoryProvenance:
    plugin_id: str
    generation_id: str
    channel_name: str
    source_revision: str
    config_revision: str
    factory_export: str

    def __post_init__(self) -> None:
        _text(self.plugin_id, "plugin_id")
        _text(self.generation_id, "generation_id")
        if not isinstance(self.channel_name, str) or _NAME.fullmatch(self.channel_name) is None:
            raise ValueError(f"factory provenance channel_name 无效: {self.channel_name}")
        if not isinstance(self.source_revision, str):
            raise ValueError("source_revision 必须是字符串")
        if not isinstance(self.config_revision, str):
            raise ValueError("config_revision 必须是字符串")
        if not isinstance(self.factory_export, str) or _FACTORY_EXPORT.fullmatch(self.factory_export) is None:
            raise ValueError("factory provenance factory_export 无效")


@dataclass(frozen=True, slots=True)
class ChannelFactoryFreezeInput:
    """Core-only input carrying source/config provenance into a freeze."""

    generation_id: str
    source_revision: str = ""
    config_revision: str = ""

    def __post_init__(self) -> None:
        _text(self.generation_id, "generation_id")
        if not isinstance(self.source_revision, str):
            raise ValueError("source_revision 必须是字符串")
        if not isinstance(self.config_revision, str):
            raise ValueError("config_revision 必须是字符串")


@dataclass(frozen=True, slots=True)
class ChannelRegistrySnapshot:
    descriptors: tuple[ChannelDescriptor, ...]
    factories: tuple[ChannelFactoryProvenance, ...]
    identity: str
    root_instance_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        descriptors = tuple(self.descriptors)
        factories = tuple(self.factories)
        if any(not isinstance(item, ChannelDescriptor) for item in descriptors):
            raise TypeError("channel registry descriptor 类型无效")
        if any(not isinstance(item, ChannelFactoryProvenance) for item in factories):
            raise TypeError("channel registry factory provenance 类型无效")
        if len({item.name for item in descriptors}) != len(descriptors):
            raise ValueError("channel registry descriptor 名称重复")
        factory_keys = tuple(_factory_sort_key(item) for item in factories)
        if len(set(factory_keys)) != len(factory_keys):
            raise ValueError("channel registry factory provenance 重复")
        if tuple(sorted(descriptors, key=lambda item: item.name)) != descriptors:
            raise ValueError("channel registry descriptors 必须按 name 排序")
        if tuple(sorted(factories, key=_factory_sort_key)) != factories:
            raise ValueError("channel registry factories 必须按 provenance 排序")
        if self.identity != _registry_identity(descriptors, factories):
            raise ValueError("channel registry identity 与内容不匹配")
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "factories", factories)


@dataclass(frozen=True, slots=True)
class CommittedChannelCatalog:
    """Own the immutable merge of Core definitions and the plugin registry."""

    plugin_registry: ChannelRegistrySnapshot | None = None
    core_definitions: tuple[CoreChannelDefinition, ...] = ()
    root_instance_token: object | None = field(default=None, repr=False, compare=False)
    registry: ChannelRegistrySnapshot = field(init=False)

    def __post_init__(self) -> None:
        raw_definitions = tuple(self.core_definitions)
        if any(not isinstance(item, CoreChannelDefinition) for item in raw_definitions):
            raise TypeError("core_definitions 必须只包含 CoreChannelDefinition")
        definitions = tuple(sorted(raw_definitions, key=lambda item: item.name))
        core_names = tuple(item.name for item in definitions)
        if len(set(core_names)) != len(core_names):
            raise ValueError("Core channel 名称重复")

        plugin_registry = self.plugin_registry
        root_token = self.root_instance_token
        if plugin_registry is not None:
            if not isinstance(plugin_registry, ChannelRegistrySnapshot):
                raise TypeError("plugin_registry 类型无效")
            if root_token is not None and root_token is not plugin_registry.root_instance_token:
                raise ValueError("CommittedChannelCatalog root token 与 plugin registry 不一致")
            root_token = plugin_registry.root_instance_token
            plugin_names = {item.name for item in plugin_registry.descriptors}
            collisions = sorted(plugin_names.intersection(core_names))
            if collisions:
                raise ValueError(
                    "Core channel 与 v3 plugin channel 名称冲突: "
                    + ", ".join(collisions)
                )
            plugin_descriptors = plugin_registry.descriptors
            plugin_factories = plugin_registry.factories
        else:
            if root_token is None:
                root_token = object()
            plugin_descriptors = ()
            plugin_factories = ()

        descriptors = tuple(
            sorted(
                tuple(item.descriptor for item in definitions) + plugin_descriptors,
                key=lambda item: item.name,
            )
        )
        factories = tuple(
            sorted(
                tuple(item.provenance for item in definitions) + plugin_factories,
                key=_factory_sort_key,
            )
        )
        registry = ChannelRegistrySnapshot(
            descriptors=descriptors,
            factories=factories,
            identity=_registry_identity(descriptors, factories),
            root_instance_token=root_token,
        )
        object.__setattr__(self, "core_definitions", definitions)
        object.__setattr__(self, "root_instance_token", root_token)
        object.__setattr__(self, "registry", registry)

    @property
    def identity(self) -> str:
        """Return the content identity of the merged committed registry."""

        return self.registry.identity

    @property
    def descriptors(self) -> tuple[ChannelDescriptor, ...]:
        """Expose the merged descriptor projection without a mutable registry."""

        return self.registry.descriptors

    def definition(self, channel_name: str) -> CoreChannelDefinition | None:
        """Resolve one Core definition by exact channel name."""

        for definition in self.core_definitions:
            if definition.name == channel_name:
                return definition
        return None


CHANNELS = ServiceKey["PluginChannels"]("core.channels")


@dataclass(slots=True)
class _ChannelRegistration:
    owner: str
    definition: ChannelDefinition
    descriptor: ChannelDescriptor
    owner_fiber: FiberHandle
    activation_token: object
    generation_id: str
    incident_reporter: Callable[[str, str], IncidentView]
    health: HealthHandle | None = None


class _ChannelDeclarations:
    """Own one Root-local declaration set until Core freezes it."""

    def __init__(self) -> None:
        self._registrations: dict[str, _ChannelRegistration] = {}
        self._frozen: ChannelRegistrySnapshot | None = None

    async def register(self, ctx: Context, definition: ChannelDefinition) -> None:
        """Validate and register one blueprint as Fiber-owned Effects."""

        normalized = _normalize_definition(definition)
        owner_fiber = ctx.fiber
        activation_token = owner_fiber.activation_token
        if activation_token is None:
            raise CompositionError(
                "INACTIVE_FIBER",
                f"{ctx.runtime.plugin_id} 当前 Fiber 没有 active activation",
            )
        registration: _ChannelRegistration | None = None

        def setup() -> Callable[[], None]:
            nonlocal registration
            registration, cleanup = self._register(
                ctx.runtime.plugin_id,
                normalized,
                owner_fiber,
                activation_token,
                ctx.generation_id,
                ctx.report_incident,
            )
            return cleanup

        registration_effect = await ctx.effect(
            setup,
            label=f"channel:{normalized.name}",
        )
        try:
            health = await ctx.health(f"channel:{normalized.name}", required=True)
        except BaseException:
            await registration_effect.aclose()
            raise
        assert registration is not None
        registration.health = health

    def freeze(
        self,
        root_instance_token: object,
        *,
        factory_provenance_by_owner: Mapping[
            str,
            ChannelFactoryFreezeInput | tuple[str, str, str],
        ]
        | None = None,
    ) -> ChannelRegistrySnapshot:
        """Freeze declarations with Core-supplied factory provenance."""

        if self._frozen is not None:
            if self._frozen.root_instance_token is not root_instance_token:
                raise RuntimeError("channel declaration registry 属于另一棵 Root")
            return self._frozen
        provenance = factory_provenance_by_owner or {}
        registrations = tuple(
            sorted(self._registrations.values(), key=lambda item: item.definition.name)
        )
        descriptors = tuple(item.descriptor for item in registrations)
        factories = tuple(
            sorted(
                (
                    _make_provenance(item, provenance.get(item.owner))
                    for item in registrations
                ),
                key=_factory_sort_key,
            )
        )
        snapshot = ChannelRegistrySnapshot(
            descriptors=descriptors,
            factories=factories,
            identity=_registry_identity(descriptors, factories),
            root_instance_token=root_instance_token,
        )
        self._frozen = snapshot
        return snapshot

    def _register(
        self,
        owner: str,
        definition: ChannelDefinition,
        owner_fiber: FiberHandle,
        activation_token: object,
        generation_id: str,
        incident_reporter: Callable[[str, str], IncidentView],
    ) -> tuple[_ChannelRegistration, Callable[[], None]]:
        if self._frozen is not None:
            raise CompositionError(
                "PLUGIN_CHANNELS_FROZEN",
                "插件 channel 声明已冻结，不能新增",
            )
        if definition.name in self._registrations:
            raise CompositionError(
                "DUPLICATE_PLUGIN_CHANNEL",
                f"插件 channel 名称重复: {definition.name}",
            )
        descriptor = ChannelDescriptor(
            owner=owner,
            name=definition.name,
            capabilities=tuple(sorted(definition.capabilities, key=lambda item: item.value)),
            factory_export=definition.factory_export,
            inbound_identity=definition.inbound_identity,
            credential_paths=definition.credential_paths,
        )
        registration = _ChannelRegistration(
            owner=owner,
            definition=definition,
            descriptor=descriptor,
            owner_fiber=owner_fiber,
            activation_token=activation_token,
            generation_id=generation_id,
            incident_reporter=incident_reporter,
        )
        self._registrations[definition.name] = registration

        def cleanup() -> None:
            if self._registrations.get(definition.name) is registration:
                del self._registrations[definition.name]

        return registration, cleanup


class PluginChannels:
    """Expose only Fiber-owned channel blueprint registration to plugins."""

    def __init__(self, root_instance_token: object) -> None:
        self._root_instance_token = root_instance_token
        self._declarations = _ChannelDeclarations()

    async def register(self, ctx: Context, definition: ChannelDefinition) -> None:
        """Register one channel blueprint through the Core-owned collector."""

        if (
            ctx._root_instance_token() is not self._root_instance_token
            or ctx.require(CHANNELS) is not self
        ):
            raise CompositionError(
                "CHANNEL_SERVICE_ROOT_MISMATCH",
                "插件 channel Service 不属于当前 Root",
            )
        await self._declarations.register(ctx, definition)


def _freeze_plugin_channels(
    value: object,
    root_instance_token: object,
    *,
    factory_provenance_by_owner: Mapping[
        str,
        ChannelFactoryFreezeInput | tuple[str, str, str],
    ]
    | None = None,
) -> ChannelRegistrySnapshot:
    """Freeze the exact Core-created channel declaration facade."""

    if not isinstance(value, PluginChannels):
        raise RuntimeError("RuntimeSnapshot channel Service 类型无效")
    if value._root_instance_token is not root_instance_token:
        raise RuntimeError("RuntimeSnapshot channel Service 不属于 exact Root")
    return value._declarations.freeze(
        root_instance_token,
        factory_provenance_by_owner=factory_provenance_by_owner,
    )


def _normalize_definition(definition: ChannelDefinition) -> ChannelDefinition:
    if not isinstance(definition, ChannelDefinition):
        raise TypeError("PluginChannels.register 只接受 ChannelDefinition")
    return ChannelDefinition(
        name=definition.name,
        capabilities=frozenset(definition.capabilities),
        factory_export=definition.factory_export,
        inbound_identity=definition.inbound_identity,
        credential_paths=tuple(definition.credential_paths),
    )


def _make_provenance(
    registration: _ChannelRegistration,
    supplied: ChannelFactoryFreezeInput | tuple[str, str, str] | None,
) -> ChannelFactoryProvenance:
    if supplied is None:
        source = ChannelFactoryFreezeInput(registration.generation_id)
        return ChannelFactoryProvenance(
            plugin_id=registration.owner,
            generation_id=source.generation_id,
            channel_name=registration.definition.name,
            source_revision=source.source_revision,
            config_revision=source.config_revision,
            factory_export=registration.definition.factory_export,
        )
    if isinstance(supplied, ChannelFactoryFreezeInput):
        result = ChannelFactoryProvenance(
            plugin_id=registration.owner,
            generation_id=supplied.generation_id,
            channel_name=registration.definition.name,
            source_revision=supplied.source_revision,
            config_revision=supplied.config_revision,
            factory_export=registration.definition.factory_export,
        )
    elif isinstance(supplied, tuple) and len(supplied) == 3:
        result = ChannelFactoryProvenance(
            plugin_id=registration.owner,
            generation_id=supplied[0],
            channel_name=registration.definition.name,
            source_revision=supplied[1],
            config_revision=supplied[2],
            factory_export=registration.definition.factory_export,
        )
    else:
        raise TypeError("channel factory provenance 输入类型无效")
    return result


def _factory_sort_key(item: ChannelFactoryProvenance) -> tuple[str, str, str, str, str, str]:
    return (
        item.plugin_id,
        item.generation_id,
        item.channel_name,
        item.source_revision,
        item.config_revision,
        item.factory_export,
    )


def _registry_identity(
    descriptors: tuple[ChannelDescriptor, ...],
    factories: tuple[ChannelFactoryProvenance, ...],
) -> str:
    payload = {
        "descriptors": [
            {
                "owner": item.owner,
                "name": item.name,
                "capabilities": [capability.value for capability in item.capabilities],
                "factory_export": item.factory_export,
                "inbound_identity": (
                    None
                    if item.inbound_identity is None
                    else item.inbound_identity.value
                ),
                "credential_paths": list(item.credential_paths),
            }
            for item in descriptors
        ],
        "factories": [
            {
                "plugin_id": item.plugin_id,
                "generation_id": item.generation_id,
                "channel_name": item.channel_name,
                "source_revision": item.source_revision,
                "config_revision": item.config_revision,
                "factory_export": item.factory_export,
            }
            for item in factories
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _credential_paths(value: object, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, tuple) or (not allow_empty and not value):
        raise ValueError("credential_paths 必须是非空 tuple")
    if not value:
        return ()
    result: list[str] = []
    for path in value:
        if not isinstance(path, str) or not path or path.strip() != path:
            raise ValueError("credential_paths 必须是非空字符串")
        if any(not part or part in {".", ".."} for part in path.split(".")):
            raise ValueError(f"credential path 无效: {path}")
        if path in result:
            raise ValueError(f"credential_paths 重复: {path}")
        result.append(path)
    return tuple(result)


def _credential_refs(
    value: Mapping[str, CredentialRef],
) -> Mapping[str, CredentialRef]:
    if not isinstance(value, Mapping):
        raise TypeError("credentials 必须是 mapping")
    result: dict[str, CredentialRef] = {}
    for path in sorted(value):
        ref = value[path]
        if not isinstance(path, str) or not isinstance(ref, CredentialRef):
            raise TypeError("credentials 必须映射到 CredentialRef")
        if path != ".".join(ref.path):
            raise ValueError(f"credential path 与 ref 不一致: {path}")
        result[path] = ref
    return MappingProxyType(result)


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


_INBOUND_STATES = {
    (InboundOwner.INGRESS, InboundState.ADMITTED),
    (InboundOwner.BUS, InboundState.BUS_QUEUED),
    (InboundOwner.LANE, InboundState.LANE_QUEUED),
    (InboundOwner.LOOP, InboundState.RUNNING),
}

_INBOUND_TRANSITIONS = {
    (InboundOwner.INGRESS, InboundState.ADMITTED): InboundOwner.BUS,
    (InboundOwner.BUS, InboundState.BUS_QUEUED): InboundOwner.LANE,
    (InboundOwner.LANE, InboundState.LANE_QUEUED): InboundOwner.LOOP,
}

_INBOUND_STATE_BY_OWNER = {
    InboundOwner.BUS: InboundState.BUS_QUEUED,
    InboundOwner.LANE: InboundState.LANE_QUEUED,
    InboundOwner.LOOP: InboundState.RUNNING,
}


def _validate_binding_lease(lease: object) -> None:
    """Check the narrow exact-binding fields before an envelope retains a lease."""

    for field_name in (
        "snapshot_lease",
        "snapshot_id",
        "generation_id",
        "channel_name",
        "binding_token",
    ):
        if not hasattr(lease, field_name):
            raise TypeError(f"ChannelBindingLease 缺少 {field_name}")
    aclose = getattr(lease, "aclose", None)
    if not callable(aclose):
        raise TypeError("ChannelBindingLease.aclose 必须是 callable")
    _text(getattr(lease, "snapshot_id"), "lease.snapshot_id")
    _text(getattr(lease, "generation_id"), "lease.generation_id")
    _text(getattr(lease, "channel_name"), "lease.channel_name")
    _text(getattr(lease, "binding_token"), "lease.binding_token")


async def _close_lease_critically(lease: ChannelBindingLease) -> bool:
    """Finish lease cleanup before restoring caller cancellation."""

    task = asyncio.ensure_future(lease.aclose())
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    # Reading the task result preserves a real lease-close failure.
    task.result()
    return cancelled


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


def _attachment_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("artifact_id 必须是 str")
    if _ATTACHMENT_ID.fullmatch(value) is None:
        raise ValueError("artifact_id 必须是安全的 opaque id")
    return value


def _attachment_filename(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("filename 必须是 str 或 None")
    if (
        not value
        or value != value.strip()
        or len(value) > 255
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("filename 必须是 1..255 字符的纯文件名")
    return value


def _attachment_media_type(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("media_type 必须是 str 或 None")
    if len(value) > 255 or _MEDIA_TYPE.fullmatch(value) is None:
        raise ValueError("media_type 必须是合法 MIME type")
    return value


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
    "ChannelAdapter",
    "ChannelCapability",
    "ChannelCommitRole",
    "ChannelAttachmentImportPort",
    "ChannelAttachmentReadPort",
    "ChannelCleanupFailure",
    "ChannelControlPort",
    "ChannelDeliveryReceipt",
    "ChannelFactoryContext",
    "ChannelIngressPort",
    "ChannelIdentityPort",
    "ChannelReady",
    "ChannelTerminalStatus",
    "ChannelPresentationPorts",
    "ChannelInboundMessage",
    "AttachmentKind",
    "AttachmentReadLease",
    "AttachmentRef",
    "CredentialRef",
    "DeliveryStatus",
    "ControlReceipt",
    "ControlResponseBodies",
    "ChannelDefinition",
    "ChannelDescriptor",
    "CoreChannelDefinition",
    "CommittedChannelCatalog",
    "ChannelFactoryFreezeInput",
    "ChannelFactoryProvenance",
    "ChannelRegistrySnapshot",
    "InboundEnvelope",
    "InboundIdentity",
    "InboundOwner",
    "InboundState",
    "JsonValue",
    "OutboundEnvelope",
    "PluginChannels",
    "ProviderClient",
    "ProviderClientFactory",
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
    "_freeze_plugin_channels",
]
