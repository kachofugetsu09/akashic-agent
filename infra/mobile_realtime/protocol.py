from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)


PROTOCOL_VERSION = 1
MAX_JSON_FRAME_BYTES = 256 * 1024
_MAX_REBASE_ACK = 1 << 62

COMMAND_TYPES = frozenset(
    {
        "session.list",
        "session.create",
        "session.open",
        "history.get",
        "message.send",
        "turn.stop",
        "attachment.begin",
        "attachment.finish",
        "attachment.download",
        "command.list",
        "plugin.ui.catalog",
        "plugin.ui.asset.get",
        "plugin.ui.query",
        "plugin.ui.cancel",
        "device.update",
        "ping",
    }
)
EVENT_TYPES = frozenset(
    {
        "session.list",
        "session.created",
        "session.updated",
        "history.page",
        "turn.started",
        "react.thinking.delta",
        "react.tool.started",
        "react.tool.completed",
        "answer.delta",
        "turn.snapshot",
        "message.final",
        "turn.interrupted",
        "message.proactive",
        "attachment.progress",
        "attachment.ready",
        "connection.degraded",
        "sync.completed",
        "sync.reset_required",
        "device.revoked",
    }
)
CONTROL_TYPES = frozenset(
    {
        "server.challenge",
        "device.proof",
        "auth.accepted",
        "resume",
        "plugin.ui.changed",
        "pair.claim",
        "pair.pending",
        "pair.accepted",
        "protocol.error",
    }
)
PRE_AUTH_CONTROL_TYPES = frozenset(
    {
        "server.challenge",
        "device.proof",
        "pair.claim",
        "pair.pending",
        "pair.accepted",
        "protocol.error",
    }
)

_FRAME_ID_PATTERN_TEXT = (
    r"^(?:[0-9A-HJKMNP-TV-Z]{26}|[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-"
    r"7[0-9A-Fa-f]{3}-[89ABab][0-9A-Fa-f]{3}-[0-9A-Fa-f]{12})$"
)
_FRAME_ID_PATTERN = re.compile(_FRAME_ID_PATTERN_TEXT)
_RFC3339_INSTANT_PATTERN_TEXT = (
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$"
)
_REPLY_TYPE_PATTERN = re.compile(
    r"^(?P<command>[a-z][a-z0-9]*(?:\.[a-z][a-z0-9_]*)+)\.(?:ok|error)$"
)
_SPECIAL_REPLY_TYPES = frozenset({"plugin.ui.catalog.not_modified"})

FrameId: TypeAlias = Annotated[
    str,
    Field(min_length=26, max_length=36, pattern=_FRAME_ID_PATTERN_TEXT),
]
ConnectionEpoch: TypeAlias = Annotated[int, Field(ge=1)]
EventSequence: TypeAlias = Annotated[int, Field(ge=1)]
NonEmptyId: TypeAlias = Annotated[str, Field(min_length=1, max_length=512)]
JsonObject: TypeAlias = dict[str, JsonValue]


class ProtocolDecodeError(ValueError):
    pass


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class MessageReplyReference(ProtocolModel):
    message_id: NonEmptyId | None = None
    client_message_id: FrameId | None = None
    legacy_role: Literal["user", "assistant"] | None = Field(
        default=None,
        alias="role",
        exclude=True,
    )
    legacy_preview: str | None = Field(
        default=None,
        alias="preview",
        max_length=512,
        exclude=True,
    )

    @model_validator(mode="after")
    def validate_identity(self) -> MessageReplyReference:
        """要求引用消息只携带一种稳定标识。"""

        identities = (self.message_id is not None) + (self.client_message_id is not None)
        if identities != 1:
            raise ValueError("reply_to 必须且只能提供一种消息标识")
        if self.client_message_id is not None:
            _validate_frame_id(self.client_message_id, "reply_to.client_message_id")
        return self


class MessageSendPayload(ProtocolModel):
    client_message_id: FrameId
    session_id: NonEmptyId
    text: str = Field(max_length=65_536)
    media_refs: list[NonEmptyId] = Field(max_length=10)
    client_created_at: str = Field(
        min_length=1,
        max_length=64,
        pattern=_RFC3339_INSTANT_PATTERN_TEXT,
    )
    reply_to: MessageReplyReference | None = None

    @field_validator("client_created_at")
    @classmethod
    def validate_client_created_at(cls, value: str) -> str:
        """校验客户端实际创建消息的时间。"""

        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("client_created_at 必须是 RFC 3339 时间") from error
        if parsed.tzinfo is None:
            raise ValueError("client_created_at 必须包含时区")
        return value

    @model_validator(mode="after")
    def validate_client_message_id(self) -> MessageSendPayload:
        _validate_frame_id(self.client_message_id, "client_message_id")
        if len(set(self.media_refs)) != len(self.media_refs):
            raise ValueError("media_refs 不能重复")
        return self


class AttachmentBeginPayload(ProtocolModel):
    attachment_id: FrameId
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class AttachmentFinishPayload(ProtocolModel):
    attachment_id: FrameId


class AttachmentDownloadPayload(ProtocolModel):
    attachment_id: FrameId
    offset: int = Field(ge=0)


class TurnSnapshotPayload(ProtocolModel):
    turn_id: NonEmptyId
    status: Literal["queued", "running", "completed", "interrupted", "failed"]
    blocks: list[JsonObject]
    content_so_far: str
    last_source_event_id: FrameId | None

    @model_validator(mode="after")
    def validate_last_source_event_id(self) -> TurnSnapshotPayload:
        if self.last_source_event_id is not None:
            _validate_frame_id(self.last_source_event_id, "last_source_event_id")
        return self


class DeltaPayload(ProtocolModel):
    delta: str = Field(min_length=1, max_length=65_536)
    block_id: NonEmptyId | None = None
    ordinal: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_process_block(self) -> DeltaPayload:
        if (self.block_id is None) != (self.ordinal is None):
            raise ValueError("delta block_id 与 ordinal 必须同时出现")
        return self


class AckPayload(ProtocolModel):
    through_event_seq: int = Field(ge=0)


class AuthAcceptedPayload(ProtocolModel):
    connection_epoch: ConnectionEpoch
    device_id: NonEmptyId


class ResumePayload(ProtocolModel):
    last_ack: int = Field(ge=0, le=_MAX_REBASE_ACK)
    active_turns: list[NonEmptyId] = Field(max_length=128)


class CommandEnvelope(ProtocolModel):
    v: Literal[1]
    kind: Literal["command"]
    type: str
    id: FrameId
    connection_epoch: ConnectionEpoch
    session_id: NonEmptyId | None = None
    turn_id: NonEmptyId | None = None
    payload: JsonObject

    @model_validator(mode="after")
    def validate_id(self) -> CommandEnvelope:
        _validate_frame_id(self.id, "id")
        return self


class MessageSendCommand(ProtocolModel):
    v: Literal[1]
    kind: Literal["command"]
    type: Literal["message.send"]
    id: FrameId
    connection_epoch: ConnectionEpoch
    session_id: NonEmptyId
    turn_id: NonEmptyId | None = None
    payload: MessageSendPayload

    @model_validator(mode="after")
    def validate_command(self) -> MessageSendCommand:
        _validate_frame_id(self.id, "id")
        if self.session_id != self.payload.session_id:
            raise ValueError("message.send envelope 与 payload 的 session_id 必须一致")
        return self


class AttachmentBeginCommand(ProtocolModel):
    v: Literal[1]
    kind: Literal["command"]
    type: Literal["attachment.begin"]
    id: FrameId
    connection_epoch: ConnectionEpoch
    session_id: NonEmptyId
    turn_id: None = None
    payload: AttachmentBeginPayload


class AttachmentFinishCommand(ProtocolModel):
    v: Literal[1]
    kind: Literal["command"]
    type: Literal["attachment.finish"]
    id: FrameId
    connection_epoch: ConnectionEpoch
    session_id: NonEmptyId
    turn_id: None = None
    payload: AttachmentFinishPayload


class AttachmentDownloadCommand(ProtocolModel):
    v: Literal[1]
    kind: Literal["command"]
    type: Literal["attachment.download"]
    id: FrameId
    connection_epoch: ConnectionEpoch
    session_id: NonEmptyId
    turn_id: None = None
    payload: AttachmentDownloadPayload


class GenericCommand(CommandEnvelope):
    type: Literal[
        "session.list",
        "session.create",
        "session.open",
        "history.get",
        "command.list",
        "plugin.ui.catalog",
        "plugin.ui.asset.get",
        "plugin.ui.query",
        "plugin.ui.cancel",
        "turn.stop",
        "device.update",
        "ping",
    ]


ClientCommand: TypeAlias = (
    MessageSendCommand
    | AttachmentBeginCommand
    | AttachmentFinishCommand
    | AttachmentDownloadCommand
    | GenericCommand
)
CommandFrame: TypeAlias = Annotated[
    ClientCommand,
    Field(discriminator="type"),
]


class ReplyFrame(ProtocolModel):
    v: Literal[1]
    kind: Literal["reply"]
    type: str = Field(min_length=4, max_length=80)
    id: FrameId
    connection_epoch: ConnectionEpoch
    session_id: NonEmptyId | None = None
    turn_id: NonEmptyId | None = None
    payload: JsonObject

    @model_validator(mode="after")
    def validate_reply(self) -> ReplyFrame:
        _validate_frame_id(self.id, "id")
        if self.type in _SPECIAL_REPLY_TYPES:
            return self
        match = _REPLY_TYPE_PATTERN.fullmatch(self.type)
        if match is None or match.group("command") not in COMMAND_TYPES:
            raise ValueError(
                "reply type 必须为已知 command 的标准或显式扩展结果"
            )
        return self


class EventEnvelope(ProtocolModel):
    v: Literal[1]
    kind: Literal["event"]
    type: str
    id: FrameId
    connection_epoch: ConnectionEpoch
    event_seq: EventSequence
    session_id: NonEmptyId | None = None
    turn_id: NonEmptyId | None = None
    payload: JsonObject

    @model_validator(mode="after")
    def validate_id(self) -> EventEnvelope:
        _validate_frame_id(self.id, "id")
        return self


class ThinkingDeltaEvent(EventEnvelope):
    type: Literal["react.thinking.delta"]
    payload: DeltaPayload


class AnswerDeltaEvent(EventEnvelope):
    type: Literal["answer.delta"]
    payload: DeltaPayload


class TurnSnapshotEvent(ProtocolModel):
    v: Literal[1]
    kind: Literal["event"]
    type: Literal["turn.snapshot"]
    id: FrameId
    connection_epoch: ConnectionEpoch
    event_seq: EventSequence
    session_id: NonEmptyId | None = None
    turn_id: NonEmptyId
    payload: TurnSnapshotPayload

    @model_validator(mode="after")
    def validate_event(self) -> TurnSnapshotEvent:
        _validate_frame_id(self.id, "id")
        if self.turn_id != self.payload.turn_id:
            raise ValueError("turn.snapshot envelope 与 payload 的 turn_id 必须一致")
        return self


class GenericEvent(EventEnvelope):
    type: Literal[
        "session.list",
        "session.created",
        "session.updated",
        "history.page",
        "turn.started",
        "react.tool.started",
        "react.tool.completed",
        "message.final",
        "turn.interrupted",
        "message.proactive",
        "attachment.progress",
        "attachment.ready",
        "connection.degraded",
        "sync.completed",
        "sync.reset_required",
        "device.revoked",
    ]


EventFrame: TypeAlias = Annotated[
    ThinkingDeltaEvent | AnswerDeltaEvent | TurnSnapshotEvent | GenericEvent,
    Field(discriminator="type"),
]


class AckFrame(ProtocolModel):
    v: Literal[1]
    kind: Literal["ack"]
    type: Literal["event.ack"]
    connection_epoch: ConnectionEpoch
    payload: AckPayload


class ControlEnvelope(ProtocolModel):
    v: Literal[1]
    kind: Literal["control"]
    type: str
    id: FrameId | None = None
    connection_epoch: ConnectionEpoch | None = None
    payload: JsonObject

    @model_validator(mode="after")
    def validate_optional_id(self) -> ControlEnvelope:
        if self.id is not None:
            _validate_frame_id(self.id, "id")
        return self


class AuthenticatedControlEnvelope(ProtocolModel):
    v: Literal[1]
    kind: Literal["control"]
    type: str
    id: FrameId | None = None
    connection_epoch: ConnectionEpoch
    payload: JsonObject

    @model_validator(mode="after")
    def validate_optional_id(self) -> AuthenticatedControlEnvelope:
        if self.id is not None:
            _validate_frame_id(self.id, "id")
        return self


class AuthAcceptedControl(AuthenticatedControlEnvelope):
    type: Literal["auth.accepted"]
    payload: AuthAcceptedPayload

    @model_validator(mode="after")
    def validate_epoch_match(self) -> AuthAcceptedControl:
        if self.connection_epoch != self.payload.connection_epoch:
            raise ValueError("auth.accepted envelope 与 payload 的 connection_epoch 必须一致")
        return self


class ResumeControl(AuthenticatedControlEnvelope):
    type: Literal["resume"]
    payload: ResumePayload


class PluginUiChangedControl(AuthenticatedControlEnvelope):
    type: Literal["plugin.ui.changed"]


class GenericControl(ControlEnvelope):
    type: Literal[
        "server.challenge",
        "device.proof",
        "pair.claim",
        "pair.pending",
        "pair.accepted",
        "protocol.error",
    ]


ControlFrame: TypeAlias = Annotated[
    AuthAcceptedControl | ResumeControl | PluginUiChangedControl | GenericControl,
    Field(discriminator="type"),
]

MobileFrame: TypeAlias = Annotated[
    CommandFrame | ReplyFrame | EventFrame | AckFrame | ControlFrame,
    Field(discriminator="kind"),
]

FRAME_ADAPTER: TypeAdapter[MobileFrame] = TypeAdapter(MobileFrame)


def parse_frame(data: bytes | str) -> MobileFrame:
    """严格解码一条 JSON 文本帧并建立协议类型。"""

    # 1. 在传输边界限制 UTF-8 编码后的真实帧大小。
    raw = data if isinstance(data, bytes) else data.encode("utf-8")
    if len(raw) > MAX_JSON_FRAME_BYTES:
        raise ProtocolDecodeError(f"JSON frame 超过 {MAX_JSON_FRAME_BYTES} bytes")

    # 2. 拒绝非标准常量、重复字段和非 object 顶层值。
    try:
        decoded = raw.decode("utf-8")
        payload = json.loads(
            decoded,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolDecodeError("JSON frame 不是合法 UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ProtocolDecodeError("JSON frame 顶层必须是 object")

    # 3. kind/type discriminator 负责选择唯一的严格 schema。
    return FRAME_ADAPTER.validate_python(payload, strict=True)


def frame_to_json(frame: MobileFrame) -> str:
    """把已验证帧编码为确定性的紧凑 JSON。"""
    return json.dumps(
        FRAME_ADAPTER.dump_python(frame, mode="json", exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _validate_frame_id(value: str, field: str) -> None:
    if _FRAME_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} 必须是 ULID 或 UUIDv7")


def _reject_json_constant(value: str) -> None:
    raise ProtocolDecodeError(f"JSON frame 不允许非标准常量: {value}")


def _unique_json_object(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    """构造 JSON object，并拒绝会产生歧义的重复字段。"""
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolDecodeError(f"JSON frame 包含重复字段: {key}")
        result[key] = value
    return result


def validation_issues(error: ValidationError) -> list[JsonObject]:
    """返回可安全写入 protocol.error 的稳定校验问题。"""
    return cast(
        list[JsonObject],
        error.errors(include_url=False, include_context=False),
    )
