"""Ledger 的值、读写授权与同库事务合同；实现随插件 generation 生灭。"""
from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from itertools import islice
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast, overload, runtime_checkable

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey
from core.common.frozen_json import freeze_json, json_value

if TYPE_CHECKING:
    from plugins.channels.contract import InboundEnvelope, RawInbound

_T = TypeVar("_T")


MAX_METADATA_BYTES = 64 * 1024


def freeze_metadata(value: Mapping[str, object]) -> Mapping[str, object]:
    """附加信息只接受有界 JSON 对象；插件内部结构不参与消息类型校验。"""
    if not isinstance(value, Mapping):
        raise TypeError("消息 metadata 必须是 JSON 对象")
    frozen = cast(Mapping[str, object], freeze_json(value))
    if any(not key for key in frozen):
        raise ValueError("消息 metadata 命名空间不能为空")
    # JSON 编码同时固定字节预算；不按 Python 对象大小或字符数计算。
    payload = json.dumps(frozen, default=_json_container, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(payload.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError("消息 metadata 超过 64 KiB")
    return frozen


def _json_container(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, object], value))
    raise TypeError("消息 metadata 包含非 JSON 值")


@dataclass(frozen=True, slots=True)
class ContentPart:
    """内容类型及其不可变载荷；具体 schema 由声明该类型的能力校验。"""

    kind: str
    value: object

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind or self.kind == "tool_call":
            raise ValueError("内容类型不能为空")
        object.__setattr__(self, "value", freeze_json(self.value))


@dataclass(frozen=True, slots=True)
class ContentReferences:
    """内容检查声明的耐久链接；附件保留正文出现次序和重复项。"""

    binding_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for values in (self.binding_ids, self.artifact_ids):
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise TypeError("内容引用必须是非空 ID 的 tuple")


@dataclass(frozen=True, slots=True)
class ToolCall:
    binding_id: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.binding_id, str) or not self.binding_id:
            raise ValueError("工具调用必须固定 binding_id")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("工具参数必须是 JSON 对象")
        object.__setattr__(self, "arguments", freeze_json(self.arguments))


@dataclass(frozen=True, slots=True)
class CallRef:
    message_id: str
    part_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.message_id, str)
            or not self.message_id
            or type(self.part_index) is not int
            or self.part_index < 0
        ):
            raise ValueError("调用引用需要 message_id 和非负 part_index")


type Part = ContentPart | ToolCall


@dataclass(frozen=True, slots=True)
class Input:
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("Input 只能包含内容块")
        object.__setattr__(self, "parts", parts)


@dataclass(frozen=True, slots=True)
class Output:
    parts: tuple[Part, ...]
    finish: Literal["continue", "complete", "quiet"]

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        if any(not isinstance(part, (ContentPart, ToolCall)) for part in parts):
            raise TypeError("Output 只能包含内容块或工具调用")
        object.__setattr__(self, "parts", parts)
        if self.finish not in {"continue", "complete", "quiet"}:
            raise ValueError("Output finish 无效")
        if self.finish != "continue" and any(
            isinstance(part, ToolCall) for part in self.parts
        ):
            raise ValueError("含工具调用的 Output 必须是 continue")


@dataclass(frozen=True, slots=True)
class ToolResult:
    # 仅持久化解码器记录旧表示；运行时 outcome 仍遵守当前值域。
    _legacy_unknown: bool = field(default=False, init=False, repr=False, compare=False)
    call_ref: CallRef
    outcome: Literal["success", "denied", "error", "interrupted"]
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.call_ref, CallRef):
            raise TypeError("工具结果需要有效的调用引用")
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("ToolResult 只能包含内容块")
        object.__setattr__(self, "parts", parts)
        if self.outcome not in {"success", "denied", "error", "interrupted"}:
            raise ValueError("ToolResult outcome 无效")


@dataclass(frozen=True, slots=True)
class Control:
    action: Literal["pause", "resume", "abandon", "failure"]
    through_seq: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.action not in {"pause", "resume", "abandon", "failure"}:
            raise ValueError("Control action 无效")
        if type(self.through_seq) is not int or self.through_seq < 0:
            raise ValueError("Control through_seq 必须非负")
        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError("Control reason 必须是文本")


type Body = Input | Output | ToolResult | Control


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Message:
    """一条已接纳事实；作者、来源与消息用途分别表达独立信息。"""

    message_id: str
    session_id: str
    seq: int
    recorded_at: datetime
    author: str
    source: str
    body: Body
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        except (TypeError, ValueError) as error:
            raise type(error)(
                f"Session {self.session_id} Message {self.message_id} metadata 损坏: {error}"
            ) from error
        if not all(
            isinstance(value, str) and value
            for value in (self.message_id, self.session_id, self.author, self.source)
        ):
            raise ValueError("消息身份、Session、作者和来源不能为空")
        if type(self.seq) is not int or self.seq < 0:
            raise ValueError("消息 seq 必须非负")
        if (
            not isinstance(self.recorded_at, datetime)
            or self.recorded_at.utcoffset() is None
        ):
            raise ValueError("消息接纳时间必须包含时区")
        if not isinstance(self.body, (Input, Output, ToolResult, Control)):
            raise TypeError("消息 body 类型无效")
        if isinstance(self.body, Control) and self.body.through_seq >= self.seq:
            raise ValueError("Control 只能指向之前的已接纳前缀")


def body_to_dict(body: Body) -> dict[str, object]:
    """返回当前运行时消息字段，供展示和上下文使用。"""
    data: dict[str, object]
    if isinstance(body, Control):
        data = {
            "kind": "control",
            "action": body.action,
            "through_seq": body.through_seq,
            "reason": body.reason,
        }
    else:
        parts: list[dict[str, object]] = [
            (
                {
                    "kind": "tool_call",
                    "binding_id": part.binding_id,
                    "arguments": json_value(part.arguments),
                }
                if isinstance(part, ToolCall)
                else {"kind": part.kind, "value": json_value(part.value)}
            )
            for part in body.parts
        ]
        if isinstance(body, Input):
            data = {"kind": "input", "parts": parts}
        elif isinstance(body, Output):
            data = {
                "kind": "output",
                "parts": parts,
                "finish": body.finish,
            }
        else:
            data = {
                "kind": "tool_result",
                "parts": parts,
                "outcome": body.outcome,
                "call_ref": {
                    "message_id": body.call_ref.message_id,
                    "part_index": body.call_ref.part_index,
                },
            }
    return data


def encode_body(body: Body, *, allow_legacy: bool = True) -> str:
    """保留已读消息的历史编码；新写入可显式拒绝旧表示。"""
    data = body_to_dict(body)
    if isinstance(body, ToolResult) and body._legacy_unknown:
        if not allow_legacy:
            raise ValueError("旧 unknown 工具结果只能重放已有消息，不能作为新消息写入")
        data["outcome"] = "unknown"
    return json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _object(value: object, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(cast(dict[object, object], value)) != fields:
        raise ValueError(f"消息对象字段必须为 {sorted(fields)}")
    return cast(dict[str, object], value)


def _part(value: object) -> Part:
    if not isinstance(value, dict):
        raise ValueError("消息 part 必须是对象")
    raw = cast(dict[str, object], value)
    if raw.get("kind") == "tool_call":
        data = _object(raw, {"kind", "binding_id", "arguments"})
        return ToolCall(
            cast(str, data["binding_id"]), cast(Mapping[str, object], data["arguments"])
        )
    data = _object(raw, {"kind", "value"})
    return ContentPart(cast(str, data["kind"]), data["value"])


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"消息 JSON 包含重复字段: {key}")
        result[key] = value
    return result


def decode_body(payload: str) -> Body:
    """在持久化边界拒绝未知结构；值域与不变量归消息构造函数。"""
    raw: object = json.loads(payload, object_pairs_hook=_unique_fields)
    if not isinstance(raw, dict):
        raise ValueError("消息 body 必须是对象")
    data = cast(dict[str, object], raw)
    kind = data.get("kind")
    if kind == "control":
        row = _object(data, {"kind", "action", "through_seq", "reason"})
        return Control(
            cast(Literal["pause", "resume", "abandon", "failure"], row["action"]),
            cast(int, row["through_seq"]),
            cast(str | None, row["reason"]),
        )
    fields = {
        "input": {"kind", "parts"},
        "output": {"kind", "parts", "finish"},
        "tool_result": {"kind", "parts", "call_ref", "outcome"},
    }
    if not isinstance(kind, str) or kind not in fields:
        raise ValueError("消息 body kind 无效")
    row = _object(data, fields[kind])
    raw_parts = row["parts"]
    if not isinstance(raw_parts, list):
        raise ValueError("消息 parts 必须是数组")
    parts = tuple(_part(part) for part in cast(list[object], raw_parts))
    if kind == "input":
        return Input(cast(tuple[ContentPart, ...], parts))
    if kind == "output":
        return Output(
            parts,
            cast(Literal["continue", "complete", "quiet"], row["finish"]),
        )
    call = _object(row["call_ref"], {"message_id", "part_index"})
    result = ToolResult(
        CallRef(cast(str, call["message_id"]), cast(int, call["part_index"])),
        cast(Literal["success", "denied", "error", "interrupted"],
             "error" if row["outcome"] == "unknown" else row["outcome"]),
        cast(tuple[ContentPart, ...], parts),
    )

    if row["outcome"] == "unknown":
        object.__setattr__(result, "_legacy_unknown", True)
    return result


_ATTACHMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_MEDIA_TYPE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AttachmentKind(StrEnum):
    FILE = "file"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class AttachmentRef:
    """标识不可变附件，不暴露存储路径。"""

    artifact_id: str
    kind: AttachmentKind
    filename: str | None
    media_type: str | None
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _ = check_artifact_id(self.artifact_id)
        if not isinstance(self.kind, AttachmentKind):
            raise TypeError("kind 必须是 AttachmentKind")
        _ = _attachment_filename(self.filename)
        _ = _attachment_media_type(self.media_type)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes 必须是 int")
        if self.size_bytes < 0:
            raise ValueError("size_bytes 不能是负数")
        if not isinstance(self.sha256, str):
            raise TypeError("sha256 必须是 str")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("sha256 必须是 64 位小写十六进制字符串")


def check_artifact_id(value: object) -> str:
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


class AttachmentReadLease(Protocol):
    @property
    def ref(self) -> AttachmentRef: ...

    async def read_bytes(self, *, max_bytes: int) -> bytes: ...

    async def read_chunk(self, *, offset: int, max_bytes: int) -> bytes: ...

    async def aclose(self) -> None: ...


class AttachmentReadPort(Protocol):
    async def acquire(self, ref: AttachmentRef) -> AttachmentReadLease: ...


@dataclass(frozen=True, slots=True)
class SessionDeleteResult:
    session_key: str
    deleted: bool
    deleted_at: str | None


@dataclass(frozen=True, slots=True)
class SessionTitleResult:
    session_key: str
    title: str | None


class AppendChecks(Protocol):
    async def register(
        self, ctx: Context, check: Callable[[Message, MessageReader], None],
    ) -> Effect:
        """在提交事务中同步检查新消息；异常回滚。reader 只读，回调不得保留它或执行外部效果。"""
        ...


class MessageWriters(Protocol):
    async def register_metadata(
        self, ctx: Context, *, keys: frozenset[str],
        update: Callable[[Body], Mapping[str, object | None]],
    ) -> Effect: ...

    def bind(
        self,
        ctx: Context,
        *,
        author: str,
        source: str,
        body_types: tuple[type[Input] | type[Output] | type[ToolResult] | type[Control], ...],
        content: Mapping[str, Callable[[ContentPart], ContentReferences]],
        check_call: Callable[[ToolCall], None] | None = None,
        update_metadata: Callable[[Body], Mapping[str, object | None]] | None = None,
        check_metadata: Callable[[Mapping[str, object]], None] | None = None,
    ) -> Callable[..., MessageWriter]: ...


class OwnerState(Protocol):
    def open(self, ctx: Context) -> OwnerStore: ...

    def open_scoped(self, ctx: Context, scope: str) -> OwnerStore: ...


class SessionAdmin(Protocol):
    async def set_deleted(self, session_key: str, *, deleted: bool) -> SessionDeleteResult: ...

    async def set_title(self, session_key: str, title: str | None) -> SessionTitleResult: ...

    async def set_title_if_unset(self, session_key: str, title: str) -> bool: ...


class SessionAdmission(Protocol):
    async def register_initializer(
        self, ctx: Context, *, name: str,
        initialize: Callable[[str, SessionAttributes, OwnerTransaction], None],
    ) -> Effect: ...

    async def register_dimension(
        self, ctx: Context, *, name: str, check: Callable[[str], None],
    ) -> Effect: ...

    def ensure(self, ctx: Context, session_id: str, attributes: SessionAttributes) -> SessionAttributes: ...

    async def ensure_async(self, ctx: Context, session_id: str, attributes: SessionAttributes) -> SessionAttributes: ...


_SCOPE_DIMENSION = re.compile(r"[a-z][a-z0-9_]{0,31}")
_SCOPE_VALUE_LIMIT = 128


class _MessagePrefix(Protocol):
    @property
    def session_id(self) -> str: ...
    @property
    def revision(self) -> int | None: ...
    @property
    def messages(self) -> Sequence[Message]: ...


@dataclass(frozen=True, slots=True)
class SessionAttributes:
    """会话接纳时固定的独立事实；存储不替展示或学习消费者作决定。

    scope 是宽键中已声明的维度；缺失维度即 default，Core 不解释维度含义。
    """

    visibility: Literal["listed", "internal"] = "listed"
    learning: Literal["eligible", "excluded"] = "eligible"
    scope: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.visibility not in ("listed", "internal") or self.learning not in ("eligible", "excluded"):
            raise ValueError("Session 属性无效")
        names = [name for name, _ in self.scope]
        if names != sorted(set(names)):
            raise ValueError("Session scope 维度必须唯一且有序")
        for name, value in self.scope:
            if not isinstance(name, str) or _SCOPE_DIMENSION.fullmatch(name) is None:
                raise ValueError(f"Session scope 维度名无效: {name!r}")
            if (
                not isinstance(value, str) or not value or value == "default"
                or len(value) > _SCOPE_VALUE_LIMIT or value != value.strip()
            ):
                raise ValueError(f"Session scope 维度值无效: {name}")

    @classmethod
    def scoped(
        cls, dimensions: Mapping[str, str], *,
        visibility: Literal["listed", "internal"] = "listed",
        learning: Literal["eligible", "excluded"] = "eligible",
    ) -> SessionAttributes:
        return cls(visibility, learning, tuple(sorted(dimensions.items())))

    def dimension(self, name: str) -> str:
        """缺失维度按 default 解析；新增维度不需要迁移旧 Session。"""
        return dict(self.scope).get(name, "default")


@dataclass(frozen=True, slots=True)
class SessionEntry:
    """目录中的只读事实；不持有执行状态，也不替 UI 生成标题。"""

    session_id: str
    created_at: datetime
    updated_at: datetime
    attributes: SessionAttributes
    metadata: Mapping[str, object] | None
    head_seq: int
    message_count: int
    first_message: Message | None
    """显式标题覆盖；None 表示由表示边界按首条消息推导。"""
    title: str | None = None


@dataclass(frozen=True, slots=True)
class SessionPage:
    items: tuple[SessionEntry, ...]
    total: int
    next_cursor: tuple[str, str] | None


@dataclass(frozen=True, slots=True)
class MessagePage:
    """同一读取快照中的有序消息、引用和固定上界，不保存副本或消费进度。"""

    messages: tuple[Message, ...]
    attachments: Mapping[str, tuple[AttachmentRef, ...]]
    bindings: Mapping[str, Mapping[str, object]]
    through_seq: int
    has_more: bool


class InvalidPage(ValueError):
    """调用者的分页范围或游标无效；与持久记录损坏区分。"""


class MessageConflict(ValueError):
    """消息身份、引用或来源前缀发生冲突。"""


class SourceHeadConflict(MessageConflict):
    """来源 head 的 CAS 失败，事务未提交；调用者可重新选择前缀。"""


class WriterExpired(RuntimeError):
    """任务已释放写入权，不能再提交新的输出。"""


@dataclass(frozen=True, slots=True)
class MessageSnapshot(Sequence[Message]):
    """固定消息读面；只提供消息和前缀关系，不持有数据库或写入能力。"""

    _prefix: _MessagePrefix
    _count: int
    through_seq: int

    @property
    def session_id(self) -> str:
        return self._prefix.session_id

    @property
    def prefix_revision(self) -> int | None:
        return self._prefix.revision

    def extends(self, previous: MessageSnapshot) -> bool:
        """相同存储读面只追加；截短或前缀变化不能复用旧投影。"""
        return self._prefix is previous._prefix and self._count >= previous._count

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[Message]:
        return islice(self._prefix.messages, self._count)

    @overload
    def __getitem__(self, index: int) -> Message: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Message, ...]: ...

    def __getitem__(self, index: int | slice) -> Message | tuple[Message, ...]:
        if isinstance(index, slice):
            return tuple(self._prefix.messages[i] for i in range(*index.indices(self._count)))
        position = index + self._count if index < 0 else index
        if not 0 <= position < self._count:
            raise IndexError(index)
        return self._prefix.messages[position]


@dataclass(frozen=True, slots=True, weakref_slot=True)
class OwnerRecord:
    version: int
    value: Mapping[str, object]


class PreparedAppend(Protocol):
    @property
    def session_id(self) -> str: ...


class MessageCatalog(Protocol):
    def snapshot_heads(
        self, *, prefix: str = "", visibility: Literal["listed", "internal"] | None = None,
    ) -> Mapping[str, int]: ...

    def reader(self, session_id: str) -> MessageReader: ...

    def attributes(self, session_id: str) -> SessionAttributes: ...

    def exists(self, session_id: str) -> bool: ...

    def sessions(
        self, *, prefix: str = "", visibility: Literal["listed", "internal"] | None = None,
        after: tuple[str, str] | None = None, limit: int = 50,
    ) -> SessionPage: ...

    def snapshot_attributes(self) -> Mapping[str, SessionAttributes]: ...

    def follow_metadata(self) -> AsyncGenerator[None, None]: ...

    def follow(
        self, *, poll_interval: float | None = None, wake_on: type[Body] | tuple[type[Body], ...] | None = None,
        prefix: str = "", visibility: Literal["listed", "internal"] | None = None,
    ) -> AsyncGenerator[Mapping[str, int]]: ...


@runtime_checkable
class MessageReader(Protocol):
    def incremental(self) -> MessageReader: ...

    def committed_snapshot(self, *, through_seq: int | None = None) -> MessageSnapshot: ...

    async def committed_snapshot_async(self, *, through_seq: int | None = None) -> MessageSnapshot: ...

    async def read_async(self, consume: Callable[[MessageReader], _T]) -> _T: ...

    async def snapshot_async(self, *, through_seq: int) -> tuple[Message, ...]: ...

    def read_snapshot(self) -> AbstractContextManager[None]: ...

    def source_changed(self, source: str, through_seq: int) -> bool: ...

    @property
    def session_id(self) -> str: ...

    def metadata(self) -> Mapping[str, object] | None: ...

    @property
    def attributes(self) -> SessionAttributes: ...

    @property
    def deleted(self) -> bool: ...

    @property
    def title(self) -> str | None: ...

    def read(
        self,
        *,
        after_seq: int = -1,
        through_seq: int | None = None,
        source: str | None = None,
        limit: int = 1000,
    ) -> tuple[Message, ...]: ...

    def source_names(self) -> frozenset[str]: ...

    def latest_input(self, source: str, *, through_seq: int) -> Message | None: ...

    def latest_input_seq(self, source: str, *, through_seq: int) -> int | None: ...

    def latest_finished_output_seq(
        self, source: str, *, after_seq: int, through_seq: int,
    ) -> int | None: ...

    def scan_controls(
        self, consume: Callable[[Iterable[tuple[int, Control]]], _T], *,
        source: str, after_seq: int, through_seq: int,
    ) -> _T: ...

    def latest_control(self, source: str, *, through_seq: int) -> Message | None: ...

    def scan(
        self, consume: Callable[[Iterable[Message]], _T], *, after_seq: int = -1,
        through_seq: int | None = None, source: str | None = None,
    ) -> _T: ...

    def snapshot(self, *, after_seq: int = -1, through_seq: int | None = None) -> tuple[Message, ...]: ...

    def read_page(
        self, *, after_seq: int = -1, through_seq: int | None = None, limit: int = 50,
    ) -> MessagePage: ...

    def read_tail(
        self, *, before_seq: int | None = None, through_seq: int | None = None, limit: int = 50,
    ) -> MessagePage: ...

    def get(self, message_id: str) -> Message | None: ...

    def attachments(self, message_id: str) -> tuple[AttachmentRef, ...]: ...

    def attachments_for(self, message_ids: tuple[str, ...]) -> tuple[AttachmentRef, ...]: ...

    def head(self, *, source: str | None = None) -> int: ...

    def follow_heads(self) -> AsyncGenerator[int, None]: ...

    def follow(
        self, *, after_seq: int = -1, poll_interval: float | None = None
    ) -> AsyncGenerator[Message, None]: ...


class MessageWriter(Protocol):
    @property
    def session_id(self) -> str: ...

    @property
    def source(self) -> str: ...

    def check(self, body: Body) -> None: ...

    def expire(self) -> None: ...

    def append(
        self, message_id: str, body: Body, *, expected_source_head: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Message: ...

    async def prepare_async(
        self, message_id: str, body: Body, *, metadata: Mapping[str, object] | None = None,
    ) -> PreparedAppend: ...

    async def append_async(
        self, message_id: str, body: Body, *,
        expected_source_head: int | None = None, metadata: Mapping[str, object] | None = None,
        on_commit: Callable[[Message, bool], None] | None = None,
    ) -> Message: ...


class OwnerStore(Protocol):
    def check_access(self, *capabilities: MessageReader | MessageWriter) -> None: ...

    def read(self, key: str) -> OwnerRecord | None: ...

    def list(self) -> tuple[tuple[str, OwnerRecord], ...]: ...

    def scan(self, *, start: str, stop: str, limit: int = 100) -> tuple[tuple[str, OwnerRecord], ...]: ...

    def snapshot(self, callback: Callable[[], _T]) -> _T: ...

    def transact(self, callback: Callable[[OwnerTransaction], _T]) -> _T: ...

    async def transact_async(
        self, callback: Callable[[OwnerTransaction], _T], *,
        on_commit: Callable[[_T], None] | None = None,
    ) -> _T: ...


class OwnerTransaction(Protocol):
    def source_changed(
        self, reader: MessageReader, source: str, through_seq: int,
    ) -> bool: ...

    def read(self, key: str) -> OwnerRecord | None: ...

    def save(
        self, key: str, value: Mapping[str, object], *, expected_version: int | None
    ) -> OwnerRecord: ...

    def append(
        self,
        writer: MessageWriter,
        message_id: str,
        body: Body,
        *,
        expected_source_head: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Message: ...

    def append_prepared(
        self, prepared: PreparedAppend, *, expected_source_head: int | None = None,
    ) -> Message: ...


class MessageEmbeddings(Protocol):
    def bind(self, text: Callable[[Message], str]) -> EmbeddingRecords: ...


class EmbeddingRecords(Protocol):
    def read(self, message: Message, *, model: str, dimension: int) -> tuple[float, ...] | None: ...

    def save(self, message: Message, *, model: str, embedding: Sequence[float]) -> None: ...


MESSAGE_WRITERS = ServiceKey[MessageWriters]("ledger.message_writers.v1")
OWNER_STATE = ServiceKey[OwnerState]("ledger.owner_state.v1")

MESSAGE_CATALOG = ServiceKey[MessageCatalog]("ledger.message_catalog.v1")

SESSION_ADMIN = ServiceKey[SessionAdmin]("ledger.session_admin.v1")

MESSAGE_EMBEDDINGS = ServiceKey[MessageEmbeddings]("ledger.message_embeddings.v1")
SESSION_ADMISSION = ServiceKey[SessionAdmission]("ledger.session_admission.v1")


class Bindings(Protocol):
    """固定业务选择与来源证据，并借用当前 provider。"""

    def bind(
        self, service: ServiceKey[Any], metadata: Mapping[str, object], *,
        contributors: tuple[Context, ...] = (),
    ) -> str: ...

    def describe(self, identity: str, service: ServiceKey[Any]) -> Mapping[str, object]: ...

    def open(
        self, identity: str, service: ServiceKey[_T],
    ) -> AbstractAsyncContextManager[tuple[_T, Mapping[str, object]]]: ...


BINDINGS = ServiceKey[Bindings]("ledger.bindings.v1")


class ArtifactRead(Protocol):
    """只授予有界读取，不暴露附件路径或 repository。"""

    async def acquire(self, ref: AttachmentRef) -> AttachmentReadLease: ...


class ArtifactImport(Protocol):
    """导入来源并取得不可变引用，不授予消息写入或删除权。"""

    async def import_source(self, source: str, kind: AttachmentKind) -> AttachmentRef: ...


ARTIFACT_READ = ServiceKey[ArtifactRead]("ledger.artifact_read.v1")
ARTIFACT_IMPORT = ServiceKey[ArtifactImport]("ledger.artifact_import.v1")


class PendingInputs(Protocol):
    def __call__(self, *, channel: str, session_key: str, provider_message_id: str) -> bool: ...


class PendingAttachments(Protocol):
    def __call__(self, *, channel: str, session_key: str, provider_message_id: str) -> tuple[AttachmentRef, ...] | None: ...


class RejectInput(Protocol):
    def __call__(self, *, channel: str, session_key: str, provider_message_id: str) -> Awaitable[None]: ...


@dataclass(frozen=True, slots=True)
class InputCustody:
    prepare_channel_input: Callable[[InboundEnvelope], Coroutine[Any, Any, None]]
    complete_channel_input: Callable[[InboundEnvelope], Coroutine[Any, Any, None]]
    retain_channel_input: Callable[[InboundEnvelope], Coroutine[Any, Any, None]]
    reserve_durable_inbound: Callable[[RawInbound], Coroutine[Any, Any, bool]]
    defer_durable_inbound: Callable[[str], Awaitable[bool]]
    settle_rejected_inbound: RejectInput
    has_pending_durable_inbound: PendingInputs
    pending_durable_attachment_refs: PendingAttachments
    recover_durable_inbounds: Callable[[Callable[[RawInbound], Coroutine[Any, Any, bool]]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ChannelIdentity:
    resolve: Callable[[str, str], str | None]
    remember: Callable[[str, str, str], Coroutine[Any, Any, object]]
    rollback: Callable[[object], Awaitable[bool]]


class ImportAttachment(Protocol):
    async def __call__(self, data: bytes, *, kind: AttachmentKind, filename: str | None, media_type: str | None) -> AttachmentRef: ...


class AcquireAttachment(Protocol):
    async def __call__(self, ref: AttachmentRef) -> AttachmentReadLease: ...


@dataclass(frozen=True, slots=True)
class ChannelAttachmentImport:
    import_bytes: ImportAttachment


@dataclass(frozen=True, slots=True)
class ChannelAttachmentRead:
    resolve_refs: Callable[[tuple[str, ...]], tuple[AttachmentRef, ...]]
    acquire: AcquireAttachment


INPUT_CUSTODY = ServiceKey[InputCustody]("ledger.input_custody.v1")
CHANNEL_IDENTITY = ServiceKey[ChannelIdentity]("ledger.channel_identity.v1")
CHANNEL_ATTACHMENT_IMPORT = ServiceKey[ChannelAttachmentImport]("ledger.channel_attachment_import.v1")
CHANNEL_ATTACHMENT_READ = ServiceKey[ChannelAttachmentRead]("ledger.channel_attachment_read.v1")

APPEND_CHECKS = ServiceKey[AppendChecks]("ledger.append_checks.v1")

def detect_supported_image_mime(head: bytes) -> str | None:
    """根据文件签名识别 Akashic 支持的图片格式。"""

    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head.startswith(b"BM"):
        return "image/bmp"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp"
    return None
