from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from contextlib import AbstractContextManager
from itertools import islice
import re
from typing import Literal, Protocol, TypeVar, overload, runtime_checkable


from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey
from session.artifacts import AttachmentRef
from session.message import Body, CallRef, ContentPart, ContentReferences, Control, Input, Message, Output, ToolCall, ToolResult


@dataclass(frozen=True, slots=True)
class SessionDeleteResult:
    session_key: str
    deleted: bool
    deleted_at: str | None


@dataclass(frozen=True, slots=True)
class SessionTitleResult:
    session_key: str
    title: str | None


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


__all__ = [
    "EmbeddingRecords",
    "InvalidPage",
    "MESSAGE_CATALOG",
    "MESSAGE_EMBEDDINGS",
    "MESSAGE_WRITERS",
    "MessageCatalog",
    "MessageConflict",
    "MessageEmbeddings",
    "MessageReader",
    "MessageSnapshot",
    "MessageWriter",
    "MessageWriters",
    "OWNER_STATE",
    "OwnerRecord",
    "OwnerState",
    "OwnerStore",
    "OwnerTransaction",
    "SESSION_ADMIN",
    "SESSION_ADMISSION",
    "SessionAdmin",
    "SessionAdmission",
    "SessionAttributes",
    "SessionDeleteResult",
]

_T = TypeVar("_T")

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


MESSAGE_WRITERS = ServiceKey[MessageWriters]("core.message_writers")
OWNER_STATE = ServiceKey[OwnerState]("core.owner_state")

MESSAGE_CATALOG = ServiceKey[MessageCatalog]("core.message_catalog")

SESSION_ADMIN = ServiceKey[SessionAdmin]("core.session_admin")

MESSAGE_EMBEDDINGS = ServiceKey[MessageEmbeddings]("core.message_embeddings")
SESSION_ADMISSION = ServiceKey[SessionAdmission]("core.session_admission")
