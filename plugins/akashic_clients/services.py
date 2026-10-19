"""Akashic clients 的窄服务合同。

这个模块只描述客户端插件消费的结构能力。宿主可以用任意实现提供这些
Protocol；客户端插件不导入 bootstrap、Core runtime 或其他插件实现。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from pathlib import Path
from typing import Literal, Protocol, TypeVar, runtime_checkable

from plugins.channels.contract import AttachmentRef, ChannelInboundMessage
from plugins.ui.contract import MessageDisplayReader
from plugins.ledger.contract import (
    InvalidPage,
    MessageConflict,
    SessionAttributes,
    SessionDeleteResult,
    SessionTitleResult,
)
from plugins.ui.contract import (
    WebUiProvider as WebUiProvider,
)
from plugins.ui.contract import (
    PluginUiPluginUnavailable,  # noqa: F401 - 显式再导出给本插件消费者。
    PluginUiQueryOverloaded,  # noqa: F401 - 显式再导出给本插件消费者。
    PluginUiQueryTimeout,  # noqa: F401 - 显式再导出给本插件消费者。
    PluginUiRpcExecutionError,  # noqa: F401 - 显式再导出给本插件消费者。
    PluginUiRpcInvalidRequest,  # noqa: F401 - 显式再导出给本插件消费者。
    PluginUiStaleRevision,  # noqa: F401 - 显式再导出给本插件消费者。
)
from plugins.ledger.contract import Message
from plugins.ui.contract import (
    PluginUiProvider as PluginUiProvider,
)

_T = TypeVar("_T")


class MessagePagePort(Protocol):
    """只读消息页的字段合同；具体数据库页由宿主适配。"""

    messages: tuple[Message, ...]
    attachments: Mapping[str, tuple[AttachmentRef, ...]]
    bindings: Mapping[str, Mapping[str, object]]
    through_seq: int
    has_more: bool


class SessionEntryPort(Protocol):
    """会话目录条目的只读字段合同。"""

    session_id: str
    created_at: object
    updated_at: object
    message_count: int
    head_seq: int
    first_message: Message | None
    title: str | None


class SessionPagePort(Protocol):
    """会话目录页的只读字段合同。"""

    items: tuple[SessionEntryPort, ...]
    total: int
    next_cursor: tuple[str, str] | None


class SteerPort(Protocol):
    """生成中的插话；对话插件缺席时 stage 返回 False，客户端按普通消息发送。"""

    async def stage(self, session_id: str, message_id: str, message: ChannelInboundMessage) -> bool: ...

    async def withdraw(self, session_id: str, message_id: str) -> bool: ...


class AttachmentStorePort(Protocol):
    """客户端临时上传与 旧上传分片共享的文件 owner。"""

    root: Path
    def create_path(self, prefix: str, suffix: str) -> Path: ...
    def create_persistent_path(self, prefix: str, suffix: str) -> Path: ...
    def create_staging_path(self, *, prefix: str = ".upload-", suffix: str = ".part") -> Path: ...
    def publish_staging(self, staging: Path, *, prefix: str, suffix: str) -> Path: ...
    def write_bytes(self, data: bytes, *, prefix: str, suffix: str) -> Path: ...


class ArtifactReadLeasePort(Protocol):
    ref: AttachmentRef
    async def read_bytes(self, *, max_bytes: int) -> bytes: ...
    async def read_chunk(self, *, offset: int, max_bytes: int) -> bytes: ...
    async def aclose(self) -> None: ...


@runtime_checkable
class ArtifactStorePort(Protocol):
    """Core artifact owner 的只读/导入投影。"""

    async def import_bytes(self, data: bytes, *, kind: object, filename: str | None, media_type: str | None) -> AttachmentRef: ...
    def resolve_refs(self, artifact_ids: tuple[str, ...]) -> tuple[AttachmentRef, ...]: ...
    async def acquire(self, ref: AttachmentRef) -> ArtifactReadLeasePort: ...


class MessageReaderPort(Protocol):
    @property
    def attributes(self) -> SessionAttributes: ...
    @property
    def deleted(self) -> bool: ...
    async def deleted_async(self) -> bool: ...
    @property
    def title(self) -> str | None: ...
    @property
    def session_id(self) -> str: ...
    def head(self) -> int: ...
    async def head_async(self) -> int: ...
    def follow(self, *, after_seq: int = -1) -> AsyncGenerator[Message, None]: ...
    def metadata(self) -> Mapping[str, object] | None: ...
    def get(self, message_id: str) -> Message | None: ...
    def read_tail(self, *, before_seq: int | None, through_seq: int | None, limit: int) -> MessagePagePort: ...
    async def read_tail_async(self, *, before_seq: int | None, through_seq: int | None, limit: int) -> MessagePagePort: ...
    def read_page(self, *, after_seq: int = -1, through_seq: int | None = None, limit: int = 50) -> MessagePagePort: ...
    async def read_async(self, consume: Callable[[MessageReaderPort], _T]) -> _T: ...


class SessionAdminPort(Protocol):
    """用户显式的会话数据管理操作；软删/恢复与标题覆盖，没有消息或物理删除权限。"""

    async def set_deleted(self, session_key: str, *, deleted: bool) -> SessionDeleteResult: ...
    async def set_title(self, session_key: str, title: str | None) -> SessionTitleResult: ...


class MessageCatalogPort(Protocol):
    def reader(self, session_id: str) -> MessageReaderPort: ...
    def sessions(self, *, prefix: str, visibility: str, after: tuple[str, str] | None, limit: int) -> SessionPagePort: ...
    async def sessions_async(self, *, prefix: str, visibility: str, after: tuple[str, str] | None, limit: int) -> SessionPagePort: ...
    def follow_metadata(self) -> AsyncGenerator[None, None]: ...
    def follow(self, *, poll_interval: float | None = None,
               prefix: str = "", visibility: Literal["listed", "internal"] | None = None,
               ) -> AsyncGenerator[Mapping[str, int], None]: ...


ReplyStatusPort = Callable[[str], AsyncGenerator[dict[str, object], None]]
# 产出 None 表示回复插件未加载，无法判断哪些会话在运行。
ActiveSessionsFollowPort = Callable[[], AsyncGenerator[frozenset[str] | None, None]]




__all__ = [
    "ArtifactReadLeasePort", "ArtifactStorePort", "AttachmentStorePort",
    "InvalidPage", "MessageCatalogPort", "MessageConflict", "MessageDisplayReader",
    "MessagePagePort", "MessageReaderPort", "SessionAdminPort",
    "SessionEntryPort", "SessionPagePort", "Message",
]
