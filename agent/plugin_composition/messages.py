from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol


from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey
from session.embedding_store import (
    EmbeddingRecords,
    MessageEmbeddings,
)
from session.log import (
    MessagePage as MessagePage,
    SessionEntry as SessionEntry,
    MessageReader as MessageReader,
    MessageSnapshot as MessageSnapshot,
    MessageCatalog as MessageCatalog,
    MessageWriter as MessageWriter,
    OwnerStore as OwnerStore,
    OwnerTransaction as OwnerTransaction,
    OwnerRecord as OwnerRecord,
    SessionAttributes as SessionAttributes,
    WriterExpired as WriterExpired,
    MessageConflict as MessageConflict,
    SourceHeadConflict as SourceHeadConflict,
    InvalidPage as InvalidPage,
)
from session.message import Body, CallRef, ContentPart, ContentReferences, Control, Input, Output, ToolCall, ToolResult


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


MESSAGE_WRITERS = ServiceKey[MessageWriters]("core.message_writers")
OWNER_STATE = ServiceKey[OwnerState]("core.owner_state")

MESSAGE_CATALOG = ServiceKey[MessageCatalog]("core.message_catalog")

SESSION_ADMIN = ServiceKey[SessionAdmin]("core.session_admin")

MESSAGE_EMBEDDINGS = ServiceKey[MessageEmbeddings]("core.message_embeddings")
SESSION_ADMISSION = ServiceKey[SessionAdmission]("core.session_admission")


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
