from __future__ import annotations

from collections.abc import Callable, Mapping

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey
# EmbeddingRecords/MessageEmbeddings are service vocabulary.  The repair path
# also needs the explicit store writer for a supplied sessions database.
from session.embedding_store import (
    EmbeddingRecords,
    MessageEmbeddings,
    MessageEmbeddingStore,
)
from session.log import (
    MessageLog as _MessageLog,
    MessageReader as MessageReader,
    MessageCatalog as MessageCatalog,
    MessageWriter as MessageWriter,
    OwnerStore as OwnerStore,
    OwnerTransaction as OwnerTransaction,
    OwnerRecord as OwnerRecord,
    SessionAttributes as SessionAttributes,
    WriterExpired as WriterExpired,
    MessageConflict as MessageConflict,
    InvalidPage as InvalidPage,
)
from session.message import Body, CallRef, ContentPart, ContentReferences, Control, Input, Output, ToolCall, ToolResult



class MessageWriters:
    """正式组合固定写入范围，消费者只得到现有的窄 MessageWriter。"""

    def __init__(self, log: _MessageLog | None):
        self._log = log
        self._metadata: dict[str, tuple[Context, Callable[[Body], Mapping[str, object | None]]]] = {}

    async def register_metadata(
        self, ctx: Context, *, keys: frozenset[str],
        update: Callable[[Body], Mapping[str, object | None]],
    ) -> Effect:
        """注册每个元数据键的唯一 owner 和纯投影；卸载只释放内存注册。"""
        if ctx.require(MESSAGE_WRITERS) is not self:
            raise PermissionError("metadata 注册不属于当前 MessageWriters")
        if not keys:
            raise ValueError("metadata 注册需要明确的键")
        def setup():
            conflicts = keys & self._metadata.keys()
            if conflicts:
                raise ValueError(f"Session metadata 已有 owner: {sorted(conflicts)}")
            grant = (ctx, update)
            for key in keys:
                self._metadata[key] = grant
            def cleanup() -> None:
                for key in keys:
                    del self._metadata[key]
            return cleanup
        return await ctx.effect(setup, label="session-metadata:" + ",".join(sorted(keys)))

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
    ) -> Callable[..., MessageWriter]:
        """固定身份、类型和检查器；打开时只选择 Session 和可选 exact call。"""
        log = self._log
        if log is None:
            raise RuntimeError("candidate 验证期禁止签发消息 writer")
        owner = ctx.require_runtime_owner(MESSAGE_WRITERS, self)
        grants = {
            key: grant for key, grant in self._metadata.items()
            if grant[0] is ctx and grant[1] is update_metadata
        }
        if update_metadata is not None and not grants:
            raise PermissionError("metadata 投影未登记给当前 owner")
        def project_metadata(body: Body) -> Mapping[str, object | None]:
            if any(self._metadata.get(key) is not grant for key, grant in grants.items()):
                raise WriterExpired("metadata 授权已释放")
            assert update_metadata is not None
            return update_metadata(body)
        content = dict(content)
        body_types = tuple(body_types)

        def open(session_id: str, *, call_ref: CallRef | None = None) -> MessageWriter:
            _ = ctx.require_runtime_owner(MESSAGE_WRITERS, self)
            return log.writer(
                session_id, author=author, source=source, body_types=body_types,
                content=content, call_ref=call_ref, check_call=check_call,
                metadata_keys=frozenset(grants),
                update_metadata=project_metadata if update_metadata is not None else None,
                message_metadata_keys=frozenset({owner}),
                check_metadata=check_metadata,
            )

        return open


class OwnerState:
    """按实际插件 owner 分配同库事务空间；没有任意 namespace 或 SQL 参数。"""

    def __init__(self, log: _MessageLog | None):
        self._log = log

    @property
    def available(self) -> bool:
        """Tell providers whether this Root has formal owner storage."""
        return self._log is not None

    def open(self, ctx: Context) -> OwnerStore:
        if self._log is None:
            raise RuntimeError("candidate 验证期禁止访问正式 owner state")
        return self._log.owner("plugin:" + ctx.require_runtime_owner(OWNER_STATE, self))

    def open_scoped(self, ctx: Context, scope: str) -> OwnerStore:
        """同一 owner 的独立子空间；其他消费者的 key 扫描互不可见。"""
        if self._log is None:
            raise RuntimeError("candidate 验证期禁止访问正式 owner state")
        if not isinstance(scope, str) or not scope or ":" in scope:
            raise ValueError("owner state 子空间名必须是非空且不含冒号的字符串")
        owner = ctx.require_runtime_owner(OWNER_STATE, self)
        return self._log.owner(f"plugin:{owner}:{scope}")


class SessionAdmission:
    """仅授予固定属性的 create-once，不带元数据改写、删除或消息权限。"""

    def __init__(self, log: _MessageLog | None):
        self._log = log
        self._dimensions: dict[str, tuple[Context, Callable[[str], None]]] = {}

    # 每个 scope 维度只有一个 owner 负责校验取值；卸载只释放内存注册。
    async def register_dimension(
        self, ctx: Context, *, name: str, check: Callable[[str], None],
    ) -> Effect:
        if ctx.require(SESSION_ADMISSION) is not self:
            raise PermissionError("维度注册不属于当前 SessionAdmission")
        _ = SessionAttributes(scope=((name, "probe"),))
        def setup():
            if name in self._dimensions:
                raise ValueError(f"Session 维度已有 owner: {name}")
            self._dimensions[name] = (ctx, check)
            def cleanup() -> None:
                del self._dimensions[name]
            return cleanup
        return await ctx.effect(setup, label="session-dimension:" + name)

    def ensure(self, ctx: Context, session_id: str, attributes: SessionAttributes) -> SessionAttributes:
        if self._log is None:
            raise RuntimeError("candidate 验证期禁止接纳正式 Session")
        _ = ctx.require_runtime_owner(SESSION_ADMISSION, self)
        # 1. 维度值只在首次接纳时由其 owner 校验；已有 Session 只比较固定事实。
        if attributes.scope and not self._admitted(session_id):
            for name, value in attributes.scope:
                grant = self._dimensions.get(name)
                if grant is None:
                    raise PermissionError(f"Session 维度没有 owner: {name}")
                grant[1](value)
        # 2. 固定事实写入与冲突检查仍由同一个 create-once 事务完成。
        return self._log.ensure_session(session_id, attributes)

    def _admitted(self, session_id: str) -> bool:
        assert self._log is not None
        try:
            _ = self._log.catalog().attributes(session_id)
        except ValueError:
            return False
        return True


MESSAGE_WRITERS = ServiceKey[MessageWriters]("core.message_writers")
OWNER_STATE = ServiceKey[OwnerState]("core.owner_state")

MESSAGE_CATALOG = ServiceKey[MessageCatalog]("core.message_catalog")

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
    "MessageEmbeddingStore",
    "MessageEmbeddings",
    "MessageReader",
    "MessageWriter",
    "MessageWriters",
    "OWNER_STATE",
    "OwnerRecord",
    "OwnerState",
    "OwnerStore",
    "OwnerTransaction",
    "SESSION_ADMISSION",
    "SessionAdmission",
    "SessionAttributes",
]
