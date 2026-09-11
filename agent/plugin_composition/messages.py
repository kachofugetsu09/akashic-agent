from __future__ import annotations

from collections.abc import Callable, Mapping

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey
# 同 `session.log`：这三个名字是 embedding 服务的接口词汇（`core.message_embeddings`
# 返回的就是 `MessageEmbeddings`），插件需要在注解与维护路径里命名它们。
# `MessageEmbeddingStore` 是 Core 权威存储实现类，仍由本模块再导出 —— akasha 的
# 维护路径（`plugins/akasha/repair.py`）会按显式给定的 sessions 路径重建 store。
from session.embedding_store import (
    EmbeddingRecords,
    MessageEmbeddings,
    MessageEmbeddingStore,
)
# 这些名字是 Core 消息服务的**接口词汇**：`core.message_catalog` 返回的就是
# `MessageCatalog`，`core.owner_state` 返回的就是 `OwnerStore`。插件需要在注解与
# 异常捕获里命名它们，但不应该 import `session/` 的存储实现模块。服务定义在本文件，
# 因此接口词汇也在这里再导出，作为插件的唯一入口。
from session.log import (
    InvalidPage,
    MessageCatalog,
    MessageConflict,
    MessageLog,
    MessageReader,
    MessageWriter,
    OwnerRecord,
    OwnerStore,
    OwnerTransaction,
    SessionAttributes,
    WriterExpired,
)
from session.message import Body, CallRef, ContentPart, ContentReferences, Control, Input, Output, ToolCall, ToolResult



class MessageWriters:
    """正式组合固定写入范围，消费者只得到现有的窄 MessageWriter。"""

    def __init__(self, log: MessageLog | None):
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

    def __init__(self, log: MessageLog | None):
        self._log = log

    def open(self, ctx: Context) -> OwnerStore:
        if self._log is None:
            raise RuntimeError("candidate 验证期禁止访问正式 owner state")
        return self._log.owner("plugin:" + ctx.require_runtime_owner(OWNER_STATE, self))


class SessionAdmission:
    """仅授予固定属性的 create-once，不带元数据改写、删除或消息权限。"""

    def __init__(self, log: MessageLog | None):
        self._log = log

    def ensure(self, ctx: Context, session_id: str, attributes: SessionAttributes) -> SessionAttributes:
        if self._log is None:
            raise RuntimeError("candidate 验证期禁止接纳正式 Session")
        _ = ctx.require_runtime_owner(SESSION_ADMISSION, self)
        return self._log.ensure_session(session_id, attributes)


MESSAGE_WRITERS = ServiceKey[MessageWriters]("core.message_writers")
OWNER_STATE = ServiceKey[OwnerState]("core.owner_state")

MESSAGE_CATALOG = ServiceKey[MessageCatalog]("core.message_catalog")

MESSAGE_EMBEDDINGS = ServiceKey[MessageEmbeddings]("core.message_embeddings")
SESSION_ADMISSION = ServiceKey[SessionAdmission]("core.session_admission")


__all__ = [
    "InvalidPage",
    "MESSAGE_CATALOG",
    "EmbeddingRecords",
    "MESSAGE_EMBEDDINGS",
    "MESSAGE_WRITERS",
    "OWNER_STATE",
    "SESSION_ADMISSION",
    "MessageCatalog",
    "MessageEmbeddings",
    "MessageEmbeddingStore",
    "MessageConflict",
    "MessageReader",
    "MessageWriters",
    "MessageWriter",
    "OwnerRecord",
    "OwnerState",
    "OwnerStore",
    "OwnerTransaction",
    "SessionAdmission",
    "SessionAttributes",
    "WriterExpired",
]
