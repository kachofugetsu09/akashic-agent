from __future__ import annotations

from collections.abc import Callable, Mapping

from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey
from session.embedding_store import MessageEmbeddings
from session.log import MessageCatalog, MessageLog, MessageWriter, OwnerStore
from session.message import CallRef, ContentPart, ContentReferences, Control, Input, Output, ToolCall, ToolResult



class MessageWriters:
    """正式组合固定写入范围，消费者只得到现有的窄 MessageWriter。"""

    def __init__(self, log: MessageLog | None):
        self._log = log

    def bind(
        self,
        ctx: Context,
        *,
        author: str,
        source: str,
        body_types: tuple[type[Input] | type[Output] | type[ToolResult] | type[Control], ...],
        content: Mapping[str, Callable[[ContentPart], ContentReferences]],
        check_call: Callable[[ToolCall], None] | None = None,
    ) -> Callable[..., MessageWriter]:
        """固定身份、类型和检查器；打开时只选择 Session 和可选 exact call。"""
        log = self._log
        if log is None:
            raise RuntimeError("candidate 验证期禁止签发消息 writer")
        _ = ctx.require_runtime_owner(MESSAGE_WRITERS, self)
        content = dict(content)
        body_types = tuple(body_types)

        def open(session_id: str, *, call_ref: CallRef | None = None) -> MessageWriter:
            _ = ctx.require_runtime_owner(MESSAGE_WRITERS, self)
            return log.writer(
                session_id, author=author, source=source, body_types=body_types,
                content=content, call_ref=call_ref, check_call=check_call,
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


MESSAGE_WRITERS = ServiceKey[MessageWriters]("core.message_writers")
OWNER_STATE = ServiceKey[OwnerState]("core.owner_state")

MESSAGE_CATALOG = ServiceKey[MessageCatalog]("core.message_catalog")

MESSAGE_EMBEDDINGS = ServiceKey[MessageEmbeddings]("core.message_embeddings")
