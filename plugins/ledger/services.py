from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack, AsyncExitStack
from dataclasses import dataclass

from core.common.file_io import run_file_io

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from .log import (
    MessageLog as _MessageLog,
    MessageWriter as MessageWriter,
    SessionAttributes as SessionAttributes,
    WriterExpired as WriterExpired,
)
from plugins.ledger.contract import Body, CallRef, ContentPart, ContentReferences, Control, Input, Output, ToolCall, ToolResult


from plugins.ledger.contract import (
    OwnerStore, OwnerTransaction,
    MESSAGE_WRITERS, APPEND_CHECKS, Message, MessageReader,
    OWNER_STATE,
    SESSION_ADMISSION,
    SessionDeleteResult,
    SessionTitleResult,
)


class AppendChecks:
    """检查注册随消费者 Effect 撤回；检查者不取得 writer 或 SQL。"""

    def __init__(self, log: _MessageLog):
        self._log = log

    async def register(self, ctx: Context, check: Callable[[Message, MessageReader], None]) -> Effect:
        ctx.require_runtime_identity(APPEND_CHECKS, self)
        async def setup():
            registered: list[Callable[[], None]] = []
            def add() -> None:
                registered.append(self._log.check_appends(check))
            try:
                await run_file_io(add)
            except BaseException:
                # 取消可能晚于实际登记；Effect 尚未取得 cleanup，必须先撤回。
                if registered:
                    await run_file_io(registered[0])
                raise
            remove = registered[0]
            async def cleanup():
                await run_file_io(remove)
            return cleanup
        return await ctx.effect(setup, label="ledger.append-check")


class MessageWriters:
    """正式组合固定写入范围，消费者只得到现有的窄 MessageWriter。"""

    def __init__(self, log: _MessageLog):
        self._log = log
        self._metadata: dict[str, tuple[Context, Callable[[Body], Mapping[str, object | None]]]] = {}
        self._part_metadata: dict[str, _PartMetadataGrant] = {}

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
            conflicts = keys & (self._metadata.keys() | self._part_metadata_keys())
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
        part_metadata: bool = False,
    ) -> Callable[..., MessageWriter]:
        """固定身份、类型和检查器；打开时只选择 Session 和可选 exact call。"""
        log = self._log
        owner = ctx.require_runtime_identity(MESSAGE_WRITERS, self).plugin_id
        grants = {
            key: grant for key, grant in self._metadata.items()
            if grant[0] is ctx and grant[1] is update_metadata
        }
        if update_metadata is not None and not grants:
            raise PermissionError("metadata 投影未登记给当前 owner")
        parts = dict(self._part_metadata) if part_metadata else {}
        def project_metadata(body: Body) -> Mapping[str, object | None]:
            if any(self._metadata.get(key) is not grant for key, grant in grants.items()):
                raise WriterExpired("metadata 授权已释放")
            # 1. 普通回调只能写自己登记的键；片段 owner 的键只能由对应片段派生。
            changes = dict(update_metadata(body)) if update_metadata is not None else {}
            if not changes.keys() <= grants.keys():
                raise PermissionError(f"metadata 投影写入了未授权的键: {sorted(changes.keys() - grants.keys())}")
            # 2. 片段投影各自校验 owner 授权与键集合。
            changes.update(self._project_parts(parts, body))
            return changes
        projected = update_metadata is not None or bool(parts)
        metadata_keys = frozenset(grants).union(*(grant.keys for grant in parts.values()))
        content = dict(content)
        body_types = tuple(body_types)

        def open(session_id: str, *, call_ref: CallRef | None = None) -> MessageWriter:
            _ = ctx.require_runtime_identity(MESSAGE_WRITERS, self)
            return log.writer(
                session_id, author=author, source=source, body_types=body_types,
                content=content, call_ref=call_ref, check_call=check_call,
                metadata_keys=metadata_keys,
                update_metadata=project_metadata if projected else None,
                message_metadata_keys=frozenset({owner}),
                check_metadata=check_metadata,
            )

        return open

    async def register_part_metadata(
        self, ctx: Context, *, kind: str, keys: frozenset[str],
        update: Callable[[ContentPart], Mapping[str, object | None]],
    ) -> Effect:
        """登记某个 kind 片段的会话元数据投影；卸载只释放内存登记。"""
        if ctx.require(MESSAGE_WRITERS) is not self:
            raise PermissionError("metadata 注册不属于当前 MessageWriters")
        if not kind or not keys:
            raise ValueError("片段 metadata 注册需要 kind 和明确的键")
        def setup():
            if kind in self._part_metadata:
                raise ValueError(f"片段 metadata 已有 owner: {kind}")
            conflicts = keys & (self._metadata.keys() | self._part_metadata_keys())
            if conflicts:
                raise ValueError(f"Session metadata 已有 owner: {sorted(conflicts)}")
            grant = _PartMetadataGrant(keys, update)
            self._part_metadata[kind] = grant
            def cleanup() -> None:
                if self._part_metadata.get(kind) is grant:
                    del self._part_metadata[kind]
            return cleanup
        return await ctx.effect(setup, label="session-part-metadata:" + kind)

    def _part_metadata_keys(self) -> frozenset[str]:
        empty: frozenset[str] = frozenset()
        return empty.union(*(grant.keys for grant in self._part_metadata.values()))

    # 写入者 bind 时取到的片段投影：owner 已卸载则写入失效，投影越权写键则拒绝。
    def _project_parts(
        self, parts: Mapping[str, _PartMetadataGrant], body: Body,
    ) -> Mapping[str, object | None]:
        if not parts or not isinstance(body, Input):
            return {}
        changes: dict[str, object | None] = {}
        for part in body.parts:
            grant = parts.get(part.kind)
            if grant is None:
                continue
            if self._part_metadata.get(part.kind) is not grant:
                raise WriterExpired("片段 metadata 授权已释放")
            delta = grant.update(part)
            if not set(delta) <= grant.keys:
                raise PermissionError(f"片段 metadata 投影越权: {part.kind}")
            changes.update(delta)
        return changes


@dataclass(frozen=True, slots=True)
class _PartMetadataGrant:
    keys: frozenset[str]
    update: Callable[[ContentPart], Mapping[str, object | None]]


class OwnerState:
    """按实际插件 owner 分配同库事务空间；没有任意 namespace 或 SQL 参数。"""

    def __init__(self, log: _MessageLog):
        self._log = log

    def open(self, ctx: Context) -> OwnerStore:
        return self._log.owner("plugin:" + ctx.require_runtime_identity(OWNER_STATE, self).plugin_id)

    def open_scoped(self, ctx: Context, scope: str) -> OwnerStore:
        """同一 owner 的独立子空间；其他消费者的 key 扫描互不可见。"""
        if not isinstance(scope, str) or not scope or ":" in scope:
            raise ValueError("owner state 子空间名必须是非空且不含冒号的字符串")
        owner = ctx.require_runtime_identity(OWNER_STATE, self).plugin_id
        return self._log.owner(f"plugin:{owner}:{scope}")





class SessionAdmin:
    """会话软删/恢复、标题覆盖与空标题条件写入，不授予消息减少权限。"""

    def __init__(self, log: _MessageLog):
        self._log = log

    async def set_deleted(self, session_key: str, *, deleted: bool) -> SessionDeleteResult:
        deleted_at = await run_file_io(
            lambda: self._log.set_session_deleted(session_key, deleted=deleted)
        )
        return SessionDeleteResult(session_key, deleted=deleted_at is not None, deleted_at=deleted_at)

    async def set_title(self, session_key: str, title: str | None) -> SessionTitleResult:
        stored = await run_file_io(
            lambda: self._log.set_session_title(session_key, title)
        )
        return SessionTitleResult(session_key, stored)

    async def set_title_if_unset(self, session_key: str, title: str) -> bool:
        """仅在未命名且未软删时写入；竞争失败返回 False，不重试。"""
        return await run_file_io(lambda: self._log.set_session_title_if_unset(session_key, title))


class SessionAdmission:
    """仅授予固定属性的 create-once，不带元数据改写、删除或消息权限。"""

    def __init__(self, log: _MessageLog):
        self._log = log
        self._dimensions: dict[str, tuple[Context, Callable[[str], None]]] = {}
        self._initializers: dict[str, tuple[Context, OwnerStore, Callable[[str, SessionAttributes, OwnerTransaction], None]]] = {}

    async def register_initializer(
        self, ctx: Context, *, name: str,
        initialize: Callable[[str, SessionAttributes, OwnerTransaction], None],
    ) -> Effect:
        """Admit plugin state only with a new Session, using that owner's transaction."""
        if ctx.require(SESSION_ADMISSION) is not self:
            raise PermissionError("初始化注册不属于当前 SessionAdmission")
        store = ctx.require(OWNER_STATE).open(ctx)
        if not name or not callable(initialize):
            raise ValueError("Session 初始化需要名称和同步函数")

        def setup():
            if name in self._initializers:
                raise ValueError(f"Session 初始化已有 owner: {name}")
            self._initializers[name] = (ctx, store, initialize)
            return lambda: self._initializers.pop(name)

        return await ctx.effect(setup, label="session-initializer:" + name)

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
        _ = ctx.require_runtime_identity(SESSION_ADMISSION, self)
        # 1. 维度值只在首次接纳时由其 owner 校验；已有 Session 只比较固定事实。
        if attributes.scope and not self._admitted(session_id):
            self._check_dimensions(attributes)
        # 2. 固定事实写入与冲突检查仍由同一个 create-once 事务完成。
        with ExitStack() as scopes:
            registered = tuple(self._initializers.values())
            for owner, _, _ in registered:
                _ = scopes.enter_context(owner.call_scope())
            return self._log.ensure_session(session_id, attributes, initializers=tuple(
                (store, initialize) for _, store, initialize in registered
            ))

    async def ensure_async(self, ctx: Context, session_id: str, attributes: SessionAttributes) -> SessionAttributes:
        """原 scope 校验维度；完整 create-once 事务进入文件线程并排空取消。"""
        log = self._log
        _ = ctx.require_runtime_identity(SESSION_ADMISSION, self)
        if attributes.scope and not await run_file_io(lambda: self._admitted(session_id)):
            self._check_dimensions(attributes)
        async with AsyncExitStack() as scopes:
            registered = tuple(self._initializers.values())
            for owner, _, _ in registered:
                _ = await scopes.enter_async_context(owner.runtime_scope())
            return await log.ensure_session_async(session_id, attributes, initializers=tuple(
                (store, initialize) for _, store, initialize in registered
            ))

    def check_scope(self, dimension: str, value: str) -> None:
        self._check_dimensions(SessionAttributes(scope=((dimension, value),)))

    def _check_dimensions(self, attributes: SessionAttributes) -> None:
        for name, value in attributes.scope:
            grant = self._dimensions.get(name)
            if grant is None:
                raise PermissionError(f"Session 维度没有 owner: {name}")
            with grant[0].call_scope():
                grant[1](value)

    def _admitted(self, session_id: str) -> bool:
        try:
            _ = self._log.catalog().attributes(session_id)
        except ValueError:
            return False
        return True
