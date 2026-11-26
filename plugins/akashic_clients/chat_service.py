"""聊天目录领域操作：HTTP 路由与 CHAT_SESSIONS 消费者调用同一段代码。"""
from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from plugins.ui.contract import read_message_rows, session_row

from .services import ArtifactContent, ChatRequestError
from .navigation import NavigationPreferences, PinUpdate, session_pin_row
from .services import InvalidPage, MessageCatalogPort, SessionAdminPort
from .web_chat import MAX_UPLOAD_BYTES, UploadTooLargeError, WebChatChannel

CatalogOpener = Callable[[], AbstractAsyncContextManager[MessageCatalogPort]]
AdminOpener = Callable[[], AbstractAsyncContextManager[SessionAdminPort]]
ScopeCheck = Callable[[str, str], Awaitable[None]]


# 分页参数边界与界面 Query 声明一致；直接调用方也受同一约束。
def _bounded(name: str, value: int, low: int, high: int | None = None) -> int:
    if value < low or (high is not None and value > high):
        raise ChatRequestError(422, f"{name} 超出范围 [{low}, {high if high is not None else '∞'}]")
    return value


class ChatService:
    """一条 Web 通道 binding 的聊天目录操作；每次调用短借目录或管理端口。"""

    def __init__(
        self, *, channel: WebChatChannel, open_catalog: CatalogOpener, open_admin: AdminOpener | None,
        navigation: NavigationPreferences | None, check_scope: ScopeCheck | None,
    ) -> None:
        self._channel = channel
        self._open_catalog = open_catalog
        self._open_admin = open_admin
        self._navigation = navigation
        self._check_scope = check_scope

    @property
    def channel_name(self) -> str:
        return self._channel.name

    @property
    def _prefix(self) -> str:
        return f"{self._channel.name}:"

    async def list_sessions(
        self, *, page_size: int = 50, after_time: str | None = None, after_key: str | None = None,
    ) -> dict[str, object]:
        if (after_time is None) != (after_key is None):
            raise ChatRequestError(422, "目录 cursor 需要时间与会话 ID")
        limit = _bounded("page_size", page_size, 1, 200)
        after = None if after_time is None or after_key is None else (after_time, after_key)
        async with self._open_catalog() as catalog:
            # 纪元在查询前读取：客户端据此判断首连基线通知是否已被这次读取覆盖。
            epoch = catalog.metadata_epoch
            try:
                page = await catalog.sessions_async(prefix=self._prefix, visibility="listed", after=after, limit=limit)
            except InvalidPage as error:
                raise ChatRequestError(422, str(error)) from error
        return {"items": [session_row(cast(Any, entry)) for entry in page.items], "total": page.total,
                "catalog_epoch": epoch,
                "next_cursor": None if page.next_cursor is None else {
                    "updated_at": page.next_cursor[0], "session_id": page.next_cursor[1]}}

    async def list_messages(
        self, session_key: str, *, page_size: int = 50, before_seq: int | None = None,
        through_seq: int | None = None, after_seq: int | None = None,
    ) -> dict[str, object]:
        # after_seq 往新的方向读（查找跳到较早位置后向下补齐）；不带时照旧从尾部往前翻。
        if after_seq is not None and before_seq is not None:
            raise ChatRequestError(422, "after_seq 与 before_seq 不能同时使用")
        limit = _bounded("page_size", page_size, 1, 200)
        for name, value, low in (("before_seq", before_seq, 0), ("through_seq", through_seq, -1), ("after_seq", after_seq, -1)):
            if value is not None:
                _ = _bounded(name, value, low)
        async with self._open_catalog() as catalog:
            try:
                reader = catalog.reader(session_key)
                if after_seq is not None:
                    newer = await reader.read_async(lambda current: current.read_page(
                        after_seq=after_seq, through_seq=through_seq, limit=limit))
                    rows = await read_message_rows(cast(Any, newer), display_only=True, reader=self._channel.message_display)
                    return {"version": 2, "items": rows, "through_seq": newer.through_seq,
                            "has_more": newer.has_more, "after_seq": after_seq}
                page = await reader.read_tail_async(before_seq=before_seq, through_seq=through_seq, limit=limit)
                items = await read_message_rows(cast(Any, page), display_only=True, reader=self._channel.message_display)
                # 软删会话仍物理存在：如实返回消息并标记 deleted，供前端只读展示。
                deleted = await reader.deleted_async()
            except KeyError as error:
                raise ChatRequestError(404, "会话不存在") from error
            except InvalidPage as error:
                raise ChatRequestError(422, str(error)) from error
        return {"version": 2, "items": items, "through_seq": page.through_seq,
                "has_more": page.has_more, "deleted": deleted,
                "before_seq": page.messages[0].seq if page.has_more else None}

    # 会话内查找：按完整历史匹配正文，返回命中消息的位置与摘要；空查询只建好索引。
    async def search_messages(self, session_key: str, *, q: str = "", limit: int = 200) -> dict[str, object]:
        if len(q) > 200:
            raise ChatRequestError(422, "q 超过 200 字")
        bounded = _bounded("limit", limit, 1, 500)
        async with self._open_catalog() as catalog:
            try:
                found = await catalog.reader(session_key).find_text_async(q, limit=bounded)
            except KeyError as error:
                raise ChatRequestError(404, "会话不存在") from error
        return {"version": 1, "query": q, "total": found.total, "capped": found.capped,
                "through_seq": found.through_seq,
                "items": [{"message_id": hit.message_id, "seq": hit.seq, "kind": hit.kind,
                           "timestamp": hit.timestamp, "count": hit.count,
                           "before": hit.before, "match": hit.match, "after": hit.after} for hit in found.hits]}

    # 侧栏目录的可见性：本通道前缀 + listed；软删会话仍可只读查看。
    async def ensure_listed(self, session_key: str) -> None:
        if not session_key.startswith(self._prefix):
            raise ChatRequestError(400, "只能读取当前聊天目录中的会话")
        async with self._open_catalog() as catalog:
            try:
                attributes = await catalog.reader(session_key).read_async(lambda row: row.attributes)
            except KeyError as error:
                raise ChatRequestError(404, "会话不存在") from error
        if attributes.visibility != "listed":
            raise ChatRequestError(404, "会话不存在")

    async def set_deleted(self, session_key: str, *, deleted: bool) -> dict[str, object]:
        self._require_managed(session_key)
        try:
            async with self._admin() as admin:
                result = await admin.set_deleted(session_key, deleted=deleted)
        except KeyError as error:
            raise ChatRequestError(404, "会话不存在") from error
        return {"key": result.session_key, "deleted": result.deleted, "deleted_at": result.deleted_at}

    # 覆盖会话显示标题；空标题清除覆盖回到首条消息推导，幂等。
    async def rename(self, session_key: str, title: str) -> dict[str, object]:
        self._require_managed(session_key)
        if len(title) > 512:
            raise ChatRequestError(422, "标题超过 512 字")
        try:
            async with self._admin() as admin:
                result = await admin.set_title(session_key, title)
        except KeyError as error:
            raise ChatRequestError(404, "会话不存在") from error
        except ValueError as error:
            raise ChatRequestError(422, str(error)) from error
        return {"key": result.session_key, "title": result.title}

    async def read_pins(self) -> dict[str, object]:
        navigation = self._require_navigation()
        async with self._open_catalog() as catalog:
            refs = navigation.read()
            sessions = [row for ref in refs if ref.kind == "session"
                        if (row := await session_pin_row(catalog, ref.id)) is not None]
        return {"pins": [ref.model_dump() for ref in refs], "sessions": sessions}

    async def update_pin(self, *, kind: str, id: str, pinned: bool) -> dict[str, object]:
        navigation = self._require_navigation()
        try:
            reference = PinUpdate(kind=kind, id=id, pinned=pinned).reference()
        except ValidationError as error:
            raise ChatRequestError(422, str(error)) from error
        async with self._open_catalog() as catalog:
            # Replaying an already committed pin must not revalidate a temporarily missing target.
            if pinned and reference not in navigation.read():
                if reference.kind == "session":
                    if await session_pin_row(catalog, reference.id) is None:
                        raise ChatRequestError(400, "只能置顶可见的会话")
                else:
                    await self._check_scope_pin(reference.kind, reference.id)
            navigation.update(reference, pinned=pinned)
        return await self.read_pins()

    # 附件登记：有界读取后交给通道 artifact owner 导入，返回可随消息发送的 artifact 身份。
    async def save_upload(self, chunks: AsyncIterable[bytes], filename: str) -> dict[str, object]:
        clean_name = Path(filename).name or "upload.bin"
        try:
            return await self._channel.save_upload_stream(chunks, clean_name, max_bytes=MAX_UPLOAD_BYTES)
        except UploadTooLargeError as error:
            raise ChatRequestError(413, str(error)) from error
        except ValueError as error:
            raise ChatRequestError(400, str(error)) from error

    async def read_artifact(self, artifact_id: str) -> ArtifactContent:
        try:
            data, media_type, filename = await self._channel.read_artifact(artifact_id)
        except (RuntimeError, ValueError) as error:
            raise ChatRequestError(404, "附件不存在") from error
        return ArtifactContent(artifact_id, data, media_type, filename)

    def _require_managed(self, session_key: str) -> None:
        if not session_key.startswith(self._prefix):
            raise ChatRequestError(400, "只能管理当前聊天目录中的会话")

    def _admin(self) -> AbstractAsyncContextManager[SessionAdminPort]:
        if self._open_admin is None:
            raise ChatRequestError(503, "会话管理暂不可用")
        return self._open_admin()

    def _require_navigation(self) -> NavigationPreferences:
        if self._navigation is None:
            raise ChatRequestError(503, "置顶偏好暂不可用")
        return self._navigation

    # 维度值置顶交给该维度 owner 的接纳判定；owner 缺席或判定拒绝都不写入偏好。
    async def _check_scope_pin(self, dimension: str, value: str) -> None:
        if self._check_scope is None:
            raise ChatRequestError(503, "会话维度判定暂不可用")
        try:
            await self._check_scope(dimension, value)
        except PermissionError as error:
            raise ChatRequestError(400, f"没有插件负责这个维度: {dimension}") from error
        except ValueError as error:
            raise ChatRequestError(400, str(error)) from error


class ChatSlot:
    """CHAT_SESSIONS 的提供者：当代通道启动后登记服务，停止时撤下。"""

    def __init__(self) -> None:
        self._service: ChatService | None = None

    def current(self) -> ChatService:
        if self._service is None:
            raise ChatRequestError(503, "聊天通道尚未就绪")
        return self._service

    def bind(self, service: ChatService) -> None:
        self._service = service

    def release(self, service: ChatService) -> None:
        if self._service is service:
            self._service = None
