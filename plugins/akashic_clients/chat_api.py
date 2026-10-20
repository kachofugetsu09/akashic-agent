from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from core.common.file_io import run_file_io

from .static import current_chat_build, register_chat_assets
from plugins.ui.contract import PluginUiSlot, accepts_gzip_header, read_message_rows, session_row
from .navigation import NavigationPreferences, PinUpdate, session_pin_row
from .notifications import NotificationFeed, NotificationRequest, notification_events
from .read_marks import ReadMarks
from .session_activity import follow_session_activity
from .services import AttachmentStorePort as AttachmentStore
from .services import (
    ActiveSessionsFollowPort,
    InvalidPage,
    MessageCatalogPort as MessageCatalog,
    MessageDisplayReader,
    PluginUiPluginUnavailable,
    PluginUiProvider,
    PluginUiQueryOverloaded,
    PluginUiQueryTimeout,
    PluginUiRpcExecutionError,
    PluginUiRpcInvalidRequest,
    PluginUiStaleRevision,
    SessionAdminPort,
)
from .services import ArtifactStorePort as ChannelAttachmentArtifactStore
from .web_chat import (
    MAX_UPLOAD_BYTES,
    UploadTooLargeError,
    WebChatChannel,
)


class WebPluginUiQueryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    plugin_id: str = Field(min_length=1, max_length=128)
    plugin_revision: str = Field(min_length=1, max_length=128)
    method: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,255}$")
    payload: dict[str, object]
    slot: PluginUiSlot
    session_id: str | None = Field(default=None, max_length=512)
    turn_id: str | None = Field(default=None, max_length=128)


class RenameSessionRequest(BaseModel):
    """标题覆盖请求：空串或全空白清除覆盖；长度上限与服务端合同一致。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(default="", max_length=512)


class WebUiProvider(Protocol):
    async def bootstrap(self) -> bytes: ...

    async def state(self) -> dict[str, str | bool]: ...


# bootstrap 载荷按目录身份缓存(plugins/ui/plugin.py);同一载荷的 gzip 结果也缓存。
# plugin-ui 模块同样按内容哈希(sha256 参数即内容指纹)缓存压缩结果。
# 容量覆盖当前目录的模块数量加 bootstrap,内容更新后旧条目随即被淘汰。
_gzip_cache: dict[str, bytes] = {}


def _gzip_cached(payload_sha256: str, payload: bytes) -> bytes:
    body = _gzip_cache.get(payload_sha256)
    if body is None:
        body = gzip.compress(payload, compresslevel=6, mtime=0)
        _gzip_cache[payload_sha256] = body
        while len(_gzip_cache) > 24:
            del _gzip_cache[next(iter(_gzip_cache))]
    return body


class ReadMarkUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(min_length=1, max_length=512)
    seq: int = Field(ge=-1)


def create_chat_app(
    *,
    workspace: Path,
    channel: WebChatChannel,
    navigation: NavigationPreferences | None = None,
    check_scope: Callable[[str, str], Awaitable[None]] | None = None,
    read_marks: ReadMarks | None = None,
    message_display: MessageDisplayReader | None = None,
    plugin_ui_provider: PluginUiProvider | None = None,
    plugin_ui_scope: Callable[[], Any] | None = None,
    web_ui_provider: WebUiProvider | None = None,
    messages: MessageCatalog | None = None,
    reply_status: Callable[[str], AsyncGenerator[dict[str, object], None]] | None = None,
    follow_active_sessions: ActiveSessionsFollowPort | None = None,
    message_scope: Callable[[], Any] | None = None,
    session_admin_scope: Callable[[], Any] | None = None,
    attachment_store: AttachmentStore | None = None,
    artifact_store: ChannelAttachmentArtifactStore | None = None,
) -> FastAPI:
    if messages is not None and message_scope is not None:
        raise ValueError("chat API 不能同时绑定直接消息 provider 与 request scope")
    if plugin_ui_provider is not None and plugin_ui_scope is not None:
        raise ValueError("chat API 不能同时绑定直接 Plugin UI provider 与 request scope")
    if messages is not None:
        channel.bind_message_readers(messages, reply_status)
    if message_display is not None:
        channel.bind_message_display(message_display)
    if attachment_store is not None:
        channel.bind_attachment_store(attachment_store)
    if artifact_store is not None:
        channel.bind_artifact_store(artifact_store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        yield

    app = FastAPI(title="Akashic Chat API", lifespan=lifespan)
    app.state.workspace = workspace
    app.state.channel = channel

    @asynccontextmanager
    async def open_message_catalog() -> AsyncGenerator[MessageCatalog, None]:
        """Hold the message catalog only while one HTTP operation is running."""

        if message_scope is not None:
            async with message_scope() as catalog:
                yield cast(MessageCatalog, catalog)
            return
        if messages is None:
            raise HTTPException(status_code=503, detail="会话日志不可用")
        yield messages

    async def session_activity_stream() -> AsyncGenerator[dict[str, object], None]:
        """短借目录后独立跟随；订阅不持有请求 scope 或消息写权限。"""
        async def reply_activity() -> AsyncGenerator[frozenset[str] | None, None]:
            if follow_active_sessions is None:
                yield None
                return
            async with aclosing(follow_active_sessions()) as frames:
                async for frame in frames:
                    yield frame

        marks = await ready_read_marks()
        async with open_message_catalog() as catalog:
            heads = catalog.follow(prefix=f"{channel.name}:", visibility="listed", poll_interval=60)
        async with aclosing(follow_session_activity(
            heads, reply_activity(), prefix=f"{channel.name}:", reads=None if marks is None else marks.follow(),
        )) as frames:
            async for frame in frames:
                yield frame

    # 首次使用时以当时的会话水位为基线；之后出现的新会话从头算未读。
    async def ready_read_marks() -> ReadMarks | None:
        if read_marks is None:
            return None
        if not read_marks.has_baseline():
            read_marks.ensure_baseline(await NotificationFeed(open_message_catalog, prefix=f"{channel.name}:").baseline())
        return read_marks

    if messages is not None or message_scope is not None:
        channel.bind_session_activity(session_activity_stream)

    @asynccontextmanager
    async def open_session_admin() -> AsyncGenerator[SessionAdminPort, None]:
        """会话软删/恢复是显式数据管理操作，只在一次 HTTP 操作内借用窄端口。"""

        if session_admin_scope is None:
            raise HTTPException(status_code=503, detail="会话管理暂不可用")
        async with session_admin_scope() as admin:
            yield cast(SessionAdminPort, admin)

    static_dir = Path(__file__).resolve().parent / "static/chat"
    register_chat_assets(app, static_dir)

    @app.get("/api/shell/state")
    async def client_state() -> dict[str, object]:
        try:
            async with open_message_catalog():
                ready = True
        except RuntimeError:
            ready = False
        return {"status": "ready" if ready else "unavailable",
                "configured": True, "chatReady": ready, "build": await current_chat_build(static_dir)}

    @app.get("/api/chat/health")
    async def chat_health() -> dict[str, str]:
        try:
            async with open_message_catalog():
                return {"status": "ready"}
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail="聊天请求接纳不可用") from error

    @app.get("/api/chat/web-ui/bootstrap")
    async def web_ui_bootstrap(request: Request) -> Response:
        if web_ui_provider is None:
            raise HTTPException(status_code=503, detail="Web 插件界面服务不可用")
        try:
            payload = await web_ui_provider.bootstrap()
        except RuntimeError as error:
            raise HTTPException(
                status_code=503,
                detail="Web 插件界面服务暂不可用",
            ) from error
        # 每次仍读取当前 snapshot；只有完整响应字节相同才复用浏览器缓存。
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        compressed = accepts_gzip_header(request.headers.get("accept-encoding", ""))
        etag = f'"{payload_sha256}{"-gzip" if compressed else ""}"'
        headers = {
            "Cache-Control": "private, no-cache",
            "ETag": etag,
            "Vary": "Accept-Encoding",
            "X-Content-Type-Options": "nosniff",
        }
        # 压缩代理可能把强 ETag 改为弱 ETag；GET 的条件校验按弱比较匹配。
        validators = request.headers.get("if-none-match", "").split(",")
        if any(value.strip().removeprefix("W/") in {"*", etag} for value in validators):
            return Response(status_code=304, headers=headers)
        # 目录字节是 JSON 内嵌的压缩模块,线上约 1MB;gzip 后约剩 1/4。
        # 强 ETag 区分编码；固定 gzip 时间戳，让相同载荷跨进程仍得到相同字节。
        body = payload
        if compressed:
            # 首次压缩约 35ms,排水到文件 worker,不占用事件循环;命中缓存的线程往返可忽略。
            body = await run_file_io(lambda: _gzip_cached(payload_sha256, payload))
            headers["Content-Encoding"] = "gzip"
        return Response(content=body, media_type="application/json", headers=headers)

    @app.get("/api/chat/web-ui/state")
    async def web_ui_state() -> Response:
        if web_ui_provider is None:
            raise HTTPException(status_code=503, detail="Web 插件界面服务不可用")
        try:
            state = await web_ui_provider.state()
        except RuntimeError as error:
            raise HTTPException(
                status_code=503,
                detail="Web 插件界面服务暂不可用",
            ) from error
        return Response(
            content=json.dumps(state, ensure_ascii=False, separators=(",", ":")),
            media_type="application/json",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/api/chat/sessions")
    async def list_sessions(
        page_size: int = Query(50, ge=1, le=200),
        after_time: str | None = Query(default=None),
        after_key: str | None = Query(default=None),
    ) -> dict[str, object]:
        if (after_time is None) != (after_key is None):
            raise HTTPException(status_code=422, detail="目录 cursor 需要时间与会话 ID")
        after = None if after_time is None or after_key is None else (after_time, after_key)
        async with open_message_catalog() as catalog:
            try:
                page = await catalog.sessions_async(
                    prefix=f"{channel.name}:", visibility="listed", after=after, limit=page_size)
            except InvalidPage as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        return {"items": [session_row(cast(Any, entry)) for entry in page.items], "total": page.total,
                "next_cursor": None if page.next_cursor is None else {
                    "updated_at": page.next_cursor[0], "session_id": page.next_cursor[1]}}

    async def _set_session_deleted(session_key: str, *, deleted: bool) -> dict[str, object]:
        if not session_key.startswith(f"{channel.name}:"):
            raise HTTPException(status_code=400, detail="只能管理当前聊天目录中的会话")
        try:
            async with open_session_admin() as admin:
                result = await admin.set_deleted(session_key, deleted=deleted)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="会话不存在") from error
        return {"key": result.session_key, "deleted": result.deleted,
                "deleted_at": result.deleted_at}

    @app.post("/api/chat/sessions/{session_key:path}/delete")
    async def delete_session(session_key: str) -> dict[str, object]:
        """软删当前聊天目录中的会话；幂等，消息物理保留，可随时恢复。"""

        return await _set_session_deleted(session_key, deleted=True)

    @app.post("/api/chat/sessions/{session_key:path}/undelete")
    async def undelete_session(session_key: str) -> dict[str, object]:
        """恢复已软删的会话；幂等，只清除 deleted_at 标记。"""

        return await _set_session_deleted(session_key, deleted=False)

    @app.post("/api/chat/sessions/{session_key:path}/rename")
    async def rename_session(session_key: str, payload: RenameSessionRequest) -> dict[str, object]:
        """覆盖会话显示标题；空标题清除覆盖回到首条消息推导，幂等。"""
        if not session_key.startswith(f"{channel.name}:"):
            raise HTTPException(status_code=400, detail="只能管理当前聊天目录中的会话")
        try:
            async with open_session_admin() as admin:
                result = await admin.set_title(session_key, payload.title)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="会话不存在") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"key": result.session_key, "title": result.title}

    @app.post("/api/chat/notifications/stream")
    async def notification_stream(payload: NotificationRequest) -> StreamingResponse:
        feed = NotificationFeed(open_message_catalog, prefix=f"{channel.name}:")
        cursor = await feed.baseline() if payload.cursor is None else payload.cursor
        marks = await ready_read_marks()
        return StreamingResponse(
            notification_events(feed, cursor, reads=None if marks is None else marks.read_all,
                                read_events=payload.read_events and marks is not None),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/chat/sessions/read")
    async def mark_session_read(payload: ReadMarkUpdate) -> dict[str, object]:
        marks = await ready_read_marks()
        if marks is None:
            raise HTTPException(status_code=503, detail="已读状态暂不可用")
        if not payload.session_id.startswith(f"{channel.name}:"):
            raise HTTPException(status_code=400, detail="只能标记本通道的会话")
        # 水位不超过已提交消息，避免客户端把尚未出现的消息提前标为已读。
        async with open_message_catalog() as catalog:
            try:
                reader = catalog.reader(payload.session_id)
                deleted = await reader.deleted_async()
                head = await reader.head_async()
            except (KeyError, ValueError) as error:
                raise HTTPException(status_code=404, detail="会话不存在") from error
            if deleted:
                raise HTTPException(status_code=404, detail="会话不存在")
        seq = marks.mark(payload.session_id, min(payload.seq, head))
        return {"session_id": payload.session_id, "seq": seq}

    async def read_pins() -> dict[str, object]:
        if navigation is None:
            raise HTTPException(status_code=503, detail="置顶偏好暂不可用")
        async with open_message_catalog() as catalog:
            refs = navigation.read()
            sessions = [row for ref in refs if ref.kind == "session"
                        if (row := await session_pin_row(catalog, ref.id)) is not None]
        return {"pins": [ref.model_dump() for ref in refs], "sessions": sessions}

    @app.get("/api/chat/navigation/pins")
    async def navigation_pins() -> JSONResponse:
        return JSONResponse(await read_pins(), headers={"Cache-Control": "no-store"})

    @app.post("/api/chat/navigation/pins")
    async def update_navigation_pin(request: PinUpdate) -> dict[str, object]:
        if navigation is None:
            raise HTTPException(status_code=503, detail="置顶偏好暂不可用")
        async with open_message_catalog() as catalog:
            reference = request.reference()
            # Replaying an already committed pin must not revalidate a temporarily missing target.
            if request.pinned and reference not in navigation.read():
                if reference.kind == "session":
                    if await session_pin_row(catalog, reference.id) is None:
                        raise HTTPException(status_code=400, detail="只能置顶可见的会话")
                else:
                    await _check_scope_pin(check_scope, reference.kind, reference.id)
            navigation.update(reference, pinned=request.pinned)
        return await read_pins()

    @app.get("/api/chat/navigation")
    def chat_navigation() -> dict[str, str]:
        return {"dashboard_path": "/"}

    @app.get("/api/chat/plugin-ui/catalog")
    async def plugin_ui_catalog() -> dict[str, object]:
        if plugin_ui_scope is not None:
            async with plugin_ui_scope() as provider:
                return await provider.catalog()
        return await _require_plugin_ui_provider(plugin_ui_provider).catalog()

    @app.get("/api/chat/plugin-ui/asset")
    async def plugin_ui_asset(
        request: Request,
        plugin_id: str = Query(..., min_length=1, max_length=128),
        plugin_revision: str = Query(..., min_length=1, max_length=128),
        kind: Literal["module", "stylesheet"] = Query(...),
        sha256: str = Query(..., pattern=r"^[0-9a-f]{64}$"),
    ) -> Response:
        try:
            if plugin_ui_scope is not None:
                async with plugin_ui_scope() as provider:
                    asset = await provider.asset(
                        plugin_id,
                        plugin_revision,
                        kind,
                        sha256,
                    )
            else:
                asset = await _require_plugin_ui_provider(plugin_ui_provider).asset(
                    plugin_id,
                    plugin_revision,
                    kind,
                    sha256,
                )
        except (PluginUiPluginUnavailable, PluginUiStaleRevision) as error:
            raise _plugin_ui_http_error(error) from error
        headers = {
            "Cache-Control": "private, max-age=31536000, immutable",
            "Vary": "Accept-Encoding",
        }
        # 模块字节以 sha256 参数为内容指纹且 immutable;接受 gzip 时送缓存的压缩变体。
        content = str(asset["content"])
        body: str | bytes = content
        if accepts_gzip_header(request.headers.get("accept-encoding", "")):
            body = await run_file_io(lambda: _gzip_cached(sha256, content.encode("utf-8")))
            headers["Content-Encoding"] = "gzip"
        return Response(
            content=body,
            media_type="text/javascript" if kind == "module" else "text/css",
            headers=headers,
        )

    @app.post("/api/chat/plugin-ui/query")
    async def plugin_ui_query(
        request: WebPluginUiQueryPayload,
    ) -> dict[str, object]:
        try:
            encoded = json.dumps(
                request.payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except ValueError as error:
            raise HTTPException(status_code=400, detail="插件参数不是有效 JSON") from error
        if len(encoded) > 64 * 1024:
            raise HTTPException(status_code=413, detail="插件参数超过 64 KiB")
        # 插件按 session_id 读写会话数据；入口先把它限定在本通道，插件不必各自重做通道授权。
        if request.session_id and not request.session_id.startswith(f"{channel.name}:"):
            raise HTTPException(status_code=400, detail="插件查询只能指向本通道的会话")
        try:
            if plugin_ui_scope is not None:
                async with plugin_ui_scope() as provider:
                    return await provider.query(
                        request.plugin_id,
                        request.plugin_revision,
                        request.method,
                        request.payload,
                        session_id=request.session_id,
                        turn_id=request.turn_id,
                    )
            return await _require_plugin_ui_provider(plugin_ui_provider).query(
                request.plugin_id,
                request.plugin_revision,
                request.method,
                request.payload,
                session_id=request.session_id,
                turn_id=request.turn_id,
            )
        except (
            PluginUiPluginUnavailable,
            PluginUiStaleRevision,
            PluginUiQueryOverloaded,
            PluginUiQueryTimeout,
            PluginUiRpcInvalidRequest,
            PluginUiRpcExecutionError,
        ) as error:
            raise _plugin_ui_http_error(error) from error

    @app.get("/api/chat/sessions/{session_key:path}/messages")
    async def list_messages(
        session_key: str,
        page_size: int = Query(50, ge=1, le=200),
        before_seq: int | None = Query(default=None, ge=0),
        through_seq: int | None = Query(default=None, ge=-1),
    ) -> dict[str, object]:
        async with open_message_catalog() as catalog:
            try:
                reader = catalog.reader(session_key)
                page = await reader.read_tail_async(
                    before_seq=before_seq, through_seq=through_seq, limit=page_size)
                items = await read_message_rows(
                    cast(Any, page),
                    display_only=True,
                    reader=channel.message_display,
                )
                # 软删会话仍物理存在：如实返回消息并标记 deleted，供前端只读展示。
                deleted = await reader.deleted_async()
            except KeyError as error:
                raise HTTPException(status_code=404, detail="会话不存在") from error
            except InvalidPage as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        return {"version": 2, "items": items, "through_seq": page.through_seq,
                "has_more": page.has_more, "deleted": deleted,
                "before_seq": page.messages[0].seq if page.has_more else None}

    @app.websocket("/ws")
    async def chat_ws(websocket: WebSocket, watch_sessions: bool = False) -> None:
        await channel.handle_websocket(websocket, watch_sessions=watch_sessions)

    @app.post("/api/chat/uploads")
    async def upload_file(
        request: Request,
        filename: str = Query(default="upload.bin"),
    ) -> dict[str, object]:
        declared_length = request.headers.get("content-length")
        if declared_length is not None:
            try:
                declared = int(declared_length)
                if declared < 0:
                    raise ValueError("负数")
                if declared > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="上传内容超过 50MB 限制")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Content-Length 非法") from exc
        clean_name = Path(filename).name or "upload.bin"
        try:
            return await channel.save_upload_stream(
                request.stream(),
                clean_name,
                max_bytes=MAX_UPLOAD_BYTES,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/chat/artifacts/{artifact_id}")
    async def read_artifact(artifact_id: str) -> Response:
        try:
            data, media_type, filename = await channel.read_artifact(artifact_id)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=404, detail="附件不存在") from error
        headers: dict[str, str] = {}
        if filename:
            safe_filename = Path(filename).name.replace('"', "").replace("\r", "").replace("\n", "")
            if safe_filename:
                encoded_filename = quote(safe_filename, safe="")
                headers["Content-Disposition"] = (
                    f"inline; filename*=utf-8''{encoded_filename}"
                    if encoded_filename != safe_filename
                    else f'inline; filename="{safe_filename}"'
                )
        # Any attachment can be opened as a document, including XHTML and XML.
        headers["Content-Security-Policy"] = "sandbox allow-scripts"
        headers["X-Content-Type-Options"] = "nosniff"
        return Response(
            content=data,
            media_type=media_type or "application/octet-stream",
            headers=headers,
        )

    @app.get("/api/chat/media")
    def read_media(path: str = Query(...)) -> FileResponse:
        requested = Path(path).expanduser().resolve()
        if not _can_read_media(channel, requested):
            raise HTTPException(status_code=404, detail="文件不存在")
        if not requested.is_file():
            raise HTTPException(status_code=404, detail="文件不存在")
        return FileResponse(requested)

    return app


def build_chat_server(
    *,
    workspace: Path,
    channel: WebChatChannel,
    navigation: NavigationPreferences | None = None,
    check_scope: Callable[[str, str], Awaitable[None]] | None = None,
    read_marks: ReadMarks | None = None,
    message_display: MessageDisplayReader | None = None,
    plugin_ui_provider: PluginUiProvider | None = None,
    plugin_ui_scope: Callable[[], Any] | None = None,
    web_ui_provider: WebUiProvider | None = None,
    messages: MessageCatalog | None = None,
    reply_status: Callable[[str], AsyncGenerator[dict[str, object], None]] | None = None,
    follow_active_sessions: ActiveSessionsFollowPort | None = None,
    message_scope: Callable[[], Any] | None = None,
    session_admin_scope: Callable[[], Any] | None = None,
    attachment_store: AttachmentStore | None = None,
    artifact_store: ChannelAttachmentArtifactStore | None = None,
    uds: str,
) -> uvicorn.Server:
    config = uvicorn.Config(
        create_chat_app(
            workspace=workspace,
            channel=channel,
            navigation=navigation,
            check_scope=check_scope,
            read_marks=read_marks,
            message_display=message_display,
            plugin_ui_provider=plugin_ui_provider,
            plugin_ui_scope=plugin_ui_scope,
            web_ui_provider=web_ui_provider,
            messages=messages,
            reply_status=reply_status,
            follow_active_sessions=follow_active_sessions,
            message_scope=message_scope,
            session_admin_scope=session_admin_scope,
            attachment_store=attachment_store,
            artifact_store=artifact_store,
        ),
        uds=uds,
        log_level="warning",
        access_log=False,
        timeout_graceful_shutdown=10,
    )
    return ChatApiServer(config)


class ChatApiServer(uvicorn.Server):
    """Chat API 的退出请求立即进入 uvicorn 优雅关闭。"""

    def __init__(self, config: uvicorn.Config) -> None:
        self._exit_loop: asyncio.AbstractEventLoop | None = None
        self._exit_event: asyncio.Event | None = None
        self._exit_requested = False
        super().__init__(config)

    @property
    def should_exit(self) -> bool:
        return self._exit_requested

    # uvicorn 只轮询该标志；置位时同时通知主循环，可能来自信号处理器。
    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self._exit_requested = value
        loop, event = self._exit_loop, self._exit_event
        if value and loop is not None and event is not None and not loop.is_closed():
            _ = loop.call_soon_threadsafe(event.set)

    # 未请求退出时保持 uvicorn 每 0.1s 一次的 on_tick 节拍。
    async def main_loop(self) -> None:
        self._exit_event = asyncio.Event()
        self._exit_loop = asyncio.get_running_loop()
        counter = 0
        should_exit = await self.on_tick(counter)
        while not should_exit:
            counter = (counter + 1) % 864000
            try:
                _ = await asyncio.wait_for(self._exit_event.wait(), timeout=0.1)
            except TimeoutError:
                pass
            should_exit = await self.on_tick(counter)


# 维度值置顶交给该维度 owner 的接纳判定；owner 缺席或判定拒绝都不写入偏好。
async def _check_scope_pin(
    check: Callable[[str, str], Awaitable[None]] | None, dimension: str, value: str,
) -> None:
    if check is None:
        raise HTTPException(status_code=503, detail="会话维度判定暂不可用")
    try:
        await check(dimension, value)
    except PermissionError as error:
        raise HTTPException(status_code=400, detail=f"没有插件负责这个维度: {dimension}") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _require_plugin_ui_provider(
    provider: PluginUiProvider | None,
) -> PluginUiProvider:
    if provider is None:
        raise HTTPException(status_code=503, detail="插件界面服务不可用")
    return provider


def _plugin_ui_http_error(error: Exception) -> HTTPException:
    if isinstance(error, PluginUiPluginUnavailable):
        return HTTPException(status_code=404, detail={
            "code": "plugin_ui_unavailable", "message": "此插件界面已卸载或暂不可用。",
        })
    if isinstance(error, PluginUiStaleRevision):
        return HTTPException(status_code=409, detail={
            "code": "plugin_ui_stale_revision", "message": "插件界面版本已变更。",
        })
    if isinstance(error, PluginUiQueryOverloaded):
        return HTTPException(status_code=429, detail=str(error))
    if isinstance(error, PluginUiQueryTimeout):
        return HTTPException(status_code=504, detail=str(error))
    if isinstance(error, PluginUiRpcInvalidRequest):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=502, detail=str(error))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        _ = path.relative_to(root)
        return True
    except ValueError:
        return False


def _can_read_media(channel: WebChatChannel, path: Path) -> bool:
    if any(_is_relative_to(path, root.resolve()) for root in channel.upload_roots()):
        return True
    if channel.has_media(path):
        return True
    return False
