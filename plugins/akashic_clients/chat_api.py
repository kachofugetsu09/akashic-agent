from __future__ import annotations

from dataclasses import asdict
from plugins.models.contract import ModelCallStats

import hashlib
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition.models import ModelControlUnavailable, ModelError

from .static import register_chat_assets
from agent.plugin_composition.message_view import read_message_rows, session_row
from .navigation import NavigationPreferences, PinUpdate, check_project_pin, session_pin_row
from .notifications import NotificationFeed, NotificationRequest, notification_events
from .services import AttachmentStorePort as AttachmentStore
from .services import (
    InvalidPage,
    MessageCatalogPort as MessageCatalog,
    MessageDisplayReader,
    ModelCatalogSnapshot,
    ChatModelSelection,
    ModelCatalogUnavailable,
    PluginUiPluginUnavailable,
    PluginUiProvider,
    PluginUiQueryOverloaded,
    PluginUiQueryTimeout,
    PluginUiRpcExecutionError,
    PluginUiRpcInvalidRequest,
    PluginUiStaleRevision,
    SessionAdminPort,
    default_chat_model_id,
    project_chat_runtimes,
    project_unavailable_chat_runtimes,
)
from .services import ArtifactStorePort as ChannelAttachmentArtifactStore
from .web_chat import (
    MAX_UPLOAD_BYTES,
    UploadTooLargeError,
    WebChatChannel,
)
from .runtime_inspection import (
    RuntimeInspectionError,
    RuntimeInspectionService,
)


class WebPluginUiQueryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    plugin_id: str = Field(min_length=1, max_length=128)
    plugin_revision: str = Field(min_length=1, max_length=128)
    method: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,255}$")
    payload: dict[str, object]
    slot: Literal[
        "turn.before_reasoning",
        "turn.before_tool",
        "turn.after_answer",
        "drawer.panel",
    ]
    session_id: str | None = Field(default=None, max_length=512)
    turn_id: str | None = Field(default=None, max_length=128)


class RenameSessionRequest(BaseModel):
    """标题覆盖请求：空串或全空白清除覆盖；长度上限与服务端合同一致。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(default="", max_length=512)


class WebUiProvider(Protocol):
    async def bootstrap(self) -> bytes: ...

    async def state(self) -> dict[str, str | bool]: ...


def create_chat_app(
    *,
    workspace: Path,
    channel: WebChatChannel,
    navigation: NavigationPreferences | None = None,
    runtime_inspection: RuntimeInspectionService | None = None,
    message_display: MessageDisplayReader | None = None,
    plugin_ui_provider: PluginUiProvider | None = None,
    plugin_ui_scope: Callable[[], Any] | None = None,
    web_ui_provider: WebUiProvider | None = None,
    model_catalog_reader: Callable[[], Awaitable[ModelCatalogSnapshot]] | None = None,
    model_call_stats_reader: Callable[[str], Awaitable[ModelCallStats]] | None = None,
    model_selection_reader: Callable[
        [Mapping[str, object]], Awaitable[ChatModelSelection]
    ] | None = None,
    messages: MessageCatalog | None = None,
    reply_status: Callable[[str], AsyncGenerator[dict[str, object], None]] | None = None,
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

    @asynccontextmanager
    async def open_session_admin() -> AsyncGenerator[SessionAdminPort, None]:
        """会话软删/恢复是显式数据管理操作，只在一次 HTTP 操作内借用窄端口。"""

        if session_admin_scope is None:
            raise HTTPException(status_code=503, detail="会话管理暂不可用")
        async with session_admin_scope() as admin:
            yield cast(SessionAdminPort, admin)

    register_chat_assets(app, Path(__file__).resolve().parent / "static/chat")

    @app.get("/api/shell/state")
    async def client_state() -> dict[str, object]:
        try:
            async with open_message_catalog():
                ready = True
        except RuntimeError:
            ready = False
        return {"status": "ready" if ready else "unavailable",
                "configured": True, "chatReady": ready}

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
        etag = f'"{hashlib.sha256(payload).hexdigest()}"'
        headers = {
            "Cache-Control": "private, no-cache",
            "ETag": etag,
            "X-Content-Type-Options": "nosniff",
        }
        # 压缩代理可能把强 ETag 改为弱 ETag；GET 的条件校验按弱比较匹配。
        validators = request.headers.get("if-none-match", "").split(",")
        if any(value.strip().removeprefix("W/") in {"*", etag} for value in validators):
            return Response(status_code=304, headers=headers)
        return Response(content=payload, media_type="application/json", headers=headers)

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
                page = catalog.sessions(prefix=f"{channel.name}:", visibility="listed", after=after, limit=page_size)
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
        return StreamingResponse(
            notification_events(feed, cursor),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    async def read_pins() -> dict[str, object]:
        if navigation is None:
            raise HTTPException(status_code=503, detail="置顶偏好暂不可用")
        async with open_message_catalog() as catalog:
            refs = navigation.read()
            sessions = [row for ref in refs if ref.kind == "session"
                        if (row := session_pin_row(catalog, ref.id)) is not None]
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
                    if session_pin_row(catalog, reference.id) is None:
                        raise HTTPException(status_code=400, detail="只能置顶没有项目归属的可见会话")
                else:
                    try:
                        if plugin_ui_scope is not None:
                            async with plugin_ui_scope() as provider:
                                await check_project_pin(provider, reference.id)
                        else:
                            await check_project_pin(_require_plugin_ui_provider(plugin_ui_provider), reference.id)
                    except ValueError as error:
                        raise HTTPException(status_code=400, detail=str(error)) from error
                    except (PluginUiPluginUnavailable, PluginUiStaleRevision, PluginUiQueryOverloaded,
                            PluginUiQueryTimeout, PluginUiRpcInvalidRequest, PluginUiRpcExecutionError) as error:
                        raise _plugin_ui_http_error(error) from error
            navigation.update(reference, pinned=request.pinned)
        return await read_pins()

    @app.get("/api/chat/navigation")
    def chat_navigation() -> dict[str, str]:
        return {"dashboard_path": "/"}

    @app.get("/api/chat/models")
    async def chat_models(session_key: str = Query(default="")) -> dict[str, object]:
        if model_catalog_reader is None:
            raise HTTPException(status_code=503, detail="模型注册表不可用")
        session_override = ""
        session_effort = ""
        if session_key:
            if model_selection_reader is None:
                raise HTTPException(status_code=503, detail="模型选择服务不可用")
            async with open_message_catalog() as catalog:
                metadata = catalog.reader(session_key).metadata()
            try:
                selection = await model_selection_reader(
                    metadata if metadata is not None else {}
                )
            except RuntimeError as error:
                if not ModelError.matches(error, ModelControlUnavailable):
                    raise
                raise HTTPException(
                    status_code=503,
                    detail="模型选择服务不可用",
                ) from error
            session_override = selection.model_id or ""
            session_effort = selection.reasoning_effort or ""
        try:
            current = await model_catalog_reader()
        except ModelCatalogUnavailable as error:
            raise HTTPException(status_code=503, detail="模型注册表不可用") from error
        return {
            "generationId": current.revision,
            "defaultRuntime": default_chat_model_id(current),
            "sessionOverride": session_override,
            "sessionSelection": {
                "modelRef": session_override,
                "reasoningEffort": session_effort,
            },
            "runtimes": project_chat_runtimes(current),
            "unavailableRuntimes": project_unavailable_chat_runtimes(current),
        }

    @app.get("/api/chat/plugin-ui/catalog")
    async def plugin_ui_catalog() -> dict[str, object]:
        if plugin_ui_scope is not None:
            async with plugin_ui_scope() as provider:
                return await provider.catalog()
        return await _require_plugin_ui_provider(plugin_ui_provider).catalog()

    @app.get("/api/chat/plugin-ui/asset")
    async def plugin_ui_asset(
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
        return Response(
            content=str(asset["content"]),
            media_type="text/javascript" if kind == "module" else "text/css",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
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

    @app.get("/api/chat/model-calls/{call_id}")
    async def model_call_stats(call_id: str) -> dict[str, object]:
        """只返回模型 owner 的公开统计，不转发模型管理命令。"""
        if model_call_stats_reader is None:
            raise HTTPException(status_code=503, detail="模型调用统计不可用")
        try:
            return asdict(await model_call_stats_reader(call_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="模型调用记录不存在") from error
        except RuntimeError as error:
            if not ModelError.matches(error, ModelControlUnavailable):
                raise
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/api/chat/runtime/documents")
    async def list_runtime_documents() -> dict[str, object]:
        try:
            return await _require_runtime_inspection(runtime_inspection).list_documents()
        except RuntimeInspectionError as error:
            raise _runtime_http_error(error) from error

    @app.get("/api/chat/runtime/documents/{document_id}")
    async def read_runtime_document(document_id: str) -> dict[str, object]:
        try:
            return await _require_runtime_inspection(runtime_inspection).get_document(
                document_id
            )
        except RuntimeInspectionError as error:
            raise _runtime_http_error(error) from error

    @app.get("/api/chat/runtime/jobs")
    async def list_runtime_jobs() -> dict[str, object]:
        try:
            return await _require_runtime_inspection(runtime_inspection).list_jobs()
        except RuntimeInspectionError as error:
            raise _runtime_http_error(error) from error

    @app.get("/api/chat/runtime/jobs/{job_id}")
    async def read_runtime_job(job_id: str) -> dict[str, object]:
        try:
            return await _require_runtime_inspection(runtime_inspection).get_job(job_id)
        except RuntimeInspectionError as error:
            raise _runtime_http_error(error) from error

    @app.get("/api/chat/runtime/capabilities")
    async def list_runtime_capabilities() -> dict[str, object]:
        try:
            return await _require_runtime_inspection(
                runtime_inspection
            ).list_capabilities()
        except RuntimeInspectionError as error:
            raise _runtime_http_error(error) from error

    @app.get("/api/chat/runtime/mcp")
    async def read_runtime_mcp(
        owner_id: str = Query(...),
        name: str = Query(...),
    ) -> dict[str, object]:
        try:
            return await _require_runtime_inspection(runtime_inspection).get_mcp(
                owner_id,
                name,
            )
        except RuntimeInspectionError as error:
            raise _runtime_http_error(error) from error

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
                page = reader.read_tail(
                    before_seq=before_seq, through_seq=through_seq, limit=page_size)
                items = await read_message_rows(
                    cast(Any, page),
                    display_only=True,
                    reader=channel.message_display,
                )
                # 软删会话仍物理存在：如实返回消息并标记 deleted，供前端只读展示。
                deleted = reader.deleted
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
                headers["Content-Disposition"] = f'inline; filename="{safe_filename}"'
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
    runtime_inspection: RuntimeInspectionService | None = None,
    message_display: MessageDisplayReader | None = None,
    plugin_ui_provider: PluginUiProvider | None = None,
    plugin_ui_scope: Callable[[], Any] | None = None,
    web_ui_provider: WebUiProvider | None = None,
    model_catalog_reader: Callable[[], Awaitable[ModelCatalogSnapshot]] | None = None,
    model_call_stats_reader: Callable[[str], Awaitable[ModelCallStats]] | None = None,
    model_selection_reader: Callable[
        [Mapping[str, object]], Awaitable[ChatModelSelection]
    ] | None = None,
    messages: MessageCatalog | None = None,
    reply_status: Callable[[str], AsyncGenerator[dict[str, object], None]] | None = None,
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
            runtime_inspection=runtime_inspection,
            message_display=message_display,
            plugin_ui_provider=plugin_ui_provider,
            plugin_ui_scope=plugin_ui_scope,
            web_ui_provider=web_ui_provider,
            model_catalog_reader=model_catalog_reader,
            model_call_stats_reader=model_call_stats_reader,
            model_selection_reader=model_selection_reader,
            messages=messages,
            reply_status=reply_status,
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
    return uvicorn.Server(config)


def _require_runtime_inspection(
    service: RuntimeInspectionService | None,
) -> RuntimeInspectionService:
    if service is None:
        raise HTTPException(status_code=503, detail="运行时检查服务不可用")
    return service


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


def _runtime_http_error(error: RuntimeInspectionError) -> HTTPException:
    if error.code == "inspection_unavailable":
        return HTTPException(status_code=503, detail=str(error))
    status_code = 404 if error.code.endswith("_not_found") else 409
    return HTTPException(status_code=status_code, detail=str(error))


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
