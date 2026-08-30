from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from agent.plugins.mobile_ui import (
    MobileUiPluginUnavailable,
    MobileUiProvider,
    MobileUiQueryOverloaded,
    MobileUiQueryTimeout,
    MobileUiRpcExecutionError,
    MobileUiRpcInvalidRequest,
    MobileUiStaleRevision,
)
from agent.model_runtime.session_selection import read_session_model_selection
from agent.plugin_composition import ModelCatalogSnapshot
from agent.plugins.model_catalog import (
    ModelCatalogUnavailable,
    default_chat_model_id,
    project_chat_runtimes,
)
from infra.channels.base import AttachmentStore
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from infra.channels.web_chat_channel import (
    MAX_UPLOAD_BYTES,
    UploadTooLargeError,
    WebChatChannel,
)
from infra.mobile_realtime.pairing import PairingError
from infra.mobile_realtime.runtime_inspection import (
    RuntimeInspectionError,
    RuntimeInspectionService,
)
from infra.mobile_realtime.storage import PairingStateError
from session.store import SessionStore

if TYPE_CHECKING:
    from agent.plugin_composition.model_settings_http import ModelControl
    from infra.mobile_realtime.gateway import MobilePairingAdmin


class PairingApprovalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    confirmation_code: str = Field(pattern=r"^[0-9]{6}$")


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


class WebUiProvider(Protocol):
    async def bootstrap(self) -> bytes: ...

    async def state(self) -> dict[str, str]: ...


def create_chat_app(
    *,
    workspace: Path,
    channel: WebChatChannel,
    mobile_pairing_admin: MobilePairingAdmin | None = None,
    runtime_inspection: RuntimeInspectionService | None = None,
    plugin_ui_provider: MobileUiProvider | None = None,
    web_ui_provider: WebUiProvider | None = None,
    model_catalog_reader: Callable[[], Awaitable[ModelCatalogSnapshot]] | None = None,
    model_control: ModelControl | None = None,
) -> FastAPI:
    channel.bind_attachment_store(AttachmentStore(workspace / "uploads"))
    channel_context = channel._ctx
    if channel.artifact_store is None and channel_context is not None:
        session_store = channel_context.session_manager.control_store
        if isinstance(session_store, SessionStore):
            channel.bind_artifact_store(
                ChannelAttachmentArtifactStore(
                    workspace=workspace,
                    session_store=session_store,
                    max_import_bytes=MAX_UPLOAD_BYTES,
                )
            )
    app = FastAPI(title="Akashic Chat API")
    if model_control is not None:
        from agent.plugin_composition.model_settings_http import create_model_settings_router

        app.include_router(create_model_settings_router(model_control))
    app.state.workspace = workspace
    app.state.channel = channel
    project_root = Path(__file__).resolve().parent.parent
    static_dir = project_root / "static" / "chat"
    index_file = static_dir / "index.html"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/assets",
        StaticFiles(directory=static_dir, check_dir=False),
        name="chat_assets",
    )

    @app.get("/", response_model=None)
    def chat_index() -> FileResponse | dict[str, str]:
        if index_file.exists():
            return FileResponse(index_file)
        return {"status": "ok", "channel": channel.name}

    @app.get("/api/chat/health")
    def chat_health() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/api/chat/web-ui/bootstrap")
    async def web_ui_bootstrap() -> Response:
        if web_ui_provider is None:
            raise HTTPException(status_code=503, detail="Web 插件界面服务不可用")
        try:
            payload = await web_ui_provider.bootstrap()
        except RuntimeError as error:
            raise HTTPException(
                status_code=503,
                detail="Web 插件界面服务暂不可用",
            ) from error
        return Response(
            content=payload,
            media_type="application/json",
            headers={
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
                "X-Content-Type-Options": "nosniff",
            },
        )

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
    def list_sessions(page: int = Query(1), page_size: int = Query(50)) -> dict[str, Any]:
        ctx = channel._require_ctx()
        items, total = ctx.session_manager._store.list_sessions_for_dashboard(
            channel=channel.name,
            page=page,
            page_size=page_size,
        )
        visible = [
            item
            for item in items
            if str(item.get("first_message_content") or "").strip()
        ]
        return {"items": visible, "total": len(visible)}

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
            session = channel._require_ctx().session_manager.get_or_create(session_key)
            selection = read_session_model_selection(session.metadata)
            session_override = selection.model_ref
            session_effort = selection.reasoning_effort
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
        }

    @app.get("/api/chat/plugin-ui/catalog")
    def plugin_ui_catalog() -> dict[str, object]:
        return _require_plugin_ui_provider(plugin_ui_provider).catalog()

    @app.get("/api/chat/plugin-ui/asset")
    def plugin_ui_asset(
        plugin_id: str = Query(..., min_length=1, max_length=128),
        plugin_revision: str = Query(..., min_length=1, max_length=128),
        kind: Literal["module", "stylesheet"] = Query(...),
        sha256: str = Query(..., pattern=r"^[0-9a-f]{64}$"),
    ) -> Response:
        try:
            asset = _require_plugin_ui_provider(plugin_ui_provider).asset(
                plugin_id,
                plugin_revision,
                kind,
                sha256,
            )
        except (MobileUiPluginUnavailable, MobileUiStaleRevision) as error:
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
            return await _require_plugin_ui_provider(plugin_ui_provider).query(
                request.plugin_id,
                request.plugin_revision,
                request.method,
                request.payload,
                session_id=request.session_id,
                turn_id=request.turn_id,
            )
        except (
            MobileUiPluginUnavailable,
            MobileUiStaleRevision,
            MobileUiQueryOverloaded,
            MobileUiQueryTimeout,
            MobileUiRpcInvalidRequest,
            MobileUiRpcExecutionError,
        ) as error:
            raise _plugin_ui_http_error(error) from error

    @app.get("/api/chat/runtime/documents")
    def list_runtime_documents() -> dict[str, object]:
        return _require_runtime_inspection(runtime_inspection).list_documents()

    @app.get("/api/chat/runtime/documents/{document_id}")
    def read_runtime_document(document_id: str) -> dict[str, object]:
        try:
            return _require_runtime_inspection(runtime_inspection).get_document(
                document_id
            )
        except RuntimeInspectionError as error:
            raise _runtime_http_error(error) from error

    @app.get("/api/chat/runtime/jobs")
    def list_runtime_jobs() -> dict[str, object]:
        return _require_runtime_inspection(runtime_inspection).list_jobs()

    @app.get("/api/chat/runtime/jobs/{job_id}")
    def read_runtime_job(job_id: str) -> dict[str, object]:
        try:
            return _require_runtime_inspection(runtime_inspection).get_job(job_id)
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
    def list_messages(
        session_key: str,
        page_size: int = Query(50, ge=1, le=200),
        before_seq: int | None = Query(default=None, ge=0),
    ) -> dict[str, Any]:
        ctx = channel._require_ctx()
        items, total, has_more = ctx.session_manager._store.list_chat_history_page(
            session_key=session_key,
            page_size=page_size,
            before_seq=before_seq,
        )
        _project_message_attachments(channel, items)
        next_before_seq = items[0]["seq"] if items and has_more else None
        return {
            "items": items,
            "total": total,
            "has_more": has_more,
            "before_seq": next_before_seq,
        }

    @app.websocket("/ws")
    async def chat_ws(websocket: WebSocket) -> None:
        await channel.handle_websocket(websocket)

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

    if mobile_pairing_admin is not None:

        @app.post("/api/chat/mobile-pairing")
        def create_mobile_pairing() -> dict[str, object]:
            return mobile_pairing_admin.create_offer()

        @app.get("/api/chat/mobile-pairing/{pairing_id}")
        def read_mobile_pairing(pairing_id: str) -> dict[str, object]:
            claim = mobile_pairing_admin.pending_claim(pairing_id)
            if claim is None:
                return {"pairing_id": pairing_id, "status": "waiting_for_phone"}
            return {**claim, "status": "waiting_for_desktop_confirmation"}

        @app.post("/api/chat/mobile-pairing/{pairing_id}/approve")
        def approve_mobile_pairing(
            pairing_id: str,
            payload: PairingApprovalPayload,
        ) -> dict[str, object]:
            try:
                return mobile_pairing_admin.approve(
                    pairing_id,
                    payload.confirmation_code,
                )
            except (PairingError, PairingStateError) as error:
                raise HTTPException(status_code=409, detail=str(error)) from error

    return app


def _project_message_attachments(
    channel: WebChatChannel,
    items: list[dict[str, Any]],
) -> None:
    """把 durable message artifact identity 投影为不含路径的 Web descriptor。"""

    # 1. Only messages with durable bindings require the Core artifact owner.
    for item in items:
        raw_ids = item.get("attachment_ids")
        if raw_ids is None:
            continue
        if not isinstance(raw_ids, list) or not all(
            isinstance(artifact_id, str) and artifact_id for artifact_id in raw_ids
        ):
            raise RuntimeError("chat message attachment_ids 投影无效")
        store = channel.artifact_store
        if store is None:
            raise RuntimeError("Web artifact store 尚未绑定")

        # 2. Preserve durable order and expose no filesystem path.
        artifact_ids = tuple(raw_ids)
        refs = store.resolve_refs(artifact_ids)
        if tuple(ref.artifact_id for ref in refs) != artifact_ids:
            raise RuntimeError("chat message attachment_ids 无法 exact resolve")
        item["attachments"] = [
            channel.artifact_descriptor(ref) for ref in refs
        ]


def build_chat_server(
    *,
    workspace: Path,
    channel: WebChatChannel,
    mobile_pairing_admin: MobilePairingAdmin | None = None,
    runtime_inspection: RuntimeInspectionService | None = None,
    plugin_ui_provider: MobileUiProvider | None = None,
    web_ui_provider: WebUiProvider | None = None,
    model_catalog_reader: Callable[[], Awaitable[ModelCatalogSnapshot]] | None = None,
    model_control: ModelControl | None = None,
    uds: str,
) -> uvicorn.Server:
    config = uvicorn.Config(
        create_chat_app(
            workspace=workspace,
            channel=channel,
            mobile_pairing_admin=mobile_pairing_admin,
            runtime_inspection=runtime_inspection,
            plugin_ui_provider=plugin_ui_provider,
            web_ui_provider=web_ui_provider,
            model_catalog_reader=model_catalog_reader,
            model_control=model_control,
        ),
        uds=uds,
        log_level="warning",
        access_log=False,
    )
    return uvicorn.Server(config)


def _require_runtime_inspection(
    service: RuntimeInspectionService | None,
) -> RuntimeInspectionService:
    if service is None:
        raise HTTPException(status_code=503, detail="运行时检查服务不可用")
    return service


def _require_plugin_ui_provider(
    provider: MobileUiProvider | None,
) -> MobileUiProvider:
    if provider is None:
        raise HTTPException(status_code=503, detail="插件界面服务不可用")
    return provider


def _plugin_ui_http_error(error: Exception) -> HTTPException:
    if isinstance(error, MobileUiPluginUnavailable):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, MobileUiStaleRevision):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, MobileUiQueryOverloaded):
        return HTTPException(status_code=429, detail=str(error))
    if isinstance(error, MobileUiQueryTimeout):
        return HTTPException(status_code=504, detail=str(error))
    if isinstance(error, MobileUiRpcInvalidRequest):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=502, detail=str(error))


def _runtime_http_error(error: RuntimeInspectionError) -> HTTPException:
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
    try:
        ctx = channel._require_ctx()
    except RuntimeError:
        return False
    store = ctx.session_manager._store
    media_path_exists = getattr(store, "media_path_exists", None)
    if callable(media_path_exists):
        return bool(media_path_exists(path))
    return False
