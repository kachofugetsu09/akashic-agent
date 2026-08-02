from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from infra.channels.base import AttachmentStore
from infra.channels.web_chat_channel import (
    MAX_UPLOAD_BYTES,
    UploadTooLargeError,
    WebChatChannel,
)
from infra.mobile_realtime.pairing import PairingError
from infra.mobile_realtime.storage import PairingStateError

if TYPE_CHECKING:
    from infra.mobile_realtime.gateway import MobilePairingAdmin


class PairingApprovalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    confirmation_code: str = Field(pattern=r"^[0-9]{6}$")


def create_chat_app(
    *,
    workspace: Path,
    channel: WebChatChannel,
    mobile_pairing_admin: MobilePairingAdmin | None = None,
) -> FastAPI:
    channel.bind_attachment_store(AttachmentStore(workspace / "uploads"))
    app = FastAPI(title="Akashic Chat API")
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
    def chat_navigation() -> dict[str, int]:
        return {"dashboard_port": _public_dashboard_port()}

    @app.get("/api/chat/sessions/{session_key:path}/messages")
    def list_messages(
        session_key: str,
        page: int = Query(1),
        page_size: int = Query(50),
        sort_by: str = Query("seq"),
        sort_order: str = Query("asc"),
    ) -> dict[str, Any]:
        ctx = channel._require_ctx()
        items, total = ctx.session_manager._store.list_messages_for_dashboard(
            session_key=session_key,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return {"items": items, "total": total}

    @app.websocket("/ws")
    async def chat_ws(websocket: WebSocket) -> None:
        await channel.handle_websocket(websocket)

    @app.post("/api/chat/uploads")
    async def upload_file(
        request: Request,
        filename: str = Query(default="upload.bin"),
    ) -> dict[str, str]:
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


def build_chat_server(
    *,
    workspace: Path,
    channel: WebChatChannel,
    mobile_pairing_admin: MobilePairingAdmin | None = None,
    host: str = "127.0.0.1",
    port: int = 6322,
) -> uvicorn.Server:
    config = uvicorn.Config(
        create_chat_app(
            workspace=workspace,
            channel=channel,
            mobile_pairing_admin=mobile_pairing_admin,
        ),
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    return uvicorn.Server(config)


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


def _public_dashboard_port() -> int:
    raw_port = os.environ.get("AKASHIC_DASHBOARD_PUBLIC_PORT", "2236")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise RuntimeError(
            "AKASHIC_DASHBOARD_PUBLIC_PORT 必须是 1 到 65535 的整数"
        ) from error
    if not 1 <= port <= 65535:
        raise RuntimeError(
            "AKASHIC_DASHBOARD_PUBLIC_PORT 必须是 1 到 65535 的整数"
        )
    return port
