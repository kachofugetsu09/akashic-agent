from __future__ import annotations

from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from infra.channels.web_chat_channel import WebChatChannel


def create_chat_app(
    *,
    workspace: Path,
    channel: WebChatChannel,
) -> FastAPI:
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
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="上传内容不能为空")
        clean_name = Path(filename).name or "upload.bin"
        return channel.save_upload(data, clean_name)

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
    host: str = "127.0.0.1",
    port: int = 6322,
) -> uvicorn.Server:
    config = uvicorn.Config(
        create_chat_app(
            workspace=workspace,
            channel=channel,
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
