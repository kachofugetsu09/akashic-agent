from __future__ import annotations

from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket

from infra.channels.web_chat_channel import WebChatChannel


def create_chat_app(
    *,
    workspace: Path,
    channel: WebChatChannel,
) -> FastAPI:
    app = FastAPI(title="Akashic Chat API")
    app.state.workspace = workspace
    app.state.channel = channel

    @app.get("/")
    def chat_index() -> dict[str, str]:
        return {"status": "ok", "channel": channel.name}

    @app.get("/api/chat/sessions")
    def list_sessions(page: int = Query(1), page_size: int = Query(50)) -> dict[str, Any]:
        ctx = channel._require_ctx()
        items, total = ctx.session_manager._store.list_sessions_for_dashboard(
            channel=channel.name,
            page=page,
            page_size=page_size,
        )
        return {"items": items, "total": total}

    @app.get("/api/chat/sessions/{session_key:path}/messages")
    def list_messages(session_key: str, page: int = Query(1), page_size: int = Query(50)) -> dict[str, Any]:
        ctx = channel._require_ctx()
        items, total = ctx.session_manager._store.list_messages_for_dashboard(
            session_key=session_key,
            page=page,
            page_size=page_size,
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
        log_level="info",
    )
    return uvicorn.Server(config)
