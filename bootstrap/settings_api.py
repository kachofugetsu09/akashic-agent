from __future__ import annotations

import logging
import socket
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


class SettingsServer(uvicorn.Server):
    """Publish one deterministic startup result across the server thread."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.startup_event = threading.Event()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        try:
            await super().startup(sockets=sockets)
        finally:
            self.startup_event.set()


def create_settings_app() -> FastAPI:
    """Serve the model-plugin UI without owning model or memory state."""

    static_dir = Path(__file__).resolve().parent.parent / "static" / "chat"
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(Exception)
    async def internal_error(_request: Request, error: Exception) -> JSONResponse:
        logger.exception("[settings] unexpected failure", exc_info=error)
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "设置操作失败，请查看服务端日志",
            },
        )

    @app.middleware("http")
    async def secure_static_response(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'self' "
            "http://127.0.0.1:5173 http://localhost:5173"
        )
        return response

    @app.get("/settings")
    @app.get("/settings/")
    async def retired_settings() -> RedirectResponse:
        return RedirectResponse(url="/#models", status_code=308)

    @app.get("/chat")
    @app.get("/chat/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount(
        "/assets",
        StaticFiles(directory=static_dir, check_dir=False),
        name="settings-assets",
    )
    return app


__all__ = ["SettingsServer", "create_settings_app"]
