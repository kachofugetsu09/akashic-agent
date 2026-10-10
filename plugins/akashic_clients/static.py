"""聊天制品只属于客户端插件；读取和缓存不改变业务事实。"""
from pathlib import Path
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from core.common.file_io import run_file_io

# Vite 产物文件名内嵌内容哈希，命中后才允许永久缓存；入口 HTML 仍需每次校验。
_HASHED_ASSET = re.compile(r"/assets/[^/]*-[\w-]{8}\.[\w]+$")


def _cache_control_for(path: str) -> str:
    if _HASHED_ASSET.search(path):
        return "public, max-age=31536000, immutable"
    if path.endswith(".html") or path in ("/", "/chat", "/chat/"):
        return "no-cache"
    return "no-store"


def register_chat_assets(app: FastAPI, static_dir: Path) -> None:
    """在客户端 owner 的监听器提供插件制品和既有缓存合同。"""

    @app.middleware("http")
    async def secure_static_response(request: Request, call_next):
        response = await call_next(request)
        if request.url.path not in {"/", "/chat", "/chat/", "/settings", "/settings/"} and not request.url.path.startswith("/assets/"):
            return response
        cache_control = _cache_control_for(request.url.path)
        response.headers["Cache-Control"] = cache_control
        if cache_control == "no-store":
            response.headers["Pragma"] = "no-cache"
        else:
            del response.headers["Pragma"]
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; connect-src 'self' data: blob:; "
            "frame-ancestors 'self' "
            "http://127.0.0.1:5173 http://localhost:5173"
        )
        return response

    @app.get("/settings")
    @app.get("/settings/")
    async def settings() -> RedirectResponse:
        return RedirectResponse(url="/#models", status_code=308)

    @app.get("/")
    @app.get("/chat")
    @app.get("/chat/")
    async def index() -> FileResponse:
        path = static_dir / "index.html"
        try:
            info = await run_file_io(path.stat)
        except FileNotFoundError as error:
            raise HTTPException(503, "聊天前端资产尚未构建") from error
        return FileResponse(path, stat_result=info)

    app.mount(
        "/assets",
        StaticFiles(directory=static_dir, check_dir=False),
        name="chat_assets",
    )
