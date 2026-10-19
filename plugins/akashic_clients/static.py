"""聊天制品只属于客户端插件；读取和缓存不改变业务事实。"""
from base64 import b64encode
from hashlib import sha256
from pathlib import Path
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from core.common.file_io import run_file_io

# Vite 产物文件名内嵌内容哈希，命中后才允许永久缓存；入口 HTML 仍需每次校验。
_HASHED_ASSET = re.compile(r"/assets/[^/]*-[\w-]{8}\.[\w]+$")
# 插件共享运行时经 import map 用稳定 URL 引用，内容随版本变化；可以缓存，但每次按 ETag 校验。
_SHARED_RUNTIME = re.compile(r"^/assets/sdk/[\w-]+\.js$")


# 聊天入口唯一的内联脚本是插件共享运行时的 import map；按实际产物内容放行它的 hash，不放宽为 unsafe-inline。
_IMPORT_MAP = re.compile(rb'<script type="importmap">(.*?)</script>', re.S)


def _import_map_sources(index: bytes) -> str:
    return "".join(f" 'sha256-{b64encode(sha256(match.group(1)).digest()).decode()}'"
                   for match in _IMPORT_MAP.finditer(index))


def _cache_control_for(path: str) -> str:
    if _HASHED_ASSET.search(path):
        return "public, max-age=31536000, immutable"
    if path.endswith(".html") or path in ("/", "/chat", "/chat/") or _SHARED_RUNTIME.match(path):
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
        # 入口处理器把与正文同一次读取算出的 hash 放在 request.state，正文和 CSP 不会来自两个版本。
        scripts = getattr(request.state, "import_map_sources", "")
        response.headers["Content-Security-Policy"] = (
            f"default-src 'self'; script-src 'self'{scripts}; style-src 'self' 'unsafe-inline'; "
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
    async def index(request: Request) -> Response:
        try:
            body = await run_file_io((static_dir / "index.html").read_bytes)
        except FileNotFoundError as error:
            raise HTTPException(503, "聊天前端资产尚未构建") from error
        request.state.import_map_sources = _import_map_sources(body)
        etag = f'"{sha256(body).hexdigest()}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        return Response(body, media_type="text/html; charset=utf-8", headers={"ETag": etag})

    app.mount(
        "/assets",
        StaticFiles(directory=static_dir, check_dir=False),
        name="chat_assets",
    )
