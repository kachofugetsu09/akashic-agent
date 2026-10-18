"""聊天制品只属于客户端插件；读取和缓存不改变业务事实。"""
from base64 import b64encode
from hashlib import sha256
from pathlib import Path
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from core.common.file_io import run_file_io

# Vite 产物文件名内嵌内容哈希，命中后才允许永久缓存；入口 HTML 仍需每次校验。
_HASHED_ASSET = re.compile(r"/assets/[^/]*-[\w-]{8}\.[\w]+$")


# 聊天入口唯一的内联脚本是插件共享运行时的 import map；按实际产物内容放行它的 hash，不放宽为 unsafe-inline。
_IMPORT_MAP = re.compile(rb'<script type="importmap">(.*?)</script>', re.S)


def _import_map_sources(index: bytes) -> str:
    return "".join(f" 'sha256-{b64encode(sha256(match.group(1)).digest()).decode()}'"
                   for match in _IMPORT_MAP.finditer(index))


def _cache_control_for(path: str) -> str:
    if _HASHED_ASSET.search(path):
        return "public, max-age=31536000, immutable"
    if path.endswith(".html") or path in ("/", "/chat", "/chat/"):
        return "no-cache"
    return "no-store"


def register_chat_assets(app: FastAPI, static_dir: Path) -> None:
    """在客户端 owner 的监听器提供插件制品和既有缓存合同。"""

    index_path = static_dir / "index.html"
    import_maps: dict[tuple[int, int], str] = {}

    # 按入口文件版本缓存 hash；入口缺失时只保留 'self'。
    async def import_map_sources() -> str:
        try:
            info = await run_file_io(index_path.stat)
        except FileNotFoundError:
            return ""
        version = (info.st_mtime_ns, info.st_size)
        if version not in import_maps:
            import_maps.clear()
            import_maps[version] = _import_map_sources(await run_file_io(index_path.read_bytes))
        return import_maps[version]

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
        scripts = await import_map_sources() if request.url.path in {"/", "/chat", "/chat/"} else ""
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
