"""聊天制品只属于客户端插件；读取和缓存不改变业务事实。"""
from pathlib import Path
import re
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from core.common.file_io import run_file_io
from plugins.ui.contract import CompressedStaticFiles

# Vite 产物文件名内嵌内容哈希，命中后才允许永久缓存。
_HASHED_ASSET = re.compile(r"/assets/[^/]*-[\w-]{8}\.[\w]+$")
# 聊天挂载入口、其唯一静态依赖分块与样式用稳定 URL 引用，内容随版本变化；可以缓存，但每次按 ETag 校验。
_STABLE_ENTRY = re.compile(r"^/assets/chat-app(?:-mount)?\.(?:js|css)$")


# 外壳页面引用稳定入口时带 ?v=<构建版本>（frontend/chat/vite.config.ts 写入）；版本与当前构建一致时内容就是这一版，
# 可以永久缓存，热启动不再逐个等 304。版本不一致（页面比服务端旧或新）仍按 ETag 校验，免得把别的版本存成这一版。
def _cache_control_for(path: str, version: str | None = None, build: str | None = None) -> str:
    if _HASHED_ASSET.search(path):
        return "public, max-age=31536000, immutable"
    if _STABLE_ENTRY.match(path):
        return "public, max-age=31536000, immutable" if version and version == build else "no-cache"
    return "no-store"


# 页面版本取挂载模块构建时写下的版本：聊天里有一份，/api/shell/state 给当前一份；两者不同说明服务端已换了前端。
async def current_chat_build(static_dir: Path) -> str | None:
    try:
        build = await run_file_io(lambda: (static_dir / "chat-app.build").read_text(encoding="ascii"))
    except FileNotFoundError:
        return None
    return build.strip() or None


def register_chat_assets(app: FastAPI, static_dir: Path) -> None:
    """在客户端 owner 的监听器提供聊天制品和既有缓存合同。"""

    @app.middleware("http")
    async def secure_static_response(request: Request, call_next):
        response = await call_next(request)
        if request.url.path not in {"/chat", "/chat/", "/settings", "/settings/"} and not request.url.path.startswith("/assets/"):
            return response
        path = request.url.path
        version = request.query_params.get("v")
        build = await current_chat_build(static_dir) if version and _STABLE_ENTRY.match(path) else None
        cache_control = _cache_control_for(path, version, build)
        response.headers["Cache-Control"] = cache_control
        if cache_control == "no-store":
            response.headers["Pragma"] = "no-cache"
        else:
            del response.headers["Pragma"]
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/settings")
    @app.get("/settings/")
    async def settings() -> RedirectResponse:
        return RedirectResponse(url="/#models", status_code=308)

    # 聊天已并入外壳页面（ADR 0126）；旧链接与通知里的 /chat?session=… 带着查询参数回到外壳，由聊天自己读取。
    @app.get("/chat")
    @app.get("/chat/")
    async def chat(request: Request) -> RedirectResponse:
        query = request.url.query
        return RedirectResponse(url=f"/?{query}" if query else "/", status_code=308)

    app.mount(
        "/assets",
        CompressedStaticFiles(directory=static_dir, check_dir=False),
        name="chat_assets",
    )
