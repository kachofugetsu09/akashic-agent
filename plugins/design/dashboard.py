"""设计标签的浏览器入口：按会话列出、新建、读写设计文档，并交付按需加载的编辑器资产。"""
from __future__ import annotations

import os
import re
from dataclasses import asdict
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response, WebSocket
from starlette.types import Scope

from agent.plugin_composition import DashboardContext
from plugins.ui.contract import CompressedStaticFiles

from .live import DESIGN_LIVE
from .store import DESIGN_STORE, MAX_DOCUMENT_BYTES, DesignDocument, DesignStore, DesignStoreError, RevisionConflict

inject = (DESIGN_STORE, DESIGN_LIVE)
ASSETS = "/design/assets/"
_SESSION = Query(min_length=1, max_length=512)
_DOCUMENT = re.compile(r"[0-9a-f]{16}")
# 编辑器页面比宿主默认策略多放行 CanvasKit 的 WASM 编译；不放行内联脚本与 eval。
EDITOR = "editor.html"
EDITOR_POLICY = (
    "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
    "font-src 'self' data: blob:; connect-src 'self' data: blob:; "
    "frame-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
)


# Agent 写的 JSX 在这一页里求值：不透明源沙箱，只放行内联脚本与 eval，不能连网、取同源资源或被别的站点嵌入。
SANDBOX = "jsx.html"
SANDBOX_POLICY = (
    "sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; connect-src 'none'; "
    "img-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
)


# 构建产物按内容哈希命名，可以长期缓存；入口页 editor.html 每次都要回源，换版本时才能拿到新的文件名。
class _EditorAssets(CompressedStaticFiles):
    def file_response(self, full_path: os.PathLike[str] | str, stat_result: os.stat_result,
                      scope: Scope, status_code: int = 200) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        hashed = not str(full_path).endswith(".html")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable" if hashed else "no-cache"
        policy = {EDITOR: EDITOR_POLICY, SANDBOX: SANDBOX_POLICY}.get(Path(full_path).name)
        if policy:
            response.headers["Content-Security-Policy"] = policy
        if policy and policy is not EDITOR_POLICY:
            response.headers["Referrer-Policy"] = "no-referrer"
        return response


def register(app: FastAPI, context: DashboardContext) -> None:
    assets = _EditorAssets(directory=context.plugin_dir / "static", check_dir=False)
    for name in asset_names(context.plugin_dir):
        _asset_route(app, assets, name)
    _catalog_routes(app, context)
    _content_routes(app, context)

    # 标签模块为当前会话保持一条连接：别处保存的新修订号与 Agent 的设计工具调用都经它送达画布。
    @app.websocket("/api/dashboard/design/live")
    async def live(socket: WebSocket, session: str = _SESSION) -> None:
        await context.require(DESIGN_LIVE).serve(socket, session)


# 能力只在路由的租约内解析：插件换代后旧路由拿不到新一代的存储。
def _store(context: DashboardContext) -> DesignStore:
    return context.require(DESIGN_STORE)


# 当前会话的文档列表、新建与改名
def _catalog_routes(app: FastAPI, context: DashboardContext) -> None:
    @app.get("/api/dashboard/design/documents")
    async def documents(session: str = _SESSION) -> dict[str, object]:
        return {"documents": [_view(item) for item in await _store(context).list(session)]}

    @app.post("/api/dashboard/design/documents")
    async def create(session: str = _SESSION, name: str = Body("", embed=True, max_length=200)) -> dict[str, object]:
        try:
            return _view(await _store(context).create(session, name))
        except DesignStoreError as error:
            raise HTTPException(409, str(error)) from error

    @app.patch("/api/dashboard/design/documents/{document}")
    async def rename(document: str, session: str = _SESSION,
                     name: str = Body(embed=True, max_length=200)) -> dict[str, object]:
        try:
            return _view(await _store(context).rename(session, _checked(document), name))
        except DesignStoreError as error:
            raise HTTPException(404, str(error)) from error


# 文档内容的读写：读取时修订号随 ETag 返回，写入要带上自己基于的修订号
def _content_routes(app: FastAPI, context: DashboardContext) -> None:
    @app.get("/api/dashboard/design/documents/{document}")
    async def read(document: str, session: str = _SESSION) -> Response:
        try:
            meta, data = await _store(context).read(session, _checked(document))
        except DesignStoreError as error:
            raise HTTPException(404, str(error)) from error
        # 修订号放在 ETag 里：宿主代理只转发白名单里的响应头。
        return Response(data, media_type="application/octet-stream", headers={
            "Cache-Control": "no-store", "ETag": f'"{meta.revision}"'})

    @app.put("/api/dashboard/design/documents/{document}")
    async def write(document: str, request: Request, session: str = _SESSION,
                    base: int = Query(ge=0)) -> dict[str, object]:
        data = await _body(request)
        try:
            saved = await _store(context).write(session, _checked(document), data, base=base)
        except RevisionConflict as error:
            raise HTTPException(409, {"message": str(error), "revision": error.current}) from error
        except DesignStoreError as error:
            raise HTTPException(404, str(error)) from error
        await context.require(DESIGN_LIVE).changed(session, saved.id, saved.revision)
        return _view(saved)


# iframe 页面不带 Web 模块身份，只能走宿主放行的导航路径：每个构建产物各登记一条无参数的 GET 路由。
def asset_names(plugin_dir: Path) -> tuple[str, ...]:
    folder = plugin_dir / "static"
    return tuple(sorted(path.name for path in folder.iterdir() if path.is_file())) if folder.is_dir() else ()


def _asset_route(app: FastAPI, assets: _EditorAssets, name: str) -> None:
    async def asset(request: Request) -> Response:
        return await assets.get_response(name, request.scope)

    app.add_api_route(f"{ASSETS}{name}", asset, methods=["GET"], include_in_schema=False)


def _checked(document: str) -> str:
    if not _DOCUMENT.fullmatch(document):
        raise HTTPException(400, "文档 ID 无效")
    return document


# 边读边数：超过上限立即拒绝，不先把整个请求体收进内存。
async def _body(request: Request) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_DOCUMENT_BYTES:
            raise HTTPException(413, f"设计文档超过 {MAX_DOCUMENT_BYTES // 1024 // 1024} MB")
        chunks.append(chunk)
    return b"".join(chunks)


def _view(item: DesignDocument) -> dict[str, object]:
    return asdict(item)
