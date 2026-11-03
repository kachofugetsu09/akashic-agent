"""设计标签的浏览器入口：按会话列出、新建、读写设计文档。"""
from __future__ import annotations

import re
from dataclasses import asdict

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response

from agent.plugin_composition import DashboardContext

from .store import DESIGN_STORE, MAX_DOCUMENT_BYTES, DesignDocument, DesignStore, DesignStoreError, RevisionConflict

inject = (DESIGN_STORE,)
_SESSION = Query(min_length=1, max_length=512)
_DOCUMENT = re.compile(r"[0-9a-f]{16}")


def register(app: FastAPI, context: DashboardContext) -> None:
    # 能力只在路由的租约内解析：插件换代后旧路由拿不到新一代的存储。
    def store() -> DesignStore:
        return context.require(DESIGN_STORE)

    @app.get("/api/dashboard/design/documents")
    async def documents(session: str = _SESSION) -> dict[str, object]:
        return {"documents": [_view(item) for item in await store().list(session)]}

    @app.post("/api/dashboard/design/documents")
    async def create(session: str = _SESSION, name: str = Body("", embed=True, max_length=200)) -> dict[str, object]:
        try:
            return _view(await store().create(session, name))
        except DesignStoreError as error:
            raise HTTPException(409, str(error)) from error

    @app.patch("/api/dashboard/design/documents/{document}")
    async def rename(document: str, session: str = _SESSION,
                     name: str = Body(embed=True, max_length=200)) -> dict[str, object]:
        try:
            return _view(await store().rename(session, _checked(document), name))
        except DesignStoreError as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/api/dashboard/design/documents/{document}")
    async def read(document: str, session: str = _SESSION) -> Response:
        try:
            meta, data = await store().read(session, _checked(document))
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
            saved = await store().write(session, _checked(document), data, base=base)
        except RevisionConflict as error:
            raise HTTPException(409, {"message": str(error), "revision": error.current}) from error
        except DesignStoreError as error:
            raise HTTPException(404, str(error)) from error
        return _view(saved)


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
