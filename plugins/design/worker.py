"""Agent 的设计工具在服务端执行：每个会话一块无头画布（与网页同一份编辑器代码），直接读写存储里的文档，用户不在也能做设计。

并发按乐观方式处理：改完带着读入时的修订号写回，期间别处（用户的画布）先保存了就读入最新版本、把这次调用重做一遍。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlsplit

from fastapi import HTTPException
from playwright.async_api import Error, Page

from . import dashboard
from .headless import HeadlessPages, PageSpec, Reply
from .live import DesignLive
from .store import DesignDocument, DesignStore, RevisionConflict

MOUNT = "/design/assets/"
ENTRY = "worker.html"
ATTEMPTS = 3
CALL_TIMEOUT = 120.0


class DesignWorkerError(RuntimeError):
    pass


@dataclass(eq=False)
class _Session:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Agent 正在改的文档；进程重启后丢失，回落到最近修改的那份
    document: str | None = None
    # 哪块无头页面里装着哪份文档的哪个修订：页面被回收或重开后对不上，就重新读入
    loaded: tuple[Page, str, int] | None = None


class DesignWorker:
    def __init__(self, store: DesignStore, live: DesignLive, pages: HeadlessPages, plugin_dir: Path, data_dir: Path) -> None:
        self._store = store
        self._live = live
        self._pages = pages
        self._root = plugin_dir / "static"
        self._fonts = data_dir / "fonts"
        self._sessions: dict[str, _Session] = {}

    # design_document：列出、打开或新建；打开或新建的那份成为 Agent 之后调用的目标
    async def document(self, session: str, turn: str, action: str, name: str, wanted: str) -> dict[str, object]:
        state = self._sessions.setdefault(session, _Session())
        async with state.lock:
            documents = await self._store.list(session)
            if action == "create":
                state.document = (await self._store.create(session, name)).id
                documents = await self._store.list(session)
            elif action == "open":
                found = next((item for item in documents if item.id == wanted or (name and item.name == name)), None)
                if found is None:
                    raise DesignWorkerError(f"没有这份设计文档：{wanted or name}；先用 action=list 查看")
                state.document = found.id
            current = next((item for item in documents if item.id == state.document), None)
            if current is not None and action != "list":
                await self._live.agent(session, turn, current.id)
            return {"documents": [{"id": item.id, "name": item.name, "updated_at": item.updated_at} for item in documents],
                    "open": {"id": current.id, "name": current.name} if current else None}

    # 1. 定目标文档并让画布跟上；2. 执行；改了文档就带修订号写回，冲突时读入最新版本重做
    async def call(self, session: str, turn: str, tool: str, arguments: dict[str, object]) -> dict[str, object]:
        state = self._sessions.setdefault(session, _Session())
        async with state.lock:
            target = await self._target(session, state)
            await self._live.agent(session, turn, target.id)
            page = await self._page(session)
            for _ in range(ATTEMPTS):
                meta = await self._sync(page, session, state, target.id)
                # 执行期间不认为画布还等于这份修订：失败的调用可能已经改了一半，下次整份重读，残留不会混进之后的保存
                state.loaded = None
                reply = cast(dict[str, object], await self._evaluate(
                    page, "([tool, args]) => akashicDesign.call(tool, args)", [tool, arguments]))
                if reply.get("ok") is not True:
                    return reply
                if reply.get("mutated") is not True:
                    state.loaded = (page, meta.id, meta.revision)
                    return reply
                if await self._commit(page, session, state, meta):
                    return reply
            raise DesignWorkerError("文档一直在被别处修改，这次修改没能保存；稍后再试")

    # design_show_html 要求导入图层：在无头画布里把网页转成图层，写进那份文档
    async def import_html(self, session: str, turn: str, document: DesignDocument) -> int:
        state = self._sessions.setdefault(session, _Session())
        async with state.lock:
            state.document = document.id
            page = await self._page(session)
            html = await self._store.preview(session, document.id)
            for _ in range(ATTEMPTS):
                meta = await self._sync(page, session, state, document.id)
                state.loaded = None
                count = cast(int, await self._evaluate(page, "([html, name]) => akashicDesign.importHtml(html, name)",
                                                       [html, meta.name]))
                if await self._commit(page, session, state, meta):
                    await self._live.agent(session, turn, document.id)
                    return count
            raise DesignWorkerError("文档一直在被别处修改，导入没能保存；稍后再试")

    # 网页预览发布后：之后的设计调用默认改这一份，开着的画布切到它的网页预览
    async def preview(self, session: str, turn: str, document: str) -> None:
        self._sessions.setdefault(session, _Session()).document = document
        await self._live.agent(session, turn, document, "preview")

    async def aclose(self) -> None:
        await self._pages.aclose()

    async def _target(self, session: str, state: _Session) -> DesignDocument:
        documents = await self._store.list(session)
        found = next((item for item in documents if item.id == state.document), None) or (documents[0] if documents else None)
        if found is None:
            found = await self._store.create(session, "")
        state.document = found.id
        return found

    # 无头画布里不是存储中的最新修订就整份读入；读入后图层 ID 会重新编号，Agent 拿到的是不变的 GUID，不受影响。
    async def _sync(self, page: Page, session: str, state: _Session, document: str) -> DesignDocument:
        meta = await self._store.find(session, document)
        if meta is not None and state.loaded == (page, meta.id, meta.revision):
            return meta
        state.loaded = None
        meta, data = await self._store.read(session, document)
        payload = base64.b64encode(data).decode("ascii") if data else None
        _ = await self._evaluate(page, "(data) => akashicDesign.open(data)", payload)
        state.loaded = (page, meta.id, meta.revision)
        return meta

    async def _commit(self, page: Page, session: str, state: _Session, meta: DesignDocument) -> bool:
        data = base64.b64decode(cast(str, await self._evaluate(page, "() => akashicDesign.save()")))
        try:
            saved = await self._store.write(session, meta.id, data, base=meta.revision)
        except RevisionConflict:
            state.loaded = None
            return False
        state.loaded = (page, saved.id, saved.revision)
        await self._live.changed(session, saved.id, saved.revision)
        return True

    async def _page(self, session: str) -> Page:
        spec = PageSpec(self._root, MOUNT, ENTRY, "() => Boolean(window.akashicDesign)",
                        handlers={"/api/dashboard/design/fonts": self._font},
                        headers={dashboard.SANDBOX: {"Content-Security-Policy": dashboard.SANDBOX_POLICY},
                                 ENTRY: {"Content-Security-Policy": dashboard.EDITOR_POLICY}},
                        external_origins=frozenset({dashboard.ICON_ORIGIN}))
        key = "design-" + hashlib.sha256(session.encode("utf-8")).hexdigest()[:24]
        page = await self._pages.page(key, spec)
        self._pages.touch(key)
        return page

    async def _evaluate(self, page: Page, expression: str, argument: object = None) -> object:
        try:
            return await asyncio.wait_for(page.evaluate(expression, argument), CALL_TIMEOUT)
        except TimeoutError as error:
            await page.close()
            raise DesignWorkerError(f"无头画布 {CALL_TIMEOUT:.0f} 秒内没有完成，已重开") from error
        except Error as error:
            # 页面可能已经崩溃或处在未知状态：关掉，下次重开
            await page.close()
            raise DesignWorkerError(f"无头画布出错：{error.message.splitlines()[0]}") from error

    async def _font(self, path: str, _body: bytes | None) -> Reply:
        url = parse_qs(urlsplit(path).query).get("url", [""])[0]
        try:
            response = await dashboard.fetch_font(self._fonts, url)
        except HTTPException as error:
            return Reply(error.status_code, "text/plain; charset=utf-8", str(error.detail).encode())
        return Reply(response.status_code, response.media_type or "application/octet-stream", bytes(response.body))

