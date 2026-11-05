"""在无头 Chromium 里托管一份静态网页产物：资源从本地目录取，指定前缀的请求交给 Python 处理，其余网络一律拒绝；闲置的页面自动关闭。

设计插件用它在服务端跑编辑器（不依赖用户开着网页）；浏览器用部署里已有的 Playwright Chromium。接口里没有设计专属的概念，
别的插件也要在服务端跑自己的前端代码时，把它提成公共能力（新增 Core 公开模块要过插件边界评审）。
"""
from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from playwright.async_api import Browser, BrowserContext, Error, Page, Playwright, Route, async_playwright

logger = logging.getLogger(__name__)

IDLE_CHECK_SECONDS = 30.0


class HeadlessUnavailable(RuntimeError):
    """浏览器起不来或页面没能就绪：操作者需要修复运行环境。"""


@dataclass(frozen=True, slots=True)
class Reply:
    status: int
    content_type: str
    body: bytes


# 路径（含查询串）与请求体 → 回复；抛出的异常按 502 回给页面。
Handler = Callable[[str, bytes | None], Awaitable[Reply]]


# root 里的文件挂在 mount 路径下（与产物里写死的资源地址一致），打开的是 mount 下的 entry；ready 是页面就绪时为真的 JS 表达式。
@dataclass(frozen=True, slots=True)
class PageSpec:
    root: Path
    mount: str
    entry: str
    ready: str
    handlers: Mapping[str, Handler] = field(default_factory=lambda: {})
    headers: Mapping[str, Mapping[str, str]] = field(default_factory=lambda: {})
    # 允许页面直连的外部域名（图标来自 Iconify）；不在这里的一律拒绝。
    external: frozenset[str] = frozenset()
    ready_timeout: float = 60.0


@dataclass(eq=False)
class _Open:
    context: BrowserContext
    page: Page
    used: float


class HeadlessPages:
    """按 key 复用页面：同一个 key 拿到的是同一块页面（状态留在页面里），各 key 之间不共享存储与缓存。"""

    def __init__(self, *, idle_seconds: float = 600.0, executable_path: str | None = None) -> None:
        self._idle = idle_seconds
        self._executable_path = executable_path
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._pages: dict[str, _Open] = {}
        self._lock = asyncio.Lock()
        self._reaper: asyncio.Task[None] | None = None

    # 1. 已有且还活着的页面直接用；2. 否则新开一个隔离的 context，按 spec 接管全部请求，等页面报告就绪
    async def page(self, key: str, spec: PageSpec) -> Page:
        async with self._lock:
            current = self._pages.get(key)
            if current is not None and not current.page.is_closed():
                current.used = time.monotonic()
                return current.page
            if current is not None:
                await self._drop(key)
            browser = await self._browser_ready()
            context = await browser.new_context()
            try:
                page = await self._load(context, key, spec)
            except BaseException:
                await context.close()
                if not self._pages:
                    await self._stop_browser()
                raise
            self._pages[key] = _Open(context, page, time.monotonic())
            if self._reaper is None or self._reaper.done():
                self._reaper = asyncio.create_task(self._reap())
            return page

    def touch(self, key: str) -> None:
        current = self._pages.get(key)
        if current is not None:
            current.used = time.monotonic()

    async def close(self, key: str) -> None:
        async with self._lock:
            await self._drop(key)

    async def aclose(self) -> None:
        async with self._lock:
            for key in list(self._pages):
                await self._drop(key)
            await self._stop_browser()
        if self._reaper is not None:
            _ = self._reaper.cancel()

    async def _load(self, context: BrowserContext, key: str, spec: PageSpec) -> Page:
        origin = f"https://{key}.headless.invalid"
        await context.route("**/*", lambda route: self._serve(route, origin, spec))
        # WebSocket 不经过 route：一律接管并直接关闭，不连真正的服务器
        await context.route_web_socket(re.compile(".*"), lambda socket: socket.close())
        page = await context.new_page()
        page.on("pageerror", lambda error: logger.warning("headless %s: %s", key, str(error)[:300]))
        try:
            _ = await page.goto(f"{origin}{spec.mount}{spec.entry}", wait_until="load")
            _ = await page.wait_for_function(spec.ready, timeout=spec.ready_timeout * 1000)
        except Error as error:
            raise HeadlessUnavailable(f"无头页面没能就绪：{error}") from error
        return page

    async def _serve(self, route: Route, origin: str, spec: PageSpec) -> None:
        url = route.request.url
        if not url.startswith(f"{origin}/"):
            # 页面只能取自己托管的东西；spec 里点名的域名（图标接口）放行，其余一律拒绝。
            return await route.continue_() if urlsplit(url).hostname in spec.external else await route.abort()
        path = url[len(origin):]
        name = urlsplit(path).path
        try:
            prefix = next((item for item in spec.handlers if name.startswith(item)), None)
            if prefix is not None:
                body = route.request.post_data_buffer
                reply = await spec.handlers[prefix](path, body)
            elif name.startswith(spec.mount):
                reply = await asyncio.to_thread(_static, spec.root, name[len(spec.mount):])
            else:
                reply = Reply(404, "text/plain; charset=utf-8", b"not found")
        except Exception as error:  # 处理器的错误回给页面，不让请求悬着
            reply = Reply(502, "text/plain; charset=utf-8", f"{type(error).__name__}: {error}".encode())
        base = name.rsplit("/", 1)[-1]
        await route.fulfill(status=reply.status, body=reply.body,
                            headers={"Content-Type": reply.content_type, **spec.headers.get(base, {})})

    async def _browser_ready(self) -> Browser:
        if self._browser is not None and self._browser.is_connected():
            return self._browser
        await self._stop_browser()
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                executable_path=self._executable_path, chromium_sandbox=True, timeout=15_000,
                # 不透明源沙箱 iframe 放进独立进程：里面死循环时页面自己的超时还能触发（无头外壳默认不隔离）
                args=["--site-per-process", "--enable-features=IsolateSandboxedIframes"])
        except Error as error:
            await self._stop_browser()
            raise HeadlessUnavailable(f"无头浏览器不可用：{error}") from error
        return self._browser

    # 闲置超时的页面关掉；一块都不剩时连浏览器一起退出，下次用到再起。
    async def _reap(self) -> None:
        while True:
            await asyncio.sleep(IDLE_CHECK_SECONDS)
            async with self._lock:
                now = time.monotonic()
                for key in [key for key, item in self._pages.items() if now - item.used > self._idle]:
                    await self._drop(key)
                if not self._pages:
                    await self._stop_browser()
                    return

    async def _drop(self, key: str) -> None:
        current = self._pages.pop(key, None)
        if current is not None:
            try:
                await current.context.close()
            except Error as error:
                logger.warning("headless %s: 关闭页面失败 %s", key, error)

    async def _stop_browser(self) -> None:
        browser, playwright = self._browser, self._playwright
        self._browser = None
        self._playwright = None
        try:
            if browser is not None:
                await browser.close()
        except Error as error:
            logger.warning("headless: 关闭浏览器失败 %s", error)
        finally:
            if playwright is not None:
                await playwright.stop()


# 只给根目录里真实存在的文件；路径先解析再判断，../ 与符号链接都出不去。
def _static(root: Path, name: str) -> Reply:
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        return Reply(404, "text/plain; charset=utf-8", b"not found")
    kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    if target.suffix == ".wasm":
        kind = "application/wasm"
    elif target.suffix == ".html":
        kind = "text/html; charset=utf-8"
    return Reply(200, kind, target.read_bytes())
