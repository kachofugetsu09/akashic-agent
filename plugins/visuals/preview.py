"""复用浏览器，按工具调用隔离页面；预检失败不发布附件。"""
from __future__ import annotations

import asyncio
from playwright.async_api import Browser, BrowserContext, Error, Playwright, async_playwright


_PAGE_PROBLEM = """() => {
    if (!document.body.innerText.trim() && !document.body.querySelector('svg:not(#ak-defs),canvas')) return '文档没有可见内容';
    if (document.documentElement.scrollWidth > innerWidth + 1) return '内容超出可用宽度';
    if ([...document.querySelectorAll('output,[data-numeric]')].some(node => /\\b(?:NaN|Infinity)\\b/.test(node.textContent))) return '数值输出出现非有限数值';
    return null;
}"""


class PreviewFailed(ValueError):
    """调用者可修正文档或浏览器配置后重试。"""


class PreviewUnavailable(RuntimeError):
    """操作者需要修复浏览器环境，修改生成内容无法恢复。"""


class Preview:
    def __init__(self, executable_path: str | None = None) -> None:
        self._executable_path = executable_path
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(2)

    async def close(self) -> None:
        """作用域收束后关闭所有预检资源。"""
        try:
            if self._browser is not None:
                await self._browser.close()
        finally:
            if self._playwright is not None:
                await self._playwright.stop()
            self._browser = None
            self._playwright = None

    async def _get_browser(self) -> Browser:
        """首个调用启动浏览器；崩溃后的新调用重新建立资源。"""
        async with self._lock:
            if self._browser is not None and self._browser.is_connected():
                return self._browser
            await self.close()
            self._playwright = await async_playwright().start()
            try:
                self._browser = await self._playwright.chromium.launch(
                    executable_path=self._executable_path, chromium_sandbox=True, timeout=10_000,
                )
            except Error as error:
                await self.close()
                raise PreviewUnavailable(f"浏览器环境不可用，需要操作者修复；请勿通过改写 HTML 重试：{error}") from error
            except asyncio.CancelledError:
                await self.close()
                raise
            return self._browser

    async def check(self, document: str) -> bytes:
        """以桌面和窄屏宽度检查将要保存的完整文档：真实脚本、布局及滑块边界；通过后返回桌面宽度截图。"""
        # 1. 不积累无界排队；整个调用有截止时间。
        if self._slots.locked():
            raise PreviewFailed("可视化预检繁忙，请稍后重试")
        async with self._slots:
            try:
                async with asyncio.timeout(20):
                    return await self._check_document(document)
            except (Error, TimeoutError) as error:
                raise PreviewFailed(f"可视化预检失败：{error}") from error

    async def _check_document(self, document: str) -> bytes:
        """每次调用创建新 context，不携带账号、缓存或持久存储。"""
        browser = await self._get_browser()
        context = await browser.new_context(viewport={"width": 720, "height": 900}, service_workers="block")
        inspection = asyncio.create_task(self._inspect(context, document))
        try:
            # 先关闭 context 让浏览器调用真实结束，避免取消 Playwright RPC 留下未取异常。
            return await asyncio.shield(inspection)
        finally:
            await context.close()
            await asyncio.gather(inspection, return_exceptions=True)

    async def _inspect(self, context: BrowserContext, document: str) -> bytes:
        """检查真实页面；外层始终等待它随 context 关闭而结束。"""
        errors: list[str] = []
        # 2. 服务端拒绝全部网络，预检的就是将要保存的同一份文件。
        await context.route("**/*", lambda route: route.abort())
        page = await context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)[:300]) if len(errors) < 8 else None)
        page.on("console", lambda message: errors.append(message.text[:300])
                if message.type == "error" and len(errors) < 8 else None)
        await page.set_content(document, wait_until="load")
        for width in (720, 280):
            await page.set_viewport_size({"width": width, "height": 900})
            await page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
            problem = await page.evaluate(_PAGE_PROBLEM)
            if problem:
                errors.append(f"{width}px: {problem}")
        # 3. 触发真实 input/change；不把任意业务含义推断成统一响应合同。
        sliders = page.locator('input[type="range"]')
        if await sliders.count() > 12:
            raise PreviewFailed("交互图最多支持 12 个滑块")
        for index, slider in enumerate(await sliders.all(), 1):
            initial = await slider.input_value()
            for edge in ("min", "max", "initial"):
                await slider.evaluate("""(input, {edge, initial}) => {
                    input.value = edge === 'initial' ? initial : input[edge] || (edge === 'min' ? '0' : '100');
                    input.dispatchEvent(new Event('input', {bubbles:true}));
                    input.dispatchEvent(new Event('change', {bubbles:true}));
                }""", {"edge": edge, "initial": initial})
                await page.evaluate("() => new Promise(resolve => requestAnimationFrame(resolve))")
                problem = await page.evaluate(_PAGE_PROBLEM)
                if problem:
                    errors.append(f"滑块 {index} ({edge}): {problem}")
        if errors:
            raise PreviewFailed("；".join(errors[:8]))
        # 4. 滑块已回到初始值；按桌面宽度截整页给模型自检，超长页面只截前 2400px。
        await page.set_viewport_size({"width": 720, "height": 900})
        await page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        height = min(2400, await page.evaluate("() => Math.ceil(document.documentElement.scrollHeight)"))
        return await page.screenshot(type="png", clip={"x": 0, "y": 0, "width": 720, "height": max(1, height)}, full_page=True)
