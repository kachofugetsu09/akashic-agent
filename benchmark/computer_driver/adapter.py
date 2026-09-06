"""把 Cua 原有任务的 DesktopSession 接到 Computer Gateway。"""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from urllib.parse import urlsplit

import aiohttp
import cua_bench as cb
from cua_bench.computers.webtop import WebDesktopSession
from cua_bench.types import Snapshot, WindowSnapshot
from playwright.async_api import async_playwright


class GatewaySession(WebDesktopSession):
    """复用上游元素点击 helper；所有鼠标键盘动作只走 /input。"""

    def __init__(self, gateway: str, cdp: str, css: str, suppress: bool):
        super().__init__()
        self.gateway, self.cdp, self.css, self.suppress = gateway, cdp, css, suppress
        self.actions = []

    async def start(self, config=None, headless=None):
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40))
        async with self.http.get(self.cdp + "/json/version") as response:
            response.raise_for_status()
            info = await response.json()
        websocket = urlsplit(info["webSocketDebuggerUrl"])
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.connect_over_cdp(
            "ws://" + urlsplit(self.cdp).netloc + websocket.path
        )
        self.context = self.browser.contexts[0]
        self._page = await self.context.new_page()
        self._page.set_default_timeout(5000)

    async def launch_window(self, *, html, title, width, height):
        """把上游 HTML 原样放入真实 Chromium，固定桌面坐标原点。"""
        # 1. 同一容器内的测试标签全屏；不连接用户桌面。
        cdp = await self.context.new_cdp_session(self.page)
        window = await cdp.send("Browser.getWindowForTarget")
        if window["bounds"]["windowState"] != "fullscreen":
            await cdp.send(
                "Browser.setWindowBounds",
                {
                    "windowId": window["windowId"],
                    "bounds": {"windowState": "fullscreen"},
                },
            )
        await self.page.bring_to_front()
        await self.page.wait_for_function(
            "innerWidth === 1280 && innerHeight === 800 && devicePixelRatio === 1"
        )
        # 2. 只固定随机种子与测试样式依赖，任务内容和判分器不变。
        await self.page.evaluate(
            "Math.random = (() => { let s = 42; return () => ((s = (1664525*s + 1013904223) >>> 0) / 4294967296); })()"
        )
        await self.page.set_content(html)
        await self.page.add_script_tag(content=self.css)
        await self.page.wait_for_function("!!document.querySelector('style')")
        await self.page.evaluate(
            "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
        )
        return "fixture"

    async def execute_javascript(self, pid, javascript):
        return await self.page.evaluate(javascript)

    async def get_element_rect(self, pid, selector, *, space="window", timeout=0.5):
        box = await self.page.locator(selector).bounding_box()
        if box is None:
            raise NotImplementedError(
                f"Original solver target has no Chromium screen rectangle: {selector}"
            )
        return box

    async def screenshot(self):
        async with self.http.get(self.gateway + "/screenshot?quiet=1") as response:
            response.raise_for_status()
            return await response.read()

    async def get_snapshot(self):
        return Snapshot(
            windows=[
                WindowSnapshot(
                    window_type="webview", pid="fixture", width=1280, height=800
                )
            ]
        )

    async def execute_action(self, action):
        """转换上游动作类型，不实现另一套输入或降级路径。"""
        values = asdict(action)
        match action:
            case cb.ClickAction():
                payload = {"action": "click", **values}
            case cb.DoubleClickAction():
                payload = {"action": "double_click", **values}
            case cb.TypeAction():
                payload = {"action": "type", **values}
            case cb.KeyAction():
                payload = {"action": "key", **values}
            case cb.HotkeyAction():
                payload = {"action": "key", "key": "+".join(action.keys)}
            case cb.DragAction():
                payload = {
                    "action": "drag",
                    "x": round(action.from_x),
                    "y": round(action.from_y),
                    "to_x": round(action.to_x),
                    "to_y": round(action.to_y),
                }
            case cb.WaitAction():
                payload = {"action": "wait", "ms": round(action.seconds * 1000)}
            case _:
                raise NotImplementedError(
                    f"No faithful Gateway mapping: {type(action).__name__}"
                )
        start = perf_counter()
        if self.suppress:
            self.actions.append({"input": payload, "suppressed": True})
            return
        async with self.http.post(self.gateway + "/input", json=payload) as response:
            result = await response.json()
            self.actions.append(
                {
                    "input": payload,
                    "output": result,
                    "http_status": response.status,
                    "seconds": perf_counter() - start,
                }
            )
            response.raise_for_status()

    async def close(self):
        try:
            if self.page is not None:
                await self.page.close()
        finally:
            if self.playwright is not None:
                await self.playwright.stop()
            await self.http.close()


class GatewayEnvironment(cb.Environment):
    session_type = GatewaySession

    async def create_sandbox(self, provider, provider_config=None, setup_config=None):
        self.session = self.session_type(*self.gateway_settings)
        self.session.env = self
        await self.session.start()
        self.page = self.session.page

    @classmethod
    def load(cls, task, gateway, cdp, css, suppress):
        """使用上游加载器提供的四个回调，不复制题目和解法。"""
        original = cb.make(str(task))
        env = cls(
            original.env_name,
            tasks_config_fn=original.tasks_config_fn,
            setup_task_fn=original.setup_task_fn,
            solve_task_fn=original.solve_task_fn,
            evaluate_task_fn=original.evaluate_task_fn,
        )
        env.gateway_settings = gateway, cdp, css, suppress
        return env
