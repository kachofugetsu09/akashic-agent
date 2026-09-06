"""原 Cua 解法映射到源码驱动；与原版 Browser + Sky 对照使用相同动作绑定。"""

import asyncio
import json
import uuid
from time import perf_counter

import cua_bench as cb
from adapter import GatewaySession


class SourceSession(GatewaySession):
    async def run_code(self, code="", *, end_turn=False):
        """只通过正式 gateway 执行，夹具身份不加载 Akashic Agent。"""
        payload = {
            "context": {
                "session_id": "cua-fixture",
                "turn_id": self.turn,
                "call_id": uuid.uuid4().hex,
            },
            "code": code,
            "endTurn": end_turn,
            "timeoutMs": 30000,
        }
        started = perf_counter()
        async with self.http.post(
            self.gateway + "/driver/run", json=payload
        ) as response:
            result = await response.json()
            self.actions.append(
                {
                    "driver_input": payload,
                    "output": result,
                    "http_status": response.status,
                    "seconds": perf_counter() - started,
                }
            )
            response.raise_for_status()
            return result

    async def start(self, config=None, headless=None):
        await super().start(config, headless)
        self.turn = uuid.uuid4().hex
        await self.page.close()
        async with self.context.expect_page() as created:
            result = await self.run_code(
                "nodeRepl.write((await browser.tabs.new()).id);"
            )
        self._page = await created.value
        self.driver_tab = next(
            item["text"] for item in result["content"] if item["type"] == "text"
        )
        self._page.set_default_timeout(5000)

    def tab(self):
        return f"(await browser.tabs.get({json.dumps(self.driver_tab)}))"

    async def click_element(self, pid, selector, **kwargs):
        if self.suppress:
            self.actions.append({"click_element": selector, "suppressed": True})
            return
        if await self.page.locator(selector).evaluate('(e)=>e.tagName==="OPTION"'):
            value = await self.page.locator(selector).get_attribute("value")
            code = f"{self.tab()}.playwright.locator({json.dumps(selector + ' >> xpath=..')} ).selectOption({json.dumps(value)})"
        else:
            code = f"{self.tab()}.playwright.locator({json.dumps(selector)}).click()"
        await self.run_code("await " + code)

    async def execute_action(self, action):
        if self.suppress:
            self.actions.append({"action": type(action).__name__, "suppressed": True})
            return
        match action:
            case cb.ClickAction() | cb.DoubleClickAction() | cb.RightClickAction():
                method, data = "click", {"x": action.x, "y": action.y}
                if isinstance(action, cb.RightClickAction):
                    data["mouse_button"] = "right"
                elif isinstance(action, cb.DoubleClickAction):
                    data["click_count"] = 2
            case cb.DragAction():
                method, data = (
                    "drag",
                    {
                        "path": [
                            {"x": round(action.from_x), "y": round(action.from_y)},
                            {"x": round(action.to_x), "y": round(action.to_y)},
                        ]
                    },
                )
            case cb.TypeAction():
                kind = await self.page.evaluate(
                    'document.activeElement?.tagName==="INPUT" ? document.activeElement.type : null'
                )
                if kind in (
                    "date",
                    "time",
                    "datetime-local",
                    "month",
                    "week",
                    "range",
                    "color",
                ):
                    await self.run_code(
                        f'await {self.tab()}.playwright.locator(":focus").fill({json.dumps(action.text)});'
                    )
                    return
                method, data = "type_text", {"text": action.text}
            case cb.KeyAction():
                method, data = "press_key", {"key": action.key}
            case cb.HotkeyAction():
                method, data = "press_key", {"key": "+".join(action.keys)}
            case cb.WaitAction():
                await asyncio.sleep(action.seconds)
                return
            case _:
                raise NotImplementedError(type(action).__name__)
        await self.run_code(f"await sky[{json.dumps(method)}]({json.dumps(data)});")

    async def close(self):
        try:
            if hasattr(self, "driver_tab"):
                await self.run_code(end_turn=True)
        finally:
            await super().close()
