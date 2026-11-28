#!/usr/bin/env python3
"""在验收网络内的浏览器容器里走新用户路径：设密码、引导、装包、对话与工具。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable
from urllib.request import Request, urlopen

from playwright.sync_api import Page, sync_playwright

PASSWORD = "accept-pass-2026"
SHELL_MARK = "ACC-SHELL"
FILE_TEXT = "中文内容 ACC-FILE 第二行"


@dataclass
class Case:
    """一轮对话：下发 emulate 规则，发消息，核对界面回复与工具真实结果。"""
    name: str
    prompt: str
    calls: list[tuple[str, dict[str, object]]]
    reply: str
    expect_results: list[str] = field(default_factory=list)


CASES = [
    Case("text", "你好", [], "ACC-TEXT-OK"),
    Case("shell", "跑个命令", [("shell", {"command": f"echo {SHELL_MARK}-$(id -u) && uname -s",
                                            "description": "验收"})],
         "ACC-SHELL-OK", [f"{SHELL_MARK}-"]),
    Case("files", "写个文件再读出来", [
        ("write_file", {"path": "acceptance/hello.txt", "content": FILE_TEXT}),
        ("read_file", {"path": "acceptance/hello.txt"})], "ACC-FILE-OK", [FILE_TEXT]),
    Case("web_fetch", "抓一下网页", [("web_fetch", {"url": "https://example.com"})],
         "ACC-FETCH-OK", ["documentation examples"]),
    Case("visual", "画个图", [
        ("load_tools", {"plugin": "visuals@release"}),
        ("tool_call", {"name": "render_visual", "arguments": {
            "title": "验收图", "summary": "中文渲染验收", "draft": True,
            "html": "<h1 style='font-size:48px'>中文标题 ACC</h1>"}})], "ACC-VISUAL-OK"),
]


class Emulate:
    """emulate 的控制接口：每个用例前换规则，结束后读调用记录。"""

    def __init__(self, base: str) -> None:
        self._base = base.rstrip("/")

    def _send(self, method: str, path: str, body: object | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = Request(self._base + path, data=data, method=method,
                          headers={"content-type": "application/json"})
        with urlopen(request, timeout=10) as response:
            raw = response.read()
        return json.loads(raw) if raw else None

    # emulate 容器首次要现下 npm 包，开浏览器前等它能应答。
    def wait_ready(self, timeout_s: float = 180) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self._send("GET", "/_ai/rules")
                return
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(2)

    # 按顺序：每个工具调用用一次，拿到全部结果后回文本。带 tool 条件，标题等无工具请求不消耗规则。
    def program(self, case: Case) -> None:
        rules: list[dict[str, object]] = []
        for index, (tool, arguments) in enumerate(case.calls):
            rules.append({"id": f"{case.name}-{index}", "times": 1,
                          "match": {"tool": tool if tool != "tool_call" else "load_tools",
                                    "hasToolResult": index > 0},
                          "reply": [{"type": "tool_call", "name": tool, "arguments": arguments}]})
        rules.append({"id": f"{case.name}-reply", "match": {"tool": "shell"},
                      "reply": [{"type": "text", "text": case.reply}]})
        self._send("POST", "/_ai/reset")
        self._send("PUT", "/_ai/rules", {"strict": False, "models": ["akashic-mock"], "rules": rules})

    # 只看本轮：最后一次请求里，用户这条消息之后的工具结果。
    def tool_results(self, prompt: str) -> list[str]:
        calls = self._send("GET", "/_ai/calls") or []
        texts: list[str] = []
        for call in reversed(calls):
            messages = call.get("messages", [])
            starts = [i for i, m in enumerate(messages) if m.get("role") == "user"
                      and any(prompt == part.get("text") for part in m.get("parts", []))]
            if not starts:
                continue
            for message in messages[starts[-1]:]:
                texts += [json.dumps(part, ensure_ascii=False) for part in message.get("parts", [])
                          if part.get("type") == "tool_result"]
            break
        return texts


class Run:
    """记录每一步的结果、截图和失败原因。"""

    def __init__(self, page: Page, output: Path) -> None:
        self.page = page
        self.output = output
        self.steps: list[dict[str, object]] = []

    def step(self, name: str, action: Callable[[], object]) -> bool:
        started = time.monotonic()
        try:
            detail = action()
            ok = True
        except Exception as error:  # 记录任何失败并继续后续可独立的步骤。
            detail, ok = f"{type(error).__name__}: {error}"[:2000], False
        self.page.screenshot(path=str(self.output / f"{len(self.steps) + 1:02d}-{name}.png"))
        self.steps.append({"step": name, "ok": ok, "seconds": round(time.monotonic() - started, 1),
                           "detail": detail})
        print(json.dumps(self.steps[-1], ensure_ascii=False), flush=True)
        return ok


def body_text(page: Page) -> str:
    return page.locator("body").inner_text()


def wait_text(page: Page, text: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if text in body_text(page):
            return
        page.wait_for_timeout(1000)
    raise TimeoutError(f"{timeout_s:.0f}s 内没有出现：{text}")


# 一次性登录链接；首次进入时把临时密码换成自己的。
def login(page: Page, url: str) -> str:
    page.goto(url)
    page.wait_for_timeout(2500)
    if "设置你自己的密码" in body_text(page):
        fields = page.locator("input[type=password]:visible")
        fields.nth(0).fill(PASSWORD)
        fields.nth(1).fill(PASSWORD)
        page.get_by_role("button", name="保存并进入").click()
        page.wait_for_timeout(4000)
    return page.url


# 引导第一步：用"自定义 API"连上 emulate，确认它成为默认对话模型。
def connect_model(page: Page, model_base: str) -> str:
    page.goto(page.url.split("#")[0] + "#onboarding")
    page.wait_for_timeout(2500)
    if page.get_by_role("button", name="开始").count():
        page.get_by_role("button", name="开始").click()
        page.wait_for_timeout(2000)
    page.get_by_role("button", name="自定义 API").click()
    page.get_by_label("名称").fill("Emulate")
    page.get_by_label("地址").fill(model_base)
    page.get_by_role("textbox", name="API Key").fill("test-key")
    page.get_by_role("button", name="添加", exact=True).click()
    wait_text(page, "akashic-mock", 60)
    page.get_by_role("button", name="继续", exact=True).click()
    page.wait_for_timeout(3000)
    return "默认对话模型：akashic-mock"


# 引导第二步：默认全选安装官方扩展包；读取列表失败时像用户一样点重试。
def install_packages(page: Page, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    retries = 0
    while "正在读取官方扩展包" in body_text(page) or page.get_by_role("button", name="重试").count():
        if page.get_by_role("button", name="重试").count():
            retries += 1
            page.get_by_role("button", name="重试").click()
        page.wait_for_timeout(3000)
        if time.monotonic() > deadline:
            raise TimeoutError(f"读不到官方扩展包列表（重试 {retries} 次）")
    button = page.get_by_role("button", name="安装 ").first
    label = button.inner_text()
    button.click()
    while time.monotonic() < deadline:
        page.wait_for_timeout(5000)
        if page.get_by_role("button", name="继续", exact=True).is_enabled() and "正在" not in body_text(page):
            return f"{label}；列表重试 {retries} 次"
    raise TimeoutError("安装没有在期限内结束")


# 一轮对话：规则先就位，回复出现在界面上，工具结果里有可区分的真实输出。
def chat(page: Page, emulate: Emulate, case: Case) -> str:
    emulate.program(case)
    box = page.get_by_role("textbox", name="消息")
    box.fill(case.prompt)
    page.keyboard.press("Enter")
    wait_text(page, case.reply, 120)
    results = emulate.tool_results(case.prompt)
    missing = [needle for needle in case.expect_results if not any(needle in r for r in results)]
    errors = [r for r in results if "工具状态: error" in r]
    if len(results) < len(case.calls) or missing or errors:
        raise AssertionError(f"工具结果 {len(results)}/{len(case.calls)}，缺少 {missing}：{errors or results}"[:1500])
    return "；".join(r[:200] for r in results) or "无工具"


# 重建容器后：同一会话和消息仍在，还能继续对话。
def after_restart(page: Page, emulate: Emulate) -> str:
    page.goto(page.url.split("#")[0])
    page.wait_for_timeout(3000)
    page.get_by_text("ACC-TEXT-OK").first.click(timeout=15000)
    wait_text(page, "ACC-SHELL-OK", 30)
    chat(page, emulate, Case("text-again", "还在吗", [], "ACC-AGAIN-OK"))
    return "历史保留并可继续对话"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fresh", "restart"), required=True)
    parser.add_argument("--login", required=True)
    parser.add_argument("--emulate", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--install-timeout", type=float, default=1800)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    emulate = Emulate(args.emulate)
    emulate.wait_ready()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_context(viewport={"width": 1280, "height": 900}, locale="zh-CN").new_page()
        run = Run(page, args.output)
        if run.step("login", lambda: login(page, args.login)):
            if args.phase == "fresh":
                if run.step("model", lambda: connect_model(page, args.model_base)):
                    run.step("packages", lambda: install_packages(page, args.install_timeout))
                    page.goto(page.url.split("#")[0])
                    page.wait_for_timeout(3000)
                    for case in CASES:
                        run.step(f"chat-{case.name}", lambda case=case: chat(page, emulate, case))
            else:
                run.step("restart", lambda: after_restart(page, emulate))
        browser.close()
    (args.output / "steps.json").write_text(json.dumps(run.steps, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
