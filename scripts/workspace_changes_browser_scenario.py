"""在 Chromium 中加载真实改动标签，控制宿主响应时序，验证会话与 diff 的归属。"""
from __future__ import annotations

import argparse
import asyncio
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from playwright.async_api import async_playwright


async def scenario(url: str, output: Path, chromium: str) -> dict[str, object]:
    """使用构建产物与最小宿主合同夹具，所有断言观察 DOM 和实际引用请求。"""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=chromium, headless=True, args=["--no-sandbox"])
        page = await browser.new_page()
        try:
            await page.goto(url)
            await page.evaluate("""async () => {
                const {activate} = await import('/plugins/workspace_changes/web_module.js');
                document.body.replaceChildren();
                const host = document.createElement('div'); document.body.append(host);
                window.requests = []; window.quotes = []; window.current = 'A';
                const ctx = {
                    http: {request: url => new Promise((resolve, reject) => requests.push({url, resolve, reject}))},
                    ui: {inject: (_, apply) => apply({register: entry => window.entry = entry})},
                };
                const view = {
                    active: false, session: 'A',
                    onSessionChange: fn => {window.changeSession = session => {current = session; fn(session)}; fn('A'); return () => {}},
                    onActiveChange: fn => {fn(false); return () => {}},
                    onIntent: fn => {window.openFile = fn; return () => {}},
                    quote: value => {if (value.session !== current) return 'stale'; quotes.push(value); return 'inserted'},
                };
                activate(ctx); entry.render(host, null, view);
                window.reply = (kind, session, data, failed = false) => {
                    const at = requests.findIndex(request => {
                        const url = new URL(request.url, location.href);
                        return url.pathname.endsWith('/' + kind) && url.searchParams.get('session') === session;
                    });
                    if (at < 0) throw new Error('missing request: ' + kind + ' ' + session);
                    const request = requests.splice(at, 1)[0];
                    if (failed) request.reject(new Error('old request failed'));
                    else request.resolve({ok: true, json: async () => data});
                };
                window.scope = session => ({status: 'ok', before: 'a', after: 'b', commits: [],
                    files: [{path: session + '.txt', status: 'M', added: 1, removed: 1}]});
                window.patch = session => ({status: 'ok', patch:
                    'diff --git a/' + session + '.txt b/' + session + '.txt\\n@@ -1 +1 @@\\n-old\\n+' + session + ' text\\n'});
            }""")

            async def show(session: str) -> None:
                await page.evaluate("s => reply('scope', s, scope(s))", session)
                await page.wait_for_function("s => requests.some(r => r.url.includes('/patch?') && r.url.includes('session=' + s))", arg=session)
                await page.evaluate("s => reply('patch', s, patch(s))", session)
                await page.wait_for_function("s => document.querySelector('.changes-pane').textContent.includes('+' + s + ' text')", arg=session)

            # 1. 正常评论有效；切到 B 的同步瞬间即撤下 A 的文件和评论入口。
            await show("A")
            await page.locator(".changes-row[data-kind=add]").click()
            await page.locator("textarea").fill("A comment")
            await page.locator("textarea").press("Enter")
            assert await page.evaluate("quotes.length === 1 && quotes[0].session === 'A'")
            await page.evaluate("changeSession('B')")
            assert await page.locator(".changes-row").count() == 0
            assert await page.locator(".changes-comment").count() == 0
            await page.evaluate("reply('scope', 'B', {status:'ok', before:'c', after:'c', files:[]})")
            await page.wait_for_function("document.querySelector('.changes-pane').children.length === 0")

            # 2. 同一对树和路径再次出现时，要重新显示内容，不能命中已清空的 DOM 缓存。
            await page.evaluate("changeSession('A')")
            await show("A")
            assert await page.locator(".changes-row[data-kind=add]").count() == 1

            # 3. C 的 patch 迟到，不能覆盖已经显示的 D。
            await page.evaluate("changeSession('C'); reply('scope', 'C', scope('C'))")
            await page.wait_for_function("requests.some(r => r.url.includes('/patch?') && r.url.includes('session=C'))")
            await page.evaluate("changeSession('D')")
            await show("D")
            await page.evaluate("reply('patch', 'C', patch('C'))")
            await page.evaluate("() => new Promise(resolve => requestAnimationFrame(resolve))")
            assert await page.locator(".changes-pane").get_attribute("data-path") == "D.txt"

            # 4. 旧请求的失败同样不能把新会话的内容变成错误。
            await page.evaluate("changeSession('E'); reply('scope', 'E', scope('E'))")
            await page.wait_for_function("requests.some(r => r.url.includes('/patch?') && r.url.includes('session=E'))")
            await page.evaluate("changeSession('F')")
            await show("F")
            await page.evaluate("reply('patch', 'E', null, true)")
            await page.evaluate("() => new Promise(resolve => requestAnimationFrame(resolve))")
            assert await page.locator(".changes-pane").get_attribute("data-path") == "F.txt"
            assert await page.locator(".is-error").count() == 0

            # 5. 文件意图等待列表期间换会话，旧意图不得继续请求新会话的其它范围。
            await page.evaluate("""() => {
                window.intentStatus = null;
                openFile({name:'workspace.file.open.v1', session:'F', payload:{path:'missing.txt'}})
                    .then(status => window.intentStatus = status);
                changeSession('G');
                reply('scope', 'F', {status:'ok', files:[]});
            }""")
            await page.wait_for_function("intentStatus !== null")
            assert await page.evaluate("intentStatus") == "invalid"
            assert await page.evaluate("requests.filter(r => r.url.includes('session=G')).length") == 1
            await show("G")
            await page.locator(".changes-row[data-kind=add]").click()
            await page.locator("textarea").fill("G comment")
            await page.locator("textarea").press("Enter")
            assert await page.evaluate("quotes.length === 2 && quotes[1].session === 'G' && quotes[1].source.includes('G.txt')")
            await page.screenshot(path=str(output / "changes.png"))
            return {"old_rows_removed": True, "same_patch_restored": True, "late_patch_ignored": True,
                    "late_error_ignored": True, "old_intent_stopped": True, "quotes": await page.evaluate("quotes")}
        finally:
            await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="执行 build:web-plugins 后的源码根目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chromium", default="/usr/bin/chromium")
    args = parser.parse_args()
    args.output.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(args.assets.resolve())))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        report = asyncio.run(scenario(f"http://127.0.0.1:{server.server_port}/", args.output, args.chromium))
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
