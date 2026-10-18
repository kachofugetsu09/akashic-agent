"""在真实聊天 CSP、附件 HTTP 接口和插件文档入口下检查图表启动。"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
import time

from fastapi.responses import HTMLResponse, Response
from fastapi.routing import APIRoute
from playwright.async_api import Browser, async_playwright
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.plugin_composition import DashboardContext
from plugins.akashic_clients.chat_api import create_chat_app
from plugins.akashic_clients.web_chat import WebChatChannel
from plugins.ledger.artifact_store import ArtifactStore
from plugins.ledger.attachments import ChannelAttachmentArtifactStore
from plugins.ledger.contract import AttachmentKind
from plugins.visuals import dashboard


async def check_direct_artifacts(browser: Browser, files: ChannelAttachmentArtifactStore, base: str) -> None:
    """Open active attachment formats and prove scripts cannot read the app origin."""
    # 1. Seed private data through the real attachment owner and a browser cookie.
    private = await files.import_bytes(b"private-artifact-marker", kind=AttachmentKind.FILE,
                                       filename="private.txt", media_type="text/plain")
    context = await browser.new_context()
    await context.add_cookies([{"name": "private_marker", "value": "secret-cookie", "url": base}])
    script = ("(async()=>{const result={};try{result.cookie=document.cookie}"
              "catch(e){result.cookie=e.name}try{result.read=await(await fetch("
              + json.dumps(base + "/api/chat/artifacts/" + private.artifact_id)
              + ")).text()}catch(e){result.read=e.name}window.probe=result})()")
    try:
        # 2. MIME variants must keep scripts working while removing app privileges.
        for filename, media_type in (("page.xhtml", "application/xhtml+xml"), ("page.xml", "text/xml"),
                                     ("page.html", "text/html"), ("page.svg", "image/svg+xml")):
            document = (f'<svg xmlns="http://www.w3.org/2000/svg"><script><![CDATA[{script}]]></script></svg>'
                        if filename.endswith(".svg") else
                        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Probe</title></head>'
                        f'<body><script>{script}</script></body></html>')
            ref = await files.import_bytes(document.encode(), kind=AttachmentKind.FILE,
                                           filename=filename, media_type=media_type)
            page = await context.new_page()
            await page.goto(base + "/api/chat/artifacts/" + ref.artifact_id)
            await page.wait_for_function("window.probe !== undefined")
            result = await page.evaluate("window.probe")
            assert result == {"cookie": "SecurityError", "read": "TypeError"}, (filename, result)
            print(json.dumps({"direct_attachment": filename, "app_origin_blocked": True}))
            await page.close()
    finally:
        await context.close()


async def run(workspace: Path, executable: str | None, artifact: Path | None) -> None:
    """只替换聊天页的展示夹具；安全中间件、附件存储和文档路由使用真实实现。"""
    # 1. 发布真实不可变附件，构造最小 renderer 宿主。
    metadata = ArtifactStore(workspace / "artifacts.db")
    files = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=metadata)
    document = json.loads(artifact.read_text()) if artifact else {
        "schema": "akashic.visual.v1", "title": "斜率与直线", "summary": "拖动参数",
        "html": '<h2>斜率与直线</h2><label>斜率 <output id="value">1</output>'
                '<input type="range" min="1" max="5" value="1" '
                'oninput="document.getElementById(\'value\').textContent=this.value"></label>',
    }
    content = json.dumps(document, ensure_ascii=False).encode()
    ref = await files.import_bytes(content, kind=AttachmentKind.FILE,
                                   filename=document["title"] + ".akashic-visual.json", media_type="application/json")
    app = create_chat_app(workspace=workspace, channel=WebChatChannel(), artifact_store=files)
    dashboard.register(app, DashboardContext(plugin_id="visuals@scenario", plugin_dir=ROOT / "plugins/visuals",
                                             data_root=workspace, validation=True))
    block = {"artifact_id": ref.artifact_id, "size_bytes": ref.size_bytes,
             "kind": "file", "media_type": "application/json", "filename": ref.filename,
             "origin": {"kind": "tool_result", "tool": "render_visual"}}
    page_html = ('<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">'
                 '<div id="host"></div><script type="module" src="/scenario-entry.js"></script>')
    app.router.routes.insert(0, APIRoute("/chat", lambda: HTMLResponse(page_html), methods=["GET"]))

    @app.get("/scenario-entry.js")
    async def entry() -> Response:
        return Response('import plugin from "/scenario-module.js";'
                        'window.visualError=null;window.disposeVisual=plugin.slots["message.attachment"].mount('
                        f'document.getElementById("host"),{{block:{json.dumps(block)},'
                        'reportError(message){window.visualError=message}});', media_type="text/javascript")

    @app.get("/scenario-module.js")
    async def module() -> Response:
        return Response((ROOT / "plugins/visuals/message_ui.js").read_bytes(), media_type="text/javascript")

    # 2. 使用真实 HTTP 服务和 Chromium，保留主页禁止内联脚本的策略。
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                await asyncio.sleep(0.01)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(executable_path=executable)
            try:
                await check_direct_artifacts(browser, files, f"http://127.0.0.1:{port}")
                for width in (390, 1024):
                    page = await browser.new_page(viewport={"width": width, "height": 844})
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    start = time.monotonic()
                    response = await page.goto(f"http://127.0.0.1:{port}/chat")
                    assert response is not None and "script-src 'self';" in response.headers["content-security-policy"]
                    await page.locator("iframe").wait_for(state="visible", timeout=8000)
                    assert await page.evaluate("window.visualError") is None
                    elapsed = round(time.monotonic() - start, 3)
                    frame = page.frames[1]
                    assert await frame.locator("h2").count() > 0
                    # 3. 交互和主题更新不重建文档；脚本仍无法读取父页面。
                    assert await frame.evaluate('() => {try {parent.document.body;return false} catch(error) {return error.name === "SecurityError"}}')
                    slider = frame.locator('input[type="range"]').first
                    await slider.press("End")
                    value = await slider.input_value()
                    assert float(value) == float(await slider.get_attribute("max") or "100")
                    await page.evaluate('document.documentElement.setAttribute("data-theme", "dark")')
                    assert await slider.input_value() == value
                    assert await page.evaluate("window.visualError") is None
                    assert not errors, errors
                    print(json.dumps({"width": width, "ready_seconds": elapsed, "interaction": "passed", "opaque_origin": True}))
                    await page.evaluate("window.disposeVisual()")
                    assert await page.locator("iframe").count() == 0
                    await page.close()
            finally:
                await browser.close()
    finally:
        server.should_exit = True
        await serving
        listener.close()
        metadata.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable-path")
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="visuals-frame-") as directory:
        asyncio.run(run(Path(directory), args.executable_path, args.artifact))
