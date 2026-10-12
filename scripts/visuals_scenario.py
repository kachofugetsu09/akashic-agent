"""通过真实插件安装、工具消息与附件导入验证可视化生命周期。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBE = '''from agent.plugin_composition import ServiceKey
from plugins.content.contract import CONTENT
from plugins.ledger.contract import BINDINGS, MESSAGE_CATALOG, MESSAGE_WRITERS, SESSION_ADMISSION, SessionAttributes, Output, Input, ContentPart, ToolCall, CallRef
from plugins.tools.contract import ALL_TOOLS, TOOLS, TOOL_PROGRAM_V2
api_version = 3
name = "visual_probe"
version = "1.0.0"
inject = (CONTENT, BINDINGS, TOOLS, MESSAGE_CATALOG, MESSAGE_WRITERS, SESSION_ADMISSION, ALL_TOOLS, TOOL_PROGRAM_V2)
async def apply(ctx):
    ctx.require(SESSION_ADMISSION).ensure(ctx, "visual:scenario", SessionAttributes())
    reader = ctx.require(MESSAGE_CATALOG).reader("visual:scenario")
    content = ctx.require(CONTENT)
    checks = {"text": content.check_text, "artifact_ref": content.check_artifact}
    async def call(identity, arguments):
        view = ctx.require(ALL_TOOLS)()
        selected = next(ref for ref in view.refs if ref.name == "render_visual")
        from plugins.tools.contract import ToolView
        async def authorize(binding, arguments):
            return {"decision": "allowed"}
        def check_start(transaction):
            pass
        menu = await ctx.require(TOOL_PROGRAM_V2).create_menu(reader, "scenario",
            content=checks, check_start=check_start, authorize=authorize, view=ToolView((selected,)))
        writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="agent", source="scenario",
            body_types=(Input, Output), content=checks, check_call=menu.check_call)("visual:scenario")
        writer.append(identity + ":input", Input((ContentPart("text", "解释关系，并让我动手试一试"),)))
        steps = arguments if isinstance(arguments, list) else [("先观察斜率，再调节参数。", arguments)]
        for index, (text, values) in enumerate(steps):
            message_id = identity if index == 0 else f"{identity}:{index}"
            binding = menu.bind("render_visual")
            writer.append(message_id, Output((ContentPart("text", text), ToolCall(binding, values)), "continue"))
            result = await menu.execute(CallRef(message_id, 1))
            if result.outcome != "success":
                break
        writer.append(identity + ":end", Output((ContentPart("text", "这些图可以独立调节；改变参数后，观察数值和图形是否一致。"),), "complete"))
        writer.expire()
        return result
    async def hold(entered, release, arguments):
        selected = next(ref for ref in ctx.require(ALL_TOOLS)().refs if ref.name == "render_visual")
        bindings = ctx.require(BINDINGS)
        identity = await ctx.require(TOOLS).bind(selected, bindings)
        async with bindings.open(identity, TOOLS) as (catalog, metadata):
            async with catalog.open(metadata) as target:
                prepared = await target.prepare(arguments)
                assert not isinstance(prepared, str), prepared
                entered.set()
                await release.wait()
                return await target.invoke("held-preview", prepared)
    await ctx.provide(ServiceKey("scenario.visual.hold"), ctx.entrypoint(hold))
    await ctx.provide(ServiceKey("scenario.visual.call"), ctx.entrypoint(call))
'''

HTML = '''<h2>斜率如何改变直线</h2><p>拖动参数，观察同一横坐标对应的高度。</p>
<svg viewBox="0 0 260 140" role="img" aria-label="线性关系图" style="width:100%;height:200px">
<path d="M30 10V115H250" fill="none" stroke="var(--ak-color-border-default)"/>
<path id="curve" fill="none" stroke="var(--ak-chart-1)" stroke-width="3"/>
<text x="235" y="135">x</text><text x="12" y="18">y</text></svg>
<label for="slope">斜率 <output id="value">1.0</output></label>
<input id="slope" type="range" min="0.2" max="2" step="0.1" value="1" aria-label="斜率">
<p>当 x = 2 时，y = <output id="result">2.0</output></p>
<script>const slope=document.getElementById('slope');function draw(){const a=Number(slope.value);document.getElementById('value').textContent=a.toFixed(1);document.getElementById('result').textContent=(2*a).toFixed(1);document.getElementById('curve').setAttribute('d',`M30 115L230 ${115-45*a}`)}slope.addEventListener('input',draw);draw();</script>'''


def rows(workspace: Path):
    with sqlite3.connect(workspace / "sessions.db") as connection:
        return connection.execute("SELECT * FROM messages ORDER BY seq").fetchall()


async def run(directory: Path, executable_path: str | None) -> dict[str, object]:
    """正式安装插件到隔离 home；保留数据库与导出文件供浏览器验收。"""
    from agent.plugin_composition import ServiceKey
    from agent.plugin_composition.config_input import save_config
    from agent.plugins.bundles import set_plugin_choice
    from agent.plugins.install import install_git_plugin
    from agent.plugins.manager import PluginManager
    from agent.plugins.selection import PluginSelection
    from plugins.ledger.contract import ARTIFACT_READ, MESSAGE_CATALOG
    from plugins.ui.contract import PLUGIN_UI, MESSAGE_DISPLAY

    workspace, home, sources = directory / "workspace", directory / "home", directory / "sources"
    workspace.mkdir(parents=True)
    home.mkdir()
    PluginSelection(workspace).initialize()
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION="")
    for name in ("ledger", "content", "tools", "ui"):
        shutil.copytree(ROOT / "plugins" / name, sources / name, ignore=shutil.ignore_patterns("__pycache__"))
    probe = sources / "visual_probe"
    probe.mkdir()
    (probe / "plugin.py").write_text(PROBE)
    provider = directory / "visuals"
    shutil.copytree(ROOT / "plugins/visuals", provider, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(provider)], check=True)
    subprocess.run(["git", "-C", str(provider), "add", "."], check=True)
    subprocess.run(["git", "-C", str(provider), "-c", "user.name=Scenario", "-c",
                    "user.email=scenario@example.invalid", "commit", "-qm", "visuals"], check=True)
    install_git_plugin(workspace=workspace, source=str(provider), marketplace="lab", plugins_home=home)
    save_config(workspace / "plugin-data/visuals-lab", {"executable_path": executable_path})
    build = lambda: PluginManager([sources], workspace=workspace, installed_cache_root=home / "cache")
    host = build()
    timings = []
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        for identity in ("first", "warm"):
            start = time.monotonic()
            async with root.context.open_service(ServiceKey("scenario.visual.call")) as call:
                result = await call(identity, {"title": "斜率与直线", "summary": "斜率增大时，同一 x 对应的 y 随之增大。", "html": HTML})
            timings.append(round(time.monotonic() - start, 3))
            assert result.outcome == "success", result
        reader = root.context.require(MESSAGE_CATALOG).reader("visual:scenario")
        attachments = reader.attachments_for(tuple(message.message_id for message in reader.snapshot()))
        assert len(attachments) == 2
        ref = attachments[0]
        async with root.context.open_service(ARTIFACT_READ) as files:
            lease = await files.acquire(ref)
            try:
                document = await lease.read_bytes(max_bytes=ref.size_bytes)
            finally:
                await lease.aclose()
        (directory / "document.json").write_bytes(document)
        async with root.context.open_service(PLUGIN_UI) as ui:
            catalog = await ui.catalog()
            item = next(item for item in cast(list[dict[str, str]], catalog["items"]) if item["id"] == "visuals@lab")
            module = await ui.asset(item["id"], item["revision"], "module", item["module_sha256"])
            for name, digest in cast(dict[str, str], item["scripts"]).items():
                script = await ui.asset(item["id"], item["revision"], f"script:{name}", digest)
                (directory / name).write_text(cast(str, script["content"]))
        (directory / "module.js").write_text(cast(str, module["content"]))
        (directory / "catalog.json").write_text(json.dumps(catalog))
        # 真实 HTTP 路由复用当前服务作用域，校验脚本身份和停用后的拒绝行为。
        from httpx import ASGITransport, AsyncClient
        from plugins.akashic_clients.chat_api import create_chat_app
        from plugins.akashic_clients.web_chat import WebChatChannel
        app = create_chat_app(workspace=workspace, channel=WebChatChannel(),
                              plugin_ui_scope=lambda: root.context.open_service(PLUGIN_UI))
        params = {"plugin_id": item["id"], "plugin_revision": item["revision"], "name": name}
        script_url = f"/api/chat/plugin-ui/script/{digest}.js"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://scenario") as http:
            response = await http.get(script_url, params=params)
            assert response.status_code == 200 and response.text == script["content"]
            assert (await http.get("/api/chat/plugin-ui/script/bad.js", params=params)).status_code == 422
            assert (await http.get(script_url, params={**params, "name": "missing.js"})).status_code == 404
            assert (await http.get(script_url, params={**params, "plugin_revision": "stale"})).status_code == 409
        async with root.context.open_service(MESSAGE_DISPLAY) as display:
            page = await display(reader.read_page(after_seq=-1, limit=100), display_only=False)
        (directory / "messages.json").write_text(json.dumps(page, ensure_ascii=False))
        from visuals.examples import SCALING, SCATTER, CANVAS, INTRO, EXPLANATION
        async with root.context.open_service(ServiceKey("scenario.visual.call")) as call:
            showcase = await call("showcase", [(INTRO, {"title": "同样的参数量，训练数据越多，Loss 越低", "summary": EXPLANATION, "html": SCALING}),
                (EXPLANATION + "\n\n图表语法也能自由组合散点、回归线和误差带。", {"title": "散点与回归", "summary": "散点与回归线展示整体趋势及波动。", "html": SCATTER}),
                ("并非所有解释都是统计图。下面直接用 Canvas 计算梯度下降。", {"title": "梯度下降", "summary": "每一步沿梯度下降，损失逐步降低。", "html": CANVAS})])
        assert showcase.outcome == "success", showcase
        async with root.context.open_service(MESSAGE_DISPLAY) as display:
            page = await display(reader.read_page(after_seq=-1, limit=100), display_only=False)
        (directory / "showcase-messages.json").write_text(json.dumps(page[8:], ensure_ascii=False))
        for attachment in reader.attachments_for(tuple(message.message_id for message in reader.snapshot())):
            async with root.context.open_service(ARTIFACT_READ) as files:
                lease = await files.acquire(attachment)
                try:
                    (directory / f"{attachment.artifact_id}.json").write_bytes(await lease.read_bytes(max_bytes=attachment.size_bytes))
                finally:
                    await lease.aclose()
        before = rows(workspace)
        attachment_count = len(reader.attachments_for(tuple(message.message_id for message in reader.snapshot())))
        async with root.context.open_service(ServiceKey("scenario.visual.call")) as call:
            invalid = await call("invalid", {"title": "损坏脚本", "summary": "不应发布", "html": "<script>throw new Error('deliberate failure')</script>"})
        assert invalid.outcome == "error" and "deliberate failure" in str(invalid.parts)
        assert len(reader.attachments_for(tuple(message.message_id for message in reader.snapshot()))) == attachment_count
        assert rows(workspace)[:len(before)] == before
        before = rows(workspace)
        # 停用只关闭能力；已有消息、附件字节与读取 owner 均保留。
        entered, release = asyncio.Event(), asyncio.Event()
        async def held_call():
            async with root.context.open_service(ServiceKey("scenario.visual.hold")) as hold:
                return await hold(entered, release, {"title": "在途图", "summary": "已接纳调用完成", "html": HTML})
        active = asyncio.create_task(held_call())
        await asyncio.wait_for(entered.wait(), 10)
        generation = host.generation("visuals@lab")
        assert generation is not None and generation.fiber is not None
        async with generation.fiber.context.runtime_scope():
            closing = generation.fiber.context.capture_runtime_scope()
        set_plugin_choice(workspace, "visuals@lab", enabled=False)
        stopping = asyncio.create_task(host.reconcile_changed())
        async with closing:
            await asyncio.wait_for(closing.wait_admission_closed(), 10)
        assert not stopping.done(), "插件不能在真实工具 scope 退出前关闭浏览器"
        release.set()
        assert (await active).outcome == "success"
        await stopping
        async with root.context.open_service(PLUGIN_UI) as ui:
            assert all(item["id"] != "visuals@lab" for item in cast(list[dict[str, str]], (await ui.catalog())["items"]))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://scenario") as http:
            assert (await http.get(script_url, params=params)).status_code == 404
        from plugins.tools.contract import ALL_TOOLS
        assert all(ref.name != "render_visual" for ref in root.context.require(ALL_TOOLS)().refs)
        async with root.context.open_service(ARTIFACT_READ) as files:
            lease = await files.acquire(ref)
            try:
                assert await lease.read_bytes(max_bytes=ref.size_bytes) == document
            finally:
                await lease.aclose()
        assert rows(workspace) == before
        await host.install(source=str(provider), marketplace="lab", ref_name="", sparse_paths=[], update_id="reenable")
        await host.wait_idle()
        assert host.read_update("reenable").state == "active"
        async with root.context.open_service(PLUGIN_UI) as ui:
            assert any(item["id"] == "visuals@lab" for item in cast(list[dict[str, str]], (await ui.catalog())["items"]))
        assert rows(workspace) == before
        # UI 缺席时工具仍能生成附件，不等待可选展示服务。
        set_plugin_choice(workspace, "ui", enabled=False)
        await host.reconcile_changed()
        async with root.context.open_service(ServiceKey("scenario.visual.call")) as call:
            headless = await call("headless", {"title": "无界面图", "summary": "无需 UI 依赖", "html": HTML})
        assert headless.outcome == "success"
        assert rows(workspace)[:len(before)] == before
        before = rows(workspace)
    finally:
        await host.terminate_all()
    host = build()
    try:
        await host.load_all()
        assert rows(workspace) == before
    finally:
        await host.terminate_all()
    return {"first_seconds": timings[0], "warm_seconds": timings[1], "artifact_id": ref.artifact_id,
            "artifact_sha256": ref.sha256, "artifact_bytes": ref.size_bytes,
            "checks": ["installed plugin and UI asset", "HTTP script identity and withdrawal", "real tool messages", "Plot marks and native Canvas in one reply", "preview errors do not publish",
                       "disable withdraws UI and tool", "disable preserves messages and artifact bytes",
                       "reenable restores UI", "inflight owner drains before browser closes",
                       "headless tool works", "restart preserves all message rows"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable-path", help="受管理 Chromium 的路径；默认使用 Playwright 自带版本")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = asyncio.run(run(args.output, args.executable_path))
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))
