"""真实插件组合、HTTP 与内存分配场景；所有状态只在临时目录中。"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tracemalloc

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBE = '''from agent.plugin_composition import ServiceKey
api_version = 3
name = "diagnostics_probe"
version = "1.0.0"
async def apply(ctx):
    held = []
    async def allocate(megabytes):
        with ctx.diagnostics.operation("scenario.allocate"):
            held.append(bytearray(megabytes * 1024 * 1024))
        ctx.diagnostics.measure("scenario.held", sum(map(len, held)), unit="bytes")
    async def release():
        held.clear()
    async def burst():
        for i in range(2300):
            ctx.diagnostics.measure("scenario.counter", i)
    await ctx.provide(ServiceKey("scenario.allocate"), ctx.entrypoint(allocate))
    await ctx.provide(ServiceKey("scenario.release"), ctx.entrypoint(release))
    await ctx.provide(ServiceKey("scenario.burst"), ctx.entrypoint(burst))
'''


async def run(directory: Path, port: int | None) -> dict[str, bool]:
    """实际 UI 许可、HTTP、插件生命周期与可分配内存共同构成验收边界。"""
    from agent.plugin_composition import ServiceKey
    from agent.plugins.bundles import set_plugin_choice
    from agent.plugins.manager import PluginManager
    from agent.plugins.install import install_git_plugin
    from agent.plugins.selection import PluginSelection
    from agent.plugins.static_manifest import load_static_plugin_manifest
    from plugins.runtime_diagnostics.contract import DIAGNOSTICS
    from plugins.ui.contract import WEB_UI

    # 1. 提供完整合同源码，只启用诊断、UI 和场景负载；不打开正式数据。
    workspace, home, sources = directory / "workspace", directory / "home", directory / "sources"
    workspace.mkdir()
    home.mkdir()
    PluginSelection(workspace).initialize()
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION="",
                      AKASHIC_EXTRA_PLUGIN_DIRS="", AKASHIC_EXECUTION_MODE="local")
    shutil.copytree(ROOT / "plugins", sources, ignore=shutil.ignore_patterns("__pycache__"))
    if port is not None:
        shutil.copytree(ROOT / "static/dashboard", sources / "ui/static/dashboard", dirs_exist_ok=True)
    for path in sources.iterdir():
        if (path / "plugin.py").is_file():
            name = load_static_plugin_manifest(path).name
            set_plugin_choice(workspace, name, enabled=name in {"ui", "shell-ui"})
    provider = directory / "runtime_diagnostics"
    shutil.copytree(ROOT / "plugins/runtime_diagnostics", provider, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(provider)], check=True)
    subprocess.run(["git", "-C", str(provider), "add", "."], check=True)
    subprocess.run(["git", "-C", str(provider), "-c", "user.name=Scenario", "-c", "user.email=scenario@example.invalid",
                    "commit", "-qm", "isolated diagnostics source"], check=True)
    install_git_plugin(workspace=workspace, source=str(provider), marketplace="lab", plugins_home=home)
    probe = sources / "diagnostics_probe"
    probe.mkdir()
    (probe / "plugin.py").write_text(PROBE)
    logging.getLogger("akashic.plugin.diagnostics").setLevel(logging.INFO)
    before_callbacks = tuple(gc.callbacks)
    logger = logging.getLogger("akashic.plugin.diagnostics")
    before_handlers = tuple(logger.handlers)
    manager = PluginManager([sources], workspace=workspace, installed_cache_root=home / "cache")
    checks: dict[str, bool] = {}
    try:
        await manager.load_all()
        root = manager.live_root
        assert root is not None
        endpoint = next(item for item in root.endpoints() if item.name == "dashboard")
        catalog = json.loads(await root.context.require(WEB_UI).bootstrap())
        module = next(item for item in catalog["modules"] if item["pluginId"] == "runtime_diagnostics@lab")
        headers = {"x-akashic-web-snapshot": catalog["snapshotId"], "x-akashic-web-catalog": catalog["catalogId"],
                   "x-akashic-web-module": module["pluginId"], "x-akashic-web-generation": module["generationId"]}
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=endpoint.address), base_url="http://local") as web:
            async def get(after: int = 0):
                response = await web.get("/api/dashboard/runtime-diagnostics", headers=headers, params={"after": after})
                assert response.status_code == 200, response.text
                return response.json()

            async def post(path: str, values: dict[str, object] | None = None, status: int = 200):
                response = await web.post("/api/dashboard/runtime-diagnostics/" + path, headers=headers, json=values)
                assert response.status_code == status, response.text
                return response.json()

            async def call(name: str, *args):
                async with root.context.open_service(ServiceKey("scenario." + name)) as function:
                    await function(*args)

            if port is not None:
                # 浏览器使用实际 Dashboard 及其 generation 头；这里只把 TCP 转发到实际 UDS。
                app = FastAPI()
                @app.get("/api/chat/web-ui/bootstrap")
                async def bootstrap():
                    async with root.context.open_service(WEB_UI) as ui:
                        return Response(await ui.bootstrap(), media_type="application/json")
                @app.get("/api/chat/web-ui/state")
                async def state():
                    async with root.context.open_service(WEB_UI) as ui:
                        return await ui.state()
                @app.post("/__scenario/allocate")
                async def allocate():
                    await call("allocate", 24)
                    return {"allocated": True}
                @app.api_route("/{path:path}", methods=["GET", "POST"])
                async def forward(request: Request, path: str):
                    reply = await web.request(request.method, "/" + path, params=request.query_params,
                        content=await request.body(), headers={key: value for key, value in request.headers.items() if key != "host"})
                    return Response(reply.content, status_code=reply.status_code,
                                    headers={key: value for key, value in reply.headers.items() if key not in {"content-length", "content-encoding"}})
                await uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")).serve()
                return {"browser_fixture_closed": True}

            # 2. 实际权限、边界校验与 OS 指标；追踪真实分配，不替换采样函数。
            assert (await web.get("/api/dashboard/runtime-diagnostics")).status_code == 403
            await post("profile", {"seconds": 0}, status=422)
            await post("profile", {"seconds": True}, status=422)
            async with asyncio.timeout(10):
                while not (baseline := await get())["samples"]:
                    await asyncio.sleep(.05)
            assert baseline["samples"][-1]["rss_bytes"] > 0
            checks["http_scope_and_real_samples"] = True
            await post("profile", {"seconds": 10})
            await post("profile", {"seconds": 10}, status=409)
            await call("allocate", 24)
            async with asyncio.timeout(20):
                while (report := await get())["profile"]["state"] != "complete":
                    await asyncio.sleep(.1)
            profile = report["profile"]
            assert profile["traced_bytes"] > 24 * 1024 * 1024
            assert any(frame["file"].endswith("diagnostics_probe/plugin.py")
                       for item in profile["allocations"] for frame in item["frames"]), profile
            assert not tracemalloc.is_tracing()
            assert max(row["rss_bytes"] for row in report["samples"]) > baseline["samples"][-1]["rss_bytes"] + 16 * 1024 * 1024
            checks["allocation_site_and_auto_stop"] = True
            assert any(row.get("operation") == "scenario.allocate" for row in report["events"])
            await call("release")
            # 3. 外部慢读者沿公开只读接口得到缺口；导出不夹带日志正文。
            cursor = report["through_seq"]
            await call("burst")
            async with root.context.open_service(DIAGNOSTICS) as reader:
                delta = reader.read(after=cursor)
            assert delta["gap"]["events"] and delta["evicted"]["events"] > 0
            assert len(delta["events"]) == 2048
            leaked = "private-diagnostic-sentinel"
            logger.info(leaked, extra={"akashic_fields": {"event": "plugin.operation.done", "note": leaked}})
            exported = await web.get("/api/dashboard/runtime-diagnostics/export", headers=headers)
            assert leaked not in exported.text and "attachment" in exported.headers["content-disposition"]
            checks["bounded_reader_export_and_redaction"] = True
            tracemalloc.start()
            await post("profile", {"seconds": 10}, status=409)
            await post("profile/stop")
            assert tracemalloc.is_tracing()
            tracemalloc.stop()
            await post("profile", {"seconds": 120})
            # 真实安装更新会停止旧代剖析，并撤销旧页面的 generation 许可。
            old_stream = report["stream_id"]
            with (provider / "plugin.py").open("a") as file:
                file.write("\n# isolated source update\n")
            subprocess.run(["git", "-C", str(provider), "add", "."], check=True)
            subprocess.run(["git", "-C", str(provider), "-c", "user.name=Scenario", "-c", "user.email=scenario@example.invalid",
                            "commit", "-qm", "isolated update"], check=True)
            await manager.install(source=str(provider), marketplace="lab", ref_name="", sparse_paths=[], update_id="diagnostics-update")
            await manager.wait_idle()
            assert manager.read_update("diagnostics-update").state == "active"
            assert not tracemalloc.is_tracing()
            assert (await web.get("/api/dashboard/runtime-diagnostics", headers=headers)).status_code == 409
            catalog = json.loads(await root.context.require(WEB_UI).bootstrap())
            module = next(item for item in catalog["modules"] if item["pluginId"] == "runtime_diagnostics@lab")
            headers.update({"x-akashic-web-snapshot": catalog["snapshotId"], "x-akashic-web-catalog": catalog["catalogId"],
                            "x-akashic-web-generation": module["generationId"]})
            assert (await get())["stream_id"] != old_stream
            assert len(gc.callbacks) == len(before_callbacks) + 1
            assert len(logger.handlers) == len(before_handlers) + 1
            checks["installed_update_stops_old_profile"] = True
            await post("profile", {"seconds": 120})
            assert (await post("profile/stop"))["state"] == "complete"
            assert not tracemalloc.is_tracing()
            checks["manual_stop"] = True
            await post("profile", {"seconds": 120})
        # 4. 停用后采样、GC、日志与剖析全部退出；无 UI 时读端口仍可用。
        await manager.terminate_all()
        assert tuple(gc.callbacks) == before_callbacks
        assert tuple(logger.handlers) == before_handlers
        assert not tracemalloc.is_tracing()
        assert not any(task.get_name() == "diagnostics-profile" for task in asyncio.all_tasks())
        assert not (workspace / "sessions.db").exists()
        checks["cleanup_without_business_database"] = True
        set_plugin_choice(workspace, "ui", enabled=False)
        set_plugin_choice(workspace, "shell-ui", enabled=False)
        manager = PluginManager([sources], workspace=workspace, installed_cache_root=home / "cache")
        await manager.load_all()
        assert manager.live_root is not None
        async with manager.live_root.context.open_service(DIAGNOSTICS) as reader:
            assert reader.read()["stream_id"] != report["stream_id"]
        checks["headless_and_new_stream"] = True
    finally:
        await manager.terminate_all()
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="akashic-diagnostics-") as temporary:
        print(json.dumps(asyncio.run(run(Path(temporary), args.port))))
