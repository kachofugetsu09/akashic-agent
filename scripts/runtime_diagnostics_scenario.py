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
import threading
import tracemalloc

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBE = '''from agent.plugin_composition import ServiceKey
from .diagnostics_probe_allocations import allocate_bytes, block_loop
api_version = 3
name = "diagnostics_probe"
version = "1.0.0"
async def apply(ctx):
    held = []
    async def allocate(megabytes):
        with ctx.diagnostics.operation("scenario.allocate"):
            held.append(allocate_bytes(megabytes))
        ctx.diagnostics.measure("scenario.held", sum(map(len, held)), unit="bytes")
    async def release():
        held.clear()
    async def stall():
        block_loop()
    async def burst():
        for i in range(2300):
            ctx.diagnostics.measure("scenario.counter", i)
    await ctx.provide(ServiceKey("scenario.allocate"), ctx.entrypoint(allocate))
    await ctx.provide(ServiceKey("scenario.release"), ctx.entrypoint(release))
    await ctx.provide(ServiceKey("scenario.burst"), ctx.entrypoint(burst))
    await ctx.provide(ServiceKey("scenario.stall"), ctx.entrypoint(stall))
'''


async def run(directory: Path, port: int | None) -> dict[str, bool]:
    """实际 HTTP 访问约束、插件生命周期与可分配内存共同构成验收边界。"""
    from agent.plugin_composition import ServiceKey
    from agent.plugins.bundles import set_plugin_choice
    from agent.plugins.manager import PluginManager
    from agent.plugins.install import install_git_plugin
    from agent.plugins.selection import PluginSelection
    from agent.plugins.static_manifest import load_static_plugin_manifest
    from plugins.runtime_diagnostics.contract import DIAGNOSTICS
    from plugins.runtime_diagnostics.stalls import BEAT_SECONDS
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
            set_plugin_choice(workspace, name, enabled=name in {"ui", "shell-ui"}
                              or (port is not None and name == "conversation-ui"))
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
    # 独立真实文件让脱敏后仍能精确定位本次分配，不依赖临时目录名。
    (probe / "diagnostics_probe_allocations.py").write_text(
        "def allocate_bytes(megabytes):\n    return bytearray(megabytes * 1024 * 1024)\n\n\n"
        "def block_loop():\n    import time\n    time.sleep(2.8)\n"
    )
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
        assert all(item["pluginId"] != "runtime_diagnostics@lab" for item in catalog["modules"])
        checks["no_frontend_module"] = True
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=endpoint.address), base_url="http://local") as web:
            async def get(after: int = 0):
                response = await web.get("/api/dashboard/runtime-diagnostics", params={"after": after})
                assert response.status_code == 200, response.text
                return response.json()

            async def post(path: str, values: dict[str, object] | None = None, status: int = 200):
                response = await web.post("/api/dashboard/runtime-diagnostics/" + path, json=values)
                assert response.status_code == status, response.text
                return response.json()

            async def call(name: str, *args):
                async with root.context.open_service(ServiceKey("scenario." + name)) as function:
                    await function(*args)

            if port is not None:
                # 浏览器加载实际宿主；这里只把 TCP 转发到实际 UDS。
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
            assert (await web.get("/api/dashboard/runtime-diagnostics",
                                  headers={"sec-fetch-site": "same-origin"})).status_code == 403
            assert (await web.post("/api/dashboard/runtime-diagnostics/profile",
                                   headers={"sec-fetch-site": "same-site"}, json={"seconds": 10})).status_code == 403
            assert (await web.post("/api/dashboard/runtime-diagnostics/profile/stop",
                                   headers={"sec-fetch-site": "cross-site"})).status_code == 403
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
            assert any(item["bytes"] >= 24 * 1024 * 1024
                       and any(frame["file"] == "diagnostics_probe_allocations.py" and frame["line"] == 2
                               for frame in item["frames"])
                       for item in profile["allocations"]), profile
            assert not tracemalloc.is_tracing()
            # mallinfo2 只在显式剖析首尾各读一次，两端都带分配器视图。
            for allocator in (profile["allocator_start"], profile["allocator_end"]):
                assert allocator["malloc_status"] == "unsupported" or allocator["malloc_in_use_bytes"] > 0, profile
            assert max(row["rss_bytes"] for row in report["samples"]) > baseline["samples"][-1]["rss_bytes"] + 16 * 1024 * 1024
            checks["allocation_site_and_auto_stop"] = True
            # 事件循环同步阻塞时，看门狗线程在停顿期间抓到循环线程正停在哪一行。
            await call("stall")
            async with asyncio.timeout(5):
                while not (stalls := (await get(report["through_seq"]))["stalls"]):
                    await asyncio.sleep(.1)
            stall = stalls[-1]
            # The record measures heartbeat delay; the next beat can be up to one interval away.
            assert (2.8 - BEAT_SECONDS) * 1000 <= stall["duration_ms"] < 4000, stall
            loop_thread = stall["captures"][0]["threads"][0]
            assert loop_thread["loop"] and loop_thread["frames"][0] == "diagnostics_probe_allocations.py:7 in block_loop", stall
            assert not any("malloc_status" in row for row in (await get())["samples"])
            checks["stall_stack_and_allocator"] = True
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
            exported = await web.get("/api/dashboard/runtime-diagnostics/export")
            assert leaked not in exported.text and "attachment" in exported.headers["content-disposition"]
            checks["bounded_reader_export_and_redaction"] = True
            tracemalloc.start()
            await post("profile", {"seconds": 10}, status=409)
            await post("profile/stop")
            assert tracemalloc.is_tracing()
            tracemalloc.stop()
            await post("profile", {"seconds": 120})
            # 真实安装更新会停止旧代剖析，不再发布前端模块。
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
            catalog = json.loads(await root.context.require(WEB_UI).bootstrap())
            assert all(item["pluginId"] != "runtime_diagnostics@lab" for item in catalog["modules"])
            assert (await get())["stream_id"] != old_stream
            assert len(gc.callbacks) == len(before_callbacks) + 1
            assert len(logger.handlers) == len(before_handlers) + 1
            assert [thread.name for thread in threading.enumerate()].count("diagnostics-stall-watch") == 1
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
        assert not any(thread.name == "diagnostics-stall-watch" for thread in threading.enumerate())
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
