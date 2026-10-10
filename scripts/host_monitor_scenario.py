"""实际 Bridge 与安装插件验证探测、HTTP、换代、重启和卸载清理。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def commit(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Scenario",
                    "-c", "user.email=scenario@example.invalid", "commit", "-qm", "scenario"], check=True)


class Server(uvicorn.Server):
    async def startup(self, sockets=None):
        await super().startup(sockets)
        self.ready.set()


async def run(base: Path) -> dict[str, bool]:
    """真实协议探测只操作本场景的 boot 和 Unix socket。"""
    from agent.host_bridge.server import HostBridgeService
    from agent.host_bridge.transport import Server as BridgeServer
    from agent.host_bridge.boot import claim_host_bridge_boot
    from agent.plugin_composition import ServiceKey
    from agent.plugins.install import install_git_plugin
    from agent.plugins.manager import PluginManager
    from agent.plugins.selection import PluginSelection
    from bootstrap.dashboard_api import create_dashboard_app

    home, workspace = base / "home", base / "workspace"
    workspace.mkdir()
    PluginSelection(workspace).initialize()
    socket = base / "bridge.sock"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION="",
                      AKASHIC_EXECUTION_MODE="host-bridge", AKASHIC_HOST_BRIDGE_SOCKET=str(socket),
                      AKASHIC_HOST_BRIDGE_TOKEN="scenario", AKASHIC_BOOT_ID="monitor-scenario",
                      AKASHIC_RUNTIME_COMMIT="a" * 40, AKASHIC_HOST_TOOLCHAIN_DIGEST="b" * 64,
                      AKASHIC_WORKLOAD_SOCKET="")
    service = HostBridgeService("scenario", 60, base / "artifacts", release_commit="a" * 40,
                               toolchain_digest="b" * 64, runtime_checkout=ROOT,
                               bridge_python=Path(sys.executable))
    bridge = BridgeServer(service)
    await bridge.start(socket)
    assert await claim_host_bridge_boot() is not None
    sources = base / "sources"
    for name in ("ui", "host_execution"):
        path = sources / name
        shutil.copytree(ROOT / "plugins" / name, path, ignore=shutil.ignore_patterns("__pycache__"))
        subprocess.run(["git", "init", "-q", "--initial-branch=source", str(path)], check=True)
        commit(path)
        install_git_plugin(workspace=workspace, source=str(path), marketplace="lab", plugins_home=home)
    observer = sources / "observer"
    observer.mkdir()
    (observer / "plugin.py").write_text('''from agent.plugin_composition import ServiceKey
from plugins.host_execution.contract import HOST_STATUS
api_version = 3
name = "observer"
version = "1.0.0"
async def apply(ctx):
    async def read():
        with ctx.borrow(HOST_STATUS) as status:
            return None if status is None else status.snapshot()
    await ctx.provide(ServiceKey("scenario.status"), ctx.entrypoint(read))
''')
    host = None
    dashboard = None
    dashboard_task = None

    def monitors():
        return [task for task in asyncio.all_tasks() if task.get_name() == "host-bridge-monitor"]

    async def wait_status(web, state):
        async with asyncio.timeout(15):
            while True:
                response = await web.get("/api/runtime/host-bridge")
                assert response.status_code == 200, response.text
                if response.json()["state"] == state:
                    return
                await asyncio.sleep(0.02)  # 实际定时探测通过 HTTP 观察，没有伪造回执。

    try:
        for boot in range(2):
            host = PluginManager([observer], workspace=workspace, installed_cache_root=home / "cache")
            app = create_dashboard_app(workspace, plugin_manager=host)
            await host.load_all()
            dashboard = Server(uvicorn.Config(app, uds=str(base / "dashboard.sock"), log_level="error"))
            dashboard.ready = asyncio.Event()
            dashboard_task = asyncio.create_task(dashboard.serve())
            await asyncio.wait_for(dashboard.ready.wait(), 5)
            async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(base / "dashboard.sock")),
                                         base_url="http://local") as web:
                await wait_status(web, "healthy")
                assert len(monitors()) == 1 and not service._managers
                if boot == 0:
                    await bridge.stop()
                    await wait_status(web, "degraded")
                    bridge = BridgeServer(service)
                    await bridge.start(socket)
                    await wait_status(web, "healthy")
                    # 真实 release 身份冲突是永久拒绝，状态与 Incident 保留；不拖住 owner 清理。
                    await bridge.stop()
                    wrong = HostBridgeService("scenario", 60, base / "wrong-artifacts", release_commit="c" * 40,
                                              toolchain_digest="b" * 64, runtime_checkout=ROOT,
                                              bridge_python=Path(sys.executable))
                    bridge = BridgeServer(wrong)
                    await bridge.start(socket)
                    await wait_status(web, "degraded")
                    async with asyncio.timeout(15):
                        while monitors():
                            await asyncio.sleep(0.02)
                    assert (await web.get("/api/runtime/host-bridge")).json()["code"] == "PERMISSION_DENIED"
                    assert any(item.kind == "bridge.monitor.failed" for item in host.live_root.receipt().incidents)
                    await wrong.shutdown()
                    await bridge.stop()
                    bridge = BridgeServer(service)
                    await bridge.start(socket)
                    with (sources / "host_execution/plugin.py").open("a") as file:
                        file.write("\n# actual installed update\n")
                    commit(sources / "host_execution")
                    await host.install(source=str(sources / "host_execution"), marketplace="lab",
                                       ref_name="", sparse_paths=[], update_id="monitor-update")
                    await host.wait_idle()
                    assert host.read_update("monitor-update").state == "active"
                    await wait_status(web, "healthy")
                    assert len(monitors()) == 1
                else:
                    observer_fiber = host._active_generations["observer"].fiber
                    await host.uninstall("host_execution@lab")
                    await host.wait_idle()
                    assert (await web.get("/api/runtime/host-bridge")).status_code == 404
                    assert host._active_generations["observer"].fiber is observer_fiber
                    assert await host.live_root.context.require(ServiceKey("scenario.status"))() is None
                    assert not monitors()
            dashboard.should_exit = True
            await dashboard_task
            dashboard_task = None
            await host.terminate_all()
            host = None
            assert not monitors()
        assert not (workspace / "sessions.db").exists()
    finally:
        if dashboard is not None:
            dashboard.should_exit = True
        if dashboard_task is not None:
            await dashboard_task
        if host is not None:
            await host.terminate_all()
        await service.shutdown()
        await bridge.stop()
    return {name: True for name in ("real_bridge_http", "disconnect_recovery", "identity_rejected", "generation", "restart",
                                    "disabled_owner_drained", "observer_stable", "no_business_database")}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="host-monitor-") as temporary:
        print(json.dumps(asyncio.run(run(Path(temporary)))))
