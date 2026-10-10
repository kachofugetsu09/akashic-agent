"""安装实际 UI，以 HTTP/WebSocket 验证 listener 的端点和换代清理。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlencode
import httpx
from websockets.asyncio.client import unix_connect
from websockets.exceptions import ConnectionClosed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def commit(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Scenario",
                    "-c", "user.email=scenario@example.invalid", "commit", "-qm", "scenario"], check=True)


async def run(base: Path) -> dict[str, bool]:
    """经过实际安装、连接和 owner 关闭，不伪造 transport 或 Scope。"""
    from agent.plugin_composition.ui import WEB_UI
    from agent.plugins.install import install_git_plugin
    from agent.plugins.manager import PluginManager
    from agent.plugins.selection import PluginSelection

    home, workspace = base / "home", base / "workspace"
    workspace.mkdir()
    PluginSelection(workspace).initialize()
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION="")
    provider = base / "ui"
    shutil.copytree(ROOT / "plugins/ui", provider, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(provider)], check=True)
    commit(provider)
    install_git_plugin(workspace=workspace, source=str(provider), marketplace="lab", plugins_home=home)
    panel, observer = base / "sources/panel", base / "sources/observer"
    panel.mkdir(parents=True)
    observer.mkdir()
    (panel / "panel.js").write_text("export function activate(ctx) { return () => {}; }\n")
    (panel / "plugin.py").write_text('''from agent.plugin_composition.ui import UI
from . import dashboard
api_version = 3
name = "panel"
version = "1.0.0"
inject = (UI,)
async def apply(ctx):
    await ctx.require(UI).register(ctx, web="panel.js", dashboard=lambda: dashboard)
''')
    (panel / "dashboard.py").write_text('''from fastapi import FastAPI, WebSocket
from agent.plugin_composition.requests import RequestContext
inject = ()
def register(app: FastAPI, context: RequestContext):
    @app.get("/api/dashboard/probe")
    async def probe():
        return {"value": "real-route"}
    @app.websocket("/api/dashboard/stream")
    async def stream(socket: WebSocket):
        await socket.accept()
        await socket.send_json({"ready": True})
        try:
            await socket.receive_text()
        finally:
            with (context.data_root / "closed").open("a") as file:
                file.write("closed\\n")
''')
    (observer / "plugin.py").write_text('''api_version = 3
name = "observer"
version = "1.0.0"
async def apply(ctx):
    with (ctx.data_root / "applies").open("a") as file:
        file.write("apply\\n")
''')
    host = None
    address = ""

    async def identity():
        catalog = json.loads(await host.live_root.context.require(WEB_UI).bootstrap())
        item = next(row for row in catalog["modules"] if row["pluginId"] == "panel")
        return dict(zip(("snapshot", "catalog", "module", "generation"),
                        (catalog["snapshotId"], catalog["catalogId"], item["pluginId"], item["generationId"])))

    try:
        for boot in range(2):
            host = PluginManager([base / "sources"], workspace=workspace, installed_cache_root=home / "cache")
            await host.load_all()
            root = host.live_root
            endpoint = next(item for item in root.endpoints() if item.name == "dashboard")
            address = endpoint.address
            assert Path(address).is_socket()
            plan = json.loads((workspace / "runtime/endpoints.json").read_text())
            assert plan["endpoints"][0]["address"] == address
            original_observer = host._active_generations["observer"].fiber
            values = await identity()
            headers = {"x-akashic-web-" + key: value for key, value in values.items()}
            async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=address), base_url="http://local") as web:
                assert (await web.get("/api/dashboard/probe", headers=headers)).json() == {"value": "real-route"}
                assert (await web.get("/api/dashboard/probe")).status_code == 403
                assert (await web.get("/dashboard")).status_code == 503  # 无构建资产时不伪造成功页面。
                query = urlencode({"__akashic_web_" + key: value for key, value in values.items()})
                async with unix_connect(address, "ws://local/api/dashboard/stream?" + query, origin="http://local") as socket:
                    assert json.loads(await socket.recv()) == {"ready": True}
                    if boot == 0:
                        with (provider / "plugin.py").open("a") as file:
                            file.write("\n# real installed UI generation\n")
                        commit(provider)
                        await host.install(source=str(provider), marketplace="lab", ref_name="",
                                           sparse_paths=[], update_id="listener-update")
                        await host.wait_idle()
                        assert host.read_update("listener-update").state == "active"
                        assert (await web.get("/api/dashboard/probe", headers=headers)).status_code == 409
                        current = await identity()
                        assert (await web.get("/api/dashboard/probe", headers={
                            "x-akashic-web-" + key: value for key, value in current.items()})).status_code == 200
                    else:
                        path = Path(address)
                        original = path.with_name("dashboard.original.sock")
                        path.rename(original)
                        path.write_text("replacement data")
                        await host.uninstall("ui@lab")
                        try:
                            await host.wait_idle()
                        except RuntimeError as error:
                            assert "UI socket 已被另一节点替换" in str(error)
                        else:
                            raise AssertionError("不能确认释放被另一节点替换的 socket")
                        assert path.read_text() == "replacement data"
                        preserved = path.with_name("replacement.keep")
                        path.rename(preserved)
                        original.rename(path)
                        await host.reconcile_disabled_and_drain("ui@lab")
                        assert preserved.read_text() == "replacement data"
                        assert not root.endpoints() and not path.exists()
                    try:
                        await socket.recv()
                    except ConnectionClosed as error:
                        assert error.code == 1012
                    else:
                        raise AssertionError("旧 owner 的 WebSocket 未关闭")
            assert host._active_generations["observer"].fiber is original_observer
            assert (workspace / "plugin-data/observer-builtin/applies").read_text().splitlines() == ["apply"] * (boot + 1)
            assert len((workspace / "plugin-data/panel-builtin/closed").read_text().splitlines()) == boot + 1
            await host.terminate_all()
            host = None
            assert not Path(address).exists()
            assert not json.loads((workspace / "runtime/endpoints.json").read_text())["endpoints"]
            assert not any(task.get_name() == "ui-dashboard-server" for task in asyncio.all_tasks())
        assert not (workspace / "sessions.db").exists()
    finally:
        if host is not None:
            await host.terminate_all()
    return {name: True for name in ("real_http", "real_websocket", "generation", "stale_catalog",
                                    "restart", "disabled_owner_drained", "replacement_preserved", "observer_stable", "no_business_database")}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="ui-listener-") as temporary:
        print(json.dumps(asyncio.run(run(Path(temporary)))))
