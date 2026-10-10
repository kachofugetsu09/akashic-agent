"""实际安装 UI，验证资源、线程查询、换代、重启和卸载。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def commit(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Scenario",
                    "-c", "user.email=scenario@example.invalid", "commit", "-qm", "scenario"], check=True)


async def run(directory: Path) -> dict[str, object]:
    """通过真实安装链读取插件文件，观察物理 worker 的退出。"""
    from agent.plugin_composition import ServiceKey
    from agent.plugins.install import install_git_plugin
    from agent.plugins.manager import PluginManager
    from agent.plugins.selection import PluginSelection

    workspace = directory / "workspace"
    workspace.mkdir()
    PluginSelection(workspace).initialize()
    home = directory / "home"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION="")
    provider = directory / "ui"
    shutil.copytree(ROOT / "plugins/ui", provider, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(provider)], check=True)
    commit(provider)
    install_git_plugin(workspace=workspace, source=str(provider), marketplace="lab", plugins_home=home)
    sources = directory / "plugins"
    panel, reader = sources / "panel", sources / "reader"
    panel.mkdir(parents=True)
    reader.mkdir()
    (panel / "panel.js").write_text("export default {};\n")
    (panel / "plugin.py").write_text('''import threading
from plugins.ui.contract import UI_SLOTS, PluginUiDefinition
api_version = 3
name = "panel"
version = "1.0.0"
inject = (UI_SLOTS,)
async def apply(ctx):
    path = ctx.data_root / "value"
    if not path.exists():
        path.write_text("stored panel value")
    def query(method, payload, *, session_id, turn_id):
        if method != "read" or payload:
            raise ValueError("unknown request")
        return {"text": path.read_text(), "worker": threading.current_thread().name}
    await ctx.require(UI_SLOTS).register_plugin_ui(ctx, PluginUiDefinition("panel.js"), query=query)
''')
    (reader / "plugin.py").write_text('''from agent.plugin_composition import ServiceKey
from agent.plugin_contracts.ui import PLUGIN_UI
api_version = 3
name = "reader"
version = "1.0.0"
async def apply(ctx):
    with (ctx.data_root / "applies").open("a") as file:
        file.write("apply\\n")
    async def read():
        with ctx.borrow(PLUGIN_UI) as provider:
            if provider is None:
                return {"unavailable": "ui_provider_unavailable"}
            catalog = await provider.catalog()
            item = next(row for row in catalog["items"] if row["id"] == "panel")
            asset = await provider.asset(item["id"], item["revision"], "module", item["module_sha256"])
            value = await provider.query(item["id"], item["revision"], "read", {}, session_id=None, turn_id=None)
            return {"value": value, "module": asset["content"]}
    await ctx.provide(ServiceKey("scenario.ui.read"), ctx.entrypoint(read))
''')

    def build():
        return PluginManager([sources], workspace=workspace, installed_cache_root=home / "cache")

    async def read(host):
        root = host.live_root
        assert root is not None
        result = await root.context.require(ServiceKey("scenario.ui.read"))()
        assert result["value"]["text"] == "stored panel value"
        assert result["value"]["worker"].startswith("plugin-ui")
        assert result["module"] == "export default {};\n"

    def workers_closed():
        assert not any(thread.name.startswith("plugin-ui") for thread in threading.enumerate())

    host = build()
    try:
        # 1. 首次读取得到真实 JS 资产和在线程中读取的插件文件。
        await host.load_all()
        await read(host)
        observer = host._active_generations["reader"].fiber
        with (provider / "plugin.py").open("a") as file:
            file.write("\n# installed implementation update\n")
        commit(provider)
        await host.install(source=str(provider), marketplace="lab", ref_name="", sparse_paths=[], update_id="ui-update")
        assert host._operation is not None
        await host._operation.task
        assert host.read_update("ui-update").state == "active"
        assert host._active_generations["reader"].fiber is observer
        assert (workspace / "plugin-data/reader-builtin/applies").read_text().splitlines() == ["apply"]
        workers_closed()
        await read(host)
    finally:
        await host.terminate_all()
    workers_closed()
    # 2. 重开相同选择和数据，再卸载实际 UI provider。
    host = build()
    try:
        await host.load_all()
        await read(host)
        observer = host._active_generations["reader"].fiber
        await host.uninstall("ui@lab")
        assert host._operation is not None
        await host._operation.task
        root = host.live_root
        assert root is not None
        assert await root.context.require(ServiceKey("scenario.ui.read"))() == {"unavailable": "ui_provider_unavailable"}
        assert host._active_generations["reader"].fiber is observer
        assert host._active_generations["panel"].fiber.state != "active"
        assert (workspace / "plugin-data/panel-builtin/value").read_text() == "stored panel value"
        workers_closed()
    finally:
        await host.terminate_all()
    return {"catalog_asset": True, "worker_query": True, "generation_cleanup": True,
            "restart": True, "disable": True, "observer_stable": True, "threads_drained": True}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-ui-queries-") as temporary:
        print(json.dumps(asyncio.run(run(Path(temporary)))))
