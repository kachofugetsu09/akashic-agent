"""真实安装的 UI/客户端 listener 经通用 Web Shell 验证路由和换代。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def commit(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Scenario",
                    "-c", "user.email=scenario@example.invalid", "commit", "-qm", "scenario"], check=True)


async def run(base: Path) -> dict[str, bool]:
    """走正式安装、实际 Core 和 TCP→UDS；只使用一次性状态。"""
    from agent.config import Config
    from agent.plugins.install import install_git_plugin
    from bootstrap.init_workspace import init_workspace
    from bootstrap.tools import build_core_runtime
    from bootstrap.web_shell import create_web_shell_server
    from core.net.http import SharedHttpResources
    from session.message import Input

    home, workspace, config = base / "home", base / "workspace", base / "config.toml"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home),
                      AKASHIC_PLUGIN_DISTRIBUTION="", AKASHIC_EXTRA_PLUGIN_DIRS="",
                      AKASHIC_EXECUTION_MODE="local")
    config.write_text('[runtime]\n')
    init_workspace(config_path=config, workspace=workspace)
    sources = base / "sources"
    for name in ("channels", "sources", "models", "ui", "shell_ui", "onboarding", "akashic_clients"):
        path = sources / name
        shutil.copytree(ROOT / "plugins" / name, path, ignore=shutil.ignore_patterns("__pycache__"))
        if name in {"ui", "akashic_clients"}:
            assets = path / "static" / ("dashboard" if name == "ui" else "chat")
            assets.mkdir(parents=True)
            (assets / "index.html").write_text(f"<!doctype html><title>{name}</title>")
            (assets / "probe-12345678.js").write_text("export {};\n")
        subprocess.run(["git", "init", "-q", "--initial-branch=source", str(path)], check=True)
        commit(path)
        install_git_plugin(workspace=workspace, source=str(path), marketplace="lab", plugins_home=home)
    observer = sources / "observer"
    observer.mkdir()
    (observer / "plugin.py").write_text('''api_version = 3
name = "observer"
version = "1.0.0"
async def apply(ctx):
    with (ctx.data_root / "applies").open("a") as file:
        file.write("apply\\n")
''')
    http = SharedHttpResources()
    core = build_core_runtime(Config.load(config, workspace=workspace), workspace, http, plugin_dirs=[observer])
    core.message_log.writer("akashic:read-only", author="user", source="conversation",
                            body_types=(Input,), content={}).append("saved-input", Input(()))
    with sqlite3.connect(workspace / "sessions.db") as database:
        original_rows = database.execute("SELECT * FROM messages ORDER BY rowid").fetchall()
    assert len(original_rows) == 1
    shell = create_web_shell_server(workspace, host="127.0.0.1", port=0)
    shell_task = None
    stopped = False
    try:
        await core.start()
        root = core.plugin_manager.live_root
        assert root is not None
        observer_fiber = next(item for item in root.fibers() if item.name == "observer")
        observer_context = observer_fiber.context
        assert sorted(endpoint.name for endpoint in root.endpoints()) == ["client", "dashboard"]
        before = core.message_log.reader("akashic:read-only").snapshot()
        shell_task = asyncio.create_task(shell.serve(), name="scenario-public-shell")
        await asyncio.to_thread(shell.startup_event.wait, 10)
        assert shell.started
        address = shell.servers[0].sockets[0].getsockname()
        origin = f"http://127.0.0.1:{address[1]}"
        async with httpx.AsyncClient(base_url=origin, trust_env=False) as client:
            assert (await client.get("/dashboard")).text == "<!doctype html><title>ui</title>"
            page = await client.get("/chat")
            assert page.text == "<!doctype html><title>akashic_clients</title>"
            assert "frame-ancestors" in page.headers["content-security-policy"]
            assert page.headers["referrer-policy"] == "no-referrer"
            asset = await client.get("/assets/probe-12345678.js")
            assert asset.status_code == 200 and asset.headers["cache-control"].endswith("immutable")
            redirect = await client.get("/settings")
            assert redirect.status_code == 308 and redirect.headers["location"] == "/#models"
            assert redirect.headers["pragma"] == "no-cache"
            assert (await client.get("/api/shell/state")).json()["chatReady"] is True
            async with websockets.connect(origin.replace("http:", "ws:") + "/ws?watch_sessions=true", proxy=None) as ws:
                assert json.loads(await ws.recv()) == {"type": "sessions.changed", "version": 2}
                path = sources / "akashic_clients"
                (path / "static/chat/index.html").write_text("<!doctype html><title>new client</title>")
                commit(path)
                await core.plugin_manager.install(source=str(path), marketplace="lab", ref_name="",
                    sparse_paths=[], update_id="client-assets-update")
                await core.plugin_manager.wait_idle()
                assert core.plugin_manager.read_update("client-assets-update").state == "active"
                try:
                    await asyncio.wait_for(ws.recv(), 10)
                except ConnectionClosed as error:
                    assert error.rcvd is not None and error.rcvd.code == 1012
                else:
                    raise AssertionError("旧客户端 WebSocket 未关闭")
            assert (await client.get("/chat")).text == "<!doctype html><title>new client</title>"
            assert observer_fiber.context is observer_context
            assert (observer_context.data_root / "applies").read_text() == "apply\n"
            assert core.message_log.reader("akashic:read-only").snapshot() == before
            plan = workspace / "runtime/endpoints.json"
            saved = plan.read_bytes()
            plan.write_text('{"invalid": true}')
            damaged = await client.get("/chat")
            assert damaged.status_code == 503 and damaged.json()["code"] == "endpoint_plan_unavailable"
            plan.write_bytes(saved)
            assert (await client.get("/chat")).status_code == 200
            await core.stop()
            stopped = True
            page = await client.get("/", headers={"accept": "text/html"})
            assert page.status_code == 503 and "Runtime 尚未就绪" in page.text
            assert (await client.get("/api/chat/sessions")).status_code == 503
            assert json.loads(plan.read_text())["endpoints"] == []
            with sqlite3.connect(workspace / "sessions.db") as database:
                assert database.execute("SELECT * FROM messages ORDER BY rowid").fetchall() == original_rows
    finally:
        shell.should_exit = True
        if shell_task is not None:
            await shell_task
        if not stopped:
            await core.stop()
        await http.aclose()
    return {"actual_installed_listeners": True, "http_and_cache": True,
            "client_state_and_redirect": True, "websocket_update_drained": True,
            "observer_not_reapplied": True, "damaged_plan_explicit": True,
            "shell_survives_runtime_stop": True, "messages_preserved": True}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="shell-endpoint-") as folder:
        print(json.dumps(asyncio.run(run(Path(folder)))))
