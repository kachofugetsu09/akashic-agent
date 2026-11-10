"""真实 UDS listener 验证 Shell 地址、禁用状态与临时 socket 所有权。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import uvicorn
from websockets.asyncio.client import unix_connect

from agent.plugin_composition import CompositionRoot
from agent.plugin_composition.endpoints import save_endpoint_plan
from agent.plugin_composition.model import FiberState, PluginRuntime
from bootstrap.web_shell import create_web_shell_app
from docker.debug.orthology_optional_requests import install_ports
from plugins.akashic_clients import plugin as clients
from plugins.channels import plugin as channels


async def check(workspace: Path) -> None:
    root = CompositionRoot("web-endpoint-scenario", endpoint_publisher=lambda endpoints: save_endpoint_plan(
        workspace / "runtime/endpoints.json", "web-endpoint-scenario", endpoints))
    runtime = await install_ports(root, workspace)
    config = workspace / "config.toml"
    app = create_web_shell_app(workspace)
    public = workspace / "runtime/web-chat.sock"
    shell_socket = workspace / "shell.sock"
    server = uvicorn.Server(uvicorn.Config(app, uds=str(shell_socket), log_level="critical"))
    serving = asyncio.create_task(server.serve())
    async with asyncio.timeout(5):
        while not server.started:
            if serving.done():
                serving.result()
                raise AssertionError("Shell 未就绪")
            await asyncio.sleep(0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://shell") as shell:
        try:
            assert (await shell.get("/api/shell/state")).status_code == 503
            config.write_text("# temporary scenario\n")
            assert (await shell.get("/api/shell/state")).status_code == 503
            await root.mount(channels.apply, name="channels", inject=channels.inject, runtime=runtime("channels"))
            for name, target in (("custom", workspace / "custom.sock"), ("default", public)):
                plugin_runtime = PluginRuntime("clients", name, workspace, workspace / "clients", workspace,
                                               {"web": {"socket_path": str(target)}})
                chat = await root.mount(clients.apply, name=name, inject=clients.inject, runtime=plugin_runtime)
                assert chat.state is FiberState.ACTIVE, chat.error
                assert public.is_symlink() == (name == "custom")
                assert (await shell.get("/api/shell/state")).json()["status"] == "ready"
                assert (await shell.get("/api/chat/health")).status_code == 200
                async with unix_connect(str(shell_socket), uri="ws://shell/ws") as websocket:
                    await websocket.send(json.dumps({"type": "ping", "request_id": name}))
                    assert json.loads(await websocket.recv()) == {"type": "pong", "request_id": name}
                await chat.dispose()
                assert not public.exists() and not public.is_symlink()
                assert not target.exists()
                assert (await shell.get("/api/shell/state")).status_code == 503

            # 外来节点阻止发布，不被启动回滚删除。
            public.write_text("foreign")
            blocked_runtime = PluginRuntime("clients", "blocked", workspace, workspace / "clients", workspace,
                                            {"web": {"socket_path": str(workspace / "blocked.sock")}})
            blocked = await root.mount(clients.apply, name="blocked", inject=clients.inject, runtime=blocked_runtime)
            assert blocked.state is not FiberState.ACTIVE
            assert public.read_text() == "foreign" and not (workspace / "blocked.sock").exists()
            await blocked.dispose()
            public.unlink()

            # 关闭时不得删除其他 owner 替换的节点（保留旧 inode 防止复用）。
            chat = await root.mount(clients.apply, name="replacement", inject=clients.inject, runtime=blocked_runtime)
            assert chat.state is FiberState.ACTIVE, chat.error
            public.rename(workspace / "old-link")
            public.write_text("successor")
            await chat.dispose()
            assert public.read_text() == "successor"
            assert not (workspace / "blocked.sock").exists()
        finally:
            try:
                await root.dispose()
            finally:
                server.should_exit = True
                await serving


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-endpoint-") as directory:
        asyncio.run(check(Path(directory)))
    print("PASS: Shell proxy default/custom, availability, reload, collision, exact node cleanup")
