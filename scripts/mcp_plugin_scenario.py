"""实际 stdio server 验证 MCP 详情、换代、重启与可选诊断读取。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 这是提供 echo 工具的真实 stdio server；MCP client 完成握手和工具调用。
SERVER = '''import json
import os
from pathlib import Path
import sys
Path(os.environ["AKASHIC_PLUGIN_DATA_DIR"]).joinpath("server.pid").write_text(str(os.getpid()))
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {"protocolVersion": request["params"]["protocolVersion"],
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "echo", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "Return the given text",
                  "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": request["params"]["arguments"]["text"]}]}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32601, "message": "unknown method"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
'''


def commit(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Scenario",
                    "-c", "user.email=scenario@example.invalid", "commit", "-qm", "scenario"], check=True)


async def run(directory: Path) -> dict[str, object]:
    """通过实际安装链运行 provider；诊断消费者只借可选服务。"""
    from agent.plugin_composition import ServiceKey
    from agent.plugins.install import install_git_plugin
    from agent.plugins.manager import PluginManager
    from agent.plugins.selection import PluginSelection
    from agent.plugins.manifest import set_plugin_enabled

    workspace = directory / "workspace"
    workspace.mkdir()
    PluginSelection(workspace).initialize()
    home = directory / "home"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION="")
    # 合同依赖使用真实安装制品；stdio 场景不启用容器或后台进程实现。
    for name in ("workloads", "managed_processes"):
        source = directory / name
        shutil.copytree(ROOT / "plugins" / name, source, ignore=shutil.ignore_patterns("__pycache__"))
        subprocess.run(["git", "init", "-q", "--initial-branch=source", str(source)], check=True)
        commit(source)
        install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
        set_plugin_enabled(name + "@lab", enabled=False, plugins_home=home)
    provider = directory / "mcp"
    shutil.copytree(ROOT / "plugins/mcp", provider, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(provider)], check=True)
    commit(provider)
    install_git_plugin(workspace=workspace, source=str(provider), marketplace="lab", plugins_home=home)
    sources = directory / "plugins"
    target = directory / "target"
    reader = sources / "reader"
    target.mkdir(parents=True)
    reader.mkdir(parents=True)
    (target / "requirements.txt").write_text("")
    (target / "server.py").write_text(SERVER)
    (target / "plugin.py").write_text('''import sys
from agent.plugin_composition import ServiceKey
from plugins.mcp.contract import MCP_SERVERS, McpServerDefinition
api_version = 3
name = "target"
version = "1.0.0"
inject = (MCP_SERVERS,)
async def apply(ctx):
    servers = ctx.require(MCP_SERVERS)
    await servers.register(ctx, McpServerDefinition("echo", ("python", "server.py")))
    async def echo(text):
        async with servers.open(ctx, "echo") as server:
            async with server.route() as route:
                result = await route.call("echo", {"text": text})
                assert result.success
                return result.output
    await ctx.provide(ServiceKey("scenario.echo"), ctx.entrypoint(echo))
''')
    (reader / "plugin.py").write_text('''from agent.plugin_composition import ServiceKey
from plugins.mcp.contract import MCP_DETAIL
api_version = 3
name = "reader"
version = "1.0.0"
async def apply(ctx):
    async def read():
        with ctx.borrow(MCP_DETAIL) as detail:
            return {"unavailable": "mcp_provider_unavailable"} if detail is None else await detail("target@lab", "echo")
    await ctx.provide(ServiceKey("scenario.detail"), ctx.entrypoint(read))
''')

    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(target)], check=True)
    commit(target)
    install_git_plugin(workspace=workspace, source=str(target), marketplace="lab", plugins_home=home)
    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(reader)], check=True)
    commit(reader)
    install_git_plugin(workspace=workspace, source=str(reader), marketplace="lab", plugins_home=home)

    def build():
        return PluginManager([], workspace=workspace, installed_cache_root=home / "cache")

    def check_exit():
        pid = int((workspace / "plugin-data/target-lab/server.pid").read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        raise AssertionError(f"MCP server 未清理: {pid}")

    async def read(host):
        root = host.live_root
        assert root is not None
        detail = await root.context.require(ServiceKey("scenario.detail"))()
        assert detail[0]["name"] == "echo"
        check_exit()
        assert await root.context.require(ServiceKey("scenario.echo"))("actual echo") == "actual echo"
        check_exit()

    host = build()
    try:
        await host.load_all()
        await read(host)
        peer = host._active_generations["reader@lab"].fiber
        # 实现变化沿真实安装事务换代；可选诊断消费者保持 activation。
        with (provider / "plugin.py").open("a") as file:
            file.write("\n# scenario implementation update\n")
        commit(provider)
        await host.install(source=str(provider), marketplace="lab", ref_name="", sparse_paths=[], update_id="mcp-update")
        assert host._operation is not None
        await host._operation.task
        assert host.read_update("mcp-update").state == "active"
        assert host._active_generations["reader@lab"].fiber is peer
        await read(host)
    finally:
        await host.terminate_all()
    host = build()
    try:
        await host.load_all()
        await read(host)
        peer = host._active_generations["reader@lab"].fiber
        await host.uninstall("mcp@lab")
        assert host._operation is not None
        await host._operation.task
        root = host.live_root
        assert root is not None
        assert await root.context.require(ServiceKey("scenario.detail"))() == {"unavailable": "mcp_provider_unavailable"}
        assert host._active_generations["reader@lab"].fiber is peer
        assert peer.state == "active"
    finally:
        await host.terminate_all()
    return {"real_stdio_detail": True, "tool_call": True, "generation": True,
            "restart": True, "disable": True, "optional_reader_stable": True, "no_child_left": True}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="akashic-mcp-") as temporary:
        print(json.dumps(asyncio.run(run(Path(temporary)))))


if __name__ == "__main__":
    main()
