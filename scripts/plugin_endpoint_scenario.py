"""真实监听器随插件安装、换代和停用发布端点，失败清理保留 owner。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROVIDER = '''import asyncio
from agent.plugin_composition import ServiceKey
api_version = 3
name = "endpoint_probe"
version = "1.0.0"
WITHDRAW = ServiceKey("scenario.endpoint.withdraw")
async def apply(ctx):
    socket = ctx.data_root / "probe.sock"
    async def respond(reader, writer):
        try:
            await reader.readline()
            writer.write(b"first\\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
    async def start():
        server = await asyncio.start_unix_server(respond, path=socket)
        async def stop():
            server.close()
            await server.wait_closed()
        return stop
    await ctx.effect(start, label="actual-listener")
    endpoint = await ctx.endpoint("probe", protocol="line+unix", address=str(socket), routes=("/probe",))
    await ctx.provide(WITHDRAW, ctx.entrypoint(endpoint.aclose))
'''


def commit(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Scenario", "-c",
                    "user.email=scenario@example.invalid", "commit", "-qm", "scenario"], check=True)


async def run(directory: Path) -> dict[str, object]:
    """读取真实派生计划并连接登记地址；不使用伪造端点或 transport。"""
    from agent.plugin_composition import ServiceKey
    from agent.plugins.install import install_git_plugin
    from agent.plugins.manager import PluginManager
    from agent.plugins.selection import PluginSelection

    workspace = directory / "workspace"
    workspace.mkdir()
    PluginSelection(workspace).initialize()
    home = directory / "home"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION="")
    source = directory / "endpoint_probe"
    source.mkdir()
    (source / "plugin.py").write_text(PROVIDER)
    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(source)], check=True)
    commit(source)
    install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    plan = workspace / "runtime/endpoints.json"

    def build():
        return PluginManager([], workspace=workspace, installed_cache_root=home / "cache")

    async def read(expected: str) -> None:
        rows = json.loads(plan.read_text())["endpoints"]
        assert len(rows) == 1 and rows[0]["routes"] == ["/probe"]
        reader, writer = await asyncio.open_unix_connection(rows[0]["address"])
        try:
            writer.write(b"read\n")
            await writer.drain()
            assert (await reader.readline()).decode().strip() == expected
        finally:
            writer.close()
            await writer.wait_closed()

    host = build()
    try:
        # 1. 计划的地址可实际连接；换代撤下旧监听器并发布新 generation。
        await host.load_all()
        root = host.live_root
        assert root is not None
        assert root.endpoints(), root.receipt()
        await read("first")
        old_generation = json.loads(plan.read_text())["endpoints"][0]["generation_id"]
        (source / "plugin.py").write_text(PROVIDER.replace('b"first\\n"', 'b"second\\n"'))
        commit(source)
        await host.install(source=str(source), marketplace="lab", ref_name="", sparse_paths=[], update_id="endpoint-update")
        assert host._operation is not None
        await host._operation.task
        assert host.read_update("endpoint-update").state == "active"
        await read("second")
        assert json.loads(plan.read_text())["endpoints"][0]["generation_id"] != old_generation
        # 2. 真正的文件发布错误不能让内存登记假装已经撤下。
        root = host.live_root
        assert root is not None
        before = root.endpoints()
        plan.unlink()
        plan.mkdir()
        try:
            await root.context.require(ServiceKey("scenario.endpoint.withdraw"))()
        except IsADirectoryError:
            pass
        else:
            raise AssertionError("派生计划发布错误必须传播")
        assert root.endpoints() == before
        assert Path(before[0].address).is_socket()
        plan.rmdir()
        await root.context.require(ServiceKey("scenario.endpoint.withdraw"))()
        assert root.endpoints() == ()
        assert json.loads(plan.read_text())["endpoints"] == []
    finally:
        await host.terminate_all()
    socket = workspace / "plugin-data/endpoint_probe-lab/probe.sock"
    assert not socket.exists()
    # 3. 原选择重启，随后停用时计划与 socket 都撤下。
    host = build()
    try:
        await host.load_all()
        await read("second")
        await host.uninstall("endpoint_probe@lab")
        assert host._operation is not None
        await host._operation.task
        assert json.loads(plan.read_text())["endpoints"] == []
        assert not socket.exists()
    finally:
        await host.terminate_all()
    return {"actual_listener": True, "generation": True, "publication_failure_keeps_owner": True,
            "cleanup_retry": True, "restart": True, "disable": True}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="ep-") as temporary:
        print(json.dumps(asyncio.run(run(Path(temporary)))))
