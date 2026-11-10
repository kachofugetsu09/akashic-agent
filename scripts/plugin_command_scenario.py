"""独立 CLI 进程连接实际插件监听器，运行实例保持原 owner 与选择。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import shutil
import socket
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROVIDER = '''import asyncio
import os
from pathlib import Path
with Path(os.environ["ENTRY_MARKER"]).open("a") as file:
    file.write("entry\\n")
api_version = 3
name = "command_probe"
version = "1.0.0"
entrypoints = {"probe": "cli.main"}
async def apply(ctx):
    async def respond(reader, writer):
        try:
            value = await reader.readline()
            writer.write(b"first:" + value)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
    async def start():
        server = await asyncio.start_unix_server(respond, path=ctx.data_root / "probe.sock")
        async def stop():
            server.close()
            await server.wait_closed()
        return stop
    await ctx.effect(start, label="actual-listener")
    await ctx.endpoint("probe", protocol="line+unix", address=str(ctx.data_root / "probe.sock"))
'''
CLI = '''import asyncio
import json
async def main(arguments, *, workspace, config_path):
    endpoint = next(item for item in json.loads((workspace / "runtime/endpoints.json").read_text())["endpoints"]
                    if item["name"] == "probe")
    reader, writer = await asyncio.open_unix_connection(endpoint["address"])
    try:
        writer.write((arguments[0] + "\\n").encode())
        await writer.drain()
        print(json.dumps({"reply": (await reader.readline()).decode().strip()}))
    finally:
        writer.close()
        await writer.wait_closed()
    return 0
'''
CLIENT = '''import asyncio
from pathlib import Path
import sys
from agent.plugins.entrypoints import invoke_plugin_command
raise SystemExit(asyncio.run(invoke_plugin_command(sys.argv[1], ("input",),
                 workspace=Path(sys.argv[2]), config_path=Path(sys.argv[2]) / "config.toml")))
'''


def commit(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Scenario", "-c",
                    "user.email=scenario@example.invalid", "commit", "-qm", "scenario"], check=True)


async def run(directory: Path) -> dict[str, object]:
    """当前实例持锁，命令进程只读选择并取得真实服务响应。"""
    from agent.plugins.install import install_git_plugin
    from agent.plugins.manager import PluginManager
    from agent.plugins.selection import PluginSelection
    from bootstrap.workspace_lock import WorkspaceInstanceLock

    workspace, home = directory / "workspace", directory / "home"
    workspace.mkdir()
    PluginSelection(workspace).initialize()
    marker = directory / "entry-marker"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION="",
                      ENTRY_MARKER=str(marker))
    source = directory / "command_probe"
    source.mkdir()
    (source / "plugin.py").write_text(PROVIDER)
    (source / "cli.py").write_text(CLI)
    (source / "__init__.py").write_text('raise RuntimeError("package entry ran")\n')
    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(source)], check=True)
    commit(source)
    install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    unrelated = directory / "plugins/unrelated"
    unrelated.mkdir(parents=True)
    (unrelated / "plugin.py").write_text('api_version=3\nname="unrelated"\nversion="1"\nasync def apply(ctx): pass\n')
    host = PluginManager([unrelated.parent], workspace=workspace, installed_cache_root=home / "cache")
    lock = WorkspaceInstanceLock(workspace)
    lock.acquire()

    async def invoke(command: str = "probe"):
        process = await asyncio.create_subprocess_exec(sys.executable, "-c", CLIENT, command, str(workspace),
                    cwd=directory, env={**os.environ, "PYTHONPATH": str(ROOT)},
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output, error = await process.communicate()
        return process.returncode, output.decode(), error.decode()

    try:
        # 1. 同一 workspace 已持运行锁，独立 CLI 仍连接原监听器，不启动第二 Root。
        await host.load_all()
        root = host.live_root
        assert root is not None
        selection = workspace / "runtime/plugin-stable.json"
        plan = workspace / "runtime/endpoints.json"
        saved = selection.read_bytes(), plan.read_bytes(), marker.read_bytes()
        status, output, error = await invoke()
        assert status == 0, error
        assert json.loads(output) == {"reply": "first:input"}
        assert saved == (selection.read_bytes(), plan.read_bytes(), marker.read_bytes())
        # 2. 无关插件的已提交输入保持有效；损坏其实现不阻断已选命令。
        (unrelated / "plugin.py").write_text("invalid syntax!\n")
        status, output, error = await invoke()
        assert status == 0, error
        assert json.loads(output) == {"reply": "first:input"}
        (unrelated / "plugin.py").write_text('api_version=3\nname="unrelated"\nversion="1"\nasync def apply(ctx): pass\n')
        # 3. 真实安装更新命令代码和服务，调用者仍连接同一 Root 的新 generation。
        (source / "plugin.py").write_text(PROVIDER.replace('b"first:"', 'b"second:"'))
        (source / "cli.py").write_text(CLI.replace("return 0", "return 3"))
        commit(source)
        await host.install(source=str(source), marketplace="lab", ref_name="", sparse_paths=[], update_id="command-update")
        assert host._operation is not None
        await host._operation.task
        assert host.read_update("command-update").state == "active" and host.live_root is root
        status, output, error = await invoke()
        assert status == 3, error
        assert json.loads(output) == {"reply": "second:input"}
        assert marker.read_text().splitlines() == ["entry", "entry"]
        status, _, error = await invoke("absent")
        assert status != 0 and "需要唯一 provider" in error
        # 4. 宿主服务启动只选当前镜像资产，不执行 workspace 中的旧命令。
        distribution = directory / "image"
        (distribution / "sources").mkdir(parents=True)
        (distribution / "bundles").mkdir()
        shutil.copytree(source, distribution / "sources/command_probe", ignore=shutil.ignore_patterns(".git", "__pycache__"))
        image_cli = distribution / "sources/command_probe/cli.py"
        image_cli.write_text(CLI.replace('"reply"', '"image_reply"').replace("return 0", "return 7"))
        (distribution / "distribution.json").write_text(json.dumps({"source_commit": "a" * 40, "marketplace": "release",
            "plugins": [{"name": "command_probe"}]}))
        (distribution / "bundles/base.toml").write_text("schema_version = 1\n[rows]\n")
        before = selection.read_bytes(), plan.read_bytes(), marker.read_bytes()
        process = await asyncio.create_subprocess_exec(sys.executable, "-m", "agent.plugins.entrypoints",
            "--distribution", str(distribution), "--workspace", str(workspace),
            "--config", str(workspace / "config.toml"), "probe", "input",
            cwd=directory, env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output, error = await process.communicate()
        assert process.returncode == 7, error.decode()
        assert json.loads(output) == {"image_reply": "second:input"}
        assert before == (selection.read_bytes(), plan.read_bytes(), marker.read_bytes())
        assert not (workspace / "sessions.db").exists()
    finally:
        await host.terminate_all()
        lock.release()
    return {"separate_process_connected": True, "no_second_root": True, "selection_unchanged": True,
            "unrelated_damage_isolated": True, "current_command_generation": True, "missing_command_fails": True,
            "image_command_ignores_old_selection": True}


async def check_command_classes(directory: Path) -> dict[str, bool]:
    """空选择实际启动发行版命令；业务与管理命令保留明确失败。"""
    directory.mkdir()
    workspace, config = directory / "workspace", directory / "config.toml"
    environment = {**os.environ, "HOME": str(directory / "home"), "PYTHONPATH": str(ROOT),
                   "AKASHIC_PLUGIN_HOME": str(directory / "home"), "AKASHIC_PLUGIN_DISTRIBUTION": ""}

    async def command(name: str, *arguments: str, env=environment):
        return await asyncio.create_subprocess_exec(sys.executable, str(ROOT / "main.py"), name,
            "--workspace", str(workspace), "--config", str(config), *arguments,
            cwd=directory, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    process = await command("init")
    output, error = await process.communicate()
    assert process.returncode == 0, (output, error)
    kept = (workspace / "runtime/plugin-stable.json", config)
    assert not (workspace / "sessions.db").exists()
    before = tuple(path.read_bytes() for path in kept)
    # 1. 首次 runtime 前 dashboard 实际监听，缺少构建资产仍有明确 HTTP 503。
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    process = await command("dashboard", "--host", "127.0.0.1", "--port", str(port))
    try:
        async with asyncio.timeout(15):
            while True:
                assert process.returncode is None
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    break
                except ConnectionRefusedError:
                    await asyncio.sleep(0.02)
            writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            assert b"200 OK" in response or (b"503 Service Unavailable" in response and b"npm run build" in response)
    finally:
        if process.returncode is None:
            process.terminate()
        output, error = await process.communicate()
        assert process.returncode == 0, (output, error)
    # 2. workload-controller 固定使用镜像源码，workspace 来自 Core 命令上下文。
    image = directory / "image"
    (image / "bundles").mkdir(parents=True)
    (image / "bundles/base.toml").write_text("schema_version = 1\n[rows]\n")
    (image / "distribution.json").write_text(json.dumps({"source_commit": "a" * 40, "marketplace": "release",
        "plugins": [{"name": "host_execution"}]}))
    shutil.copytree(ROOT / "plugins/host_execution", image / "sources/host_execution",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for relative in ("plugin-data", "runtime/plugin-validation"):
        (workspace / relative).mkdir(parents=True, exist_ok=True)
    address = directory / "controller.sock"
    process = await command("workload-controller", "--socket", str(address), "--state", str(directory / "leases.json"),
        "--network", "scenario", "--allowed-uid", str(os.getuid()), "--socket-uid", str(os.getuid()),
        "--socket-gid", str(os.getgid()), "--workload-uid", str(os.getuid()), "--workload-gid", str(os.getgid()),
        env={**environment, "AKASHIC_PLUGIN_DISTRIBUTION": str(image)})
    try:
        async with asyncio.timeout(15):
            while True:
                assert process.returncode is None
                try:
                    reader, writer = await asyncio.open_unix_connection(address)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    await asyncio.sleep(0.02)
            writer.write(b'{"version":1,"action":"status","body":{}}\n')
            await writer.drain()
            result = json.loads(await reader.readline())
            writer.close()
            await writer.wait_closed()
            assert result["ok"] and result["body"]["controller_id"] and result["body"]["leases"] == 0
    finally:
        if process.returncode is None:
            process.terminate()
        output, error = await process.communicate()
        assert process.returncode == 0, (output, error)
    # 3. 管理缺席提示恢复；业务命令不能偷偷采用发行版 provider。
    for name in ("plugin-status", "plugin-uninstall", "plugin-install", "exec", "app-server"):
        process = await command(name)
        output, error = await process.communicate()
        assert process.returncode == 2 and b"provider" in error, (output, error)
        if name.startswith("plugin-"):
            assert b"plugin-enable gateway@" in error
    assert before == tuple(path.read_bytes() for path in kept)
    assert not (workspace / "sessions.db").exists()
    return {"fresh_dashboard_listener": True, "image_controller_listener": True,
            "empty_selection_preserved": True, "management_recovery_hint": True,
            "business_no_fallback": True}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="pc-") as temporary:
        print(json.dumps(asyncio.run(run(Path(temporary)))))
        print(json.dumps(asyncio.run(check_command_classes(Path(temporary) / "classes"))))
