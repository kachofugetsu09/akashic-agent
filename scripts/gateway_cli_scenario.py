"""真实命令插件通过原生控制连接提交输入、查询和卸载。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def commit(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Scenario",
                    "-c", "user.email=scenario@example.invalid", "commit", "-qm", "scenario"], check=True)


async def run(base: Path, listen: str) -> dict[str, bool]:
    """只写一次性 workspace；命令走正式选择和独立进程。"""
    from agent.config import Config
    from agent.plugins.install import install_git_plugin
    from agent.plugins.manifest import set_plugin_enabled, workspace_plugin_data_dir
    from agent.plugin_composition.config_input import save_config
    from bootstrap.app import AppRuntime
    from bootstrap.init_workspace import init_workspace
    from session.message import ContentPart, Input, Output
    from plugins.content.contract import CONTENT

    base.mkdir()
    home, workspace, config = base / "home", base / "workspace", base / "config.toml"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home),
                      AKASHIC_PLUGIN_DISTRIBUTION="", AKASHIC_EXTRA_PLUGIN_DIRS="",
                      AKASHIC_EXECUTION_MODE="local")
    config.write_text('[runtime]\n')
    init_workspace(config_path=config, workspace=workspace)
    sources = base / "sources"
    api_sources = ("reply", "onboarding", "workloads", "delivery")
    for name in ("gateway", "sources", "models", "content", "commands", "conversation", "programmatic", "turn_projection", "ui", *api_sources):
        path = sources / name
        shutil.copytree(ROOT / "plugins" / name, path, ignore=shutil.ignore_patterns("__pycache__"))
        subprocess.run(["git", "init", "-q", "--initial-branch=source", str(path)], check=True)
        commit(path)
        install_git_plugin(workspace=workspace, source=str(path), marketplace="lab", plugins_home=home)
        if name in api_sources:
            set_plugin_enabled(name + "@lab", enabled=False, plugins_home=home)
    save_config(workspace_plugin_data_dir(workspace, "gateway", "lab"), {"listen": listen})
    observer = sources / "observer"
    observer.mkdir()
    (observer / "plugin.py").write_text('''api_version = 3
name = "observer"
version = "1.0.0"
async def apply(ctx):
    with (ctx.data_root / "applies").open("a") as file:
        file.write("apply\\n")
''')
    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(observer)], check=True)
    commit(observer)
    install_git_plugin(workspace=workspace, source=str(observer), marketplace="lab", plugins_home=home)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "sdk/python/src")])}

    async def command(*args: str) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(sys.executable, str(ROOT / "main.py"), *args,
            "--config", str(config), "--workspace", str(workspace), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            output, error = await asyncio.wait_for(process.communicate(), 30)
        finally:
            if process.returncode is None:
                process.kill()
                await process.communicate()
        assert process.returncode is not None
        return process.returncode, output.decode(), error.decode()

    if not listen:
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(workspace / "akashic.sock"))
        stale.close()
    app = AppRuntime(Config.load(config, workspace=workspace), workspace)
    stopped = False
    try:
        # 1. Workspace lock 仍由实际 runtime 持有；远程命令不创建第二个 Core。
        await app.start()
        core = app.core
        assert core is not None
        manager = core.plugin_manager
        root = manager.live_root
        assert root is not None
        assert all(manager.generation(name + "@lab") is None for name in api_sources)
        observer_generation = manager.generation("observer@lab")
        assert observer_generation is not None and observer_generation.fiber is not None
        observer_fiber = observer_generation.fiber
        context = observer_fiber.context
        core.message_log.writer("kept", author="user", source="saved", body_types=(Input,), content={}).append("saved-input", Input(()))
        before = core.message_log.reader("kept").snapshot()
        with sqlite3.connect(workspace / "sessions.db") as db:
            saved_rows = db.execute("SELECT * FROM messages WHERE session_key = 'kept'").fetchall()
        code, output, error = await command("plugin-status")
        assert code == 0, (output, error)
        assert json.loads(output)["operation"]["state"] == "done"
        plan = json.loads((workspace / "runtime/endpoints.json").read_text())
        endpoint = next(item for item in plan["endpoints"] if item["name"] == "gateway")
        assert endpoint["address"] == (listen or str(workspace / "akashic.sock")) if not listen.startswith("127.") else endpoint["address"].startswith("127.0.0.1:"), endpoint
        token_before = (workspace / ".app-server-token").read_bytes() if listen.startswith("127.") else None

        # 2. 真实程序来源提交 Input；原 Message writer 追加终态，CLI 从 RPC 读取结果。
        code, output, error = await command("exec", "--new", "--session", "programmatic:cli", "--message-id", "input-one", "--detach", "hello")
        assert code == 0, (output, error)
        reader = core.message_log.reader("programmatic:cli")
        assert next(part.value for part in reader.get("input-one").body.parts if part.kind == "text") == "hello"
        core.message_log.writer("programmatic:cli", author="assistant", source="programmatic", body_types=(Output,),
                                content={"text": root.service_value(CONTENT).check_text}).append(
            "output-one", Output((ContentPart("text", "native answer"),), finish="complete"))
        code, output, error = await command("exec", "--session", "programmatic:cli", "--message-id", "input-one", "hello", "--final-only")
        assert code == 0 and output.strip() == "native answer", (output, error)


        process = await asyncio.create_subprocess_exec(sys.executable, str(ROOT / "main.py"),
            "exec", "--new", "--session", "programmatic:paused", "--json", "waiting",
            "--config", str(config), "--workspace", str(workspace), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            assert process.stdout is not None
            async with asyncio.timeout(15):
                while json.loads(await process.stdout.readline())["type"] != "reply.status":
                    pass
            process.send_signal(signal.SIGINT)
            output_bytes, error_bytes = await asyncio.wait_for(process.communicate(), 15)
            assert process.returncode == 130, (output_bytes, error_bytes)
            assert core.message_log.reader("programmatic:paused").snapshot()[-1].body.action == "pause"
        finally:
            if process.returncode is None:
                process.kill()
                await process.communicate()

        code, output, error = await command("exec", "--new", "--detach", "--endpoint", "192.0.2.1:1234", "rejected")
        assert code == 2 and "loopback" in error, (output, error)

        # 3. 管理命令真实提交安装/换代/卸载；无关 Fiber 与历史保持原身份。
        path = sources / "gateway"
        with (path / "cli.py").open("a") as file:
            file.write("\n# scenario source update\n")
        commit(path)
        code, output, error = await command("plugin-install", "--source", str(path), "--marketplace", "lab", "--update-id", "gateway-update")
        assert code == 0 and json.loads(output)["selection"] == "selected", (output, error)
        await manager.wait_idle()
        code, output, error = await command("plugin-status", "gateway-update")
        assert code == 0 and json.loads(output)["state"] == "active", (output, error)
        code, output, error = await command("plugin-uninstall", "gateway@lab", "--json")
        assert code == 0, (output, error)
        await manager.wait_idle()
        code, output, error = await command("plugin-status")
        assert code != 0 and "唯一 provider" in error and "plugin-enable gateway@" in error, (output, error)
        assert observer_fiber.context is context
        assert (context.data_root / "applies").read_text() == "apply\n"
        assert core.message_log.reader("kept").snapshot() == before
        if token_before is not None:
            assert (workspace / ".app-server-token").read_bytes() == token_before
        await manager.install(source=str(path), marketplace="lab", ref_name="", sparse_paths=[], update_id="gateway-reinstall")
        await manager.wait_idle()
        await app.shutdown()
        stopped = True
        assert json.loads((workspace / "runtime/endpoints.json").read_text())["endpoints"] == []
        with sqlite3.connect(workspace / "sessions.db") as db:
            assert db.execute("SELECT * FROM messages WHERE session_key = 'kept'").fetchall() == saved_rows
    finally:
        if not stopped:
            await app.shutdown()
    code, output, error = await command("plugin-status")
    assert code != 0 and "plugin-enable gateway@" in error, (output, error)
    save_config(workspace_plugin_data_dir(workspace, "gateway", "lab"),
                {"enabled": False, "listen": "192.0.2.1:9"})
    await stdio(config, workspace, env, sources / "gateway")
    return {"inactive_api_sources": True, "stdio_eof_and_failure_cleanup": True, "published_native_endpoint": True, "remote_commands_under_lock": True,
            "input_and_output_rpc": True, "sigint_commits_pause": True, "command_generation_update": True,
            "missing_command_explicit": True, "management_endpoint_recovery": True, "observer_and_history_preserved": True,
            "stop_withdraws_endpoint": True}


async def stdio(config: Path, workspace: Path, env: dict[str, str], source: Path) -> None:
    """真实插件命令启动正式宿主，EOF 和超长 frame 都必须结算锁与历史。"""
    with sqlite3.connect(workspace / "sessions.db") as db:
        rows = db.execute("SELECT * FROM messages WHERE session_key = 'kept'").fetchall()
    with (source / "cli.py").open("a") as file:
        file.write("\n# stdio source update\n")
    commit(source)
    for oversized in (False, True):
        with tempfile.TemporaryFile() as logs:
            process = await asyncio.create_subprocess_exec(sys.executable, str(ROOT / "main.py"),
                "app-server", "--stdio", "--config", str(config), "--workspace", str(workspace),
                env=env, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=logs)
            assert process.stdin is not None and process.stdout is not None
            try:
                async def request(identity: int, method: str, params: dict[str, object]) -> dict:
                    process.stdin.write((json.dumps({"jsonrpc": "2.0", "id": identity,
                        "method": method, "params": params}) + "\n").encode())
                    await process.stdin.drain()
                    try:
                        line = await asyncio.wait_for(process.stdout.readline(), 20)
                    except TimeoutError as error:
                        logs.seek(0)
                        raise TimeoutError(f"stdio {method} 未响应：{logs.read().decode()}") from error
                    if not line:
                        logs.seek(0)
                        raise AssertionError(logs.read().decode())
                    frame = json.loads(line)
                    assert frame["id"] == identity and "error" not in frame, frame
                    return frame["result"]
                result = await request(1, "initialize", {"protocolVersion": "2.0",
                    "clientInfo": {"name": "scenario", "version": "1"}})
                assert result["workspace"] == str(workspace)
                process.stdin.write(b'{"jsonrpc":"2.0","method":"initialized"}\n')
                result = await request(2, "server/status", {})
                assert result["bootId"] and result["protocolVersion"] == "2.0"
                result = await request(3, "message/read", {"session_id": "kept"})
                assert len(result["items"]) == len(rows)
                if not oversized:
                    result = await request(4, "plugin/install", {"source": str(source),
                        "marketplace": "lab", "update_id": "gateway-stdio-update"})
                    assert result["selection"] == "selected"
                output, _ = await asyncio.wait_for(process.communicate(
                    b"x" * (3 * 1024 * 1024) + b"\n" if oversized else
                    b'{"jsonrpc":"2.0","id":5,"method":"server/status","params":{}}\n'), 20)
                logs.seek(0)
                error = logs.read()
                for line in output.splitlines():
                    assert json.loads(line)["jsonrpc"] == "2.0", output
                assert (process.returncode != 0) if oversized else (process.returncode == 0), error
                if oversized:
                    assert b"separator" in error or b"limit" in error, error
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.communicate()
            assert json.loads((workspace / "runtime/endpoints.json").read_text())["endpoints"] == []
            with sqlite3.connect(workspace / "sessions.db") as db:
                assert db.execute("SELECT * FROM messages WHERE session_key = 'kept'").fetchall() == rows

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="gateway-cli-") as folder:
        base = Path(folder)
        print(json.dumps(asyncio.run(run(base / "default", ""))))
        print(json.dumps(asyncio.run(run(base / "unix", str(base / "private.sock")))))
        print(json.dumps(asyncio.run(run(base / "tcp", "127.0.0.1:0"))))
