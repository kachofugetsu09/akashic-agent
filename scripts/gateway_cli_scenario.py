"""真实命令插件通过原生控制连接提交输入、查询和卸载。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import signal
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
    from bootstrap.app import AppRuntime
    from bootstrap.init_workspace import init_workspace
    from session.message import ContentPart, Input, Output
    from agent.plugin_contracts.content import CONTENT

    base.mkdir()
    home, workspace, config = base / "home", base / "workspace", base / "config.toml"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home),
                      AKASHIC_PLUGIN_DISTRIBUTION="", AKASHIC_EXTRA_PLUGIN_DIRS="",
                      AKASHIC_EXECUTION_MODE="local")
    config.write_text('[runtime]\n[app_server]\nlisten = ' + json.dumps(listen) + '\n')
    init_workspace(config_path=config, workspace=workspace)
    sources = base / "sources"
    for name in ("gateway", "sources", "models", "content", "commands", "conversation", "programmatic", "turn_projection", "ui"):
        path = sources / name
        shutil.copytree(ROOT / "plugins" / name, path, ignore=shutil.ignore_patterns("__pycache__"))
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
        assert endpoint["address"] == str(app.app_server.endpoint)
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
        assert code != 0 and "唯一 provider" in error, (output, error)
        assert observer_fiber.context is context
        assert (context.data_root / "applies").read_text() == "apply\n"
        assert core.message_log.reader("kept").snapshot() == before
        if token_before is not None:
            assert (workspace / ".app-server-token").read_bytes() == token_before
        await app.shutdown()
        stopped = True
        assert json.loads((workspace / "runtime/endpoints.json").read_text())["endpoints"] == []
        with sqlite3.connect(workspace / "sessions.db") as db:
            assert db.execute("SELECT * FROM messages WHERE session_key = 'kept'").fetchall() == saved_rows
    finally:
        if not stopped:
            await app.shutdown()
    return {"published_native_endpoint": True, "remote_commands_under_lock": True,
            "input_and_output_rpc": True, "sigint_commits_pause": True, "command_generation_update": True,
            "missing_command_explicit": True, "observer_and_history_preserved": True,
            "stop_withdraws_endpoint": True}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="gateway-cli-") as folder:
        base = Path(folder)
        print(json.dumps(asyncio.run(run(base / "unix", str(base / "private.sock")))))
        print(json.dumps(asyncio.run(run(base / "tcp", "127.0.0.1:0"))))
