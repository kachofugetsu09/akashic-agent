"""实际插件从窄生命周期端口结束 Core，验证失败与恢复。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROVIDER = '''import asyncio
import sys
from agent.plugin_composition.tasks import RESTART_GATE
api_version = 3
name = "exit_owner"
version = "1.0.0"
inject = (RESTART_GATE,)
async def apply(ctx):
    async def read_intent():
        reader = asyncio.StreamReader()
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer)
        try:
            print("SCENARIO_READY", flush=True)
            line = await reader.readline()
            error = RuntimeError("native owner failed") if line == b"fail\\n" else None
            ctx.require(RESTART_GATE).request_shutdown(error)
            try:
                ctx.require(RESTART_GATE).check_open()
            except RuntimeError:
                print("ADMISSION_CLOSED", flush=True)
            else:
                raise AssertionError("停止后仍接纳新工作")
        finally:
            transport.close()
    await ctx.spawn(read_intent(), name="native-intent")
'''


async def run(base: Path) -> dict[str, bool]:
    """启动真实 main.py 子进程；管道协调三次 boot，不使用 sleep。"""
    from agent.plugins.install import install_git_plugin
    from bootstrap.init_workspace import init_workspace
    from session.log import MessageLog
    from session.message import Input

    home, workspace, config = base / "home", base / "workspace", base / "config.toml"
    env = {**os.environ, "HOME": str(home), "AKASHIC_PLUGIN_HOME": str(home),
           "AKASHIC_PLUGIN_DISTRIBUTION": "", "AKASHIC_EXTRA_PLUGIN_DIRS": "",
           "AKASHIC_EXECUTION_MODE": "local", "AKASHIC_SUPERVISED": "0",
           "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "sdk/python/src")])}
    os.environ.update({key: env[key] for key in ("HOME", "AKASHIC_PLUGIN_HOME", "AKASHIC_PLUGIN_DISTRIBUTION")})
    config.write_text('[runtime]\n[app_server]\nenabled = false\n')
    init_workspace(config_path=config, workspace=workspace)
    source = base / "source"
    source.mkdir()
    (source / "plugin.py").write_text(PROVIDER)
    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Scenario", "-c",
                    "user.email=scenario@example.invalid", "commit", "-qm", "scenario"], check=True)
    install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    log = MessageLog(workspace / "sessions.db")
    log.writer("saved", author="user", source="saved", body_types=(Input,), content={}).append("kept", Input(()))
    log.close()
    with sqlite3.connect(workspace / "sessions.db") as db:
        rows = db.execute("SELECT * FROM messages ORDER BY rowid").fetchall()
    for intent, failed in ((b"done\n", False), (b"fail\n", True), (b"", False)):
        process = await asyncio.create_subprocess_exec(sys.executable, str(ROOT / "main.py"), "gateway",
            "--config", str(config), "--workspace", str(workspace), env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            assert process.stdout is not None
            async with asyncio.timeout(20):
                while await process.stdout.readline() != b"SCENARIO_READY\n":
                    assert process.returncode is None, "runtime 在就绪前退出"
            output, error = await asyncio.wait_for(process.communicate(intent), 20)
            assert (process.returncode != 0) is failed, (output, error)
            assert b"ADMISSION_CLOSED" in output
            if failed:
                assert b"native owner failed" in error
        finally:
            if process.returncode is None:
                process.kill()
                await process.communicate()
        assert json.loads((workspace / "runtime/endpoints.json").read_text())["endpoints"] == []
        with sqlite3.connect(workspace / "sessions.db") as db:
            assert db.execute("SELECT * FROM messages ORDER BY rowid").fetchall() == rows
    return {"native_process_stop": True, "admission_closed": True,
            "failure_exit_and_visible_reason": True, "eof_exit_zero": True,
            "restart_after_failure": True, "messages_preserved": True}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="process-shutdown-") as path:
        print(json.dumps(asyncio.run(run(Path(path)))))
