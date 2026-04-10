import asyncio
import json
import signal
from pathlib import Path

import pytest

from agent.tools.shell import ShellTool


class _FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self._stdout = stdout.encode()
        self._stderr = stderr.encode()
        self.returncode = returncode
        self.pid = 4321

    async def communicate(self):
        return self._stdout, self._stderr

    def kill(self) -> None:
        return None


@pytest.mark.asyncio
async def test_shell_tool_runs_directly_by_default(monkeypatch):
    observed: dict[str, object] = {}

    async def _fake_create_subprocess_shell(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return _FakeProc(stdout="ok")

    monkeypatch.setattr(
        "agent.tools.shell.asyncio.create_subprocess_shell",
        _fake_create_subprocess_shell,
    )

    tool = ShellTool()
    result = json.loads(await tool.execute(command="printf ok"))

    assert observed["command"] == "printf ok"
    assert result["exit_code"] == 0
    assert result["output"] == "ok"


@pytest.mark.asyncio
async def test_shell_tool_uses_configured_working_dir(monkeypatch, tmp_path: Path):
    observed: dict[str, object] = {}

    async def _fake_create_subprocess_shell(command, **kwargs):
        observed["kwargs"] = kwargs
        return _FakeProc(stdout="ok")

    monkeypatch.setattr(
        "agent.tools.shell.asyncio.create_subprocess_shell",
        _fake_create_subprocess_shell,
    )

    tool = ShellTool(working_dir=tmp_path, restricted_dir=tmp_path)
    await tool.execute(command="ls", description="列目录")

    assert observed["kwargs"]["cwd"] == str(tmp_path)


@pytest.mark.asyncio
async def test_restricted_shell_blocks_network_and_outside_paths(tmp_path: Path):
    tool = ShellTool(
        allow_network=False,
        working_dir=tmp_path,
        restricted_dir=tmp_path,
    )

    network_result = json.loads(
        await tool.execute(command="curl https://example.com", description="联网")
    )
    outside_result = json.loads(
        await tool.execute(command="cp a ../b", description="越界")
    )

    assert "禁止网络访问" in network_result["error"]
    assert "父级路径" in outside_result["error"]


@pytest.mark.asyncio
async def test_shell_tool_cancel_kills_process_group(monkeypatch):
    proc = _FakeProc(stdout="", stderr="")
    observed: dict[str, object] = {}

    async def _fake_create_subprocess_shell(command, **kwargs):
        observed["kwargs"] = kwargs
        return proc

    async def _fake_wait_for(awaitable, timeout):
        coro = awaitable
        coro.close()
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "agent.tools.shell.asyncio.create_subprocess_shell",
        _fake_create_subprocess_shell,
    )
    monkeypatch.setattr("agent.tools.shell.asyncio.wait_for", _fake_wait_for)
    killpg_mock = []

    def _fake_killpg(pid, sig):
        killpg_mock.append((pid, sig))

    monkeypatch.setattr("agent.tools.shell.os.killpg", _fake_killpg)

    with pytest.raises(asyncio.CancelledError):
        await __import__("agent.tools.shell", fromlist=["_run"])._run("sleep 10", 5)

    assert observed["kwargs"]["start_new_session"] is True
    assert killpg_mock == [(proc.pid, signal.SIGKILL)]
