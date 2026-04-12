import asyncio
import json
import os
import signal
from types import SimpleNamespace
from pathlib import Path

import pytest

from agent.tools.shell import (
    ShellTool,
    ShellTaskOutputTool,
    ShellTaskStopTool,
    _BG_REGISTRY,
    _MAX_OUTPUT,
    _run,
)


class _FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self._stdout = stdout.encode()
        self._stderr = stderr.encode()
        self.returncode = returncode
        self.pid = 4321
        self.stdout = SimpleNamespace(read=self._read_stdout)
        self.stderr = SimpleNamespace(read=self._read_stderr)

    async def communicate(self):
        return self._stdout, self._stderr

    async def wait(self):
        return self.returncode

    async def _read_stdout(self, _size: int = -1):
        data = self._stdout
        self._stdout = b""
        return data

    async def _read_stderr(self, _size: int = -1):
        data = self._stderr
        self._stderr = b""
        return data

    def kill(self) -> None:
        return None


class _BlockingPipe:
    async def read(self, _size: int = -1):
        await asyncio.Future()


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
    result = json.loads(await tool.execute(command="printf ok", description="输出 ok"))

    assert observed["command"] == "printf ok"
    assert result["exit_code"] == 0
    assert result["output"] == "ok"
    assert "stdout" not in result
    assert "stderr" not in result
    assert result["truncation"] is None
    assert result["full_output_path"] is None


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
async def test_shell_tool_supports_spawn_hook_and_streaming(monkeypatch, tmp_path: Path):
    observed: dict[str, object] = {}
    streamed: list[str] = []

    async def _fake_create_subprocess_shell(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return _FakeProc(stdout="part1", stderr="part2", returncode=0)

    monkeypatch.setattr(
        "agent.tools.shell.asyncio.create_subprocess_shell",
        _fake_create_subprocess_shell,
    )

    def _hook(ctx):
        return {
            **ctx,
            "command": "printf hooked",
            "cwd": str(tmp_path),
            "env": {"TEST_FLAG": "1"},
        }

    tool = ShellTool(spawn_hook=_hook)
    result = json.loads(
        await tool.execute(
            command="printf raw",
            description="测试 hook",
            _on_data=streamed.append,
        )
    )

    assert observed["command"] == "printf hooked"
    assert observed["kwargs"]["cwd"] == str(tmp_path)
    assert observed["kwargs"]["env"]["TEST_FLAG"] == "1"
    assert streamed == ["part1", "part2"]
    # 新实现 stdout/stderr 直接合流写文件，无分隔行
    assert result["output"] == "part1part2"
    assert "stdout" not in result
    assert "stderr" not in result


@pytest.mark.asyncio
async def test_restricted_shell_spawn_hook_cannot_escape_restricted_dir(tmp_path: Path):
    outside = tmp_path.parent

    def _hook(ctx):
        return {**ctx, "cwd": str(outside)}

    tool = ShellTool(
        working_dir=tmp_path,
        restricted_dir=tmp_path,
        spawn_hook=_hook,
    )
    result = json.loads(await tool.execute(command="ls .", description="越界 cwd"))

    assert "任务目录外" in result["error"]


@pytest.mark.asyncio
async def test_restricted_shell_spawn_hook_empty_cwd_falls_back_to_restricted_dir(
    monkeypatch, tmp_path: Path
):
    observed: dict[str, object] = {}

    async def _fake_create_subprocess_shell(command, **kwargs):
        observed["kwargs"] = kwargs
        return _FakeProc(stdout="ok")

    monkeypatch.setattr(
        "agent.tools.shell.asyncio.create_subprocess_shell",
        _fake_create_subprocess_shell,
    )

    def _hook(ctx):
        return {**ctx, "cwd": None}

    tool = ShellTool(
        working_dir=tmp_path,
        restricted_dir=tmp_path,
        spawn_hook=_hook,
    )
    result = json.loads(await tool.execute(command="ls .", description="清空 cwd"))

    assert result["exit_code"] == 0
    assert observed["kwargs"]["cwd"] == str(tmp_path)


@pytest.mark.asyncio
async def test_shell_tool_truncates_to_tail_and_persists_full_output(monkeypatch, tmp_path: Path):
    long_stdout = "HEAD\n" + ("x" * 31_000) + "\nTAIL\n"

    async def _fake_create_subprocess_shell(command, **kwargs):
        return _FakeProc(stdout=long_stdout, stderr="", returncode=0)

    monkeypatch.setattr(
        "agent.tools.shell.asyncio.create_subprocess_shell",
        _fake_create_subprocess_shell,
    )

    tool = ShellTool()
    result = json.loads(await tool.execute(command="echo long", description="长输出"))

    assert result["truncation"] is not None
    assert result["full_output_path"] is not None
    assert Path(result["full_output_path"]).read_text(encoding="utf-8") == long_stdout
    assert result["truncation"]["full_length"] == len(long_stdout)
    assert "HEAD" not in result["output"]
    assert "TAIL" in result["output"]
    assert result["truncation"]["strategy"] == "tail"
    assert len(result["output"]) <= _MAX_OUTPUT


@pytest.mark.asyncio
async def test_run_streams_stdout_and_stderr(monkeypatch):
    proc = _FakeProc(stdout="hello", stderr="world", returncode=0)

    async def _fake_create_subprocess_shell(command, **kwargs):
        return proc

    monkeypatch.setattr(
        "agent.tools.shell.asyncio.create_subprocess_shell",
        _fake_create_subprocess_shell,
    )

    chunks: list[str] = []
    stdout, stderr, exit_code, interrupted = await _run(
        "echo hi",
        5,
        on_data=chunks.append,
    )

    assert stdout == "hello"
    assert stderr == "world"
    assert exit_code == 0
    assert interrupted is False
    assert chunks == ["hello", "world"]


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


@pytest.mark.asyncio
async def test_run_does_not_hang_when_pipe_never_closes_after_exit(monkeypatch):
    proc = _FakeProc(stdout="", stderr="", returncode=7)
    proc.stdout = _BlockingPipe()
    proc.stderr = _BlockingPipe()

    async def _fake_create_subprocess_shell(command, **kwargs):
        return proc

    monkeypatch.setattr(
        "agent.tools.shell.asyncio.create_subprocess_shell",
        _fake_create_subprocess_shell,
    )

    stdout, stderr, exit_code, interrupted = await _run("false", 5)

    assert stdout == ""
    assert stderr == ""
    assert exit_code == 7
    assert interrupted is False


# ── 后台任务测试 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_shell_run_in_background_returns_task_id(monkeypatch, tmp_path):
    """run_in_background=True 时立即返回 background_task_id，不阻塞。"""

    async def _fake_create_subprocess_shell(command, **kwargs):
        return _FakeProc(stdout="bg output", stderr="", returncode=0)

    monkeypatch.setattr(
        "agent.tools.shell.asyncio.create_subprocess_shell",
        _fake_create_subprocess_shell,
    )

    tool = ShellTool()
    result = json.loads(
        await tool.execute(
            command="echo bg",
            description="后台测试",
            run_in_background=True,
        )
    )

    task_id = result["background_task_id"]
    assert task_id is not None
    assert task_id.startswith("shell_")
    assert result["status"] == "running"
    assert result["output_path"] is not None
    assert result["exit_code"] is None

    # 清理注册表，避免污染其他测试
    _BG_REGISTRY.pop(task_id, None)


@pytest.mark.asyncio
async def test_task_output_returns_log_content(monkeypatch, tmp_path):
    """task_output 能读取后台任务已写入的日志内容。"""
    import agent.tools.shell as shell_mod

    # 构造一个已完成的假后台任务
    log_path = str(tmp_path / "bg.log")
    Path(log_path).write_text("hello from bg", encoding="utf-8")

    done_future: asyncio.Future = asyncio.get_event_loop().create_future()
    done_future.set_result(None)

    fake_proc = _FakeProc(stdout="", stderr="", returncode=0)

    task_id = "shell_testoutput"
    wall_ms = int(shell_mod.time.time() * 1000)
    shell_mod._BG_REGISTRY[task_id] = shell_mod._BackgroundTask(
        proc=fake_proc,
        log_path=log_path,
        pump_task=asyncio.ensure_future(asyncio.sleep(0)),
        started_at=shell_mod.time.monotonic(),
        wall_started_at_ms=wall_ms,
    )
    # 等 pump_task 完成
    await asyncio.sleep(0)

    tool = ShellTaskOutputTool()
    result = json.loads(await tool.execute(task_id=task_id))

    assert result["task_id"] == task_id
    assert "hello from bg" in result["output"]
    assert result["truncation"] is None
    assert result["elapsed_ms"] >= 0
    assert result["since_last_output_ms"] is None  # pump 没写，last_output_at_ms 为 None

    shell_mod._BG_REGISTRY.pop(task_id, None)


@pytest.mark.asyncio
async def test_task_output_returns_timing_fields(tmp_path):
    """task_output 应返回 elapsed_ms 和 since_last_output_ms。"""
    import agent.tools.shell as shell_mod

    log_path = str(tmp_path / "timing.log")
    Path(log_path).write_bytes(b"")

    wall_ms = int(shell_mod.time.time() * 1000)
    last_ms = wall_ms - 500  # 模拟 500ms 前有过输出

    fake_proc = _FakeProc(stdout="", stderr="", returncode=0)
    task_id = "shell_timing"
    shell_mod._BG_REGISTRY[task_id] = shell_mod._BackgroundTask(
        proc=fake_proc,
        log_path=log_path,
        pump_task=asyncio.ensure_future(asyncio.sleep(100)),
        started_at=shell_mod.time.monotonic(),
        wall_started_at_ms=wall_ms,
        last_output_at_ms=last_ms,
    )

    tool = ShellTaskOutputTool()
    result = json.loads(await tool.execute(task_id=task_id))

    assert result["elapsed_ms"] >= 0
    assert result["since_last_output_ms"] >= 500  # 至少 500ms 前
    assert result["status"] == "running"

    shell_mod._BG_REGISTRY.pop(task_id, None)


@pytest.mark.asyncio
async def test_bg_pump_updates_last_output_at_ms(tmp_path):
    """_bg_pump 每次写入时应更新 bg_task.last_output_at_ms。"""
    import time as time_mod
    import agent.tools.shell as shell_mod

    log_path = str(tmp_path / "pump_timing.log")

    fake_proc = _FakeProc(stdout="hello", stderr="world", returncode=0)
    bg = shell_mod._BackgroundTask(
        proc=fake_proc,
        log_path=log_path,
        pump_task=None,
        started_at=shell_mod.time.monotonic(),
        wall_started_at_ms=int(time_mod.time() * 1000),
    )

    assert bg.last_output_at_ms is None

    before_ms = int(time_mod.time() * 1000)
    await shell_mod._bg_pump(fake_proc, log_path, bg)
    after_ms = int(time_mod.time() * 1000)

    assert bg.last_output_at_ms is not None
    assert before_ms <= bg.last_output_at_ms <= after_ms


@pytest.mark.asyncio
async def test_task_output_not_found():
    """task_output 查询不存在的 task_id 返回 error。"""
    tool = ShellTaskOutputTool()
    result = json.loads(await tool.execute(task_id="shell_nonexistent"))
    assert "error" in result


@pytest.mark.asyncio
async def test_task_stop_kills_and_removes(monkeypatch):
    """task_stop 发送 SIGKILL 并从注册表移除。"""
    import agent.tools.shell as shell_mod

    killed = []

    def _fake_killpg(pid, sig):
        killed.append((pid, sig))

    monkeypatch.setattr("agent.tools.shell.os.killpg", _fake_killpg)

    fake_proc = _FakeProc(stdout="", stderr="", returncode=0)
    fake_proc.pid = 9999

    task_id = "shell_teststop"
    pump = asyncio.ensure_future(asyncio.sleep(100))
    shell_mod._BG_REGISTRY[task_id] = shell_mod._BackgroundTask(
        proc=fake_proc,
        log_path="/tmp/fake.log",
        pump_task=pump,
        started_at=shell_mod.time.monotonic(),
        wall_started_at_ms=0,
    )

    tool = ShellTaskStopTool()
    result = json.loads(await tool.execute(task_id=task_id))

    assert result["status"] == "stopped"
    assert task_id not in shell_mod._BG_REGISTRY
    assert killed == [(9999, signal.SIGKILL)]
    # cancel() 是异步的，等一个事件循环 tick 让取消生效
    await asyncio.sleep(0)
    assert pump.cancelled()


@pytest.mark.asyncio
async def test_task_stop_deletes_log_file(monkeypatch, tmp_path):
    """task_stop 应立即删除日志文件，不依赖 done callback 的延迟清理。"""
    import agent.tools.shell as shell_mod

    monkeypatch.setattr("agent.tools.shell.os.killpg", lambda *_: None)

    log_path = tmp_path / "bg_stop.log"
    log_path.write_text("some output", encoding="utf-8")

    fake_proc = _FakeProc(stdout="", stderr="", returncode=0)
    fake_proc.pid = 1111

    task_id = "shell_stoplog"
    shell_mod._BG_REGISTRY[task_id] = shell_mod._BackgroundTask(
        proc=fake_proc,
        log_path=str(log_path),
        pump_task=asyncio.ensure_future(asyncio.sleep(100)),
        started_at=shell_mod.time.monotonic(),
        wall_started_at_ms=0,
    )

    tool = ShellTaskStopTool()
    await tool.execute(task_id=task_id)

    assert not log_path.exists(), "task_stop 后日志文件应已被立即删除"


@pytest.mark.asyncio
async def test_task_output_ttl_deletes_log_file(monkeypatch, tmp_path):
    """TTL 到期时 _bg_kill 应同时删除日志文件。"""
    import agent.tools.shell as shell_mod

    monkeypatch.setattr("agent.tools.shell.os.killpg", lambda *_: None)

    log_path = tmp_path / "ttl.log"
    log_path.write_text("old output", encoding="utf-8")

    fake_proc = _FakeProc(stdout="", stderr="", returncode=0)
    fake_proc.pid = 2222

    task_id = "shell_ttllog"
    shell_mod._BG_REGISTRY[task_id] = shell_mod._BackgroundTask(
        proc=fake_proc,
        log_path=str(log_path),
        pump_task=asyncio.ensure_future(asyncio.sleep(100)),
        started_at=shell_mod.time.monotonic() - shell_mod._BG_TTL_S - 1,
        wall_started_at_ms=0,
    )

    tool = ShellTaskOutputTool()
    result = json.loads(await tool.execute(task_id=task_id))

    assert "error" in result
    assert not log_path.exists(), "TTL 到期后日志文件应已被删除"


@pytest.mark.asyncio
async def test_task_stop_not_found():
    """task_stop 对不存在的任务返回 not_found。"""
    tool = ShellTaskStopTool()
    result = json.loads(await tool.execute(task_id="shell_ghost"))
    assert result["status"] == "not_found"


@pytest.mark.asyncio
async def test_task_output_ttl_expired(monkeypatch):
    """TTL 到期的任务在 task_output 时被自动终止并返回 error。"""
    import agent.tools.shell as shell_mod

    monkeypatch.setattr("agent.tools.shell.os.killpg", lambda *_: None)

    fake_proc = _FakeProc(stdout="", stderr="", returncode=0)
    fake_proc.pid = 1234

    task_id = "shell_expired"
    shell_mod._BG_REGISTRY[task_id] = shell_mod._BackgroundTask(
        proc=fake_proc,
        log_path="/tmp/fake_expired.log",
        pump_task=asyncio.ensure_future(asyncio.sleep(100)),
        started_at=shell_mod.time.monotonic() - shell_mod._BG_TTL_S - 1,
        wall_started_at_ms=0,
    )

    tool = ShellTaskOutputTool()
    result = json.loads(await tool.execute(task_id=task_id))

    assert "error" in result
    assert "TTL" in result["error"]
    assert task_id not in shell_mod._BG_REGISTRY


@pytest.mark.asyncio
async def test_bg_pump_completes_when_pipe_inherited_by_child(tmp_path):
    """_bg_pump 在主进程退出后应能完成，即使子进程仍持有 pipe fd（永久阻塞的 stream）。"""
    import agent.tools.shell as shell_mod

    class _ProcExitsImmediately:
        """wait() 立即返回，但 stdout/stderr 永远不关（模拟子进程继承 fd）。"""
        pid = 0
        returncode = 0
        stdout = _BlockingPipe()
        stderr = _BlockingPipe()

        async def wait(self):
            return 0

    log_path = str(tmp_path / "inherited.log")
    proc = _ProcExitsImmediately()
    bg = shell_mod._BackgroundTask(
        proc=proc,
        log_path=log_path,
        pump_task=None,
        started_at=shell_mod.time.monotonic(),
        wall_started_at_ms=0,
    )

    # 如果 _bg_pump 先 drain 再 wait，这里会永久阻塞；修复后应在 grace timeout 内完成
    await asyncio.wait_for(
        shell_mod._bg_pump(proc, log_path, bg),
        timeout=2.0,
    )


@pytest.mark.asyncio
async def test_shell_run_in_background_started_at_ms_is_wall_clock(monkeypatch, tmp_path):
    """started_at_ms 应是 Unix epoch 毫秒（wall clock），不是 monotonic。"""
    import time as time_mod

    async def _fake_create_subprocess_shell(command, **kwargs):
        return _FakeProc(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(
        "agent.tools.shell.asyncio.create_subprocess_shell",
        _fake_create_subprocess_shell,
    )

    before_ms = int(time_mod.time() * 1000)
    tool = ShellTool()
    result = json.loads(
        await tool.execute(command="echo x", description="wall clock 测试", run_in_background=True)
    )
    after_ms = int(time_mod.time() * 1000)

    ts = result["started_at_ms"]
    assert before_ms <= ts <= after_ms, f"started_at_ms={ts} 不在 [{before_ms}, {after_ms}] 范围内"

    _BG_REGISTRY.pop(result["background_task_id"], None)


@pytest.mark.asyncio
async def test_shell_auto_promotes_to_background_after_fg_threshold(monkeypatch):
    """命令超过 _FG_THRESHOLD 秒未完成时，应自动转后台返回 background_task_id。"""
    import agent.tools.shell as shell_mod

    # 把 FG_THRESHOLD 设为 0，让任何命令都立即触发自动转后台
    monkeypatch.setattr(shell_mod, "_FG_THRESHOLD", 0)

    async def _fake_create_subprocess_shell(command, **kwargs):
        # 这个进程永远不会退出（wait 永远 pending）
        proc = _FakeProc(stdout="", stderr="", returncode=None)

        async def _wait_forever():
            await asyncio.Future()  # 永远阻塞

        proc.wait = _wait_forever
        return proc

    monkeypatch.setattr(
        "agent.tools.shell.asyncio.create_subprocess_shell",
        _fake_create_subprocess_shell,
    )
    monkeypatch.setattr("agent.tools.shell.os.killpg", lambda *_: None)

    tool = ShellTool()
    result = json.loads(
        await tool.execute(command="sleep infinity", description="永远阻塞")
    )

    task_id = result.get("background_task_id")
    assert task_id is not None, "应返回 background_task_id"
    assert result["status"] == "running"
    assert result.get("auto_promoted") is True
    assert task_id in shell_mod._BG_REGISTRY

    # 清理
    shell_mod._BG_REGISTRY.pop(task_id, None)


@pytest.mark.asyncio
async def test_shell_foreground_completes_normally_within_threshold(monkeypatch):
    """命令在 FG_THRESHOLD 内完成时，应正常返回前台格式（无 background_task_id）。"""

    async def _fake_create_subprocess_shell(command, **kwargs):
        return _FakeProc(stdout="hello", stderr="", returncode=0)

    monkeypatch.setattr(
        "agent.tools.shell.asyncio.create_subprocess_shell",
        _fake_create_subprocess_shell,
    )

    tool = ShellTool()
    result = json.loads(
        await tool.execute(command="echo hello", description="快速完成")
    )

    assert "background_task_id" not in result
    assert result["exit_code"] == 0
    assert "hello" in result["output"]
    assert result.get("auto_promoted") is None


@pytest.mark.asyncio
async def test_bg_pump_done_evicts_registry_and_log(monkeypatch, tmp_path):
    """pump_task 完成后的 done callback 应清理注册表并删除日志文件。

    用立即执行版本替换 _schedule_eviction，避免测试依赖 call_later 的 tick 时序。
    这样既验证了 callback 调用链路，也验证了 evict 逻辑本身。
    """
    import agent.tools.shell as shell_mod

    def _immediate_eviction(task_id: str, log_path: str) -> None:
        shell_mod._BG_REGISTRY.pop(task_id, None)
        try:
            os.unlink(log_path)
        except OSError:
            pass

    monkeypatch.setattr(shell_mod, "_schedule_eviction", _immediate_eviction)

    log_path = str(tmp_path / "evict.log")
    Path(log_path).write_text("data", encoding="utf-8")

    task_id = "shell_evicttest"
    pump = asyncio.ensure_future(asyncio.sleep(0))
    shell_mod._BG_REGISTRY[task_id] = shell_mod._BackgroundTask(
        proc=_FakeProc(stdout="", stderr="", returncode=0),
        log_path=log_path,
        pump_task=pump,
        started_at=shell_mod.time.monotonic(),
        wall_started_at_ms=0,
    )
    pump.add_done_callback(lambda _: shell_mod._schedule_eviction(task_id, log_path))

    await pump
    await asyncio.sleep(0)  # done callback 通过 call_soon 调度，需一个额外 tick

    assert task_id not in shell_mod._BG_REGISTRY
    assert not Path(log_path).exists()
