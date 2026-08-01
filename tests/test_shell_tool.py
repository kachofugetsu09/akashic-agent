from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from agent.control.context import current_turn_id
from agent.looping.core import AgentLoop
from agent.provider import LLMResponse
from agent.subagent import SubAgent
from agent.tools.shell import ShellTaskStopTool
from agent.tools.shell import ShellTool
from agent.tools.shell import ShellWriteStdinTool
from agent.tools.shell import _shell_env
from agent.tools.shell import _validate_command
from agent.tools.shell import _validate_network_command
from agent.tools.shell_command import ResolvedShell
from agent.tools.shell_command import ShellKind
from agent.tools.shell_command import detect_shell_kind
from agent.tools.shell_command import resolve_shell
from agent.tools.unified_exec import HeadTailBuffer
from agent.tools.unified_exec import ShellProcessManager, UnknownExecutionError
from agent.tools.unified_exec import clamp_initial_yield_time
from agent.tools.unified_exec import clamp_write_stdin_yield_time
from agent.tools.registry import ToolRegistry
from bus.events import InboundMessage, OutboundMessage
from core.error_context import current_session_key
from session.manager import SessionManager


def _decode(value: str) -> dict[str, Any]:
    return json.loads(value)


def _linux_process_state(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except FileNotFoundError:
        return None


def test_shell_tool_schema_guides_observable_long_running_commands() -> None:
    manager = ShellProcessManager()
    shell_schema = ShellTool(manager).to_schema()["function"]
    writer_schema = ShellWriteStdinTool(manager).to_schema()["function"]

    command_description = shell_schema["parameters"]["properties"]["command"][
        "description"
    ]
    tty_description = shell_schema["parameters"]["properties"]["tty"]["description"]
    assert "tail -n" in command_description
    assert "等待 EOF" in command_description
    assert "启动前确认输入可获得" in tty_description
    assert "不要原样重复调用" in writer_schema["description"]
    assert "检查进程、日志或产物" in writer_schema["description"]


@pytest.mark.asyncio
async def test_short_command_returns_exit_without_execution_id() -> None:
    manager = ShellProcessManager()
    try:
        result = _decode(
            await ShellTool(manager).execute(
                command="printf short",
                description="运行短命令",
                yield_time_ms=250,
            )
        )
        assert result["process_status"] == "succeeded"
        assert result["exit_code"] == 0
        assert result["output"] == "short"
        assert "execution_id" not in result
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_long_command_returns_execution_id_and_incremental_output() -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    writer = ShellWriteStdinTool(manager)
    try:
        first = _decode(
            await shell.execute(
                command="printf one; sleep 0.4; printf two; sleep 5.1; printf three",
                description="运行分段命令",
                yield_time_ms=250,
            )
        )
        execution_id = int(first["execution_id"])
        assert first["output"] == "one"

        second = _decode(
            await writer.execute(
                execution_id=execution_id,
                chars="",
                yield_time_ms=1,
            )
        )
        assert second["process_status"] == "running"
        assert second["output"] == "two"
        assert second["execution_id"] == execution_id

        third = _decode(
            await writer.execute(
                execution_id=execution_id,
                chars="",
                yield_time_ms=5_000,
            )
        )
        assert third["process_status"] == "succeeded"
        assert third["output"] == "three"
        assert "execution_id" not in third
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_initial_wait_cancellation_keeps_registered_execution() -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    writer = ShellWriteStdinTool(manager)
    task = asyncio.create_task(
        shell.execute(
            command="sleep 0.5; printf survived",
            description="验证取消续接",
            yield_time_ms=30_000,
        )
    )
    try:
        for _ in range(100):
            active = await manager.active_execution_ids()
            if active:
                break
            await asyncio.sleep(0.01)
        assert active
        execution_id = active[0]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        resumed = _decode(
            await writer.execute(
                execution_id=execution_id,
                yield_time_ms=5_000,
            )
        )
        assert resumed["output"] == "survived"
        assert resumed["process_status"] == "succeeded"
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="stdlib 版本暂不实现 ConPTY")
async def test_tty_accepts_input() -> None:
    manager = ShellProcessManager()
    try:
        opened = _decode(
            await ShellTool(manager).execute(
                command="read value; printf 'got:%s' \"$value\"",
                description="验证终端输入",
                tty=True,
                yield_time_ms=250,
            )
        )
        result = _decode(
            await ShellWriteStdinTool(manager).execute(
                execution_id=opened["execution_id"],
                chars="hello\n",
                yield_time_ms=2_000,
            )
        )
        assert "got:hello" in str(result["output"])
        assert result["process_status"] == "succeeded"
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_non_tty_rejects_input_but_ctrl_c_interrupts() -> None:
    manager = ShellProcessManager()
    writer = ShellWriteStdinTool(manager)
    try:
        opened = _decode(
            await ShellTool(manager).execute(
                command="sleep 30",
                description="验证非终端输入",
                yield_time_ms=250,
            )
        )
        execution_id = int(opened["execution_id"])
        with pytest.raises(RuntimeError, match="stdin 已关闭"):
            await writer.execute(execution_id=execution_id, chars="hello")

        interrupted = _decode(
            await writer.execute(
                execution_id=execution_id,
                chars="\x03",
                yield_time_ms=2_000,
            )
        )
        assert interrupted["process_status"] == "failed"
        assert "execution_id" not in interrupted
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_hard_timeout_terminates_process_group() -> None:
    manager = ShellProcessManager()
    try:
        opened = _decode(
            await ShellTool(manager).execute(
                command="sleep 30",
                description="验证硬超时",
                yield_time_ms=250,
                timeout=1,
            )
        )
        result = _decode(
            await ShellWriteStdinTool(manager).execute(
                execution_id=opened["execution_id"],
                yield_time_ms=5_000,
            )
        )
        assert result["process_status"] == "timed_out"
        assert result["finish_reason"] == "timeout"
        assert await manager.active_execution_ids() == []
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_task_stop_confirms_termination_and_removes_execution() -> None:
    manager = ShellProcessManager()
    try:
        opened = _decode(
            await ShellTool(manager).execute(
                command="sleep 30 & wait",
                description="验证显式停止",
                yield_time_ms=250,
            )
        )
        execution_id = int(opened["execution_id"])
        result = _decode(
            await ShellTaskStopTool(manager).execute(execution_id=execution_id)
        )
        assert result == {
            "execution_id": execution_id,
            "process_status": "stopped",
            "status": "stopped",
        }
        assert await manager.active_execution_ids() == []
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="Windows 需要 Job Object 验证")
async def test_natural_shell_exit_kills_remaining_process_group(tmp_path: Path) -> None:
    manager = ShellProcessManager()
    pid_path = tmp_path / "child.pid"
    try:
        result = _decode(
            await ShellTool(manager).execute(
                command=f"sleep 30 & echo $! > {pid_path}",
                description="验证残留子进程",
                yield_time_ms=1_000,
            )
        )
        assert result["process_status"] == "succeeded"
        child_pid = int(pid_path.read_text(encoding="utf-8").strip())
        for _ in range(100):
            child_state = _linux_process_state(child_pid)
            # 极简容器的 PID 1 可能不回收孤儿；Z 仍证明进程已被终止。
            if child_state is None or child_state == "Z":
                break
            await asyncio.sleep(0.01)
        assert child_state is None or child_state == "Z"
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_task_stop_failure_keeps_execution_registered(monkeypatch) -> None:
    manager = ShellProcessManager()
    opened = _decode(
        await ShellTool(manager).execute(
            command="sleep 30",
            description="验证停止失败",
            yield_time_ms=250,
        )
    )
    execution_id = int(opened["execution_id"])
    original = manager._terminate_confirmed

    async def fail_termination(_execution) -> None:
        raise RuntimeError("cannot kill")

    monkeypatch.setattr(manager, "_terminate_confirmed", fail_termination)
    with pytest.raises(RuntimeError, match="cannot kill"):
        await ShellTaskStopTool(manager).execute(execution_id=execution_id)
    assert await manager.active_execution_ids() == [execution_id]
    monkeypatch.setattr(manager, "_terminate_confirmed", original)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_owner_cannot_read_another_conversation_execution() -> None:
    manager = ShellProcessManager()
    try:
        opened = await manager.exec_command(
            command="sleep 30",
            argv=["/bin/sh", "-c", "sleep 30"],
            cwd=None,
            env=_shell_env(),
            tty=False,
            yield_time_ms=250,
            max_output_tokens=10_000,
            hard_timeout_s=30,
            owner_session_key="owner-a",
        )
        assert opened.execution_id is not None
        with pytest.raises(UnknownExecutionError, match="未知 execution_id"):
            await manager.write_stdin(
                execution_id=opened.execution_id,
                chars="",
                yield_time_ms=5_000,
                max_output_tokens=10_000,
                owner_session_key="owner-b",
            )
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_same_conversation_can_resume_execution_after_context_reset() -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    writer = ShellWriteStdinTool(manager)
    first_token = current_session_key.set("cli:owner-a")
    try:
        opened = _decode(
            await shell.execute(
                command="sleep 0.5; printf resumed",
                description="启动跨调用任务",
                yield_time_ms=250,
            )
        )
    finally:
        current_session_key.reset(first_token)
    execution_id = int(opened["execution_id"])
    try:
        other_token = current_session_key.set("cli:owner-b")
        try:
            with pytest.raises(UnknownExecutionError):
                await writer.execute(execution_id=execution_id, yield_time_ms=5_000)
        finally:
            current_session_key.reset(other_token)

        resumed_token = current_session_key.set("cli:owner-a")
        try:
            resumed = _decode(
                await writer.execute(
                    execution_id=execution_id,
                    yield_time_ms=5_000,
                )
            )
        finally:
            current_session_key.reset(resumed_token)
        assert resumed["output"] == "resumed"
        assert resumed["process_status"] == "succeeded"
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_agent_loop_turn_end_terminates_owner_shell() -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    tools = ToolRegistry()
    tools.register(shell)
    loop = AgentLoop.__new__(AgentLoop)
    loop.tools = tools
    loop._processing_state = None
    loop._interrupt_states = {}
    loop._resume_interrupted_message = AsyncMock(
        side_effect=lambda message, _key: (message, False)
    )
    loop._observe_turn_started = AsyncMock()

    async def process(
        _message: InboundMessage,
        _key: str,
        *,
        dispatch_outbound: bool,
    ) -> OutboundMessage:
        assert dispatch_outbound is False
        opened = _decode(
            await shell.execute(
                command="sleep 30",
                description="验证 turn 回收",
                yield_time_ms=250,
            )
        )
        assert opened["process_status"] == "running"
        return OutboundMessage(channel="cli", chat_id="owner", content="done")

    loop._core_runner = SimpleNamespace(process=process)
    message = InboundMessage(
        channel="cli",
        sender="user",
        chat_id="owner",
        content="run",
    )
    try:
        result = await loop._process(message, dispatch_outbound=False)

        assert result.content == "done"
        assert await manager.active_execution_ids() == []
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_agent_loop_preserves_turn_failure_when_shell_cleanup_fails(
    monkeypatch,
) -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    tools = ToolRegistry()
    tools.register(shell)
    loop = AgentLoop.__new__(AgentLoop)
    loop.tools = tools
    loop._processing_state = None
    loop._interrupt_states = {}
    loop._resume_interrupted_message = AsyncMock(
        side_effect=lambda message, _key: (message, False)
    )
    loop._observe_turn_started = AsyncMock()

    async def fail_process(*_args, **_kwargs) -> OutboundMessage:
        raise RuntimeError("turn failed")

    async def fail_cleanup(_owner_session_key: str) -> None:
        raise RuntimeError("cleanup failed")

    loop._core_runner = SimpleNamespace(process=fail_process)
    monkeypatch.setattr(shell, "terminate_owner", fail_cleanup)
    message = InboundMessage(
        channel="cli",
        sender="user",
        chat_id="owner",
        content="run",
    )

    with pytest.raises(RuntimeError, match="turn failed"):
        await loop._process(message, dispatch_outbound=False)


@pytest.mark.asyncio
async def test_agent_loop_returns_completed_reply_when_shell_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    tools = ToolRegistry()
    tools.register(shell)
    loop = AgentLoop.__new__(AgentLoop)
    loop.tools = tools
    loop._processing_state = None
    loop._interrupt_states = {}
    loop._resume_interrupted_message = AsyncMock(
        side_effect=lambda message, _key: (message, False)
    )
    loop._observe_turn_started = AsyncMock()
    loop._core_runner = SimpleNamespace(
        process=AsyncMock(
            return_value=OutboundMessage("mobile", "owner", "completed reply")
        )
    )

    async def fail_cleanup(_owner_session_key: str) -> None:
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(shell, "terminate_owner", fail_cleanup)
    message = InboundMessage(
        channel="mobile",
        sender="user",
        chat_id="owner",
        content="run",
    )

    with caplog.at_level("ERROR", logger="agent.loop"):
        result = await loop._process(message)

    assert result.content == "completed reply"
    assert "event=cleanup_degraded" in caplog.text
    assert "session=mobile:owner" in caplog.text


@pytest.mark.asyncio
async def test_shutdown_terminates_all_executions() -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    first = _decode(
        await shell.execute(
            command="sleep 30",
            description="启动任务一",
            yield_time_ms=250,
        )
    )
    second = _decode(
        await shell.execute(
            command="sleep 30",
            description="启动任务二",
            yield_time_ms=250,
        )
    )
    assert sorted(await manager.active_execution_ids()) == sorted(
        [int(first["execution_id"]), int(second["execution_id"])]
    )
    await manager.shutdown()
    assert await manager.active_execution_ids() == []


@pytest.mark.asyncio
async def test_subagent_owner_end_shuts_down_shell_execution() -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    await shell.execute(
        command="sleep 30",
        description="启动子任务命令",
        yield_time_ms=250,
    )
    assert await manager.active_execution_ids()

    class _Provider:
        async def chat(self, **_kwargs: Any) -> LLMResponse:
            return LLMResponse(content="done", tool_calls=[])

    subagent = SubAgent(
        provider=cast(Any, _Provider()),
        model="test",
        tools=[
            shell,
            ShellWriteStdinTool(manager),
            ShellTaskStopTool(manager),
        ],
    )
    assert await subagent.run("finish") == "done"
    assert await manager.active_execution_ids() == []


@pytest.mark.asyncio
async def test_capacity_prunes_oldest_unprotected_execution() -> None:
    manager = ShellProcessManager(max_executions=3)
    shell = ShellTool(manager)
    ids: list[int] = []
    try:
        for index in range(4):
            result = _decode(
                await shell.execute(
                    command="sleep 30",
                    description=f"启动容量任务{index}",
                    yield_time_ms=250,
                )
            )
            ids.append(int(result["execution_id"]))
        active = await manager.active_execution_ids()
        assert ids[0] not in active
        assert sorted(active) == sorted(ids[1:])
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_output_budget_keeps_head_tail_and_full_log(tmp_path: Path) -> None:
    del tmp_path
    manager = ShellProcessManager()
    output_path: Path | None = None
    try:
        result = _decode(
            await ShellTool(manager).execute(
                command='python3 -c \'print("a" * 100, end="")\'',
                description="验证输出预算",
                yield_time_ms=1_000,
                max_output_tokens=5,
            )
        )
        assert result["original_token_count"] == 25
        assert int(result["output_omitted_bytes"]) == 80
        assert "80 bytes omitted" in str(result["output"])
        output_path = Path(str(result["output_path"]))
        assert output_path.read_text(encoding="utf-8") == "a" * 100
    finally:
        await manager.shutdown()
        if output_path is not None:
            output_path.unlink(missing_ok=True)


def test_head_tail_buffer_matches_codex_drain_semantics() -> None:
    buffer = HeadTailBuffer(6)
    buffer.push_chunk(b"abc")
    buffer.push_chunk(b"defghi")
    assert buffer.to_bytes() == b"abcghi"
    assert buffer.omitted_bytes == 3
    assert buffer.total_bytes == 9
    assert buffer.to_bytes_with_omission_marker() == (
        b"abc\n... 3 bytes omitted ...\nghi"
    )

    drained = buffer.drain()
    assert drained.to_bytes() == b"abcghi"
    assert buffer.total_bytes == 0
    buffer.push_chunk(b"new")
    assert buffer.to_bytes() == b"new"


def test_yield_time_clamps_match_codex() -> None:
    assert clamp_initial_yield_time(1) == 250
    assert clamp_initial_yield_time(120_000) == 30_000
    assert clamp_write_stdin_yield_time(1, has_input=False) == 5_000
    assert clamp_write_stdin_yield_time(999_999, has_input=False) == 300_000
    assert clamp_write_stdin_yield_time(1, has_input=True) == 250
    assert clamp_write_stdin_yield_time(999_999, has_input=True) == 30_000


def test_shell_schema_removes_old_background_protocol() -> None:
    shell = ShellTool()
    properties = shell.parameters["properties"]
    assert "run_in_background" not in properties
    assert "auto_promote" not in properties
    assert "yield_time_ms" in properties
    assert "tty" in properties
    assert "shell" in properties
    assert "login" in properties
    assert shell.parameters["additionalProperties"] is False
    assert ShellWriteStdinTool(shell.manager).parameters["required"] == ["execution_id"]


@pytest.mark.asyncio
async def test_removed_background_protocol_fails_loud() -> None:
    shell = ShellTool()
    try:
        old_shell = _decode(
            await shell.execute(
                command="sleep 30",
                description="验证旧参数失败",
                run_in_background=True,
            )
        )
        old_stop = _decode(await ShellTaskStopTool(shell.manager).execute(task_id="x"))

        assert "已移除参数" in old_shell["error"]
        assert "execution_id" in old_stop["error"]
        assert await shell.manager.active_execution_ids() == []
    finally:
        await shell.shutdown()


def test_shell_command_argv_matches_codex() -> None:
    bash = ResolvedShell(ShellKind.BASH, Path("/bin/bash"))
    powershell = ResolvedShell(ShellKind.POWERSHELL, Path("pwsh.exe"))

    assert bash.derive_argv("echo hello", login=False) == [
        "/bin/bash",
        "-c",
        "echo hello",
    ]
    assert bash.derive_argv("echo hello", login=True) == [
        "/bin/bash",
        "-lc",
        "echo hello",
    ]
    assert powershell.derive_argv("echo hello", login=False) == [
        "pwsh.exe",
        "-NoProfile",
        "-Command",
        "echo hello",
    ]
    assert powershell.derive_argv("echo hello", login=True) == [
        "pwsh.exe",
        "-Command",
        "echo hello",
    ]


def test_shell_detection_matches_codex_names() -> None:
    assert detect_shell_kind("/bin/zsh") is ShellKind.ZSH
    assert detect_shell_kind("/usr/bin/bash") is ShellKind.BASH
    assert detect_shell_kind("powershell.exe") is ShellKind.POWERSHELL
    assert detect_shell_kind(r"C:\bin\pwsh.exe") is ShellKind.POWERSHELL
    assert detect_shell_kind("/bin/sh") is ShellKind.SH
    assert detect_shell_kind("fish") is None


def test_explicit_shell_rejects_unknown_or_missing_binary() -> None:
    with pytest.raises(ValueError, match="不支持"):
        resolve_shell("fish")
    with pytest.raises(ValueError, match="不存在或不可执行"):
        resolve_shell("/definitely/missing/bash")


@pytest.mark.skipif(os.name == "nt", reason="Unix passwd shell behavior")
def test_default_shell_ignores_shell_environment_variable(monkeypatch) -> None:
    import pwd

    passwd_shell = resolve_shell("bash")
    monkeypatch.setattr(
        pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(pw_shell=str(passwd_shell.path)),
    )
    monkeypatch.setenv("SHELL", "/definitely/missing/fish")

    selected = resolve_shell()

    assert selected == passwd_shell


@pytest.mark.skipif(os.name == "nt", reason="Unix passwd shell behavior")
def test_default_shell_falls_back_when_uid_has_no_passwd_record(monkeypatch) -> None:
    import pwd

    def missing_passwd_record(_uid: int) -> None:
        raise KeyError("uid not found")

    monkeypatch.setattr(pwd, "getpwuid", missing_passwd_record)
    monkeypatch.setenv("SHELL", "/definitely/missing/fish")

    selected = resolve_shell()

    assert selected.kind in {ShellKind.BASH, ShellKind.ZSH, ShellKind.SH}
    assert selected.path.is_file()


@pytest.mark.asyncio
async def test_shell_uses_explicit_login_flag_without_injecting_pipefail() -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    try:
        ordinary = _decode(
            await shell.execute(
                command="false | true",
                description="验证管道默认语义",
                shell="/bin/bash",
                login=False,
            )
        )
        explicit = _decode(
            await shell.execute(
                command="set -o pipefail; false | true",
                description="验证显式管道失败",
                shell="/bin/bash",
                login=False,
            )
        )
        assert ordinary["exit_code"] == 0
        assert explicit["exit_code"] == 1
    finally:
        await manager.shutdown()


def test_validate_command_rejects_banned_command() -> None:
    assert "不被允许" in str(
        _validate_command(
            "nc example.com 80",
            allow_network=True,
            restricted_dir=None,
        )
    )


def test_network_guard_rejects_private_target_and_upload() -> None:
    assert "内网" in str(_validate_network_command("curl http://127.0.0.1/x"))
    assert "上传" in str(
        _validate_network_command("curl -F file=@a.txt https://example.com")
    )
    assert _validate_network_command("curl https://example.com") is None


def test_restricted_shell_rejects_parent_and_pipeline(tmp_path: Path) -> None:
    assert "父级" in str(
        _validate_command(
            "ls ../outside",
            allow_network=False,
            restricted_dir=tmp_path,
            cwd=tmp_path,
        )
    )
    assert "管道" in str(
        _validate_command(
            "ls . | sort",
            allow_network=False,
            restricted_dir=tmp_path,
            cwd=tmp_path,
        )
    )


def test_shell_env_sets_noninteractive_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    env = _shell_env()
    assert env["NO_COLOR"] == "1"
    assert env["TERM"] == "dumb"
    assert env["GIT_PAGER"] == "cat"


def test_shell_env_defers_plugin_uninstall_owned_by_current_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_DEFER_PLUGIN_UNINSTALL", "stale")
    assert "AKASHIC_DEFER_PLUGIN_UNINSTALL" not in _shell_env()

    token = current_turn_id.set("turn:context-pressure-uninstall")
    try:
        assert _shell_env()["AKASHIC_DEFER_PLUGIN_UNINSTALL"] == "1"
    finally:
        current_turn_id.reset(token)


def test_old_shell_trace_reloads_as_history_without_runtime_alias(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:old-shell-history")
    session.add_message("user", "执行旧任务")
    session.add_message(
        "assistant",
        "旧任务结束",
        tool_chain=[
            {
                "text": "",
                "calls": [
                    {
                        "call_id": "old-shell",
                        "name": "shell",
                        "arguments": {
                            "command": "sleep 1",
                            "run_in_background": True,
                        },
                        "result": '{"background_task_id":"legacy-1"}',
                    },
                    {
                        "call_id": "old-output",
                        "name": "task_output",
                        "arguments": {"task_id": "legacy-1"},
                        "result": "done",
                    },
                ],
            }
        ],
    )
    manager.save(session)
    manager.close()

    reloaded = SessionManager(tmp_path)
    try:
        history = reloaded.get_existing("cli:old-shell-history").get_history()
        calls = cast(list[dict[str, Any]], history[1]["tool_calls"])
        assert [call["function"]["name"] for call in calls] == [
            "shell",
            "task_output",
        ]
        assert (
            json.loads(calls[0]["function"]["arguments"])["run_in_background"] is True
        )
        assert history[2]["content"] == '{"background_task_id":"legacy-1"}'
        assert history[3]["content"] == "done"
    finally:
        reloaded.close()
