from __future__ import annotations

import asyncio
import errno
import json
import os
import pwd
import signal
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import agent.tools.unified_exec as unified_exec_module
from agent.tools.shell import ShellTaskStopTool
from agent.tools.shell import ShellTool
from agent.tools.shell import ShellWriteStdinTool
from agent.tools.unified_exec import HeadTailBuffer
from agent.tools.unified_exec import OUTPUT_MAX_BYTES
from agent.tools.unified_exec import ShellProcessManager
from agent.tools.unified_exec import UnknownExecutionError

# Ported from Codex c7a4a7e136d96554e1fc6f66532e6060fd2aaf15.


def _decode(value: str) -> dict[str, Any]:
    return json.loads(value)


@pytest.mark.skipif(os.name == "nt", reason="Windows 不使用 POSIX 进程组")
def test_zombie_only_permission_error_does_not_fail_completed_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_group_signal(_group_id: int, _signal: int) -> None:
        raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))

    monkeypatch.setattr(unified_exec_module.os, "killpg", deny_group_signal)
    monkeypatch.setattr(
        unified_exec_module,
        "process_group_exists",
        lambda _group_id: False,
        raising=False,
    )

    process = cast(asyncio.subprocess.Process, SimpleNamespace(pid=12345))
    unified_exec_module._kill_remaining_process_group(process)


@pytest.mark.skipif(os.name == "nt", reason="Windows 不使用 POSIX 进程组")
def test_live_privileged_group_permission_error_stays_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_group_signal(_group_id: int, _signal: int) -> None:
        raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))

    monkeypatch.setattr(unified_exec_module.os, "killpg", deny_group_signal)
    monkeypatch.setattr(
        unified_exec_module,
        "process_group_exists",
        lambda _group_id: True,
        raising=False,
    )

    with pytest.raises(PermissionError, match="Operation not permitted"):
        process = cast(asyncio.subprocess.Process, SimpleNamespace(pid=12345))
        unified_exec_module._kill_remaining_process_group(process)


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() != 0,
    reason="Gate 容器需要 root 创建真实跨 UID 进程组",
)
def test_real_cross_uid_live_process_group_returns_eperm() -> None:
    """在隔离容器中复现非特权 owner 无法清理 root 后代。"""

    nobody = pwd.getpwnam("nobody")
    leader_ready_r, leader_ready_w = os.pipe()
    leader_stop_r, leader_stop_w = os.pipe()
    residual_ready_r, residual_ready_w = os.pipe()
    result_r, result_w = os.pipe()
    leader_pid = -1
    residual_pid = -1
    controller_pid = -1
    try:
        # 1. 创建 nobody 进程组 leader，再让 root sibling 加入该组。
        leader_pid = os.fork()
        if leader_pid == 0:
            os.close(leader_ready_r)
            os.close(leader_stop_w)
            os.setpgid(0, 0)
            os.setgroups([])
            os.setgid(nobody.pw_gid)
            os.setuid(nobody.pw_uid)
            os.write(leader_ready_w, b"ready")
            os.read(leader_stop_r, 1)
            os._exit(0)
        os.close(leader_ready_w)
        os.close(leader_stop_r)
        assert os.read(leader_ready_r, 5) == b"ready"

        residual_pid = os.fork()
        if residual_pid == 0:
            os.close(residual_ready_r)
            os.setpgid(0, leader_pid)
            os.write(residual_ready_w, b"ready")
            while True:
                signal.pause()
        os.close(residual_ready_w)
        assert os.read(residual_ready_r, 5) == b"ready"

        # 2. leader 退出后，目标组只剩 root 活进程。
        os.write(leader_stop_w, b"x")
        os.waitpid(leader_pid, 0)
        leader_pid = -1

        # 3. nobody controller 调用真实 cleanup，必须得到 EPERM。
        controller_pid = os.fork()
        if controller_pid == 0:
            os.close(result_r)
            os.setgroups([])
            os.setgid(nobody.pw_gid)
            os.setuid(nobody.pw_uid)
            try:
                process = cast(
                    asyncio.subprocess.Process,
                    SimpleNamespace(pid=os.getpgid(residual_pid)),
                )
                unified_exec_module._kill_remaining_process_group(
                    process
                )
            except PermissionError as exc:
                os.write(result_w, str(exc.errno).encode())
            else:
                os.write(result_w, b"success")
            os._exit(0)
        os.close(result_w)
        assert os.read(result_r, 32) == str(errno.EPERM).encode()
        os.waitpid(controller_pid, 0)
        controller_pid = -1
    finally:
        for fd in (
            leader_ready_r,
            leader_stop_w,
            residual_ready_r,
            result_r,
        ):
            try:
                os.close(fd)
            except OSError:
                pass
        if controller_pid > 0:
            os.kill(controller_pid, signal.SIGKILL)
            os.waitpid(controller_pid, 0)
        if leader_pid > 0:
            os.kill(leader_pid, signal.SIGKILL)
            os.waitpid(leader_pid, 0)
        if residual_pid > 0:
            os.kill(residual_pid, signal.SIGKILL)
            os.waitpid(residual_pid, 0)


def test_head_tail_keeps_prefix_and_suffix_when_over_budget() -> None:
    buffer = HeadTailBuffer(10)
    buffer.push_chunk(b"0123456789")
    buffer.push_chunk(b"ab")

    assert buffer.to_bytes() == b"01234789ab"
    assert buffer.omitted_bytes == 2
    assert buffer.to_bytes_with_omission_marker() == (
        b"01234\n... 2 bytes omitted ...\n789ab"
    )


def test_head_tail_max_bytes_zero_drops_everything() -> None:
    buffer = HeadTailBuffer(0)
    buffer.push_chunk(b"abc")

    assert buffer.retained_bytes == 0
    assert buffer.omitted_bytes == 3
    assert buffer.to_bytes() == b""


def test_head_tail_one_byte_keeps_only_last_byte() -> None:
    buffer = HeadTailBuffer(1)
    buffer.push_chunk(b"abc")

    assert buffer.retained_bytes == 1
    assert buffer.omitted_bytes == 2
    assert buffer.to_bytes() == b"c"


def test_head_tail_drain_and_push_preserve_omissions() -> None:
    buffer = HeadTailBuffer(10)
    buffer.push_chunk(b"0123456789")
    buffer.push_chunk(b"ab")

    drained = buffer.drain()
    collected = HeadTailBuffer(10)
    collected.push_buffer(drained)

    assert buffer.total_bytes == 0
    assert collected.to_bytes() == b"01234789ab"
    assert collected.omitted_bytes == 2
    assert collected.total_bytes == 12


def test_head_tail_large_chunk_replaces_tail_with_its_end() -> None:
    buffer = HeadTailBuffer(10)
    buffer.push_chunk(b"0123456789")
    buffer.push_chunk(b"ABCDEFGHIJK")

    assert buffer.to_bytes().startswith(b"01234")
    assert buffer.to_bytes().endswith(b"GHIJK")
    assert buffer.omitted_bytes > 0


def test_head_tail_fills_head_then_tail_across_chunks() -> None:
    buffer = HeadTailBuffer(10)
    buffer.push_chunk(b"01")
    buffer.push_chunk(b"234")
    assert buffer.to_bytes() == b"01234"

    buffer.push_chunk(b"567")
    buffer.push_chunk(b"89")
    assert buffer.to_bytes() == b"0123456789"
    assert buffer.omitted_bytes == 0

    buffer.push_chunk(b"a")
    assert buffer.to_bytes() == b"012346789a"
    assert buffer.omitted_bytes == 1


def test_head_tail_empty_and_tiny_chunks_stay_bounded() -> None:
    buffer = HeadTailBuffer(10)
    for byte in b"0123456789ab":
        buffer.push_chunk(b"")
        buffer.push_chunk(bytes([byte]))

    assert buffer.to_bytes() == b"01234789ab"
    assert buffer.retained_bytes == 10
    assert buffer.omitted_bytes == 2


@pytest.mark.asyncio
async def test_output_collection_stays_bounded_across_repeated_drains() -> None:
    manager = ShellProcessManager()
    execution = SimpleNamespace(
        output_buffer=HeadTailBuffer(),
        output_lock=asyncio.Lock(),
        output_event=asyncio.Event(),
        exit_event=asyncio.Event(),
        output_closed=asyncio.Event(),
    )

    async def produce() -> None:
        for byte in (b"a", b"b", b"c"):
            async with execution.output_lock:
                execution.output_buffer.push_chunk(byte * OUTPUT_MAX_BYTES)
                execution.output_event.set()
            for _ in range(1_000):
                async with execution.output_lock:
                    if execution.output_buffer.retained_bytes == 0:
                        break
                await asyncio.sleep(0)
            else:
                raise AssertionError("collector did not drain output")
        execution.exit_event.set()
        execution.output_closed.set()
        execution.output_event.set()

    collected, _ = await asyncio.gather(
        manager._collect_until_deadline(
            cast(Any, execution),
            time.monotonic() + 5,
        ),
        produce(),
    )

    assert collected.retained_bytes == OUTPUT_MAX_BYTES
    assert collected.total_bytes == OUTPUT_MAX_BYTES * 3
    assert collected.to_bytes().startswith(b"a")
    assert collected.to_bytes().endswith(b"c")


@pytest.mark.asyncio
async def test_output_collection_preserves_prior_omissions() -> None:
    output = HeadTailBuffer()
    output.push_chunk(b"a" * OUTPUT_MAX_BYTES)
    output.push_chunk(b"overflow")
    execution = SimpleNamespace(
        output_buffer=output,
        output_lock=asyncio.Lock(),
        output_event=asyncio.Event(),
        exit_event=asyncio.Event(),
        output_closed=asyncio.Event(),
    )
    execution.exit_event.set()
    execution.output_closed.set()

    collected = await ShellProcessManager()._collect_until_deadline(
        cast(Any, execution),
        time.monotonic() + 1,
    )

    assert collected.total_bytes == OUTPUT_MAX_BYTES + len(b"overflow")
    assert collected.omitted_bytes == len(b"overflow")
    assert collected.to_bytes().endswith(b"overflow")


@pytest.mark.asyncio
async def test_multiple_executions_keep_output_isolated() -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    writer = ShellWriteStdinTool(manager)
    try:
        first, second = await asyncio.gather(
            shell.execute(
                command="printf a1; sleep 0.4; printf a2",
                description="启动并发任务甲",
                yield_time_ms=250,
            ),
            shell.execute(
                command="printf b1; sleep 0.4; printf b2",
                description="启动并发任务乙",
                yield_time_ms=250,
            ),
        )
        a = _decode(first)
        b = _decode(second)
        assert a["output"] == "a1"
        assert b["output"] == "b1"

        a_done, b_done = await asyncio.gather(
            writer.execute(execution_id=a["execution_id"], yield_time_ms=5_000),
            writer.execute(execution_id=b["execution_id"], yield_time_ms=5_000),
        )
        assert _decode(a_done)["output"] == "a2"
        assert _decode(b_done)["output"] == "b2"
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_write_stdin_cancellation_preserves_execution() -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    writer = ShellWriteStdinTool(manager)
    try:
        opened = _decode(
            await shell.execute(
                command="sleep 0.5; printf survived",
                description="启动可取消等待",
                yield_time_ms=250,
            )
        )
        execution_id = int(opened["execution_id"])
        wait = asyncio.create_task(
            writer.execute(execution_id=execution_id, yield_time_ms=5_000)
        )
        await asyncio.sleep(0.05)
        wait.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wait

        resumed = _decode(
            await writer.execute(execution_id=execution_id, yield_time_ms=5_000)
        )
        assert resumed["output"] == "survived"
        assert resumed["process_status"] == "succeeded"
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_failed_owner_cleanup_retains_execution_and_quarantines_new_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ShellProcessManager()
    owner = "mobile:cleanup-quarantine"
    original_remove = manager._remove_execution
    try:
        opened = await manager.exec_command(
            command="sleep 30",
            argv=["/bin/sh", "-c", "sleep 30"],
            cwd=None,
            env=dict(os.environ),
            tty=False,
            yield_time_ms=250,
            max_output_tokens=100,
            hard_timeout_s=60,
            owner_session_key=owner,
        )
        assert opened.execution_id is not None

        async def deny_remove(*_args: object, **_kwargs: object) -> None:
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))

        monkeypatch.setattr(manager, "_remove_execution", deny_remove)
        report = await manager.terminate_owner(owner)

        assert report.failed_execution_ids == (opened.execution_id,)
        assert await manager.active_execution_ids() == [opened.execution_id]
        with pytest.raises(RuntimeError, match="cleanup 未确认"):
            await manager.exec_command(
                command="printf blocked",
                argv=["/bin/sh", "-c", "printf blocked"],
                cwd=None,
                env=dict(os.environ),
                tty=False,
                yield_time_ms=250,
                max_output_tokens=100,
                hard_timeout_s=60,
                owner_session_key=owner,
            )

        other = await manager.exec_command(
            command="sleep 30",
            argv=["/bin/sh", "-c", "sleep 30"],
            cwd=None,
            env=dict(os.environ),
            tty=False,
            yield_time_ms=250,
            max_output_tokens=100,
            hard_timeout_s=60,
            owner_session_key="mobile:other-owner",
        )
        assert other.execution_id is not None

        monkeypatch.setattr(manager, "_remove_execution", original_remove)
        retry = await manager.terminate_owner(owner)
        assert retry.failed_execution_ids == ()
        other_cleanup = await manager.terminate_owner("mobile:other-owner")
        assert other_cleanup.failed_execution_ids == ()
        assert await manager.active_execution_ids() == []
    finally:
        monkeypatch.setattr(manager, "_remove_execution", original_remove)
        await manager.shutdown()


@pytest.mark.asyncio
async def test_failed_owner_cleanup_retains_complete_diagnostic_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ShellProcessManager()
    owner = "mobile:cleanup-log"
    original_remove = manager._remove_execution
    output_path: Path | None = None
    try:
        opened = await manager.exec_command(
            command="printf retained-log; sleep 30",
            argv=["/bin/sh", "-c", "printf retained-log; sleep 30"],
            cwd=None,
            env=dict(os.environ),
            tty=False,
            yield_time_ms=250,
            max_output_tokens=10,
            hard_timeout_s=60,
            owner_session_key=owner,
        )
        assert opened.execution_id is not None
        assert opened.output_path is not None
        output_path = Path(opened.output_path)

        async def deny_remove(*_args: object, **_kwargs: object) -> None:
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))

        monkeypatch.setattr(manager, "_remove_execution", deny_remove)
        report = await manager.terminate_owner(owner)

        assert report.failed_execution_ids == (opened.execution_id,)
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == "retained-log"

        monkeypatch.setattr(manager, "_remove_execution", original_remove)
        retry = await manager.terminate_owner(owner)
        assert retry.failed_execution_ids == ()
        assert not output_path.exists()
    finally:
        monkeypatch.setattr(manager, "_remove_execution", original_remove)
        await manager.shutdown()


@pytest.mark.asyncio
async def test_stop_during_initial_wait_returns_terminal_state() -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    initial = asyncio.create_task(
        shell.execute(
            command="sleep 30",
            description="验证初始等待竞态",
            yield_time_ms=30_000,
        )
    )
    try:
        execution_id = await _wait_for_execution(manager)
        stopped = _decode(
            await ShellTaskStopTool(manager).execute(execution_id=execution_id)
        )
        result = _decode(await asyncio.wait_for(initial, timeout=2))

        assert stopped["process_status"] == "stopped"
        assert result["process_status"] == "failed"
        assert result["finish_reason"] == "stopped"
        assert "execution_id" not in result
        assert await manager.active_execution_ids() == []
    finally:
        if not initial.done():
            initial.cancel()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_stop_during_write_poll_returns_terminal_state() -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    writer = ShellWriteStdinTool(manager)
    try:
        opened = _decode(
            await shell.execute(
                command="sleep 30",
                description="验证后续等待竞态",
                yield_time_ms=250,
            )
        )
        execution_id = int(opened["execution_id"])
        poll = asyncio.create_task(
            writer.execute(execution_id=execution_id, yield_time_ms=300_000)
        )
        await asyncio.sleep(0.05)
        await ShellTaskStopTool(manager).execute(execution_id=execution_id)
        result = _decode(await asyncio.wait_for(poll, timeout=2))

        assert result["process_status"] == "failed"
        assert result["finish_reason"] == "stopped"
        assert "execution_id" not in result
        assert await manager.active_execution_ids() == []
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="stdlib 版本暂不实现 ConPTY")
async def test_pty_preserves_state_across_multiple_writes() -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    writer = ShellWriteStdinTool(manager)
    try:
        opened = _decode(
            await shell.execute(
                command="sh",
                description="启动交互 shell",
                tty=True,
                yield_time_ms=250,
            )
        )
        execution_id = int(opened["execution_id"])
        await writer.execute(
            execution_id=execution_id,
            chars="value=codex\n",
            yield_time_ms=250,
        )
        echoed = _decode(
            await writer.execute(
                execution_id=execution_id,
                chars="printf '<%s>\\n' \"$value\"\n",
                yield_time_ms=1_000,
            )
        )
        exited = _decode(
            await writer.execute(
                execution_id=execution_id,
                chars="exit\n",
                yield_time_ms=1_000,
            )
        )

        assert "<codex>" in echoed["output"]
        assert exited["process_status"] == "succeeded"
        with pytest.raises(UnknownExecutionError):
            await writer.execute(execution_id=execution_id, yield_time_ms=5_000)
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_completed_command_preserves_nonzero_exit_code() -> None:
    manager = ShellProcessManager()
    try:
        result = _decode(
            await ShellTool(manager).execute(
                command="exit 17",
                description="验证退出状态",
                yield_time_ms=1_000,
            )
        )
        assert result["process_status"] == "failed"
        assert result["exit_code"] == 17
        assert "execution_id" not in result
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_exit_near_initial_deadline_never_loses_final_output() -> None:
    manager = ShellProcessManager()
    shell = ShellTool(manager)
    writer = ShellWriteStdinTool(manager)
    try:
        for _ in range(8):
            initial = _decode(
                await shell.execute(
                    command="sleep 0.24; printf final",
                    description="验证截止点退出",
                    shell="/bin/sh",
                    login=False,
                    yield_time_ms=250,
                )
            )
            output = str(initial["output"])
            terminal = initial
            if "execution_id" in initial:
                terminal = _decode(
                    await writer.execute(
                        execution_id=initial["execution_id"],
                        yield_time_ms=5_000,
                    )
                )
                output += str(terminal["output"])
            assert output == "final"
            assert terminal["process_status"] == "succeeded"
            assert "execution_id" not in terminal
    finally:
        await manager.shutdown()


def test_pruning_prefers_exited_execution_outside_recent_set() -> None:
    executions = _prune_entries(exited_ids={2})
    candidate = ShellProcessManager._select_prune_candidate(cast(Any, executions))
    assert candidate is not None
    assert candidate.execution_id == 2


def test_pruning_falls_back_to_lru_when_none_exited() -> None:
    executions = _prune_entries(exited_ids=set())
    candidate = ShellProcessManager._select_prune_candidate(cast(Any, executions))
    assert candidate is not None
    assert candidate.execution_id == 1


def test_pruning_protects_recent_execution_even_when_exited() -> None:
    executions = _prune_entries(exited_ids={3, 10})
    candidate = ShellProcessManager._select_prune_candidate(cast(Any, executions))
    assert candidate is not None
    assert candidate.execution_id == 1


async def _wait_for_execution(manager: ShellProcessManager) -> int:
    for _ in range(200):
        active = await manager.active_execution_ids()
        if active:
            return active[0]
        await asyncio.sleep(0.01)
    raise AssertionError("shell execution was not registered")


def _prune_entries(*, exited_ids: set[int]) -> list[SimpleNamespace]:
    now = time.monotonic()
    return [
        SimpleNamespace(
            execution_id=index,
            last_used=now - (50 - index),
            process=SimpleNamespace(returncode=0 if index in exited_ids else None),
        )
        for index in range(1, 11)
    ]
