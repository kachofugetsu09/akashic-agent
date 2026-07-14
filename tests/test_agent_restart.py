from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.control.context import current_turn_id
from agent.control.errors import RuntimeClosedError
from agent.control.models import TurnRequest
from agent.control.protocol.router import ConnectionRouter
from agent.control.runtime import ConversationRuntime
from agent.restart import (
    RestartCoordinator,
    RestartRejectedError,
    RestartState,
    SupervisorCommitChannel,
)
import agent.supervisor as supervisor_module
from agent.supervisor import RESTART_EXIT_CODE, _wait_child, run_supervisor
from agent.tools.agent_restart import AgentRestartTool
from agent.tools.registry import ToolRegistry
from agent.tools.tool_search import ToolSearchTool
from bootstrap.runtime_readiness import RuntimeReadiness
from bootstrap.app import AppRuntime
from core.error_context import current_session_key
from infra.control.connection import NdjsonConnection
from session.store import SessionStore


class _Admission:
    def __init__(self) -> None:
        self.quiesced: list[str] = []
        self.resumed: list[str] = []

    def quiesce(self, turn_id: str) -> None:
        self.quiesced.append(turn_id)

    def resume(self, turn_id: str) -> None:
        self.resumed.append(turn_id)


def _coordinator(
    *,
    timeout: float = 1.0,
) -> tuple[RestartCoordinator, _Admission, list[str]]:
    admission = _Admission()
    commits: list[str] = []
    coordinator = RestartCoordinator(
        "boot-a",
        supervised=True,
        commit=lambda request: commits.append(request.id),
        delivery_timeout_s=timeout,
    )
    coordinator.bind_admission(
        quiesce=admission.quiesce,
        resume=admission.resume,
    )
    return coordinator, admission, commits


@pytest.mark.asyncio
async def test_restart_commits_only_after_terminal_and_delivery() -> None:
    coordinator, admission, commits = _coordinator()

    request = coordinator.arm(
        turn_id="turn-a",
        session_key="programmatic:one",
        channel="programmatic",
        chat_id="one",
        reason="reload core",
    )
    same_request = coordinator.arm(
        turn_id="turn-a",
        session_key="programmatic:one",
        channel="programmatic",
        chat_id="one",
        reason="ignored by idempotency",
    )
    with pytest.raises(RestartRejectedError):
        coordinator.arm(
            turn_id="turn-b",
            session_key="programmatic:two",
            channel="programmatic",
            chat_id="two",
            reason="competing request",
        )

    assert same_request is request
    assert admission.quiesced == ["turn-a"]
    coordinator.mark_delivered("turn-a")
    assert commits == []
    coordinator.mark_turn_terminal("turn-a", "completed")

    assert await coordinator.wait_committed() is request
    assert commits == [request.id]
    assert coordinator.state is RestartState.COMMITTED

    coordinator.mark_turn_terminal("turn-a", "completed")
    coordinator.mark_delivered("turn-a")
    assert commits == [request.id]


@pytest.mark.asyncio
async def test_restart_failure_and_timeout_restore_admission() -> None:
    coordinator, admission, _ = _coordinator(timeout=0.01)
    coordinator.arm(
        turn_id="turn-failed",
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
        reason="reload core",
    )
    coordinator.mark_turn_terminal("turn-failed", "failed")

    assert coordinator.pending is None
    assert admission.resumed == ["turn-failed"]

    coordinator.arm(
        turn_id="turn-timeout",
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
        reason="reload core",
    )
    coordinator.mark_turn_terminal("turn-timeout", "completed")
    await asyncio.sleep(0.03)

    assert coordinator.pending is None
    assert admission.resumed == ["turn-failed", "turn-timeout"]
    assert "timed out" in str(coordinator.last_error)


@pytest.mark.asyncio
async def test_conversation_runtime_drains_before_restart_commit(
    tmp_path: Path,
) -> None:
    commits: list[str] = []
    coordinator = RestartCoordinator(
        "boot-a",
        supervised=True,
        commit=lambda request: commits.append(request.id),
    )
    store = SessionStore(tmp_path / "sessions.db")
    armed = asyncio.Event()
    release = asyncio.Event()
    turn_holder: dict[str, str] = {}

    async def execute(_request: TurnRequest) -> str:
        coordinator.arm(
            turn_id=turn_holder["id"],
            session_key="programmatic:one",
            channel="programmatic",
            chat_id="one",
            reason="reload core",
        )
        armed.set()
        await release.wait()
        return "restart scheduled"

    runtime = ConversationRuntime(
        store,
        execute,
        restart_coordinator=coordinator,
    )
    coordinator.bind_admission(
        quiesce=runtime.quiesce_for_restart,
        resume=runtime.resume_after_restart_cancel,
    )
    handle = await runtime.start_turn(TurnRequest("programmatic:one", "restart"))
    turn_holder["id"] = handle.id
    await armed.wait()

    with pytest.raises(RuntimeClosedError):
        await runtime.start_turn(TurnRequest("programmatic:two", "late"))
    release.set()
    result = await handle.result()
    assert result.status.value == "completed"
    assert commits == []

    coordinator.mark_delivered(handle.id)
    assert (await coordinator.wait_committed()).turn_id == handle.id
    assert len(commits) == 1
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_router_disconnect_restores_admission_immediately() -> None:
    coordinator, admission, _ = _coordinator(timeout=60)
    coordinator.arm(
        turn_id="turn-disconnect",
        session_key="programmatic:one",
        channel="programmatic",
        chat_id="one",
        reason="reload core",
    )
    coordinator.mark_turn_terminal("turn-disconnect", "completed")
    entered = asyncio.Event()

    class _Handle:
        id = "turn-disconnect"

        async def events(self):
            entered.set()
            await asyncio.Event().wait()
            yield None

    class _Service:
        def notify_turn_delivery_failed(self, turn_id: str, reason: str) -> None:
            coordinator.mark_delivery_failed(turn_id, reason)

    async def send(_message: dict[str, object]) -> None:
        raise AssertionError("terminal frame must not be sent")

    router = ConnectionRouter(cast(Any, _Service()), send)
    task = asyncio.create_task(router._forward_events(_Handle()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert coordinator.pending is None
    assert admission.resumed == ["turn-disconnect"]
    assert "connection closed" in str(coordinator.last_error)


@pytest.mark.asyncio
async def test_agent_restart_requires_current_attempt_search_grant() -> None:
    coordinator, _, _ = _coordinator()
    registry = ToolRegistry()
    registry.register(ToolSearchTool(registry), always_on=True)
    registry.register(
        AgentRestartTool(coordinator),
        risk="external-side-effect",
        preloadable=False,
        requires_turn_search=True,
    )
    registry.set_context(channel="programmatic", chat_id="one")
    turn_token = current_turn_id.set("turn-a")
    session_token = current_session_key.set("programmatic:one")
    scope = registry.begin_turn_search_scope(
        turn_id="turn-a",
        session_key="programmatic:one",
        attempt=0,
    )
    try:
        denied = await registry.execute("agent_restart", {"reason": "reload"})
        assert "必须在当前 turn" in str(denied)

        _ = await registry.execute(
            "tool_search",
            {"query": "select:agent_restart"},
            raise_errors=True,
        )
        with pytest.raises(ValueError, match="不允许额外字段"):
            await registry.execute(
                "agent_restart",
                {"reason": "reload", "command": "rm -rf /"},
                raise_errors=True,
            )
        scheduled = await registry.execute(
            "agent_restart",
            {"reason": "reload"},
            raise_errors=True,
        )
        assert json.loads(str(scheduled))["status"] == "scheduled"
    finally:
        registry.end_turn_search_scope(scope)
        current_session_key.reset(session_token)
        current_turn_id.reset(turn_token)

    assert "agent_restart" in registry.get_non_preloadable_names()
    assert "agent_restart" not in registry.get_always_on_names()


def test_supervisor_commit_channel_uses_inherited_fd(monkeypatch: pytest.MonkeyPatch) -> None:
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv("AKASHIC_SUPERVISED", "1")
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-a")
    monkeypatch.setenv("AKASHIC_RESTART_NONCE", "n" * 64)
    monkeypatch.setenv("AKASHIC_RESTART_COMMIT_FD", str(write_fd))
    try:
        channel = SupervisorCommitChannel.from_environment()
        assert channel is not None
        assert channel.fd == write_fd
    finally:
        os.close(read_fd)
        os.close(write_fd)

    monkeypatch.setenv("AKASHIC_RESTART_COMMIT_FD", str(write_fd))
    with pytest.raises(OSError):
        SupervisorCommitChannel.from_environment()


def _spawn_supervisor_child(
    tmp_path: Path,
    *,
    boot_id: str,
    nonce: str,
    frame_count: int,
    exit_code: int = RESTART_EXIT_CODE,
) -> tuple[subprocess.Popen[bytes], int]:
    read_fd, write_fd = os.pipe()
    code = """
import json, os, pathlib, sys, time
workspace = pathlib.Path(sys.argv[1])
boot_id, nonce, write_fd, frame_count, exit_code = sys.argv[2:]
payload = {'bootId': boot_id, 'pid': os.getpid(), 'state': 'ready'}
temporary = workspace / f'.runtime-ready.{os.getpid()}.tmp'
temporary.write_text(json.dumps(payload))
os.replace(temporary, workspace / '.runtime-ready.json')
frame = {'type': 'restart_commit', 'bootId': boot_id, 'nonce': nonce, 'requestId': 'restart_test'}
for _ in range(int(frame_count)):
    os.write(int(write_fd), (json.dumps(frame) + '\\n').encode())
time.sleep(0.05)
raise SystemExit(int(exit_code))
"""
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(tmp_path),
            boot_id,
            nonce,
            str(write_fd),
            str(frame_count),
            str(exit_code),
        ],
        pass_fds=(write_fd,),
    )
    os.close(write_fd)
    return child, read_fd


@pytest.mark.parametrize(
    ("frame_count", "expected"),
    [(1, True), (0, False), (2, False)],
)
def test_real_child_exit_75_requires_unique_private_commit(
    tmp_path: Path,
    frame_count: int,
    expected: bool,
) -> None:
    child, read_fd = _spawn_supervisor_child(
        tmp_path,
        boot_id="boot-live",
        nonce="secret-nonce",
        frame_count=frame_count,
    )

    result = _wait_child(
        child,
        read_fd=read_fd,
        workspace=tmp_path,
        boot_id="boot-live",
        nonce="secret-nonce",
        readiness_timeout_s=2,
    )

    assert result.exit_code == RESTART_EXIT_CODE
    assert result.ready is True
    assert result.commit_valid is expected


def test_runtime_readiness_is_boot_and_pid_scoped(tmp_path: Path) -> None:
    readiness = RuntimeReadiness(tmp_path, "boot-current")
    readiness.mark_ready()

    assert readiness.ready is True
    payload = json.loads((tmp_path / ".runtime-ready.json").read_text())
    assert payload == {
        "bootId": "boot-current",
        "pid": os.getpid(),
        "state": "ready",
    }

    (tmp_path / ".runtime-ready.json").write_text(
        json.dumps({"bootId": "other", "pid": 1, "state": "ready"})
    )
    readiness.clear()
    assert (tmp_path / ".runtime-ready.json").exists()


def test_stale_readiness_cannot_satisfy_new_boot(tmp_path: Path) -> None:
    (tmp_path / ".runtime-ready.json").write_text(
        json.dumps({"bootId": "old", "pid": 1, "state": "ready"})
    )
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])

    result = _wait_child(
        child,
        read_fd=read_fd,
        workspace=tmp_path,
        boot_id="new",
        nonce="secret",
        readiness_timeout_s=0.05,
    )

    assert result.ready is False
    assert child.poll() is not None


class _ReadinessProbe:
    def __init__(self) -> None:
        self.ready = False
        self.marked = asyncio.Event()

    def mark_ready(self) -> None:
        self.ready = True
        self.marked.set()


def _runtime_with_probe(tmp_path: Path) -> tuple[AppRuntime, _ReadinessProbe]:
    runtime = AppRuntime(cast(Any, object()), tmp_path)
    probe = _ReadinessProbe()
    runtime.readiness = cast(Any, probe)

    async def start() -> None:
        runtime._started = True

    async def shutdown() -> None:
        runtime._started = False

    runtime.start = start  # type: ignore[method-assign]
    runtime.shutdown = shutdown  # type: ignore[method-assign]
    return runtime, probe


@pytest.mark.asyncio
async def test_runtime_schedule_failure_never_publishes_ready(tmp_path: Path) -> None:
    runtime, probe = _runtime_with_probe(tmp_path)

    def fail_schedule() -> list[asyncio.Future[Any]]:
        raise RuntimeError("schedule failed")

    runtime._schedule_runtime_tasks = fail_schedule  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="schedule failed"):
        await runtime.run()
    assert probe.ready is False


@pytest.mark.asyncio
async def test_primary_immediate_failure_never_publishes_ready(tmp_path: Path) -> None:
    runtime, probe = _runtime_with_probe(tmp_path)

    async def fail_immediately() -> None:
        raise RuntimeError("primary failed")

    runtime._schedule_runtime_tasks = (  # type: ignore[method-assign]
        lambda: [asyncio.create_task(fail_immediately())]
    )
    with pytest.raises(RuntimeError, match="primary failed"):
        await runtime.run()
    assert probe.ready is False


@pytest.mark.asyncio
async def test_runtime_publishes_ready_only_after_tasks_survive_gate(
    tmp_path: Path,
) -> None:
    runtime, probe = _runtime_with_probe(tmp_path)
    release = asyncio.Event()

    async def stay_running() -> None:
        await release.wait()

    runtime._schedule_runtime_tasks = (  # type: ignore[method-assign]
        lambda: [asyncio.create_task(stay_running())]
    )
    run_task = asyncio.create_task(runtime.run())
    assert probe.ready is False
    await probe.marked.wait()
    assert probe.ready is True
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task


class _FakeSupervisorChild:
    def __init__(self, exit_code: int) -> None:
        self.pid = 4242
        self.returncode = exit_code
        self.signals: list[int] = []

    def poll(self) -> int:
        return self.returncode

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)

    def wait(self) -> int:
        return self.returncode


class _RunningSupervisorChild(_FakeSupervisorChild):
    def __init__(self) -> None:
        super().__init__(0)
        self.running = True

    def poll(self) -> int | None:
        return None if self.running else self.returncode

    def send_signal(self, signum: int) -> None:
        super().send_signal(signum)
        self.running = False
        self.returncode = -signum

    def wait(self) -> int:
        if self.running:
            raise AssertionError("child must receive pending stop before wait")
        return self.returncode


@pytest.mark.parametrize(
    ("child_result", "expected"),
    [
        (supervisor_module._ChildResult(2, False, False), 2),
        (supervisor_module._ChildResult(RESTART_EXIT_CODE, False, False), 70),
    ],
)
def test_supervisor_exit_code_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child_result: supervisor_module._ChildResult,
    expected: int,
) -> None:
    child = _FakeSupervisorChild(child_result.exit_code)
    monkeypatch.setattr(
        supervisor_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: child,
    )

    def wait_child(_child: Any, *, read_fd: int, **_kwargs: Any):
        os.close(read_fd)
        return child_result

    monkeypatch.setattr(supervisor_module, "_wait_child", wait_child)

    assert run_supervisor(
        config_path=tmp_path / "config.toml",
        workspace=tmp_path,
    ) == expected


def test_supervisor_stop_between_generations_does_not_spawn_child_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _FakeSupervisorChild(RESTART_EXIT_CODE)
    spawns: list[_FakeSupervisorChild] = []

    def launch(*_args: Any, **_kwargs: Any) -> _FakeSupervisorChild:
        spawns.append(child)
        return child

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", launch)

    def wait_child(_child: Any, *, read_fd: int, **_kwargs: Any):
        os.close(read_fd)
        return supervisor_module._ChildResult(
            RESTART_EXIT_CODE,
            True,
            True,
        )

    monkeypatch.setattr(supervisor_module, "_wait_child", wait_child)
    uuid_calls = 0

    def next_boot_id() -> SimpleNamespace:
        nonlocal uuid_calls
        uuid_calls += 1
        if uuid_calls == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return SimpleNamespace(hex=f"boot-{uuid_calls}")

    monkeypatch.setattr(supervisor_module, "uuid4", next_boot_id)

    assert run_supervisor(
        config_path=tmp_path / "config.toml",
        workspace=tmp_path,
    ) == 0
    assert len(spawns) == 1


def test_supervisor_signal_inside_popen_waits_for_child_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _RunningSupervisorChild()
    handler_ran_inside_popen = False

    def launch(*_args: Any, **_kwargs: Any) -> _RunningSupervisorChild:
        nonlocal handler_ran_inside_popen
        os.kill(os.getpid(), signal.SIGTERM)
        handler_ran_inside_popen = bool(child.signals)
        return child

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", launch)

    def unexpected_wait_child(*_args: Any, **_kwargs: Any):
        raise AssertionError("stopping child must not enter readiness or restart gate")

    monkeypatch.setattr(supervisor_module, "_wait_child", unexpected_wait_child)

    assert run_supervisor(
        config_path=tmp_path / "config.toml",
        workspace=tmp_path,
    ) == 0
    assert handler_ran_inside_popen is False
    assert child.signals == [signal.SIGTERM]
    assert child.running is False


def test_supervisor_child_does_not_inherit_blocked_stop_signals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = subprocess.Popen
    probe_path = tmp_path / "child-status.txt"
    observed: dict[str, int] = {}

    def launch(_argv: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        code = """
import pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(pathlib.Path('/proc/self/status').read_text())
time.sleep(30)
"""
        return cast(
            "subprocess.Popen[bytes]",
            real_popen(
                [sys.executable, "-c", code, str(probe_path)],
                **kwargs,
            ),
        )

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", launch)

    def inspect_and_stop(
        child: subprocess.Popen[bytes],
        *,
        read_fd: int,
        **_kwargs: Any,
    ) -> supervisor_module._ChildResult:
        deadline = time.monotonic() + 2
        while not probe_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        status = probe_path.read_text()
        sigblk_line = next(
            line for line in status.splitlines() if line.startswith("SigBlk:")
        )
        blocked = int(sigblk_line.split()[1], 16)
        observed["blocked"] = blocked
        child.send_signal(signal.SIGTERM)
        exit_code = child.wait(timeout=2)
        os.close(read_fd)
        return supervisor_module._ChildResult(exit_code, False, False)

    monkeypatch.setattr(supervisor_module, "_wait_child", inspect_and_stop)

    assert run_supervisor(
        config_path=tmp_path / "config.toml",
        workspace=tmp_path,
    ) == 128 + signal.SIGTERM
    stop_mask = (1 << (signal.SIGINT - 1)) | (1 << (signal.SIGTERM - 1))
    assert observed["blocked"] & stop_mask == 0


class _DrainWriter:
    def __init__(self, gate: asyncio.Event, *, fail: bool = False) -> None:
        self.gate = gate
        self.fail = fail
        self.frames: list[bytes] = []

    def write(self, payload: bytes) -> None:
        self.frames.append(payload)

    async def drain(self) -> None:
        await self.gate.wait()
        if self.fail:
            raise ConnectionError("disconnected")


@pytest.mark.asyncio
async def test_ndjson_send_receipt_waits_for_writer_drain() -> None:
    connection = object.__new__(NdjsonConnection)
    connection._queue = asyncio.Queue(2)
    gate = asyncio.Event()
    connection._writer = _DrainWriter(gate)
    writer_task = asyncio.create_task(connection._write_loop())
    send_task = asyncio.create_task(connection.send({"method": "turn/completed"}))
    await asyncio.sleep(0)

    assert send_task.done() is False
    gate.set()
    await send_task
    await connection._queue.put(None)
    await writer_task
    assert connection._writer.frames == [b'{"method":"turn/completed"}\n']


@pytest.mark.asyncio
async def test_ndjson_stream_frame_does_not_wait_for_writer_drain() -> None:
    connection = object.__new__(NdjsonConnection)
    connection._queue = asyncio.Queue(2)
    gate = asyncio.Event()
    connection._writer = _DrainWriter(gate)
    writer_task = asyncio.create_task(connection._write_loop())

    await connection.send({"method": "item/completed"})
    await asyncio.sleep(0)
    assert connection._writer.frames == [b'{"method":"item/completed"}\n']

    gate.set()
    await connection._queue.put(None)
    await writer_task


@pytest.mark.asyncio
async def test_ndjson_disconnect_fails_delivery_receipt() -> None:
    connection = object.__new__(NdjsonConnection)
    connection._queue = asyncio.Queue(2)
    gate = asyncio.Event()
    connection._writer = _DrainWriter(gate, fail=True)
    writer_task = asyncio.create_task(connection._write_loop())
    send_task = asyncio.create_task(connection.send({"method": "turn/completed"}))
    await asyncio.sleep(0)
    gate.set()

    with pytest.raises(ConnectionError, match="disconnected"):
        await send_task
    with pytest.raises(ConnectionError, match="disconnected"):
        await writer_task
