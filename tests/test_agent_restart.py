from __future__ import annotations

import asyncio
import errno
import json
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import agent.background.boot_guardian as boot_guardian_module
import agent.supervisor as supervisor_module
import main as main_module
import utils.process_guard as process_guard_module
from agent.control.context import running_turn_id
from agent.control.errors import RuntimeClosedError
from agent.control.models import TurnRequest
from agent.control.protocol.router import ConnectionRouter
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from agent.restart import (
    RestartCoordinator,
    RestartRejectedError,
    RestartState,
    SupervisorCommitChannel,
)
from agent.supervisor import RESTART_EXIT_CODE, _wait_child, run_supervisor
from agent.tools.agent_restart import AgentRestartTool
from agent.tools.registry import ToolRegistry
from agent.tools.tool_search import ToolSearchTool
from bootstrap.app import AppRuntime
from bootstrap.runtime_readiness import RuntimeReadiness
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

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        coordinator.arm(
            turn_id=turn_holder["id"],
            session_key="programmatic:one",
            channel="programmatic",
            chat_id="one",
            reason="reload core",
        )
        armed.set()
        await release.wait()
        return ControlExecutionResult(response="restart scheduled")

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

        def record(self) -> dict[str, str]:
            return {"status": "completed"}

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
    turn_token = running_turn_id.set("turn-a")
    session_token = current_session_key.set("programmatic:one")
    registry.set_context(
        channel="programmatic",
        chat_id="one",
        session_key="programmatic:one",
        turn_id="turn-a",
    )
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
        running_turn_id.reset(turn_token)

    assert "agent_restart" in registry.get_non_preloadable_names()
    assert "agent_restart" not in registry.get_always_on_names()


def test_supervisor_commit_channel_uses_inherited_fd(monkeypatch: pytest.MonkeyPatch) -> None:
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv("AKASHIC_SUPERVISED", "1")
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-a")
    monkeypatch.setenv("AKASHIC_RESTART_NONCE", "n" * 64)
    monkeypatch.setenv("AKASHIC_LIFECYCLE_FD", str(write_fd))
    try:
        channel = SupervisorCommitChannel.from_environment()
        assert channel is not None
        assert channel.fd == write_fd
    finally:
        os.close(read_fd)
        os.close(write_fd)

    monkeypatch.setenv("AKASHIC_LIFECYCLE_FD", str(write_fd))
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
ready = {'type': 'ready', 'bootId': boot_id, 'pid': os.getpid()}
os.write(int(write_fd), (json.dumps(ready) + '\\n').encode())
frame = {'type': 'commit', 'bootId': boot_id, 'nonce': nonce, 'requestId': 'restart_test'}
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
        env={**os.environ, "AKASHIC_BOOT_ID": boot_id},
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

    if frame_count < 2:
        assert result.exit_code == RESTART_EXIT_CODE
    else:
        assert result.protocol_error
    assert result.ready is True
    assert result.commit_valid is expected


def test_lifecycle_parser_rejects_frame_larger_than_pipe_buf() -> None:
    state = supervisor_module._LifecycleState("boot-live", "secret-nonce")

    state.feed(b"x" * 4097)

    assert state.protocol_error == "lifecycle frame 超过 PIPE_BUF 安全上限"


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


def test_runtime_readiness_publishes_private_stage_and_ready(
    tmp_path: Path,
) -> None:
    read_fd, write_fd = os.pipe()
    channel = SupervisorCommitChannel(write_fd, "boot-current", "n" * 64)
    readiness = RuntimeReadiness(tmp_path, "boot-current", channel)
    try:
        readiness.mark_stage("core.ready")
        readiness.mark_ready()
        frames = [json.loads(line) for line in os.read(read_fd, 4096).splitlines()]
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert [frame["type"] for frame in frames] == ["stage", "ready"]
    assert frames[0]["stage"] == "core.ready"
    assert frames[1]["pid"] == os.getpid()


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
    assert result.protocol_error == (
        "Gateway 启动超时: stage=guardian.spawn deadline=0.050s"
    )
    assert child.poll() is not None


def test_stage_flood_cannot_extend_startup_deadline(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    stopped = threading.Event()

    def flood_stages() -> None:
        elapsed_ms = 0
        while not stopped.is_set():
            frame = {
                "type": "stage",
                "bootId": "new",
                "stage": "still.starting",
                "elapsedMs": elapsed_ms,
            }
            try:
                os.write(write_fd, (json.dumps(frame) + "\n").encode())
            except (BrokenPipeError, OSError):
                return
            elapsed_ms += 1

    writer = threading.Thread(target=flood_stages, daemon=True)
    writer.start()
    started_at = time.monotonic()
    try:
        result = _wait_child(
            child,
            read_fd=read_fd,
            workspace=tmp_path,
            boot_id="new",
            nonce="secret",
            readiness_timeout_s=0.05,
        )
    finally:
        stopped.set()
        os.close(write_fd)
        writer.join(timeout=1)

    assert time.monotonic() - started_at < 1
    assert result.protocol_error == (
        "Gateway 启动超时: stage=still.starting deadline=0.050s"
    )
    assert child.poll() is not None


class _ReadinessProbe:
    def __init__(self) -> None:
        self.ready = False
        self.marked = asyncio.Event()

    def mark_ready(self) -> None:
        self.ready = True
        self.marked.set()

    def mark_stage(self, _name: str) -> None:
        pass


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


@pytest.fixture
def no_settings_server(monkeypatch: pytest.MonkeyPatch) -> None:
    server = SimpleNamespace(should_exit=False)
    thread = SimpleNamespace(join=lambda timeout=None: None)
    monkeypatch.setattr(
        supervisor_module,
        "_start_settings_server",
        lambda *_args, **_kwargs: (server, thread),
    )


class _FakeSupervisorChild:
    def __init__(self, exit_code: int) -> None:
        self.pid = 4242
        self.returncode = exit_code
        self.signals: list[int] = []

    def poll(self) -> int:
        return self.returncode

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
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

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        if self.running:
            raise AssertionError("child must receive pending stop before wait")
        return self.returncode


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="依赖 /proc boot identity"
)
def test_supervisor_cleans_orphaned_boot_listener(tmp_path: Path) -> None:
    """gateway 退出后 Supervisor 必须回收独立 service 进程组及其端口。"""
    script = tmp_path / "orphan_listener.py"
    child_pid_file = tmp_path / "child.pid"
    _ = script.write_text(
        "import os, socket, time\n"
        "from pathlib import Path\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    listener = socket.socket()\n"
        "    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "    listener.bind(('127.0.0.1', int(os.environ['PORT'])))\n"
        "    listener.listen()\n"
        "    while True:\n"
        "        connection, _ = listener.accept()\n"
        "        connection.close()\n"
        "Path(os.environ['CHILD_PID_FILE']).write_text(str(pid))\n"
        "time.sleep(0.2)\n"
        "raise SystemExit(17)\n",
        encoding="utf-8",
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    boot_id = "test-orphan-listener"
    env = {
        **os.environ,
        "AKASHIC_BOOT_ID": boot_id,
        "CHILD_PID_FILE": str(child_pid_file),
        "PORT": str(port),
    }
    leader = subprocess.Popen(
        [sys.executable, str(script)],
        env=env,
        start_new_session=True,
    )
    try:
        assert leader.wait(timeout=3) == 17
        child_pid = int(child_pid_file.read_text())
        _wait_for_listener(port)

        supervisor_module._cleanup_boot_processes(
            boot_id=boot_id,
            gateway_group_id=999_999_999,
            timeout_s=2,
        )

        assert not supervisor_module._pid_exists(child_pid)
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=0.2)
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="依赖 /proc boot identity"
)
def test_boot_cleanup_never_kills_unknown_listener(tmp_path: Path) -> None:
    script = tmp_path / "unknown_listener.py"
    _ = script.write_text(
        "import socket, sys\n"
        "listener = socket.socket()\n"
        "listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "listener.bind(('127.0.0.1', int(sys.argv[1])))\n"
        "listener.listen()\n"
        "while True:\n"
        "    connection, _ = listener.accept()\n"
        "    connection.close()\n",
        encoding="utf-8",
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    unknown = subprocess.Popen(
        [sys.executable, str(script), str(port)],
        start_new_session=True,
    )
    try:
        _wait_for_listener(port)
        supervisor_module._cleanup_boot_processes(
            boot_id="unrelated-boot",
            gateway_group_id=None,
            timeout_s=0.1,
        )
        assert unknown.poll() is None
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            pass
    finally:
        try:
            os.killpg(unknown.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        unknown.wait(timeout=2)


def _wait_for_listener(port: int) -> None:
    deadline = time.monotonic() + 2
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise AssertionError("listener did not start before timeout")
            time.sleep(0.02)


def _wait_for_pid_file(path: Path, deadline: float) -> int:
    """Wait until a child publishes one complete, parseable PID."""

    while time.monotonic() < deadline:
        try:
            return int(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            time.sleep(0.01)
    raise AssertionError(f"PID 文件未完整发布: {path}")


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="依赖 Linux subreaper 与 /proc"
)
def test_boot_guardian_reaps_adopted_zombie_while_gateway_runs(
    tmp_path: Path,
) -> None:
    """Guardian 运行期间持续收割被托管树转交的 zombie。"""

    # 1. Gateway 创建一个会被 subreaper 收养的 double-fork zombie。
    script = tmp_path / "zombie_gateway.py"
    zombie_pid_path = tmp_path / "zombie.pid"
    gateway_pid_path = tmp_path / "gateway.pid"
    _ = script.write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        "middle = os.fork()\n"
        "if middle == 0:\n"
        "    child = os.fork()\n"
        "    if child == 0:\n"
        "        Path(os.environ['ZOMBIE_PID']).write_text(str(os.getpid()))\n"
        "        os._exit(0)\n"
        "    os._exit(0)\n"
        "os.waitpid(middle, 0)\n"
        "Path(os.environ['GATEWAY_PID']).write_text(str(os.getpid()))\n"
        "while True:\n"
        "    time.sleep(60)\n",
        encoding="utf-8",
    )
    lifecycle_read_fd, lifecycle_write_fd = os.pipe()
    lease_read_fd, lease_write_fd = os.pipe()
    env = {
        **os.environ,
        "ZOMBIE_PID": str(zombie_pid_path),
        "GATEWAY_PID": str(gateway_pid_path),
    }
    guardian = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent.background.boot_guardian",
            "--main-path",
            str(script),
            "--config",
            str(tmp_path / "config.toml"),
            "--workspace",
            str(tmp_path),
            "--boot-id",
            f"zombie-reap-{os.getpid()}",
            "--nonce",
            "n" * 64,
            "--lifecycle-fd",
            str(lifecycle_write_fd),
            "--lease-fd",
            str(lease_read_fd),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        pass_fds=(lifecycle_write_fd, lease_read_fd),
        start_new_session=True,
    )
    os.close(lifecycle_write_fd)
    os.close(lease_read_fd)
    try:
        # 2. Gateway 仍存活时，adopted child 必须已经从 /proc 消失。
        deadline = time.monotonic() + 3
        zombie_pid = _wait_for_pid_file(zombie_pid_path, deadline)
        gateway_pid = _wait_for_pid_file(gateway_pid_path, deadline)
        while time.monotonic() < deadline and Path(f"/proc/{zombie_pid}").exists():
            time.sleep(0.01)
        assert supervisor_module._pid_exists(gateway_pid)
        assert not Path(f"/proc/{zombie_pid}").exists()
    finally:
        # 3. 一次性 Guardian 仍按正式生命周期清空 Gateway。
        os.close(lifecycle_read_fd)
        os.close(lease_write_fd)
        if guardian.poll() is None:
            guardian.send_signal(signal.SIGTERM)
        guardian.wait(timeout=5)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="依赖 Linux pidfd 与 subreaper"
)
@pytest.mark.parametrize("owner_failure", ["supervisor", "guardian"])
def test_boot_guardian_cleans_double_fork_listener_after_owner_failure(
    tmp_path: Path,
    owner_failure: str,
) -> None:
    """Supervisor 或 Guardian 单点退出都不能留下 setsid grandchild。"""

    script = tmp_path / "guardian_gateway.py"
    gateway_pid_path = tmp_path / "gateway.pid"
    grandchild_pid_path = tmp_path / "grandchild.pid"
    _ = script.write_text(
        "import json, os, socket, time\n"
        "from pathlib import Path\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.setsid()\n"
        "    grandchild = os.fork()\n"
        "    if grandchild > 0:\n"
        "        os._exit(0)\n"
        "    listener = socket.socket()\n"
        "    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "    listener.bind(('127.0.0.1', int(os.environ['TEST_PORT'])))\n"
        "    listener.listen()\n"
        "    Path(os.environ['GRANDCHILD_PID']).write_text(str(os.getpid()))\n"
        "    while True:\n"
        "        connection, _ = listener.accept()\n"
        "        connection.close()\n"
        "os.waitpid(pid, 0)\n"
        "Path(os.environ['GATEWAY_PID']).write_text(str(os.getpid()))\n"
        "grandchild_pid_path = Path(os.environ['GRANDCHILD_PID'])\n"
        "while True:\n"
        "    try:\n"
        "        if grandchild_pid_path.read_text().strip():\n"
        "            break\n"
        "    except FileNotFoundError:\n"
        "        pass\n"
        "    time.sleep(0.01)\n"
        "frame = {'type': 'ready', 'bootId': os.environ['AKASHIC_BOOT_ID'], "
        "'pid': os.getpid()}\n"
        "os.write(int(os.environ['AKASHIC_LIFECYCLE_FD']), "
        "(json.dumps(frame) + '\\n').encode())\n"
        "while True:\n"
        "    time.sleep(60)\n",
        encoding="utf-8",
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    boot_id = f"guardian-{owner_failure}-{os.getpid()}"
    lifecycle_read_fd, lifecycle_write_fd = os.pipe()
    lease_read_fd, lease_write_fd = os.pipe()
    env = {
        **os.environ,
        "TEST_PORT": str(port),
        "GATEWAY_PID": str(gateway_pid_path),
        "GRANDCHILD_PID": str(grandchild_pid_path),
    }
    guardian = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent.background.boot_guardian",
            "--main-path",
            str(script),
            "--config",
            str(tmp_path / "config.toml"),
            "--workspace",
            str(tmp_path),
            "--boot-id",
            boot_id,
            "--nonce",
            "n" * 64,
            "--lifecycle-fd",
            str(lifecycle_write_fd),
            "--lease-fd",
            str(lease_read_fd),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        pass_fds=(lifecycle_write_fd, lease_read_fd),
        start_new_session=True,
    )
    os.close(lifecycle_write_fd)
    os.close(lease_read_fd)
    try:
        readable, _, _ = select.select([lifecycle_read_fd], [], [], 3)
        assert readable, "Guardian Gateway 未发布 ready"
        lifecycle = os.read(lifecycle_read_fd, 4096)
        assert json.loads(lifecycle)["type"] == "ready"
        gateway_pid = int(gateway_pid_path.read_text())
        grandchild_pid = int(grandchild_pid_path.read_text())
        _wait_for_listener(port)

        if owner_failure == "supervisor":
            os.close(lease_write_fd)
            lease_write_fd = -1
            assert guardian.wait(timeout=5) == 128 + signal.SIGTERM
        else:
            os.kill(guardian.pid, signal.SIGKILL)
            assert guardian.wait(timeout=2) == -signal.SIGKILL
            supervisor_module._cleanup_boot_processes(
                boot_id=boot_id,
                gateway_group_id=None,
                timeout_s=2,
            )

        assert not supervisor_module._pid_exists(gateway_pid)
        assert not supervisor_module._pid_exists(grandchild_pid)
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=0.2)
    finally:
        os.close(lifecycle_read_fd)
        if lease_write_fd >= 0:
            os.close(lease_write_fd)
        if guardian.poll() is None:
            os.killpg(guardian.pid, signal.SIGKILL)
            guardian.wait(timeout=2)
        supervisor_module._cleanup_boot_processes(
            boot_id=boot_id,
            gateway_group_id=None,
            timeout_s=2,
        )


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
    no_settings_server: None,
) -> None:
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    child = _FakeSupervisorChild(child_result.exit_code)
    monkeypatch.setattr(
        supervisor_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: child,
    )

    def wait_child(
        _child: Any,
        *,
        read_fd: int,
        lease_fd: int,
        **_kwargs: Any,
    ):
        os.close(read_fd)
        os.close(lease_fd)
        return child_result

    monkeypatch.setattr(supervisor_module, "_wait_child", wait_child)

    assert run_supervisor(
        config_path=tmp_path / "config.toml",
        workspace=tmp_path,
    ) == expected


def test_supervisor_logs_cleanup_failure_and_still_starts_next_boot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_settings_server: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    children = [
        _FakeSupervisorChild(RESTART_EXIT_CODE),
        _FakeSupervisorChild(0),
    ]
    spawned: list[_FakeSupervisorChild] = []

    def launch(*_args: Any, **_kwargs: Any) -> _FakeSupervisorChild:
        child = children[len(spawned)]
        spawned.append(child)
        return child

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", launch)

    def wait_child(
        child: _FakeSupervisorChild,
        *,
        read_fd: int,
        lease_fd: int,
        **_kwargs: Any,
    ) -> supervisor_module._ChildResult:
        os.close(read_fd)
        os.close(lease_fd)
        if child.returncode == RESTART_EXIT_CODE:
            return supervisor_module._ChildResult(RESTART_EXIT_CODE, True, True)
        return supervisor_module._ChildResult(0, True, False)

    monkeypatch.setattr(supervisor_module, "_wait_child", wait_child)
    monkeypatch.setattr(
        boot_guardian_module,
        "_cleanup_boot_processes",
        lambda **_kwargs: (_ for _ in ()).throw(
            PermissionError(errno.EPERM, "Operation not permitted")
        ),
    )

    assert run_supervisor(
        config_path=tmp_path / "config.toml",
        workspace=tmp_path,
    ) == 0
    assert len(spawned) == 2
    assert "event=cleanup_degraded" in capsys.readouterr().err


def test_supervisor_without_config_requires_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        supervisor_module,
        "_enable_child_subreaper",
        lambda: pytest.fail("缺少配置时不应接管子进程"),
    )
    monkeypatch.setattr(
        supervisor_module._SupervisorLock,
        "acquire",
        lambda _self: pytest.fail("缺少配置时不应获取 workspace lock"),
    )
    monkeypatch.setattr(
        supervisor_module,
        "_start_settings_server",
        lambda *_args, **_kwargs: pytest.fail("缺少配置时不应启动 Web Shell"),
    )
    monkeypatch.setattr(
        supervisor_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("缺少配置时不应启动 Gateway"),
    )
    assert run_supervisor(
        config_path=tmp_path / "missing.toml",
        workspace=tmp_path,
    ) == 2
    error = capsys.readouterr().err
    assert "python main.py init" in error
    assert "python main.py setup" in error


def test_supervisor_stop_between_generations_does_not_spawn_child_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_settings_server: None,
) -> None:
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    child = _FakeSupervisorChild(RESTART_EXIT_CODE)
    spawns: list[_FakeSupervisorChild] = []

    def launch(*_args: Any, **_kwargs: Any) -> _FakeSupervisorChild:
        spawns.append(child)
        return child

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", launch)

    def wait_child(
        _child: Any,
        *,
        read_fd: int,
        lease_fd: int,
        **_kwargs: Any,
    ):
        os.close(read_fd)
        os.close(lease_fd)
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
    no_settings_server: None,
) -> None:
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
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
    no_settings_server: None,
) -> None:
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    real_popen = subprocess.Popen
    probe_path = tmp_path / "child-status.txt"
    observed: dict[str, int] = {}

    def launch(_argv: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        assert "preexec_fn" not in kwargs
        code = """
import pathlib, signal, sys, time
blocked = signal.pthread_sigmask(signal.SIG_BLOCK, [])
pathlib.Path(sys.argv[1]).write_text(','.join(str(int(sig)) for sig in blocked))
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
        lease_fd: int,
        **_kwargs: Any,
    ) -> supervisor_module._ChildResult:
        deadline = time.monotonic() + 2
        while not probe_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        blocked = {int(value) for value in probe_path.read_text().split(",") if value}
        observed["blocked"] = len(blocked)
        observed["stop_blocked"] = int(
            signal.SIGINT in blocked or signal.SIGTERM in blocked
        )
        child.send_signal(signal.SIGTERM)
        exit_code = child.wait(timeout=2)
        os.close(read_fd)
        os.close(lease_fd)
        return supervisor_module._ChildResult(exit_code, False, False)

    monkeypatch.setattr(supervisor_module, "_wait_child", inspect_and_stop)

    assert run_supervisor(
        config_path=tmp_path / "config.toml",
        workspace=tmp_path,
    ) == 128 + signal.SIGTERM
    assert observed["stop_blocked"] == 0


def test_settings_restart_bridge_waits_for_matching_ready_generation() -> None:
    bridge = supervisor_module._SettingsRestartBridge(1)
    completed: list[bool] = []

    thread = threading.Thread(
        target=lambda: (bridge.request_and_wait(), completed.append(True)),
    )
    thread.start()
    assert bridge.request_event.wait(1)
    generation = bridge.take_request()
    assert generation == 1
    bridge.complete(generation, True)
    thread.join(timeout=1)

    assert completed == [True]


def test_lifecycle_accepts_settings_reload_only_after_ready() -> None:
    state = supervisor_module._LifecycleState("boot-model", "nonce")
    state.feed(b'{"type":"ready","bootId":"boot-model","pid":123}\n')
    state.feed(
        b'{"type":"settings_reloaded","bootId":"boot-model",'
        b'"success":true,"detail":"digest"}\n'
    )

    assert state.protocol_error == ""
    assert state.settings_results == [(True, "digest")]
    assert not state.commit_valid

    invalid = supervisor_module._LifecycleState("boot-model", "nonce")
    invalid.feed(
        b'{"type":"settings_reloaded","bootId":"boot-model",'
        b'"success":true}\n'
    )
    assert invalid.protocol_error == "settings reload frame 无效"


def test_settings_restart_bridge_exposes_candidate_rejection() -> None:
    bridge = supervisor_module._SettingsRestartBridge(1)
    failures: list[str] = []

    def request() -> None:
        try:
            bridge.request_and_wait()
        except RuntimeError as exc:
            failures.append(str(exc))

    thread = threading.Thread(target=request)
    thread.start()
    assert bridge.request_event.wait(1)
    generation = bridge.take_request()
    bridge.complete(generation, False)
    thread.join(timeout=1)

    assert failures == ["Gateway 拒绝候选模型配置"]


def test_supervised_gateway_skips_duplicate_startup_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setenv("AKASHIC_SUPERVISED", "1")
    monkeypatch.setattr(
        main_module,
        "migrate_installation",
        lambda config, workspace: calls.append((config, workspace)),
    )

    assert (
        main_module._prepare_startup_migrations(
            ["gateway"],
            tmp_path / "config.toml",
            tmp_path,
        )
        is None
    )
    assert calls == []


def test_platform_boundary_exposes_supervisor_on_linux_and_macos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main_module._supervisor_supported("linux")
    assert main_module._supervisor_supported("darwin")
    assert not main_module._supervisor_supported("win32")
    assert supervisor_module._supervisor_platform_supported("darwin")
    assert boot_guardian_module._guardian_platform_supported("darwin")

    monkeypatch.setattr(supervisor_module.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="仅支持 Linux 和 macOS"):
        run_supervisor(
            config_path=tmp_path / "config.toml",
            workspace=tmp_path,
        )


def test_darwin_process_discovery_uses_exact_boot_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(boot_guardian_module.sys, "platform", "darwin")
    monkeypatch.setattr(boot_guardian_module, "_darwin_process_ids", lambda: [101, 102])
    monkeypatch.setattr(
        boot_guardian_module,
        "_darwin_process_environ",
        lambda pid: [b"AKASHIC_BOOT_ID=boot-a"] if pid == 101 else [b"OTHER=1"],
    )
    monkeypatch.setattr(boot_guardian_module.os, "getpid", lambda: 999)
    monkeypatch.setattr(boot_guardian_module.os, "getpgrp", lambda: 500)
    monkeypatch.setattr(boot_guardian_module.os, "getpgid", lambda pid: pid + 1000)
    groups: set[int] = set()
    direct_pids: set[int] = set()

    boot_guardian_module._discover_boot_targets("boot-a", groups, direct_pids)

    assert groups == {1101}
    assert direct_pids == set()


def test_darwin_subreaper_is_explicit_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boot_guardian_module.sys, "platform", "darwin")
    boot_guardian_module._enable_child_subreaper()


def test_process_ref_keeps_pidfd_on_linux_and_polls_on_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(process_guard_module.sys, "platform", "linux")
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    monkeypatch.setattr(process_guard_module, "open_pidfd", lambda _pid: read_fd)
    linux_ref = process_guard_module.open_process_ref(123)
    assert linux_ref.stable
    assert process_guard_module.process_wait_timeout(linux_ref, None) is None
    linux_ref.close()

    monkeypatch.setattr(process_guard_module.sys, "platform", "darwin")
    darwin_ref = process_guard_module.open_process_ref(123)
    assert not darwin_ref.stable
    assert process_guard_module.process_wait_timeout(darwin_ref, None) == 0.1
    darwin_ref.close()


def test_settings_server_rejects_non_loopback_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = supervisor_module._SettingsRestartBridge(1)
    monkeypatch.setenv("AKASHIC_WEB_HOST", "0.0.0.0")
    monkeypatch.delenv("AKASHIC_WEB_ALLOW_NON_LOOPBACK", raising=False)
    try:
        with pytest.raises(RuntimeError, match="只允许 127.0.0.1"):
            supervisor_module._start_settings_server(
                tmp_path / "config.toml",
                tmp_path,
                bridge,
            )
    finally:
        bridge.close()


@pytest.mark.asyncio
async def test_settings_drain_waits_for_existing_turn_without_cancelling() -> None:
    runtime = object.__new__(ConversationRuntime)
    runtime._accepting_turns = True
    runtime._admission_capacity_event = asyncio.Event()
    finished = asyncio.Event()

    async def existing_turn() -> None:
        await asyncio.sleep(0)
        finished.set()

    task = asyncio.create_task(existing_turn())
    runtime._tasks = {"turn-1": task}

    await runtime.quiesce_and_drain()

    assert runtime._accepting_turns is False
    assert finished.is_set()


class _DrainWriter:
    def __init__(self, gate: asyncio.Event, *, fail: bool = False) -> None:
        self.gate = gate
        self.fail = fail
        self.frames: list[bytes] = []
        self.closed = False

    def write(self, payload: bytes) -> None:
        self.frames.append(payload)

    async def drain(self) -> None:
        await self.gate.wait()
        if self.fail:
            raise ConnectionError("disconnected")

    def close(self) -> None:
        self.closed = True


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
async def test_ndjson_outbound_queue_overflow_closes_only_its_writer() -> None:
    connection = object.__new__(NdjsonConnection)
    connection._queue = asyncio.Queue(1)
    writer = _DrainWriter(asyncio.Event())
    connection._writer = writer

    await connection.send({"method": "item/completed", "params": {"index": 1}})
    with pytest.raises(ConnectionError, match="outbound queue is full"):
        await connection.send(
            {"method": "item/completed", "params": {"index": 2}}
        )

    assert writer.closed is True


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
