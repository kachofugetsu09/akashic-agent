from __future__ import annotations

import json
import os
import secrets
import selectors
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import IO
from uuid import uuid4

from agent.background.boot_guardian import (
    _cleanup_boot_processes,
    _cleanup_boot_processes_best_effort,
    _enable_child_subreaper,
    _pid_has_boot_identity,
    _pid_exists,
    _reap_adopted_children,
)
from utils.process_guard import (
    ProcessRef,
    open_process_ref,
    process_wait_timeout,
    signal_process_ref,
)

RESTART_EXIT_CODE = 75
SUPERVISOR_FAILURE_EXIT_CODE = 70
_GUARDIAN_CLEANUP_WAIT_SECONDS = 7.0
_MAX_LIFECYCLE_FRAME_BYTES = 4096
_MAX_LIFECYCLE_DRAIN_BYTES = 65536


class _SupervisorLock:
    """保证一个 workspace 只有一个 Supervisor。"""

    def __init__(self, workspace: Path) -> None:
        self.path = workspace / ".supervisor.lock"
        self.pid_path = workspace / ".supervisor.pid"
        self._stream: IO[str] | None = None

    def acquire(self) -> None:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError(f"workspace supervisor 已在运行: {self.path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        temporary = self.pid_path.with_name(f".{self.pid_path.name}.{os.getpid()}.tmp")
        temporary.write_text(str(os.getpid()), encoding="utf-8")
        os.replace(temporary, self.pid_path)
        self._stream = stream

    def release(self) -> None:
        import fcntl

        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            if self.pid_path.exists() and self.pid_path.read_text(
                encoding="utf-8"
            ).strip() == str(os.getpid()):
                self.pid_path.unlink()
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


@dataclass(frozen=True)
class _ChildResult:
    exit_code: int
    ready: bool
    commit_valid: bool
    settings_generation: int = 0
    gateway_pid: int | None = None
    last_stage: str = "guardian.spawn"
    protocol_error: str = ""


@dataclass
class _LifecycleState:
    boot_id: str
    nonce: str
    ready: bool = False
    gateway_pid: int | None = None
    commit_valid: bool = False
    last_stage: str = "guardian.spawn"
    last_elapsed_ms: int = -1
    protocol_error: str = ""
    buffer: bytearray = field(default_factory=bytearray)
    settings_results: list[tuple[bool, str]] = field(default_factory=list)

    def feed(self, payload: bytes, *, eof: bool = False) -> None:
        """解析完整 NDJSON frame，并保留末尾尚未完成的字节。"""

        self.buffer.extend(payload)
        while b"\n" in self.buffer:
            raw_frame, _, remainder = self.buffer.partition(b"\n")
            self.buffer = bytearray(remainder)
            if raw_frame:
                if len(raw_frame) > _MAX_LIFECYCLE_FRAME_BYTES:
                    self.protocol_error = "lifecycle frame 超过 PIPE_BUF 安全上限"
                    break
                self._accept(raw_frame)
        if len(self.buffer) > _MAX_LIFECYCLE_FRAME_BYTES and not self.protocol_error:
            self.protocol_error = "lifecycle frame 超过 PIPE_BUF 安全上限"
        if eof and self.buffer and not self.protocol_error:
            self.protocol_error = "lifecycle pipe 以不完整 frame 结束"
        if self.protocol_error:
            self.commit_valid = False

    def _accept(self, raw_frame: bytes | bytearray) -> None:
        if self.protocol_error:
            return
        try:
            frame = json.loads(raw_frame)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.protocol_error = "lifecycle frame 不是有效 JSON"
            return
        if not isinstance(frame, dict) or frame.get("bootId") != self.boot_id:
            self.protocol_error = "lifecycle frame 的 boot identity 无效"
            return
        frame_type = frame.get("type")
        if frame_type == "stage":
            self._accept_stage(frame)
        elif frame_type == "ready":
            self._accept_ready(frame)
        elif frame_type == "commit":
            self._accept_commit(frame)
        elif frame_type == "settings_reloaded":
            self._accept_settings_reloaded(frame)
        else:
            self.protocol_error = f"未知 lifecycle frame: {frame_type!r}"

    def _accept_stage(self, frame: dict[str, object]) -> None:
        stage = frame.get("stage")
        elapsed_ms = frame.get("elapsedMs")
        if (
            not isinstance(stage, str)
            or not stage.strip()
            or not isinstance(elapsed_ms, int)
            or elapsed_ms < self.last_elapsed_ms
            or self.ready
        ):
            self.protocol_error = "lifecycle stage frame 无效"
            return
        self.last_stage = stage
        self.last_elapsed_ms = elapsed_ms

    def _accept_ready(self, frame: dict[str, object]) -> None:
        pid = frame.get("pid")
        if self.ready or not isinstance(pid, int) or pid <= 0:
            self.protocol_error = "lifecycle ready frame 无效或重复"
            return
        self.ready = True
        self.gateway_pid = pid

    def _accept_commit(self, frame: dict[str, object]) -> None:
        request_id = frame.get("requestId")
        nonce = str(frame.get("nonce") or "")
        if (
            not self.ready
            or self.commit_valid
            or not isinstance(request_id, str)
            or not request_id.startswith(("restart_", "settings_"))
            or not secrets.compare_digest(nonce, self.nonce)
        ):
            self.protocol_error = "lifecycle commit frame 无效或重复"
            return
        self.commit_valid = True

    def _accept_settings_reloaded(self, frame: dict[str, object]) -> None:
        success = frame.get("success")
        detail = frame.get("detail", "")
        if (
            not self.ready
            or not isinstance(success, bool)
            or not isinstance(detail, str)
            or len(detail.encode("utf-8")) > 2048
        ):
            self.protocol_error = "settings reload frame 无效"
            return
        self.settings_results.append((success, detail))


class _SettingsRestartBridge:
    """用 wake pipe 在设置线程与 Supervisor 之间交接一次热重载。"""

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self.request_event = threading.Event()
        self._condition = threading.Condition()
        self._next_generation = 0
        self._requested_generation = 0
        self._completed: dict[int, bool] = {}
        self._read_fd, self._write_fd = os.pipe()
        os.set_blocking(self._read_fd, False)
        os.set_blocking(self._write_fd, False)

    def fileno(self) -> int:
        return self._read_fd

    def request_and_wait(self) -> None:
        with self._condition:
            self._next_generation += 1
            generation = self._next_generation
            self._requested_generation = generation
            self.request_event.set()
            os.write(self._write_fd, b"\0")
            completed = self._condition.wait_for(
                lambda: generation in self._completed,
                timeout=self.timeout_s,
            )
            if not completed:
                raise RuntimeError("Gateway 模型配置重载等待超时")
            if not self._completed.pop(generation):
                raise RuntimeError("Gateway 拒绝候选模型配置")

    def take_request(self) -> int:
        with self._condition:
            if not self.request_event.is_set():
                return 0
            generation = self._requested_generation
            self.request_event.clear()
            _drain_fd(self._read_fd)
            return generation

    def wait_request(self) -> int:
        if not self.request_event.wait(self.timeout_s):
            return 0
        return self.take_request()

    def complete(self, generation: int, success: bool) -> None:
        with self._condition:
            self._completed[generation] = success
            self._condition.notify_all()

    def close(self) -> None:
        os.close(self._read_fd)
        os.close(self._write_fd)


def run_supervisor(
    *,
    config_path: Path,
    workspace: Path,
    readiness_timeout_s: float = 15.0,
) -> int:
    """管理 boot 代际，并且只接受当前 boot 的私有重启提交。"""

    if not _supervisor_platform_supported():
        raise RuntimeError("Supervisor 仅支持 Linux 和 macOS")
    if readiness_timeout_s <= 0:
        raise ValueError("readiness_timeout_s 必须大于 0")
    project_root = Path(__file__).resolve().parent.parent
    main_path = project_root / "main.py"
    config_path = config_path.expanduser().resolve()
    workspace = workspace.expanduser().resolve()
    if not config_path.exists():
        print(
            f"未找到配置文件：{config_path}。请先运行 `python main.py init` "
            "或 `python main.py setup`。",
            file=sys.stderr,
        )
        return 2
    # 1. 建立 workspace owner、设置线程和停止信号边界。
    _enable_child_subreaper()
    lock = _SupervisorLock(workspace)
    lock.acquire()
    settings_bridge = _SettingsRestartBridge(max(30.0, readiness_timeout_s * 2))
    signal_read_fd, signal_write_fd = os.pipe()
    os.set_blocking(signal_read_fd, False)
    os.set_blocking(signal_write_fd, False)
    stopping_signal: int | None = None
    guardian: subprocess.Popen[bytes] | None = None

    def forward_stop(signum: int, _frame: FrameType | None) -> None:
        nonlocal stopping_signal
        stopping_signal = signum
        try:
            os.write(signal_write_fd, bytes((signum,)))
        except BlockingIOError:
            pass
        if guardian is not None and guardian.poll() is None:
            guardian.send_signal(signum)

    stop_signals = (signal.SIGINT, signal.SIGTERM)
    previous_handlers = {sig: signal.signal(sig, forward_stop) for sig in stop_signals}
    try:
        settings_server, settings_thread = _start_settings_server(
            config_path,
            workspace,
            settings_bridge,
        )
    except BaseException:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        settings_bridge.close()
        os.close(signal_read_fd)
        os.close(signal_write_fd)
        lock.release()
        raise

    try:
        # 2. 每一代只创建一个 Guardian，并在上一代清空后进入下一代。
        report_settings_generation = 0
        while True:
            if stopping_signal is not None:
                return 0
            boot_id = uuid4().hex
            nonce = secrets.token_hex(32)
            if stopping_signal is not None:
                return 0
            lifecycle_read_fd, lifecycle_write_fd = os.pipe()
            lease_read_fd, lease_write_fd = os.pipe()
            os.set_blocking(lifecycle_read_fd, False)
            guardian_env = os.environ.copy()
            for name in (
                "AKASHIC_SUPERVISED",
                "AKASHIC_BOOT_ID",
                "AKASHIC_LIFECYCLE_FD",
                "AKASHIC_RESTART_NONCE",
            ):
                guardian_env.pop(name, None)
            try:
                guardian = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "agent.background.boot_guardian",
                        "--main-path",
                        str(main_path),
                        "--config",
                        str(config_path),
                        "--workspace",
                        str(workspace),
                        "--boot-id",
                        boot_id,
                        "--nonce",
                        nonce,
                        "--lifecycle-fd",
                        str(lifecycle_write_fd),
                        "--lease-fd",
                        str(lease_read_fd),
                    ],
                    cwd=project_root,
                    env=guardian_env,
                    pass_fds=(lifecycle_write_fd, lease_read_fd),
                    start_new_session=True,
                )
            except BaseException:
                os.close(lifecycle_read_fd)
                os.close(lease_write_fd)
                raise
            finally:
                os.close(lifecycle_write_fd)
                os.close(lease_read_fd)

            if stopping_signal is not None and guardian.poll() is None:
                guardian.send_signal(stopping_signal)
            if stopping_signal is not None:
                os.close(lifecycle_read_fd)
                os.close(lease_write_fd)
                _stop_guardian(guardian)
                _ = _cleanup_boot_processes_best_effort(
                    boot_id=boot_id,
                    gateway_group_id=None,
                    owner="supervisor",
                )
                _reap_adopted_children()
                guardian = None
                return 0
            try:
                result = _wait_child(
                    guardian,
                    read_fd=lifecycle_read_fd,
                    lease_fd=lease_write_fd,
                    workspace=workspace,
                    boot_id=boot_id,
                    nonce=nonce,
                    readiness_timeout_s=readiness_timeout_s,
                    settings_bridge=settings_bridge,
                    report_settings_generation=report_settings_generation,
                )
            except BaseException:
                _stop_guardian(guardian)
                _ = _cleanup_boot_processes_best_effort(
                    boot_id=boot_id,
                    gateway_group_id=None,
                    owner="supervisor",
                )
                _reap_adopted_children()
                raise

            # 3. Guardian 完成后再做一次 boot-scoped 空集验证和兜底清理。
            _ = _cleanup_boot_processes_best_effort(
                boot_id=boot_id,
                gateway_group_id=None,
                owner="supervisor",
            )
            _reap_adopted_children()
            guardian = None
            if report_settings_generation:
                report_settings_generation = 0
            if stopping_signal is not None:
                return 0
            if result.protocol_error:
                print(result.protocol_error, file=sys.stderr)
                return SUPERVISOR_FAILURE_EXIT_CODE
            if result.exit_code != RESTART_EXIT_CODE:
                if result.settings_generation:
                    settings_bridge.complete(result.settings_generation, False)
                    rollback_generation = settings_bridge.wait_request()
                    if rollback_generation:
                        report_settings_generation = rollback_generation
                        continue
                return _portable_exit_code(result.exit_code)
            if not result.ready or not result.commit_valid:
                if result.settings_generation:
                    settings_bridge.complete(result.settings_generation, False)
                    rollback_generation = settings_bridge.wait_request()
                    if rollback_generation:
                        report_settings_generation = rollback_generation
                        continue
                return SUPERVISOR_FAILURE_EXIT_CODE
            report_settings_generation = result.settings_generation
    finally:
        if guardian is not None:
            _stop_guardian(guardian)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        settings_server.should_exit = True
        settings_thread.join(timeout=5)
        settings_bridge.close()
        os.close(signal_read_fd)
        os.close(signal_write_fd)
        lock.release()


def _wait_child(
    child: subprocess.Popen[bytes],
    *,
    read_fd: int,
    lease_fd: int | None = None,
    workspace: Path,
    boot_id: str,
    nonce: str,
    readiness_timeout_s: float,
    settings_bridge: _SettingsRestartBridge | None = None,
    report_settings_generation: int = 0,
) -> _ChildResult:
    """等待进程与生命周期事件，直到 Guardian 退出或启动失败。"""

    # 1. 建立本代协议状态和三个可等待的内核事件。
    _ = workspace
    owned_bridge = settings_bridge is None
    settings_bridge = settings_bridge or _SettingsRestartBridge(readiness_timeout_s)
    state = _LifecycleState(boot_id, nonce)
    deadline = time.monotonic() + readiness_timeout_s
    settings_generation = 0
    gateway_ref: ProcessRef | None = None
    child_ref = open_process_ref(child.pid)
    lifecycle_open = True
    try:
        # 2. ready 前受总体 deadline 约束，ready 后只等待事件或进程退出。
        with selectors.DefaultSelector() as selector:
            if child_ref.wait_fd is not None:
                selector.register(child_ref.wait_fd, selectors.EVENT_READ, "guardian")
            selector.register(read_fd, selectors.EVENT_READ, "lifecycle")
            while child.poll() is None:
                timeout = None
                if not state.ready:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        state.protocol_error = (
                            f"Gateway 启动超时: stage={state.last_stage} "
                            f"deadline={readiness_timeout_s:.3f}s"
                        )
                        _stop_guardian(child)
                        break
                events = selector.select(process_wait_timeout(child_ref, timeout))
                for key, _mask in events:
                    if key.data == "guardian":
                        _ = child.wait()
                    elif key.data == "lifecycle":
                        chunk = os.read(read_fd, 65536)
                        if chunk:
                            state.feed(chunk)
                            if state.settings_results:
                                if not settings_generation:
                                    state.protocol_error = "收到无对应请求的 settings reload 回执"
                                elif len(state.settings_results) != 1:
                                    state.protocol_error = "收到重复 settings reload 回执"
                                else:
                                    success, _detail = state.settings_results.pop()
                                    settings_bridge.complete(
                                        settings_generation,
                                        success,
                                    )
                                    settings_generation = 0
                        else:
                            state.feed(b"", eof=True)
                            selector.unregister(read_fd)
                            lifecycle_open = False
                    elif key.data == "settings":
                        settings_generation = settings_bridge.take_request()
                        if settings_generation:
                            assert gateway_ref is not None
                            if not gateway_ref.stable and not _pid_has_boot_identity(
                                gateway_ref.pid,
                                boot_id,
                            ):
                                state.protocol_error = (
                                    "Gateway PID 已退出或身份变化，拒绝发送设置重载信号"
                                )
                                break
                            signal_process_ref(gateway_ref, signal.SIGUSR1)

                if state.protocol_error:
                    _stop_guardian(child)
                    break
                if state.ready and gateway_ref is None:
                    assert state.gateway_pid is not None
                    if not _pid_has_boot_identity(state.gateway_pid, boot_id):
                        state.protocol_error = (
                            "ready PID 不属于当前 boot，拒绝取得信号权限"
                        )
                        state.commit_valid = False
                        _stop_guardian(child)
                        break
                    gateway_ref = open_process_ref(state.gateway_pid)
                    if report_settings_generation:
                        settings_bridge.complete(report_settings_generation, True)
                        report_settings_generation = 0
                    selector.register(
                        settings_bridge.fileno(),
                        selectors.EVENT_READ,
                        "settings",
                    )

            if lifecycle_open and not state.protocol_error:
                _read_lifecycle_to_eof(read_fd, state)
            if report_settings_generation:
                settings_bridge.complete(report_settings_generation, False)
        # 3. 只返回已解析的私有证据；调用方负责最终 boot 空集验证。
        return _ChildResult(
            (
                child.returncode
                if child.returncode is not None
                else SUPERVISOR_FAILURE_EXIT_CODE
            ),
            state.ready,
            state.commit_valid,
            settings_generation or report_settings_generation,
            state.gateway_pid,
            state.last_stage,
            state.protocol_error,
        )
    finally:
        if lease_fd is not None:
            os.close(lease_fd)
        if gateway_ref is not None:
            gateway_ref.close()
        child_ref.close()
        os.close(read_fd)
        if owned_bridge:
            settings_bridge.close()


def _supervisor_platform_supported(platform: str | None = None) -> bool:
    current = platform or sys.platform
    return current.startswith("linux") or current == "darwin"


def _read_lifecycle_to_eof(fd: int, state: _LifecycleState) -> None:
    total_bytes = 0
    while True:
        try:
            chunk = os.read(
                fd,
                min(65536, _MAX_LIFECYCLE_DRAIN_BYTES - total_bytes + 1),
            )
        except BlockingIOError:
            return
        if not chunk:
            state.feed(b"", eof=True)
            return
        total_bytes += len(chunk)
        if total_bytes > _MAX_LIFECYCLE_DRAIN_BYTES:
            state.protocol_error = "lifecycle pipe 退出排空超过固定预算"
            state.commit_valid = False
            return
        state.feed(chunk)


def _stop_guardian(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is None:
        child.send_signal(signal.SIGTERM)
    try:
        child.wait(timeout=_GUARDIAN_CLEANUP_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=2)


def _drain_fd(fd: int) -> None:
    while True:
        try:
            if not os.read(fd, 4096):
                return
        except BlockingIOError:
            return


def _portable_exit_code(code: int) -> int:
    return code if code >= 0 else 128 + abs(code)


def _start_settings_server(
    config_path: Path,
    workspace: Path,
    bridge: _SettingsRestartBridge,
):
    """启动唯一的 loopback Web Shell，并等待确定的启动事件。"""

    import asyncio

    from bootstrap.web_shell import create_web_shell_server

    host = os.environ.get("AKASHIC_WEB_HOST", "127.0.0.1")
    allow_non_loopback = os.environ.get("AKASHIC_WEB_ALLOW_NON_LOOPBACK") == "1"
    if host != "127.0.0.1" and not allow_non_loopback:
        raise RuntimeError("AKASHIC_WEB_HOST 只允许 127.0.0.1")
    raw_port = os.environ.get("AKASHIC_WEB_PORT", "2236")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise RuntimeError("AKASHIC_WEB_PORT 必须是 1 到 65535 的整数") from error
    if not 1 <= port <= 65_535:
        raise RuntimeError("AKASHIC_WEB_PORT 必须是 1 到 65535 的整数")
    server = create_web_shell_server(
        config_path,
        workspace,
        host=host,
        port=port,
    )
    thread = threading.Thread(
        target=lambda: asyncio.run(server.serve()),
        name="web-shell-server",
        daemon=True,
    )
    thread.start()
    _ = server.startup_event.wait(timeout=5)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=1)
        raise RuntimeError(f"Web Shell 无法监听 {host}:{port}")
    print(f"Akashic Web 已就绪: http://{host}:{port}", flush=True)
    return server, thread
