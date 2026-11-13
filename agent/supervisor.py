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
    open_process_ref,
    process_wait_timeout,
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
            or not request_id
            or request_id.strip() != request_id
            or not secrets.compare_digest(nonce, self.nonce)
        ):
            self.protocol_error = "lifecycle commit frame 无效或重复"
            return
        self.commit_valid = True


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
    # 1. 建立 workspace owner、Web Shell 线程和停止信号边界。
    _enable_child_subreaper()
    lock = _SupervisorLock(workspace)
    lock.acquire()
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
        web_shell_server, web_shell_thread = _start_web_shell_server(workspace)
    except BaseException:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        os.close(signal_read_fd)
        os.close(signal_write_fd)
        lock.release()
        raise

    try:
        # 2. 每一代只创建一个 Guardian，并在上一代清空后进入下一代。
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
                    boot_id=boot_id,
                    nonce=nonce,
                    readiness_timeout_s=readiness_timeout_s,
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
            if stopping_signal is not None:
                return 0
            if result.protocol_error:
                print(result.protocol_error, file=sys.stderr)
                return SUPERVISOR_FAILURE_EXIT_CODE
            if result.exit_code != RESTART_EXIT_CODE:
                return _portable_exit_code(result.exit_code)
            if not result.ready or not result.commit_valid:
                return SUPERVISOR_FAILURE_EXIT_CODE
    finally:
        if guardian is not None:
            _stop_guardian(guardian)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        web_shell_server.should_exit = True
        web_shell_thread.join(timeout=5)
        os.close(signal_read_fd)
        os.close(signal_write_fd)
        lock.release()


def _wait_child(
    child: subprocess.Popen[bytes],
    *,
    read_fd: int,
    lease_fd: int | None = None,
    boot_id: str,
    nonce: str,
    readiness_timeout_s: float,
) -> _ChildResult:
    """等待进程与生命周期事件，直到 Guardian 退出或启动失败。"""

    # 1. 建立本代协议状态和两个可等待的内核事件。
    state = _LifecycleState(boot_id, nonce)
    deadline = time.monotonic() + readiness_timeout_s
    gateway_verified = False
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
                        else:
                            state.feed(b"", eof=True)
                            selector.unregister(read_fd)
                            lifecycle_open = False

                if state.protocol_error:
                    _stop_guardian(child)
                    break
                if state.ready and not gateway_verified:
                    assert state.gateway_pid is not None
                    if not _pid_has_boot_identity(state.gateway_pid, boot_id):
                        state.protocol_error = (
                            "ready PID 不属于当前 boot，拒绝取得信号权限"
                        )
                        state.commit_valid = False
                        _stop_guardian(child)
                        break
                    gateway_verified = True

            if lifecycle_open and not state.protocol_error:
                _read_lifecycle_to_eof(read_fd, state)
        # 3. 只返回已解析的私有证据；调用方负责最终 boot 空集验证。
        return _ChildResult(
            (
                child.returncode
                if child.returncode is not None
                else SUPERVISOR_FAILURE_EXIT_CODE
            ),
            state.ready,
            state.commit_valid,
            state.protocol_error,
        )
    finally:
        if lease_fd is not None:
            os.close(lease_fd)
        child_ref.close()
        os.close(read_fd)


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


def _portable_exit_code(code: int) -> int:
    return code if code >= 0 else 128 + abs(code)


def _start_web_shell_server(workspace: Path):
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
    # Supervisor 的外壳先于 Core 监听；Core 就绪前不放行未经守卫的请求。
    server = create_web_shell_server(
        workspace,
        host=host,
        port=port,
        wait_for_ready=True,
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
