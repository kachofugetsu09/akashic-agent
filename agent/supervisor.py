from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import IO
from uuid import uuid4


RESTART_EXIT_CODE = 75
SUPERVISOR_FAILURE_EXIT_CODE = 70


class _SupervisorLock:
    """保证一个 workspace 只有一个 supervisor。"""

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


def run_supervisor(
    *,
    config_path: Path,
    workspace: Path,
    readiness_timeout_s: float = 15.0,
) -> int:
    """监管固定 gateway child，并只接受当前 boot 的私有重启提交。"""

    if readiness_timeout_s <= 0:
        raise ValueError("readiness_timeout_s 必须大于 0")
    project_root = Path(__file__).resolve().parent.parent
    main_path = project_root / "main.py"
    config_path = config_path.expanduser().resolve()
    workspace = workspace.expanduser().resolve()
    lock = _SupervisorLock(workspace)
    lock.acquire()
    stopping_signal: int | None = None
    child: subprocess.Popen[bytes] | None = None
    forwarded_stop: tuple[subprocess.Popen[bytes], int] | None = None
    stop_signals = {signal.SIGINT, signal.SIGTERM}

    def forward_stop(signum: int, _frame: FrameType | None) -> None:
        nonlocal stopping_signal, forwarded_stop
        stopping_signal = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)
            forwarded_stop = (child, signum)

    previous_handlers = {
        sig: signal.signal(sig, forward_stop)
        for sig in stop_signals
    }
    try:
        while True:
            # 1. child 代际之间收到停止信号时，禁止创建下一代进程。
            if stopping_signal is not None:
                return 0
            boot_id = uuid4().hex
            nonce = secrets.token_hex(32)
            read_fd, write_fd = os.pipe()
            os.set_blocking(read_fd, False)
            env = os.environ.copy()
            env.update(
                {
                    "AKASHIC_SUPERVISED": "1",
                    "AKASHIC_BOOT_ID": boot_id,
                    "AKASHIC_RESTART_COMMIT_FD": str(write_fd),
                    "AKASHIC_RESTART_NONCE": nonce,
                }
            )
            argv = [
                sys.executable,
                str(main_path),
                "gateway",
                "--config",
                str(config_path),
                "--workspace",
                str(workspace),
            ]
            # 2. 屏蔽 stop handler，直到 Popen 返回且 child 所有权已建立。
            spawn_blocked = False
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                stop_signals,
            )
            try:
                pending_stops = signal.sigpending() & stop_signals
                if stopping_signal is not None or pending_stops:
                    spawn_blocked = True
                else:
                    def restore_child_signal_mask() -> None:
                        signal.pthread_sigmask(
                            signal.SIG_SETMASK,
                            previous_mask,
                        )

                    try:
                        child = subprocess.Popen(
                            argv,
                            cwd=project_root,
                            env=env,
                            pass_fds=(write_fd,),
                            # supervisor 单线程；exec 前只恢复继承的 signal mask。
                            preexec_fn=restore_child_signal_mask,
                        )
                    except BaseException:
                        os.close(read_fd)
                        raise
            finally:
                _ = signal.pthread_sigmask(
                    signal.SIG_SETMASK,
                    previous_mask,
                )
                os.close(write_fd)
            if spawn_blocked:
                os.close(read_fd)
                return 0

            # 3. pending signal 恢复时 child 已有唯一 owner，可精确转发并收束。
            assert child is not None
            if stopping_signal is not None:
                if (
                    child.poll() is None
                    and forwarded_stop != (child, stopping_signal)
                ):
                    child.send_signal(stopping_signal)
                child.wait()
                os.close(read_fd)
                child = None
                return 0
            result = _wait_child(
                child,
                read_fd=read_fd,
                workspace=workspace,
                boot_id=boot_id,
                nonce=nonce,
                readiness_timeout_s=readiness_timeout_s,
            )
            child = None
            if stopping_signal is not None:
                return 0
            if result.exit_code != RESTART_EXIT_CODE:
                return _portable_exit_code(result.exit_code)
            if not result.ready or not result.commit_valid:
                return SUPERVISOR_FAILURE_EXIT_CODE
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        lock.release()


def _wait_child(
    child: subprocess.Popen[bytes],
    *,
    read_fd: int,
    workspace: Path,
    boot_id: str,
    nonce: str,
    readiness_timeout_s: float,
) -> _ChildResult:
    """等待 child 退出，同时验证 readiness 与唯一 commit frame。"""

    readiness_path = workspace / ".runtime-ready.json"
    deadline = time.monotonic() + readiness_timeout_s
    ready = False
    buffer = bytearray()
    try:
        while child.poll() is None:
            _read_available(read_fd, buffer)
            if not ready:
                ready = _matches_readiness(
                    readiness_path,
                    boot_id=boot_id,
                    pid=child.pid,
                )
                if not ready and time.monotonic() >= deadline:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)
                    return _ChildResult(
                        SUPERVISOR_FAILURE_EXIT_CODE,
                        False,
                        False,
                    )
            time.sleep(0.02)
        _read_available(read_fd, buffer)
        if not ready:
            ready = _matches_readiness(
                readiness_path,
                boot_id=boot_id,
                pid=child.pid,
            )
        return _ChildResult(
            child.returncode,
            ready,
            _valid_commit(bytes(buffer), boot_id=boot_id, nonce=nonce),
        )
    finally:
        os.close(read_fd)


def _read_available(fd: int, buffer: bytearray) -> None:
    while True:
        try:
            chunk = os.read(fd, 4096)
        except BlockingIOError:
            return
        if not chunk:
            return
        buffer.extend(chunk)


def _matches_readiness(path: Path, *, boot_id: str, pid: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload == {"bootId": boot_id, "pid": pid, "state": "ready"}


def _valid_commit(payload: bytes, *, boot_id: str, nonce: str) -> bool:
    lines = [line for line in payload.splitlines() if line]
    if len(lines) != 1:
        return False
    try:
        frame = json.loads(lines[0])
    except json.JSONDecodeError:
        return False
    return (
        isinstance(frame, dict)
        and frame.get("type") == "restart_commit"
        and frame.get("bootId") == boot_id
        and secrets.compare_digest(str(frame.get("nonce") or ""), nonce)
        and str(frame.get("requestId") or "").startswith("restart_")
    )


def _portable_exit_code(code: int) -> int:
    if code >= 0:
        return code
    return 128 + abs(code)
