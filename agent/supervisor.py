from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import IO
from uuid import uuid4


RESTART_EXIT_CODE = 75
SUPERVISOR_FAILURE_EXIT_CODE = 70
_BOOT_CLEANUP_TIMEOUT_SECONDS = 5.0


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
    settings_generation: int = 0


class _SettingsRestartBridge:
    """在设置 HTTP 线程与 supervisor owner 之间交接一次可等待重启。"""

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self.request_event = threading.Event()
        self._condition = threading.Condition()
        self._next_generation = 0
        self._requested_generation = 0
        self._completed: dict[int, bool] = {}

    def request_and_wait(self) -> None:
        with self._condition:
            self._next_generation += 1
            generation = self._next_generation
            self._requested_generation = generation
            self.request_event.set()
            completed = self._condition.wait_for(
                lambda: generation in self._completed,
                timeout=self.timeout_s,
            )
            if not completed:
                raise RuntimeError("Gateway 重启等待超时")
            if not self._completed.pop(generation):
                raise RuntimeError("Gateway 使用候选配置启动失败")

    def take_request(self) -> int:
        with self._condition:
            if not self.request_event.is_set():
                return 0
            generation = self._requested_generation
            self.request_event.clear()
            return generation

    def wait_request(self) -> int:
        if not self.request_event.wait(self.timeout_s):
            return 0
        return self.take_request()

    def complete(self, generation: int, success: bool) -> None:
        with self._condition:
            self._completed[generation] = success
            self._condition.notify_all()


def _cleanup_boot_processes(
    *,
    boot_id: str,
    gateway_group_id: int,
    timeout_s: float = _BOOT_CLEANUP_TIMEOUT_SECONDS,
) -> None:
    """清理指定 boot 的全部进程组，保证下一代启动前旧端口已释放。"""

    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(gateway_group_id), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode not in {0, 128}:
            raise RuntimeError(
                f"boot {boot_id} Windows 进程树清理失败: exit={result.returncode}"
            )
        return

    # 1. gateway 自身是独立组；Linux 再用唯一 boot ID 找到 MCP/service 子组
    groups = {gateway_group_id}
    direct_pids: set[int] = set()
    if _wait_boot_targets(
        boot_id,
        groups,
        direct_pids,
        signal.SIGTERM,
        timeout_s,
    ):
        return

    # 2. 宽限期结束后升级为 KILL，并以仍存活的 PID/PGID 作为失败证据
    if _wait_boot_targets(
        boot_id,
        groups,
        direct_pids,
        signal.SIGKILL,
        timeout_s,
    ):
        return
    alive_groups = sorted(group_id for group_id in groups if _group_exists(group_id))
    alive_pids = sorted(pid for pid in direct_pids if _pid_exists(pid))
    raise RuntimeError(
        f"boot {boot_id} 进程清理失败: groups={alive_groups}, pids={alive_pids}"
    )


def _wait_boot_targets(
    boot_id: str,
    groups: set[int],
    direct_pids: set[int],
    sig: signal.Signals,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        _discover_boot_targets(boot_id, groups, direct_pids)
        _signal_targets(groups, direct_pids, sig)
        if not any(_group_exists(group_id) for group_id in groups) and not any(
            _pid_exists(pid) for pid in direct_pids
        ):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _discover_boot_targets(
    boot_id: str,
    groups: set[int],
    direct_pids: set[int],
) -> None:
    if not sys.platform.startswith("linux"):
        return
    expected = f"AKASHIC_BOOT_ID={boot_id}".encode()
    own_group = os.getpgrp()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            environ = (entry / "environ").read_bytes().split(b"\0")
            if expected not in environ:
                continue
            group_id = os.getpgid(pid)
        except (OSError, ProcessLookupError):
            continue
        if group_id == own_group:
            direct_pids.add(pid)
        else:
            groups.add(group_id)


def _signal_targets(
    groups: set[int],
    direct_pids: set[int],
    sig: signal.Signals,
) -> None:
    for group_id in groups:
        try:
            os.killpg(group_id, sig)
        except ProcessLookupError:
            pass
    for pid in direct_pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _group_exists(group_id: int) -> bool:
    if sys.platform.startswith("linux"):
        return _linux_group_has_live_members(group_id)
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_exists(pid: int) -> bool:
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            return False
        return fields[0] != "Z"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _linux_group_has_live_members(group_id: int) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if len(fields) >= 3 and int(fields[2]) == group_id and fields[0] != "Z":
            return True
    return False


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
    settings_bridge = _SettingsRestartBridge(max(30.0, readiness_timeout_s * 2))
    try:
        settings_server, settings_thread = _start_settings_server(
            config_path,
            workspace,
            settings_bridge,
        )
    except BaseException:
        lock.release()
        raise
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
        report_settings_generation = 0
        while True:
            # 1. child 代际之间收到停止信号时，禁止创建下一代进程。
            if stopping_signal is not None:
                return 0
            if not config_path.exists():
                while report_settings_generation == 0 and stopping_signal is None:
                    if settings_bridge.request_event.wait(0.2):
                        report_settings_generation = settings_bridge.take_request()
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
                            start_new_session=True,
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
                try:
                    _cleanup_boot_processes(
                        boot_id=boot_id,
                        gateway_group_id=child.pid,
                    )
                except RuntimeError as error:
                    print(str(error), file=sys.stderr)
                    return SUPERVISOR_FAILURE_EXIT_CODE
                os.close(read_fd)
                child = None
                return 0
            try:
                result = _wait_child(
                    child,
                    read_fd=read_fd,
                    workspace=workspace,
                    boot_id=boot_id,
                    nonce=nonce,
                    readiness_timeout_s=readiness_timeout_s,
                    settings_bridge=settings_bridge,
                    report_settings_generation=report_settings_generation,
                )
            except BaseException as wait_error:
                try:
                    _cleanup_boot_processes(
                        boot_id=boot_id,
                        gateway_group_id=child.pid,
                    )
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "Supervisor 等待 gateway 与 boot 清理均失败",
                        [wait_error, cleanup_error],
                    ) from wait_error
                raise
            try:
                _cleanup_boot_processes(
                    boot_id=boot_id,
                    gateway_group_id=child.pid,
                )
            except RuntimeError as error:
                print(str(error), file=sys.stderr)
                if result.settings_generation:
                    settings_bridge.complete(result.settings_generation, False)
                return SUPERVISOR_FAILURE_EXIT_CODE
            if report_settings_generation:
                report_settings_generation = 0
            child = None
            if stopping_signal is not None:
                return 0
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
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        settings_server.should_exit = True
        settings_thread.join(timeout=5)
        lock.release()


def _wait_child(
    child: subprocess.Popen[bytes],
    *,
    read_fd: int,
    workspace: Path,
    boot_id: str,
    nonce: str,
    readiness_timeout_s: float,
    settings_bridge: _SettingsRestartBridge | None = None,
    report_settings_generation: int = 0,
) -> _ChildResult:
    """等待 child 退出，同时验证 readiness 与唯一 commit frame。"""

    settings_bridge = settings_bridge or _SettingsRestartBridge(readiness_timeout_s)
    readiness_path = workspace / ".runtime-ready.json"
    deadline = time.monotonic() + readiness_timeout_s
    ready = False
    settings_generation = 0
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
                    if report_settings_generation:
                        settings_bridge.complete(report_settings_generation, False)
                    return _ChildResult(
                        SUPERVISOR_FAILURE_EXIT_CODE,
                        False,
                        False,
                        report_settings_generation,
                    )
            if ready and report_settings_generation:
                settings_bridge.complete(report_settings_generation, True)
                report_settings_generation = 0
            if ready and settings_generation == 0 and settings_bridge.request_event.is_set():
                settings_generation = settings_bridge.take_request()
                child.send_signal(signal.SIGUSR2)
            time.sleep(0.02)
        _read_available(read_fd, buffer)
        if not ready:
            ready = _matches_readiness(
                readiness_path,
                boot_id=boot_id,
                pid=child.pid,
            )
        if report_settings_generation:
            settings_bridge.complete(report_settings_generation, False)
        return _ChildResult(
            child.returncode,
            ready,
            _valid_commit(bytes(buffer), boot_id=boot_id, nonce=nonce),
            settings_generation or report_settings_generation,
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
        and str(frame.get("requestId") or "").startswith(("restart_", "settings_"))
    )


def _portable_exit_code(code: int) -> int:
    if code >= 0:
        return code
    return 128 + abs(code)


def _start_settings_server(
    config_path: Path,
    workspace: Path,
    bridge: _SettingsRestartBridge,
):
    """在 supervisor 内启动固定设置服务，并确认端口实际进入监听。"""
    import asyncio

    from bootstrap.settings_api import create_settings_server

    server = create_settings_server(
        config_path,
        workspace,
        host=os.environ.get("AKASHIC_SETTINGS_HOST", "127.0.0.1"),
        on_applied=bridge.request_and_wait,
    )
    thread = threading.Thread(
        target=lambda: asyncio.run(server.serve()),
        name="settings-server",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=1)
        raise RuntimeError("设置服务无法监听 127.0.0.1:6321")
    return server, thread
