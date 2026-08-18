"""在 Linux 和 macOS 上持有单个 Gateway boot 的进程树生命周期。"""

from __future__ import annotations

import argparse
import ctypes
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

from core.common.diagnostic_log import diagnostic_line
from utils.process_group import process_group_exists
from utils.process_guard import open_process_ref, process_wait_timeout

GUARDIAN_FAILURE_EXIT_CODE = 70
_BOOT_CLEANUP_TIMEOUT_SECONDS = 5.0
_PR_SET_CHILD_SUBREAPER = 36


def run_boot_guardian(
    *,
    main_path: Path,
    config_path: Path,
    workspace: Path,
    boot_id: str,
    nonce: str,
    lifecycle_fd: int,
    lease_fd: int,
) -> int:
    """启动一个 Gateway，在退出或 lease 丢失时清空 boot 并返回状态。"""

    # 1. 建立平台支持的 orphan ownership 和信号唤醒边界。
    _enable_child_subreaper()
    signal_read_fd, signal_write_fd = os.pipe()
    os.set_blocking(signal_read_fd, False)
    os.set_blocking(signal_write_fd, False)
    previous_wakeup_fd = signal.set_wakeup_fd(signal_write_fd)
    stop_signals = (signal.SIGINT, signal.SIGTERM)
    handled_signals = (*stop_signals, signal.SIGCHLD)
    previous_handlers = {
        sig: signal.signal(sig, lambda _signum, _frame: None) for sig in handled_signals
    }

    gateway: subprocess.Popen[bytes] | None = None
    gateway_ref = None
    cleanup_attempted = False
    try:
        # 2. Guardian 不携带 boot identity；只有 Gateway 及其后代属于 boot。
        env = os.environ.copy()
        env.update(
            {
                "AKASHIC_SUPERVISED": "1",
                "AKASHIC_BOOT_ID": boot_id,
                "AKASHIC_LIFECYCLE_FD": str(lifecycle_fd),
                "AKASHIC_RESTART_NONCE": nonce,
            }
        )
        gateway = subprocess.Popen(
            [
                sys.executable,
                str(main_path),
                "gateway",
                "--config",
                str(config_path),
                "--workspace",
                str(workspace),
            ],
            cwd=main_path.parent,
            env=env,
            pass_fds=(lifecycle_fd,),
            start_new_session=True,
        )
        os.close(lifecycle_fd)
        lifecycle_fd = -1
        gateway_ref = open_process_ref(gateway.pid)

        # 3. Linux 事件驱动等待；macOS 定期轮询 Gateway 并等待其他 fd。
        with selectors.DefaultSelector() as selector:
            if gateway_ref.wait_fd is not None:
                selector.register(gateway_ref.wait_fd, selectors.EVENT_READ, "gateway")
            selector.register(lease_fd, selectors.EVENT_READ, "lease")
            selector.register(signal_read_fd, selectors.EVENT_READ, "signal")
            while gateway.poll() is None:
                events = selector.select(process_wait_timeout(gateway_ref, None))
                if any(key.data == "gateway" for key, _mask in events):
                    break
                if any(key.data == "lease" for key, _mask in events):
                    if os.read(lease_fd, 1) == b"":
                        break
                    raise RuntimeError("Supervisor lease 收到未知数据")
                if any(key.data == "signal" for key, _mask in events):
                    received = _drain_signal_fd(signal_read_fd)
                    if signal.SIGCHLD in received:
                        _reap_adopted_children(exclude_pids={gateway.pid})
                    if received.intersection(stop_signals):
                        break

        # 4. 无论退出来源如何，都在返回前证明当前 boot 已清空。
        gateway_exit_code = gateway.poll()
        cleanup_attempted = True
        _ = _cleanup_boot_processes_best_effort(
            boot_id=boot_id,
            gateway_group_id=gateway.pid,
            owner="guardian",
        )
        if gateway_exit_code is None:
            gateway_exit_code = gateway.wait()
        _reap_adopted_children()
        return _portable_exit_code(gateway_exit_code)
    except BaseException:
        if gateway is not None and not cleanup_attempted:
            _ = _cleanup_boot_processes_best_effort(
                boot_id=boot_id,
                gateway_group_id=gateway.pid,
                owner="guardian",
            )
        raise
    finally:
        if lifecycle_fd >= 0:
            os.close(lifecycle_fd)
        if gateway_ref is not None:
            gateway_ref.close()
        os.close(lease_fd)
        _ = signal.set_wakeup_fd(previous_wakeup_fd)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        os.close(signal_read_fd)
        os.close(signal_write_fd)


def _enable_child_subreaper() -> None:
    """Linux 成为 child subreaper；macOS 依赖独立进程组清理。"""

    if sys.platform == "darwin":
        return
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Boot Guardian 仅支持 Linux 和 macOS")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _cleanup_boot_processes(
    *,
    boot_id: str,
    gateway_group_id: int | None,
    timeout_s: float = _BOOT_CLEANUP_TIMEOUT_SECONDS,
) -> None:
    """在一个 TERM 到 KILL 总 deadline 内清空当前 boot。"""

    if not _guardian_platform_supported():
        raise RuntimeError("boot 进程树清理仅支持 Linux 和 macOS")
    if timeout_s <= 0:
        raise ValueError("boot cleanup timeout 必须大于 0")

    # 1. TERM 使用总预算的八成，为 KILL 与最终验证保留固定余量。
    groups = {gateway_group_id} if gateway_group_id is not None else set()
    direct_pids: set[int] = set()
    started_at = time.monotonic()
    deadline = started_at + timeout_s
    term_deadline = started_at + timeout_s * 0.8
    if _wait_boot_targets(
        boot_id,
        groups,
        direct_pids,
        signal.SIGTERM,
        term_deadline,
    ):
        return

    # 2. 剩余预算只用于 KILL 与证明没有非 zombie 目标。
    if _wait_boot_targets(
        boot_id,
        groups,
        direct_pids,
        signal.SIGKILL,
        deadline,
    ):
        return
    alive_groups = sorted(group_id for group_id in groups if _group_exists(group_id))
    alive_pids = sorted(pid for pid in direct_pids if _pid_exists(pid))
    raise RuntimeError(
        f"boot {boot_id} 进程清理失败: groups={alive_groups}, pids={alive_pids}"
    )


def _cleanup_boot_processes_best_effort(
    *,
    boot_id: str,
    gateway_group_id: int | None,
    owner: str,
) -> bool:
    """尽力清理 boot；权限或残留失败只输出结构化诊断。"""

    try:
        _cleanup_boot_processes(
            boot_id=boot_id,
            gateway_group_id=gateway_group_id,
        )
    except (OSError, RuntimeError) as exc:
        print(
            diagnostic_line(
                "BootCleanup",
                event="cleanup_degraded",
                flow="runtime",
                phase="boot_cleanup",
                action="continue",
                reason="boot_cleanup_unconfirmed",
                error_type=type(exc).__name__,
                note=f"owner={owner} boot_id={boot_id} error={exc}",
            ),
            file=sys.stderr,
        )
        return False
    return True


def _wait_boot_targets(
    boot_id: str,
    groups: set[int],
    direct_pids: set[int],
    sig: signal.Signals,
    deadline: float,
) -> bool:
    while True:
        _discover_boot_targets(boot_id, groups, direct_pids)
        _signal_targets(groups, direct_pids, sig)
        if not any(_group_exists(group_id) for group_id in groups) and not any(
            _pid_exists(pid) for pid in direct_pids
        ):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def _discover_boot_targets(
    boot_id: str,
    groups: set[int],
    direct_pids: set[int],
) -> None:
    expected = f"AKASHIC_BOOT_ID={boot_id}".encode()
    own_group = os.getpgrp()
    own_pid = os.getpid()
    if sys.platform.startswith("linux"):
        candidates = (
            int(entry.name)
            for entry in Path("/proc").iterdir()
            if entry.name.isdigit()
        )
    else:
        candidates = iter(_darwin_process_ids())
    for pid in candidates:
        try:
            if sys.platform.startswith("linux"):
                environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            else:
                environ = _darwin_process_environ(pid)
            if expected not in environ:
                continue
            group_id = os.getpgid(pid)
        except (OSError, ProcessLookupError):
            continue
        if pid == own_pid:
            raise RuntimeError("Boot Guardian 不得拥有 Gateway boot identity")
        if group_id == own_group:
            direct_pids.add(pid)
        else:
            groups.add(group_id)


def _pid_has_boot_identity(pid: int, boot_id: str) -> bool:
    """判断一个活进程是否携带精确的 boot token。"""

    expected = f"AKASHIC_BOOT_ID={boot_id}".encode()
    try:
        if sys.platform.startswith("linux"):
            environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        else:
            environ = _darwin_process_environ(pid)
        return expected in environ
    except OSError:
        return False


def _signal_targets(
    groups: set[int],
    direct_pids: set[int],
    sig: signal.Signals,
) -> None:
    own_group = os.getpgrp()
    if own_group in groups:
        raise RuntimeError("拒绝向 lifecycle owner 所在进程组发送清理信号")
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
    return process_group_exists(group_id)


def _pid_exists(pid: int) -> bool:
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return False
        state = result.stdout.strip()
        return result.returncode == 0 and bool(state) and not state.startswith("Z")
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return False
    return fields[0] != "Z"


def _drain_signal_fd(fd: int) -> set[int]:
    received: set[int] = set()
    while True:
        try:
            chunk = os.read(fd, 4096)
            if not chunk:
                return received
            received.update(chunk)
        except BlockingIOError:
            return received


def _reap_adopted_children(*, exclude_pids: set[int] | None = None) -> None:
    """收割 adopted zombie；运行期间保留由 Popen 持有的直接 child。"""

    excluded = exclude_pids or set()
    if excluded:
        if sys.platform == "darwin":
            return
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in excluded:
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            except (OSError, IndexError):
                continue
            if len(fields) < 2 or fields[0] != "Z" or int(fields[1]) != os.getpid():
                continue
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                continue
        return

    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _guardian_platform_supported(platform: str | None = None) -> bool:
    current = platform or sys.platform
    return current.startswith("linux") or current == "darwin"


def _darwin_process_ids() -> list[int]:
    libc = ctypes.CDLL(None, use_errno=True)
    list_all_pids = libc.proc_listallpids
    list_all_pids.argtypes = (ctypes.c_void_p, ctypes.c_int)
    list_all_pids.restype = ctypes.c_int
    count = list_all_pids(None, 0)
    if count < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    buffer = (ctypes.c_int * (count + 32))()
    actual = list_all_pids(buffer, ctypes.sizeof(buffer))
    if actual < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return [pid for pid in buffer[:actual] if pid > 0]


def _darwin_process_environ(pid: int) -> list[bytes]:
    """通过 KERN_PROCARGS2 读取 Darwin 进程的原始环境变量。"""

    if sys.platform != "darwin":
        raise RuntimeError("KERN_PROCARGS2 仅支持 macOS")
    ctl_kern = 1
    kern_procargs2 = 49
    mib = (ctypes.c_int * 3)(ctl_kern, kern_procargs2, pid)
    size = ctypes.c_size_t()
    libc = ctypes.CDLL(None, use_errno=True)
    sysctl = libc.sysctl
    sysctl.argtypes = (
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    sysctl.restype = ctypes.c_int
    if sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    buffer = ctypes.create_string_buffer(size.value)
    if sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))

    raw = buffer.raw[: size.value]
    integer_size = ctypes.sizeof(ctypes.c_int)
    if len(raw) < integer_size:
        return []
    argc = int.from_bytes(raw[:integer_size], sys.byteorder, signed=True)
    values = raw[integer_size:].split(b"\0")
    index = 1  # executable path
    while index < len(values) and not values[index]:
        index += 1
    index += max(0, argc)
    return [value for value in values[index:] if b"=" in value]


def _portable_exit_code(code: int) -> int:
    return code if code >= 0 else 128 + abs(code)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--main-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--boot-id", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--lifecycle-fd", type=int, required=True)
    parser.add_argument("--lease-fd", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        return run_boot_guardian(
            main_path=args.main_path.resolve(),
            config_path=args.config.resolve(),
            workspace=args.workspace.resolve(),
            boot_id=args.boot_id,
            nonce=args.nonce,
            lifecycle_fd=args.lifecycle_fd,
            lease_fd=args.lease_fd,
        )
    except BaseException as error:
        print(f"Boot Guardian 失败: {error}", file=sys.stderr)
        return GUARDIAN_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
