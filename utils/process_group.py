"""为本地子进程提供可重试的进程组终止语义。"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_RUNTIME_IDENTITY_ENV = ("AKASHIC_BOOT_ID", "AKASHIC_SUPERVISED")


def process_group_spawn_kwargs() -> dict[str, Any]:
    """返回让子进程成为独立进程组 owner 的平台参数。"""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def owned_process_env(overrides: dict[str, str]) -> dict[str, str]:
    """合并子进程环境，并保留 Supervisor 拥有的 boot identity。"""
    env = {**os.environ, **overrides}
    for name in _RUNTIME_IDENTITY_ENV:
        if name in os.environ:
            env[name] = os.environ[name]
    return env


@dataclass
class OwnedProcessGroup:
    """终止一个已取得 ownership 的子进程组，并等待全部成员退出。"""

    process: asyncio.subprocess.Process
    group_id: int | None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @classmethod
    def from_process(cls, process: asyncio.subprocess.Process) -> "OwnedProcessGroup":
        pid = getattr(process, "pid", None)
        return cls(process=process, group_id=pid if isinstance(pid, int) else None)

    async def terminate(self, *, timeout_s: float) -> None:
        """先发送 TERM，超时后发送 KILL，并完成 direct child wait。"""
        async with self._lock:
            if os.name == "nt" and self.group_id is not None:
                await self._terminate_windows()
                return
            if os.name != "nt" and self.group_id is not None:
                await self._terminate_posix(timeout_s=timeout_s, force=False)
                return
            await self._terminate_direct(timeout_s=timeout_s, force=False)

    async def kill(self, *, timeout_s: float) -> None:
        """立即杀死整个进程组，并完成 direct child wait。"""
        async with self._lock:
            if os.name == "nt" and self.group_id is not None:
                await self._terminate_windows()
                return
            if os.name != "nt" and self.group_id is not None:
                await self._terminate_posix(timeout_s=timeout_s, force=True)
                return
            await self._terminate_direct(timeout_s=timeout_s, force=True)

    async def _terminate_windows(self) -> None:
        """使用 taskkill /T /F 回收 Windows direct child 及其后代。"""
        assert self.group_id is not None
        taskkill = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(self.group_id),
            "/T",
            "/F",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        exit_code = await taskkill.wait()
        if exit_code != 0 and getattr(self.process, "returncode", None) is None:
            self.process.kill()
        _ = await self.process.wait()

    async def _terminate_posix(self, *, timeout_s: float, force: bool) -> None:
        """按稳定 PGID 清理 Unix 进程组，即使 leader 已经退出。"""

        # 1. 对完整进程组发信号，不能用 leader.returncode 推断后代已退出
        assert self.group_id is not None
        first_signal = signal.SIGKILL if force else signal.SIGTERM
        _signal_posix_group(self.group_id, first_signal)
        if await _wait_posix_group_exit(self.group_id, timeout_s):
            _ = await self.process.wait()
            return

        # 2. TERM 宽限期结束后升级为 KILL，并验证进程组确实消失
        if not force:
            _signal_posix_group(self.group_id, signal.SIGKILL)
            if await _wait_posix_group_exit(self.group_id, timeout_s):
                _ = await self.process.wait()
                return
        raise RuntimeError(f"进程组 {self.group_id} 在 SIGKILL 后仍未退出")

    async def _terminate_direct(self, *, timeout_s: float, force: bool) -> None:
        """为无 PGID 的测试替身和非 Unix 平台保留 direct child 回收。"""
        if getattr(self.process, "returncode", None) is not None:
            _ = await self.process.wait()
            return
        try:
            if force:
                self.process.kill()
            else:
                self.process.terminate()
        except ProcessLookupError:
            pass
        try:
            _ = await asyncio.wait_for(self.process.wait(), timeout=timeout_s)
        except TimeoutError:
            if force:
                raise RuntimeError("子进程在 SIGKILL 后仍未退出")
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
            _ = await self.process.wait()


def _signal_posix_group(group_id: int, sig: signal.Signals) -> None:
    try:
        os.killpg(group_id, sig)
    except ProcessLookupError:
        pass


async def _wait_posix_group_exit(group_id: int, timeout_s: float) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while process_group_exists(group_id):
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(0.05, remaining))
    return True


def process_group_exists(group_id: int) -> bool:
    """只在进程组仍有活成员时返回 True。"""

    if sys.platform.startswith("linux"):
        return _linux_group_has_live_members(group_id)
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _linux_group_has_live_members(group_id: int) -> bool:
    """将只剩 zombie 的组视为已释放，避免容器 PID 1 延迟回收误报。"""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        command_end = stat.rfind(")")
        if command_end < 0:
            continue
        fields = stat[command_end + 2 :].split()
        if len(fields) < 3 or int(fields[2]) != group_id:
            continue
        if fields[0] != "Z":
            return True
    return False
