from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from agent.plugin_composition.model import ServiceKey

class RestartRejectedError(RuntimeError):
    """表示当前 runtime 明确拒绝了一次重启请求。"""


class RestartPendingError(RestartRejectedError):
    """表示重启已等待提交，暂时不接纳新的外部 Root。"""


@dataclass(frozen=True, slots=True)
class ExternalRootPermit:
    """一次外部 Root 接纳；释放后才允许提交重启。"""

    _gate: "RestartGate"
    request_id: str
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        object.__setattr__(self, "_released", True)
        self._gate._release(self.request_id)

    def child(self) -> "ExternalRootPermit":
        """取得由当前已接纳 Root 明确转交给外部效果的子 permit。"""
        if self._released:
            raise RestartRejectedError("已释放的 Root permit 不能派生子 permit")
        return self._gate._acquire_child(self.request_id)


class RestartGate:
    """Core 拥有的重启闸门；只统计外部 Root permit。"""

    def __init__(
        self,
        *,
        boot_id: str,
        supervised: bool,
        commit: Callable[[str], None] | None = None,
        drain_timeout_s: float = 15.0,
        execution_enabled: bool = True,
    ) -> None:
        if not boot_id.strip() or drain_timeout_s <= 0:
            raise ValueError("restart gate 参数无效")
        if not isinstance(execution_enabled, bool):
            raise TypeError("restart gate execution_enabled 必须是 bool")
        if supervised and commit is None and execution_enabled:
            raise ValueError("supervised 与 restart commit channel 必须同时成立")
        if (not supervised or not execution_enabled) and commit is not None:
            raise ValueError("restart commit channel 只能属于可执行的 supervised gate")
        self.boot_id = boot_id
        self.supervised = supervised
        self.execution_enabled = execution_enabled
        self._commit = commit
        self._drain_timeout_s = drain_timeout_s
        self._request_id: str | None = None
        self._permits = 0
        self._accepting = True
        self._drained = asyncio.Event()
        self._drained.set()
        self._open_changed = asyncio.Event()
        self._committed = False

    @property
    def accepting(self) -> bool:
        return self._accepting

    @property
    def permit_count(self) -> int:
        return self._permits

    def check_open(self) -> None:
        """在持久接纳新 work 前核对 Core 的 admission 状态。"""
        if not self._accepting:
            raise RestartPendingError("runtime 正在等待重启，暂不接纳新 Root")

    async def wait_until_open(self) -> None:
        """等待一次 abort/reopen 通知，不保存任何来源或 Session 状态。"""
        while not self._accepting:
            changed = self._open_changed
            await changed.wait()

    def prepare(self, request_id: str) -> None:
        """立即关闭新外部 Root 接纳，并只保留 opaque request id。"""
        if not self.supervised:
            raise RestartRejectedError("当前进程未由 supervisor 托管")
        if not self.execution_enabled:
            raise RestartRejectedError("当前 runtime 不允许重启效果")
        if not request_id or request_id.strip() != request_id:
            raise ValueError("restart request id 无效")
        if self._request_id is not None:
            if self._request_id == request_id:
                return
            raise RestartRejectedError("已有重启请求等待提交")
        self._request_id = request_id
        self._accepting = False

    def acquire(self) -> ExternalRootPermit:
        """取得一个新外部 Root permit；prepare 后立即拒绝。"""
        self.check_open()
        self._permits += 1
        self._drained.clear()
        return ExternalRootPermit(self, self._request_id or "")

    def _acquire_child(self, request_id: str) -> ExternalRootPermit:
        """由已持有 Root 显式派生一个外部效果 permit。"""
        if request_id not in {"", self._request_id or ""}:
            raise RestartRejectedError("Root permit 不属于当前 gate")
        self._permits += 1
        self._drained.clear()
        return ExternalRootPermit(self, request_id)

    def _release(self, request_id: str) -> None:
        if self._permits <= 0:
            raise RuntimeError("restart Root permit 重复释放")
        if request_id not in {"", self._request_id or ""}:
            raise RuntimeError("restart Root permit 不属于当前 gate")
        self._permits -= 1
        if self._permits == 0:
            self._drained.set()

    async def commit(self, request_id: str) -> None:
        """等待既有外部 Root 排空后向 Supervisor 提交 opaque id。"""
        if not self.execution_enabled:
            raise RestartRejectedError("当前 runtime 不允许重启效果")
        if request_id != self._request_id:
            raise RestartRejectedError("restart request id 不属于当前 gate")
        if self._committed:
            return
        if self._commit is None:
            raise RestartRejectedError("当前进程没有 restart commit channel")
        try:
            async with asyncio.timeout(self._drain_timeout_s):
                await self._drained.wait()
            self._commit(request_id)
        except BaseException:
            self.abort(request_id)
            raise
        self._committed = True

    def abort(self, request_id: str) -> None:
        """取消一次准备并恢复新 Root 接纳。"""
        if request_id != self._request_id:
            return
        self._request_id = None
        self._accepting = True
        self._committed = False
        changed = self._open_changed
        self._open_changed = asyncio.Event()
        changed.set()


RESTART_GATE = ServiceKey[RestartGate]("core.restart_gate.v1")


class SupervisorCommitChannel:
    """向 Supervisor 的私有管道发布当前 boot 生命周期事件。"""

    def __init__(self, fd: int, boot_id: str, nonce: str) -> None:
        if fd <= 2:
            raise ValueError("lifecycle fd 必须是继承的私有描述符")
        if not boot_id or len(nonce) < 32:
            raise ValueError("lifecycle channel 身份无效")
        os.fstat(fd)
        self.fd = fd
        self.boot_id = boot_id
        self.nonce = nonce
        self._started_at = time.monotonic()

    @classmethod
    def from_environment(cls) -> SupervisorCommitChannel | None:
        supervised = os.environ.get("AKASHIC_SUPERVISED") == "1"
        if not supervised:
            return None
        raw_fd = os.environ.get("AKASHIC_LIFECYCLE_FD")
        boot_id = os.environ.get("AKASHIC_BOOT_ID", "")
        nonce = os.environ.get("AKASHIC_RESTART_NONCE", "")
        if raw_fd is None:
            raise RuntimeError("supervised child 缺少 lifecycle fd")
        try:
            fd = int(raw_fd)
        except ValueError as exc:
            raise RuntimeError("lifecycle fd 不是整数") from exc
        return cls(fd, boot_id, nonce)

    def commit_opaque(self, request_id: str) -> None:
        """提交调用方生成、Core 与 supervisor 都不解释的 request id。"""
        if not request_id or request_id.strip() != request_id:
            raise ValueError("opaque restart request id 无效")
        self._write_commit(request_id)

    def settings_reloaded(self, *, success: bool, detail: str = "") -> None:
        """Report one settings reload result without committing a restart."""

        self._write_frame(
            {
                "type": "settings_reloaded",
                "bootId": self.boot_id,
                "success": bool(success),
                "detail": detail[:1024],
            }
        )

    def stage(self, name: str) -> None:
        """发布可诊断但不能延长启动 deadline 的阶段事件。"""

        clean_name = name.strip()
        if not clean_name:
            raise ValueError("lifecycle stage 不能为空")
        self._write_frame(
            {
                "type": "stage",
                "bootId": self.boot_id,
                "stage": clean_name,
                "elapsedMs": int((time.monotonic() - self._started_at) * 1000),
            }
        )

    def ready(self, pid: int) -> None:
        """发布当前 Gateway 已完成全部启动阶段。"""

        if pid <= 0:
            raise ValueError("ready pid 必须大于 0")
        self._write_frame(
            {
                "type": "ready",
                "bootId": self.boot_id,
                "pid": pid,
            }
        )

    def _write_commit(self, request_id: str) -> None:
        self._write_frame(
            {
                "type": "commit",
                "bootId": self.boot_id,
                "nonce": self.nonce,
                "requestId": request_id,
            }
        )

    def _write_frame(self, frame: dict[str, object]) -> None:
        payload = (json.dumps(frame, separators=(",", ":")) + "\n").encode("utf-8")
        if len(payload) > 4096:
            raise RuntimeError("lifecycle frame 超过 PIPE_BUF 安全上限")
        written = os.write(self.fd, payload)
        if written != len(payload):
            raise RuntimeError("lifecycle pipe 发生短写")
