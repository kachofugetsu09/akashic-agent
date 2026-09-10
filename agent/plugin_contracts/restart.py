"""重启闸门的公开结构合同。

插件需要声明 `core.restart_gate.v1` 依赖、检查闸门是否接纳、申请/释放外部
Root permit，以及区分「明确拒绝」的重启错误。这些是插件与 Core 之间的结构
边界，因此本模块拥有异常类型与 `ExternalRootPermit` 值对象，并用 Protocol
描述闸门对插件可见的方法子集。

闸门的实现（状态机、commit channel、drain）留在 `agent.restart`；它按结构
满足 `RestartGate` Protocol，不需要继承。`_PermitOwner` 只是 permit 释放时
回调实现用的私有钩子，不是给插件依赖的接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class RestartRejectedError(RuntimeError):
    """表示当前 runtime 明确拒绝了一次重启请求。"""


class RestartPendingError(RestartRejectedError):
    """表示重启已等待提交，暂时不接纳新的外部 Root。"""


class _PermitOwner(Protocol):
    """permit 生命周期回调用；由 `agent.restart` 的闸门实现提供。"""

    def _release(self, request_id: str) -> None: ...

    def _acquire_child(self, request_id: str) -> ExternalRootPermit: ...


@dataclass(frozen=True, slots=True)
class ExternalRootPermit:
    """一次外部 Root 接纳；释放后才允许提交重启。"""

    _gate: _PermitOwner
    request_id: str
    _released: bool = False

    def release(self) -> None:
        """标记本次外部 Root 已结束；重复释放是 no-op。"""

        if self._released:
            return
        object.__setattr__(self, "_released", True)
        self._gate._release(self.request_id)

    def child(self) -> ExternalRootPermit:
        """取得由当前已接纳 Root 明确转交给外部效果的子 permit。"""

        if self._released:
            raise RestartRejectedError("已释放的 Root permit 不能派生子 permit")
        return self._gate._acquire_child(self.request_id)


@runtime_checkable
class RestartGate(Protocol):
    """插件可见的重启闸门方法子集。"""

    boot_id: str
    supervised: bool
    execution_enabled: bool

    @property
    def accepting(self) -> bool:
        """当前是否仍接纳新的外部 Root。"""
        ...

    @property
    def permit_count(self) -> int:
        """未释放的外部 Root 数量。"""
        ...

    def check_open(self) -> None:
        """不接纳时抛出 RestartRejectedError。"""
        ...

    async def wait_until_open(self) -> None:
        """等待闸门重新接纳。"""
        ...

    def prepare(self, request_id: str) -> None:
        """预占一次重启请求。"""
        ...

    def acquire(self) -> ExternalRootPermit:
        """接纳一个新的外部 Root。"""
        ...

    async def commit(self, request_id: str) -> None:
        """提交重启。"""
        ...

    def abort(self, request_id: str) -> None:
        """放弃一次重启请求。"""
        ...
