"""Root 的来源接纳许可；连接及具体停止协议由 provider 持有。"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from agent.plugin_composition.context import Context
from agent.plugin_composition.model import CompositionError, ServiceKey

if TYPE_CHECKING:
    from agent.plugins.snapshot import RuntimeSnapshotLease, RuntimeSnapshotStore


class SourceAdmission:
    """把来源许可绑定实际 Root，排空超时只恢复尚未释放的原实例。"""

    def __init__(self, ctx: Context, store: RuntimeSnapshotStore, *, boot_id: str, candidate: bool) -> None:
        self._root_token = ctx.root_instance_token
        self._store = store
        self.boot_id = boot_id
        self._candidate = candidate
        self.validation = candidate
        self._watchers: dict[object, tuple[Callable[[], None], Callable[[], None]]] = {}

    async def watch(self, ctx: Context, *, close: Callable[[], None], open: Callable[[], None]) -> None:
        """关闭回调归贡献 Scope；资源退出前不能解除接纳 owner。"""
        if ctx.root_instance_token is not self._root_token:
            raise CompositionError("SOURCE_ROOT_MISMATCH", "来源接纳不属于当前 Root")
        token = object()

        def setup() -> Callable[[], None]:
            self._watchers[token] = (close, open)

            def remove() -> None:
                close()
                del self._watchers[token]

            return remove

        await ctx.effect(setup, label="source-admission")

    def require_starting(self, ctx: Context) -> None:
        """只允许实际 closed 生命周期任务初始化正式来源。"""
        from agent.plugins.snapshot import get_current_runtime_lease

        lease = get_current_runtime_lease()
        if self._candidate:
            raise RuntimeError("候选装配不能启动正式来源")
        if (ctx.root_instance_token is not self._root_token or lease is None
                or lease.snapshot.composition_root is None
                or lease.snapshot.composition_root.instance_token is not self._root_token
                or lease.snapshot.accepting_leases):
            raise RuntimeError("来源初始化需要当前 Root 的 closed scope")

    def current_lease(self) -> RuntimeSnapshotLease:
        """返回当前 Task 绑定且属于本 Root 的 lease；缺席或跨 Root 时拒绝。"""

        from agent.plugins.snapshot import get_current_runtime_lease

        lease = get_current_runtime_lease()
        if (lease is None or lease.snapshot.composition_root is None
                or lease.snapshot.composition_root.instance_token is not self._root_token):
            raise RuntimeError("当前 runtime scope lease 不属于本 Root")
        return lease

    def lease(self, snapshot_id: str) -> RuntimeSnapshotLease:
        """入口只能取得本 Root 的公开 lease，不能借用恢复特权。"""
        current = self._store.current
        if (current is None or current.snapshot_id != snapshot_id
                or current.composition_root is None
                or current.composition_root.instance_token is not self._root_token):
            raise RuntimeError("来源 lease 不属于当前 Root")
        return self._store.lease(snapshot_id)

    def close(self) -> None:
        """全部来源都尝试关闭；失败保留回调及实际资源。"""
        errors: list[BaseException] = []
        for close, _ in tuple(self._watchers.values()):
            try:
                close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("来源接纳关闭失败", errors)

    def open(self) -> None:
        """整组开放失败即关闭已开放来源，不释放任何资源。"""
        try:
            for _, open in tuple(self._watchers.values()):
                open()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup("来源开放及关闭失败", [error, cleanup]) from None
            raise


SOURCE_ADMISSION = ServiceKey[SourceAdmission]("core.source_admission")
