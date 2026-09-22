"""Root 的来源接纳许可；连接及具体停止协议由 provider 持有。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from agent.plugin_composition.context import Context, RuntimeLease
from agent.plugin_composition.model import CompositionError, ServiceKey

if TYPE_CHECKING:
    from session.message import Message

    from agent.plugin_composition.channels import ChannelInboundMessage
    from agent.plugin_composition.context import CompositionRoot
    from agent.plugins.snapshot import RuntimeSnapshotStore


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

    def current_lease(self) -> RuntimeLease:
        """返回当前 Task 绑定且属于本 Root 的 opaque scope lease；缺席或跨 Root 时拒绝。"""

        from agent.plugins.snapshot import get_current_runtime_lease

        lease = get_current_runtime_lease()
        if (lease is None or lease.snapshot.composition_root is None
                or lease.snapshot.composition_root.instance_token is not self._root_token):
            raise RuntimeError("当前 runtime scope lease 不属于本 Root")
        return RuntimeLease(lease)

    def lease(self, snapshot_id: str) -> RuntimeLease:
        """入口只能取得本 Root 的公开 opaque lease，不能借用恢复特权。"""
        current = self._store.current
        if (current is None or current.snapshot_id != snapshot_id
                or current.composition_root is None
                or current.composition_root.instance_token is not self._root_token):
            raise RuntimeError("来源 lease 不属于当前 Root")
        return RuntimeLease(self._store.lease(snapshot_id))

    def _lease_root(self, lease: RuntimeLease) -> "CompositionRoot":
        """lease/root/active 检查在 composition owner 内完成；插件看不到 snapshot。"""
        if not isinstance(lease, RuntimeLease):
            raise TypeError("scope 能力必须是 admission owner 签发的 opaque lease")
        raw = lease._raw_lease()
        if not raw.active:
            raise RuntimeError("scope lease 已释放或所属 snapshot 已退役")
        root = raw.snapshot.composition_root
        if root is None or root.instance_token is not self._root_token:
            raise RuntimeError("scope lease 不属于本 Root")
        return root

    def channel_input(
        self, lease: RuntimeLease,
    ) -> Callable[[str, str, "ChannelInboundMessage"], Awaitable["Message"]]:
        """在 lease 所属 exact 且 active 的本 Root 上取已声明的 channel 输入端口。

        只允许当前 Task 绑定的 lease 解析；不提供任意 ServiceKey 查询。
        """
        from agent.plugin_composition.channels import CHANNEL_INPUT
        from agent.plugins.snapshot import get_current_runtime_lease

        root = self._lease_root(lease)
        if get_current_runtime_lease() is not lease._raw_lease():
            raise RuntimeError("scope lease 未绑定在当前 Task")
        return root.context.require(CHANNEL_INPUT)

    def require_channel_binding_owner(
        self,
        lease: RuntimeLease,
        ctx: Context,
        channels: object,
    ) -> str:
        """确认 ctx 是 lease 所属本 Root 的 CHANNELS 贡献 Context 且实现未变。"""
        from agent.plugin_composition.channels import CHANNELS

        root = self._lease_root(lease)
        if root.context.require(CHANNELS) is not channels:
            raise RuntimeError("binding 服务不属于当前 runtime scope")
        owner = root.context_owner(ctx)
        if owner is None:
            raise PermissionError("Context 不属于当前 runtime scope")
        return owner

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
