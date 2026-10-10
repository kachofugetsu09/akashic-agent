from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey


@dataclass(frozen=True, slots=True)
class UpdateStatus:
    """Project one update from its durable input and live generation facts."""

    update_id: str
    plugin_id: str
    input_ref: str | None
    selection: Literal["selected", "not_selected", "unknown"]
    generation_id: str | None
    active_input_ref: str | None
    fiber_state: str | None
    state: Literal["accepted", "active", "failed", "unknown", "restart_required"]
    error: str


from agent.plugin_composition.plugin_config import ConfigHost


class PluginInstallPort(ConfigHost, Protocol):
    """安装控制面；不暴露 Root、源码目录、数据库或任意 Manager 方法。"""
    async def install(self, *, source: str, marketplace: str, ref_name: str, sparse_paths: list[str], update_id: str) -> UpdateStatus: ...
    def read_update(self, update_id: str) -> UpdateStatus: ...
    def watch_updates(self) -> AsyncGenerator[None]: ...
    def plugin_status(self) -> dict[str, object]: ...
    async def uninstall(self, plugin_id: str) -> dict[str, object]: ...
    async def reconcile_disabled_and_drain(self, plugin_id: str) -> None: ...
    async def wait_idle(self) -> None: ...


class PluginUpdates:
    """插件管理端口；选择提交与资源排空仍由宿主拥有。"""

    def __init__(self, host: PluginInstallPort | None):
        self._host = host

    def _check(self, ctx: Context) -> PluginInstallPort:
        _ = ctx.require_runtime_identity(PLUGIN_UPDATES, self)
        if self._host is None:
            raise PermissionError("插件更新宿主不可用")
        return self._host

    def _request(self, ctx: Context, update_id: str) -> PluginInstallPort:
        host = self._check(ctx)
        if not isinstance(update_id, str) or not update_id or update_id.strip() != update_id:
            raise ValueError("更新 ID 必须是非空且无首尾空白的字符串")
        return host

    def read(self, ctx: Context, update_id: str) -> UpdateStatus | None:
        host = self._request(ctx, update_id)
        try:
            return host.read_update(update_id)
        except KeyError:
            return None

    async def install(
        self, ctx: Context, update_id: str, *, source: str, marketplace: str,
        ref: str = "", sparse: tuple[str, ...] = (),
    ) -> UpdateStatus:
        """Install one request; an existing ID is read-only and never re-run."""
        host = self._request(ctx, update_id)
        _ = await host.install(source=source, marketplace=marketplace,
            ref_name=ref, sparse_paths=list(sparse), update_id=update_id)
        status = self.read(ctx, update_id)
        assert status is not None
        return status

    def status(self, ctx: Context) -> dict[str, object]:
        """读取宿主选择、generation 与当前操作，不返回 Manager 或 Root。"""
        return self._check(ctx).plugin_status()

    async def uninstall(self, ctx: Context, plugin_id: str) -> dict[str, object]:
        """等待选择移除的 accepted；宿主继续拥有物理清理。"""
        return await self._check(ctx).uninstall(plugin_id)

    async def drain(self, ctx: Context, plugin_id: str) -> None:
        """排空用户已停用的插件；此调用不替用户修改启停决定。"""
        await self._check(ctx).reconcile_disabled_and_drain(plugin_id)

    async def wait_idle(self, ctx: Context) -> None:
        """等待当前宿主操作退出；不取消操作，也不提交新选择。"""
        await self._check(ctx).wait_idle()

    async def changes(self, ctx: Context) -> AsyncGenerator[None]:
        """通知只唤醒读取，不保存队列或持有等待发布必须排空的租约。"""
        async with ctx.runtime_scope():
            host = self._check(ctx)
        async for _ in host.watch_updates():
            yield None


PLUGIN_UPDATES = ServiceKey[PluginUpdates]("core.plugin_updates")
