from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agent.plugin_composition.bindings import BindingScope
from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey

if TYPE_CHECKING:
    from agent.plugins.manager import PluginManager


@dataclass(frozen=True, slots=True)
class UpdateStatus:
    """只读现有更新收据，不暴露安装指针、SQL 或宿主本身。"""

    update_id: str
    plugin_id: str
    phase: Literal["armed", "committed", "rolled_back"]
    ready: bool
    publishing: bool
    error: str


class PluginUpdates:
    """校验实际调用 scope；安装、验证和发布仍归原插件宿主。"""

    def __init__(self, host: PluginManager | None):
        self._host = host

    def _check(self, ctx: Context) -> PluginManager:
        _ = ctx.require_runtime_owner(PLUGIN_UPDATES, self)
        if self._host is None:
            raise PermissionError("候选验证不能管理插件更新")
        return self._host

    def _request(self, ctx: Context, update_id: str) -> PluginManager:
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
        """一次新请求准备候选；已有请求只允许 read，不重拉或重建。"""
        host = self._request(ctx, update_id)
        _ = await host.install_candidate(source=source, marketplace=marketplace,
            ref_name=ref, sparse_paths=list(sparse), update_id=update_id)
        status = self.read(ctx, update_id)
        assert status is not None
        return status

    @asynccontextmanager
    async def open_validation(self, ctx: Context, update_id: str) -> AsyncGenerator[BindingScope]:
        """程序只取得候选副本的服务；退出必须先完成真实资源清理。"""
        host = self._request(ctx, update_id)
        async with host.open_validation(update_id) as scope:
            yield scope

    def publish(self, ctx: Context, update_id: str) -> None:
        """同步提交发布请求；调用者退出 scope 后宿主才能排空并切换。"""
        self._request(ctx, update_id).start_update_publication(update_id)

    async def discard(self, ctx: Context, update_id: str, *, reason: str = "candidate behavior rejected") -> None:
        """验证拒绝后沿原 owner 清理候选并恢复旧安装状态。"""
        host = self._request(ctx, update_id)
        await host.discard_update(update_id, reason=reason)

    async def changes(self, ctx: Context) -> AsyncGenerator[None]:
        """通知只唤醒读取，不保存队列或持有等待发布必须排空的租约。"""
        async with ctx.runtime_scope():
            host = self._check(ctx)
        async for _ in host.watch_updates():
            yield None


PLUGIN_UPDATES = ServiceKey[PluginUpdates]("core.plugin_updates")
