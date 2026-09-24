from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey

if TYPE_CHECKING:
    from agent.plugins.manager import PluginManager


@dataclass(frozen=True, slots=True)
class UpdateStatus:
    """Project one update from its durable input and live generation facts."""

    update_id: str
    plugin_id: str
    input_ref: str | None
    selection: Literal["selected", "not_selected", "unknown"]
    generation_id: str | None
    archive_ref: str | None
    fiber_state: str | None
    state: Literal["accepted", "active", "failed", "unknown"]
    error: str


class PluginUpdates:
    """Expose only install, read, and change notifications."""

    def __init__(self, host: PluginManager | None):
        self._host = host

    def _check(self, ctx: Context) -> PluginManager:
        _ = ctx.require_runtime_owner(PLUGIN_UPDATES, self)
        if self._host is None:
            raise PermissionError("插件更新宿主不可用")
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
        """Install one request; an existing ID is read-only and never re-run."""
        host = self._request(ctx, update_id)
        _ = await host.install(source=source, marketplace=marketplace,
            ref_name=ref, sparse_paths=list(sparse), update_id=update_id)
        status = self.read(ctx, update_id)
        assert status is not None
        return status

    async def changes(self, ctx: Context) -> AsyncGenerator[None]:
        """通知只唤醒读取，不保存队列或持有等待发布必须排空的租约。"""
        async with ctx.runtime_scope():
            host = self._check(ctx)
        async for _ in host.watch_updates():
            yield None


PLUGIN_UPDATES = ServiceKey[PluginUpdates]("core.plugin_updates")
