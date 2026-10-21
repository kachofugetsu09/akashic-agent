"""web_search_exa 的端点、凭据与搜索选项设置归本插件；启用与否由插件行决定。"""
from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.context import FiberHandle
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.ui.contract import Configuration, ModelSettings, mount_settings

from .search import ExaConfig

SETTINGS = ServiceKey[Configuration]("web_search_exa.settings.v1")
FIELDS = ("endpoint", "search_type", "livecrawl", "timeout_seconds")


async def mount(ctx: Context, function: FiberHandle) -> None:
    # ready 反映提供方是否已登记到 standard_web；缺前置时说明先启用哪一项。
    async def status() -> dict[str, object]:
        graph = ctx.require(RUNTIME_CATALOG)(ctx)
        own = next(item for item in cast(list[dict[str, Any]], graph["plugins"]) if item["id"] == ctx.runtime.plugin_id)
        fiber = next(item for item in own["composition"]["fibers"] if item["fiber_id"] == function.fiber_id)
        reason = "需要先启用 standard_web（Web 工具）" if fiber["missing_services"] else (fiber["error"] or "")
        return {"ready": function.state.value == "active", "reason": reason}

    settings = ModelSettings(ctx, ExaConfig, FIELDS, secret="api_key", status=status)
    await mount_settings(ctx, SETTINGS, settings, lambda: import_module(".dashboard", __package__))
