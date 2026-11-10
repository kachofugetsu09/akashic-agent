"""设置工作区“插件列表”的读取与启停；选择和排空仍由宿主插件管理端口拥有。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from agent.plugin_composition import Context, DashboardContext, ServiceKey
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG

Phase = Literal["active", "pending", "failed", "disabled", "stopping", "loading"]
Row = dict[str, object]

# 插件管理页面的宿主（Dashboard、设置外框、提供网页的 Web 客户端）；它们及其传递依赖停用后，这个页面会失去入口。
MANAGEMENT_ROOTS = frozenset({"ui", "shell-ui", "akashic_clients"})


@dataclass(frozen=True, slots=True)
class PluginInventory:
    read: Callable[[], Row]
    set_enabled: Callable[[str, bool], Awaitable[Row]]


INVENTORY = ServiceKey[PluginInventory]("shell-ui.plugin_inventory")


class EnabledRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool


# 在插件自己的 Context 上绑定入口，HTTP 请求只经这个服务读写。
def bind(ctx: Context) -> PluginInventory:
    updates = ctx.require(PLUGIN_UPDATES)

    def read() -> Row:
        status = updates.status(ctx)
        catalog = ctx.require(RUNTIME_CATALOG)(ctx)
        plugins = cast(list[Row], status["plugins"])
        return {"plugins": plugin_rows(plugins, cast(list[Row], catalog["plugins"])),
                "busy": status["operation"] is not None}

    async def set_enabled(plugin_id: str, enabled: bool) -> Row:
        rows = cast(list[Row], read()["plugins"])
        if not enabled and any(row["id"] == plugin_id and row["locked"] for row in rows):
            raise PermissionError("插件管理依赖它，不能停用")
        return await updates.set_enabled(ctx, plugin_id, enabled)

    return PluginInventory(ctx.entrypoint(read), ctx.entrypoint(set_enabled))


# 合并宿主选择与运行目录：一行一个插件，只保留列表需要的事实。
def plugin_rows(plugins: list[Row], catalog: list[Row]) -> list[Row]:
    fibers = {cast(str, item["id"]): cast(list[Row], cast(Row, item["composition"])["fibers"]) for item in catalog}
    dependents: dict[str, set[str]] = {}
    providers: dict[str, set[str]] = {}
    for consumer, items in fibers.items():
        for fiber in items:
            for provider in cast(Mapping[str, str | None], fiber["dependency_providers"]).values():
                if provider is not None and provider != consumer:
                    dependents.setdefault(provider, set()).add(consumer)
                    providers.setdefault(consumer, set()).add(provider)
    locked = _closure({plugin_id for plugin_id in fibers if plugin_id.split("@", 1)[0] in MANAGEMENT_ROOTS}, providers)
    rows: list[Row] = []
    for plugin in plugins:
        plugin_id = cast(str, plugin["plugin_id"])
        own = fibers.get(plugin_id, [])
        missing = sorted({cast(str, key) for fiber in own for key in cast(list[object], fiber["missing_services"])})
        draining = cast(list[Row], plugin["draining_generations"])
        error = (plugin["load_error"] or next((fiber["error"] for fiber in own if fiber["error"]), None)
                 or next((item["load_error"] for item in draining if item["load_error"]), None))
        name = plugin_id.split("@", 1)[0]
        # 发行版带着但组合未选的插件没有选择记录，按未运行即停用处理。
        enabled = plugin["state"] is not None if plugin["enabled"] is None else plugin["enabled"] is True
        rows.append({
            "id": plugin_id,
            "name": name,
            "version": plugin["version"],
            "description": plugin["description"],
            "source": "builtin" if plugin["distribution_available"] else "external",
            "enabled": enabled,
            "phase": _phase(plugin, enabled, bool(draining), error, missing),
            "error": error,
            "missing_services": missing,
            "dependents": sorted(dependents.get(plugin_id, ())),
            "locked": "management-required" if plugin_id in locked else None,
        })
    return rows


def _closure(roots: set[str], edges: Mapping[str, set[str]]) -> set[str]:
    seen = set(roots)
    stack = list(roots)
    while stack:
        for item in edges.get(stack.pop(), ()):
            if item not in seen:
                seen.add(item)
                stack.append(item)
    return seen


def _phase(plugin: Row, enabled: bool, draining: bool, error: object, missing: list[str]) -> Phase:
    # 停用是用户意图；仍有 generation 或待排空的 owner 时，实际还没停下。
    if not enabled:
        return "stopping" if draining or plugin["state"] is not None or plugin["cleanup_pending"] else "disabled"
    if error or plugin["state"] == "failed":
        return "failed"
    if missing or plugin["fiber_state"] == "pending":
        return "pending"
    return "active" if plugin["state"] == "active" else "loading"


inject = (INVENTORY,)


def register(app: FastAPI, context: DashboardContext) -> None:
    """列表只读宿主事实；启停等待当前 Root 调和后返回结果。"""

    @app.get("/api/dashboard/plugins")
    async def list_plugins() -> Row:
        return context.require(INVENTORY).read()

    @app.post("/api/dashboard/plugins/{plugin_id}/enabled")
    async def set_enabled(plugin_id: str, request: EnabledRequest) -> Row:
        try:
            return await context.require(INVENTORY).set_enabled(plugin_id, request.enabled)
        except PermissionError as error:
            raise HTTPException(403, str(error)) from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(409, str(error)) from error
