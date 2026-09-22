"""只读命令：用 ASCII 树展示当前 stable 组合的插件与 Fiber 依赖层级。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from agent.plugin_composition import Context
from agent.plugin_composition.commands import (
    COMMANDS,
    CommandDefinition,
    CommandInvocation,
    CommandResult,
)
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG

api_version = 3
name = "stable_view"
version = "1.0.0"
desc = "命令行查看当前 stable 插件组合树"
inject = (COMMANDS, RUNTIME_CATALOG)

_FIBER_KEYS = ("name", "parent", "state", "required", "dependencies",
               "missing_services", "error")


def _short(value: object, length: int = 12) -> str:
    text = str(value or "")
    return text[:length] if len(text) > length else text


def _fiber_annotations(fiber: Mapping[str, object]) -> str:
    """汇总单个 Fiber 的依赖、缺失服务和错误为一段尾注。"""
    parts: list[str] = []
    dependencies = fiber.get("dependencies") or ()
    if dependencies:
        parts.append("→ " + ", ".join(str(item) for item in dependencies))
    missing = fiber.get("missing_services") or ()
    if missing:
        parts.append("missing: " + ", ".join(str(item) for item in missing))
    error = fiber.get("error")
    if error:
        parts.append(f"error: {error}")
    if fiber.get("required"):
        parts.append("required")
    return "  " + " · ".join(parts) if parts else ""


def _render_fiber_tree(
    lines: list[str],
    fibers: Sequence[Mapping[str, object]],
    prefix: str,
) -> None:
    """按冻结的 parent 边把一个插件内的 Fiber 渲染成子树。"""
    children: dict[str | None, list[Mapping[str, object]]] = {}
    for fiber in fibers:
        parent = fiber.get("parent")
        children.setdefault(parent if isinstance(parent, str) else None, []).append(fiber)
    for siblings in children.values():
        siblings.sort(key=lambda item: str(item.get("name") or ""))

    def walk(nodes: Sequence[Mapping[str, object]], base: str) -> None:
        for index, fiber in enumerate(nodes):
            last = index == len(nodes) - 1
            lines.append(
                f"{base}{'└─' if last else '├─'} {fiber.get('name')}"
                f"  [{fiber.get('state') or 'unknown'}]"
                f"{_fiber_annotations(fiber)}"
            )
            walk(children.get(fiber.get("name"), []), base + ("   " if last else "│  "))

    walk(children.get(None, []), prefix)


def _render_plugin(
    lines: list[str],
    item: Mapping[str, object],
    prefix: str,
    last: bool,
) -> None:
    composition = item.get("composition")
    ready = isinstance(composition, Mapping) and bool(composition.get("ready"))
    marker = "ready" if ready else "NOT READY"
    lines.append(
        f"{prefix}{'└─' if last else '├─'} {item.get('id')}"
        f"  api v{item.get('api_version')} · gen {_short(item.get('generation_id'), 8)}"
        f" · rev {_short(item.get('revision'), 8)} · {marker}"
    )
    if not isinstance(composition, Mapping):
        return
    subtree = prefix + ("   " if last else "│  ")
    fibers = composition.get("fibers")
    if isinstance(fibers, Sequence):
        _render_fiber_tree(lines, [f for f in fibers if isinstance(f, Mapping)], subtree)
    incident_count = composition.get("incident_count")
    if incident_count:
        lines.append(f"{subtree}└─ incidents: {incident_count}")


def format_stable_catalog(catalog: Mapping[str, object]) -> str:
    """把 stable 组合投影渲染成 ASCII 依赖树。"""
    lines = [
        f"stable snapshot {_short(catalog.get('snapshot_id'), 16)}",
        "plugins",
    ]
    plugins = catalog.get("plugins")
    items = [item for item in plugins or () if isinstance(item, Mapping)]
    for index, item in enumerate(items):
        _render_plugin(lines, item, "   ", index == len(items) - 1)
    servers = catalog.get("mcp_servers")
    if isinstance(servers, Sequence) and not isinstance(servers, str):
        lines.append("mcp_servers")
        entries = list(servers)
        for index, server in enumerate(entries):
            label = server.get("name") if isinstance(server, Mapping) else server
            lines.append(f"   {'└─' if index == len(entries) - 1 else '├─'} {label}")
    unavailable = catalog.get("unavailable")
    if isinstance(unavailable, Mapping):
        lines.append(
            "mcp_servers  (按调用打开，无持久会话目录"
            f" [{unavailable.get('code')}])"
        )
    return "\n".join(lines)


async def apply(ctx: Context) -> None:
    """注册 /stable 只读命令；经注入的 RUNTIME_CATALOG DTO 读取当前组合投影。"""

    read_catalog = ctx.require(RUNTIME_CATALOG)

    async def show_stable(_invocation: CommandInvocation) -> CommandResult:
        try:
            async with ctx.runtime_scope():
                catalog = read_catalog()
            return CommandResult("success", format_stable_catalog(catalog))
        except Exception as error:  # catalog owner 失败也必须如实回报
            return CommandResult("error", f"读取 stable 组合失败: {error}")

    _ = await ctx.require(COMMANDS).register(ctx, CommandDefinition(
        name="stable",
        description="显示当前 stable 组合的插件与 Fiber 依赖树",
        handler=show_stable,
        read_only=True,
    ))
