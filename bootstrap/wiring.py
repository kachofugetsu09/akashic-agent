from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from agent.context import ContextBuilder
from agent.lifecycle.facade import TurnLifecycle
from agent.tools.base import Tool
from bootstrap.toolsets.mcp import McpToolsetProvider
from bootstrap.toolsets.memory import MemoryToolsetProvider
from bootstrap.toolsets.meta import CommonMetaToolsetProvider, SpawnToolsetProvider
from bootstrap.toolsets.protocol import ToolsetProvider
from bootstrap.toolsets.schedule import SchedulerToolsetProvider
from core.memory.plugin import MemoryPlugin

if TYPE_CHECKING:
    from agent.looping.interrupt import TurnInterruptState


ContextFactory = Callable[[Path, Any], Any]
ToolsetProviderFactory = Callable[[], ToolsetProvider]
MemoryPluginFactory = Callable[[], MemoryPlugin]
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_MEMORY_WIRING: dict[str, ToolsetProviderFactory] = {
    "default": MemoryToolsetProvider,
}


def _build_default_memory_plugin() -> MemoryPlugin:
    from plugins.default_memory.memory_plugin import MemoryPlugin as DefaultMemoryPlugin

    return DefaultMemoryPlugin()


_MEMORY_PLUGIN_WIRING: dict[str, MemoryPluginFactory] = {
    "default": _build_default_memory_plugin,
}
_CONTEXT_WIRING: dict[str, ContextFactory] = {
    "default": lambda workspace, memory: ContextBuilder(workspace, memory=memory),
}
_TOOLSET_WIRING: dict[str, ToolsetProviderFactory] = {
    "spawn": SpawnToolsetProvider,
    "schedule": SchedulerToolsetProvider,
    "mcp": McpToolsetProvider,
}


def wire_turn_lifecycle(
    lifecycle: TurnLifecycle,
    *,
    active_turn_states: Mapping[str, "TurnInterruptState"],
) -> None:
    from agent.lifecycle.types import AfterStepCtx

    async def _progress_reporter(ctx: AfterStepCtx) -> None:
        state = active_turn_states.get(ctx.session_key)
        if state is None:
            return
        if ctx.partial_reply:
            state.partial_reply = ctx.partial_reply
        if ctx.partial_thinking:
            state.partial_thinking = ctx.partial_thinking
        state.tools_used = list(ctx.tools_used_so_far)
        state.tool_chain_partial = list(ctx.tool_chain_partial)

    lifecycle.on_after_step(_progress_reporter)


def resolve_memory_toolset_provider(name: str) -> ToolsetProvider:
    if name not in _MEMORY_WIRING:
        choices = ", ".join(sorted(_MEMORY_WIRING))
        raise ValueError(f"未知 memory wiring: {name}；可选值: {choices}")
    return _MEMORY_WIRING[name]()


def resolve_memory_plugin(name: str) -> MemoryPlugin:
    normalized = (name or "default").strip() or "default"
    if normalized in _MEMORY_PLUGIN_WIRING:
        return _MEMORY_PLUGIN_WIRING[normalized]()
    plugin = _load_memory_plugin_from_dir(normalized)
    if plugin is None:
        choices = ", ".join(sorted(_MEMORY_PLUGIN_WIRING))
        raise ValueError(f"未知 memory engine: {normalized}；可选值: {choices}")
    return plugin


def register_memory_plugin(
    name: str,
    factory: MemoryPluginFactory,
) -> None:
    normalized = name.strip()
    if not normalized:
        raise ValueError("memory engine 名称不能为空")
    _MEMORY_PLUGIN_WIRING[normalized] = factory


def _load_memory_plugin_from_dir(name: str) -> MemoryPlugin | None:
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"memory engine 名称非法: {name}")
    plugin_path = _PROJECT_ROOT / "plugins" / name / "memory_plugin.py"
    if not plugin_path.exists():
        return None
    module_name = f"akasic_memory_plugin_{name}"
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if hasattr(module, "create_memory_plugin"):
        plugin = module.create_memory_plugin()
    elif hasattr(module, "MemoryPlugin"):
        plugin = module.MemoryPlugin()
    else:
        raise ValueError(f"{plugin_path} 缺少 create_memory_plugin 或 MemoryPlugin")
    if not isinstance(plugin, MemoryPlugin):
        raise TypeError(f"{plugin_path} 未返回 MemoryPlugin")
    return plugin


def resolve_context_factory(name: str) -> ContextFactory:
    if name not in _CONTEXT_WIRING:
        choices = ", ".join(sorted(_CONTEXT_WIRING))
        raise ValueError(f"未知 context wiring: {name}；可选值: {choices}")
    return _CONTEXT_WIRING[name]


def resolve_toolset_provider(
    name: str, *, readonly_tools: dict[str, Tool] | None = None
) -> ToolsetProvider:
    if name == "meta_common":
        return CommonMetaToolsetProvider(readonly_tools or {})
    if name not in _TOOLSET_WIRING:
        choices = ", ".join(sorted(["meta_common", *_TOOLSET_WIRING.keys()]))
        raise ValueError(f"未知 toolset wiring: {name}；可选值: {choices}")
    return _TOOLSET_WIRING[name]()
