from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from agent.context import ContextBuilder
from agent.config_models import Config
from agent.lifecycle.facade import TurnLifecycle
from agent.provider import LLMProvider
from agent.tools.base import Tool
from bootstrap.toolsets.mcp import McpToolsetProvider
from bootstrap.toolsets.memory import MemoryToolsetProvider
from bootstrap.toolsets.meta import CommonMetaToolsetProvider, SpawnToolsetProvider
from bootstrap.toolsets.protocol import ToolsetProvider
from bootstrap.toolsets.schedule import SchedulerToolsetProvider
from core.memory.default_engine import DefaultMemoryEngine
from core.net.http import SharedHttpResources

if TYPE_CHECKING:
    from agent.looping.interrupt import TurnInterruptState
    from bus.event_bus import EventBus


ContextFactory = Callable[[Path, Any], Any]
ToolsetProviderFactory = Callable[[], ToolsetProvider]


@dataclass(frozen=True)
class MemoryEngineBuildDeps:
    config: Config
    workspace: Path
    provider: LLMProvider
    light_provider: LLMProvider | None
    http_resources: SharedHttpResources
    event_publisher: "EventBus | None" = None


MemoryEngineBuilder = Callable[[MemoryEngineBuildDeps], object]

_MEMORY_WIRING: dict[str, ToolsetProviderFactory] = {
    "default": MemoryToolsetProvider,
}


def _build_default_memory_engine(deps: MemoryEngineBuildDeps) -> DefaultMemoryEngine:
    return DefaultMemoryEngine(
        config=deps.config,
        workspace=deps.workspace,
        provider=deps.provider,
        light_provider=deps.light_provider,
        http_resources=deps.http_resources,
        event_publisher=deps.event_publisher,
    )


_MEMORY_ENGINE_WIRING: dict[str, MemoryEngineBuilder] = {
    "default": _build_default_memory_engine,
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


def resolve_memory_engine_builder(name: str) -> MemoryEngineBuilder:
    if name not in _MEMORY_ENGINE_WIRING:
        choices = ", ".join(sorted(_MEMORY_ENGINE_WIRING))
        raise ValueError(f"未知 memory_engine wiring: {name}；可选值: {choices}")
    return _MEMORY_ENGINE_WIRING[name]


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
