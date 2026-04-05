from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.context import ContextBuilder
from agent.config_models import Config
from agent.provider import LLMProvider
from bootstrap.toolsets.fitbit import FitbitToolsetProvider
from bootstrap.toolsets.mcp import McpToolsetProvider
from bootstrap.toolsets.memory import MemoryToolsetProvider
from bootstrap.toolsets.meta import CommonMetaToolsetProvider, SpawnToolsetProvider
from bootstrap.toolsets.schedule import SchedulerToolsetProvider
from core.memory.default_engine import DefaultMemoryEngine
from core.net.http import SharedHttpResources


ContextFactory = Callable[[Path, Any], Any]


@dataclass(frozen=True)
class MemoryEngineBuildDeps:
    config: Config
    workspace: Path
    provider: LLMProvider
    light_provider: LLMProvider | None
    http_resources: SharedHttpResources
    retriever: object
    post_response_worker: object | None


MemoryEngineBuilder = Callable[[MemoryEngineBuildDeps], object]

_MEMORY_WIRING = {
    "default": MemoryToolsetProvider,
}
_MEMORY_ENGINE_WIRING: dict[str, MemoryEngineBuilder] = {
    "default": lambda deps: DefaultMemoryEngine(
        retriever=deps.retriever,
        post_response_worker=deps.post_response_worker,
    ),
}
_CONTEXT_WIRING: dict[str, ContextFactory] = {
    "default": lambda workspace, memory_port: ContextBuilder(
        workspace, memory=memory_port
    ),
}
_TOOLSET_WIRING = {
    "fitbit": FitbitToolsetProvider,
    "spawn": SpawnToolsetProvider,
    "schedule": SchedulerToolsetProvider,
    "mcp": McpToolsetProvider,
}


def resolve_memory_toolset_provider(name: str):
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


def resolve_toolset_provider(name: str, *, readonly_tools: dict[str, object] | None = None):
    if name == "meta_common":
        return CommonMetaToolsetProvider(readonly_tools or {})
    if name not in _TOOLSET_WIRING:
        choices = ", ".join(sorted(["meta_common", *_TOOLSET_WIRING.keys()]))
        raise ValueError(f"未知 toolset wiring: {name}；可选值: {choices}")
    return _TOOLSET_WIRING[name]()
