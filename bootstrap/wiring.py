from __future__ import annotations

from importlib import import_module
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

from agent.context import ContextBuilder
from agent.config_models import Config
from agent.provider import LLMProvider
from agent.tools.base import Tool
from bootstrap.toolsets.fitbit import FitbitToolsetProvider
from bootstrap.toolsets.mcp import McpToolsetProvider
from bootstrap.toolsets.memory import MemoryToolsetProvider
from bootstrap.toolsets.meta import CommonMetaToolsetProvider, SpawnToolsetProvider
from bootstrap.toolsets.schedule import SchedulerToolsetProvider
from core.memory.default_engine import DefaultMemoryEngine
from core.net.http import SharedHttpResources

if TYPE_CHECKING:
    from memory2.memorizer import Memorizer
    from memory2.post_response_worker import PostResponseMemoryWorker
    from memory2.procedure_tagger import ProcedureTagger
    from memory2.retriever import Retriever


ContextFactory = Callable[[Path, Any], Any]


@dataclass(frozen=True)
class MemoryEngineBuildDeps:
    config: Config
    workspace: Path
    provider: LLMProvider
    light_provider: LLMProvider | None
    http_resources: SharedHttpResources
    retriever: "Retriever"
    memorizer: "Memorizer | None"
    tagger: "ProcedureTagger | None"
    post_response_worker: "PostResponseMemoryWorker | None"


MemoryEngineBuilder = Callable[[MemoryEngineBuildDeps], object]

_MEMORY_WIRING = {
    "default": MemoryToolsetProvider,
}


def _build_default_memory_engine(deps: MemoryEngineBuildDeps):
    return DefaultMemoryEngine(
        retriever=deps.retriever,
        memorizer=deps.memorizer,
        tagger=deps.tagger,
        post_response_worker=deps.post_response_worker,
    )


def _build_memu_memory_engine(deps: MemoryEngineBuildDeps):
    from core.memory.memu_engine import MemUMemoryEngine, MemUScopeModel
    MemoryService = cast(Any, import_module("memu.app.service").MemoryService)

    base_url = (
        deps.config.light_base_url or deps.config.base_url or "https://api.openai.com/v1"
    )
    api_key = deps.config.light_api_key or deps.config.api_key
    embed_base_url = deps.config.memory_v2.base_url or base_url
    embed_api_key = deps.config.memory_v2.api_key or api_key
    chat_model = deps.config.light_model or deps.config.model
    service = MemoryService(
        llm_profiles={
            "default": {
                "provider": "openai",
                "base_url": base_url,
                "api_key": api_key,
                "chat_model": chat_model,
                "client_backend": "sdk",
            },
            "embedding": {
                "provider": "openai",
                "base_url": embed_base_url,
                "api_key": embed_api_key,
                "embed_model": deps.config.memory_v2.embed_model,
                "client_backend": "sdk",
            },
        },
        blob_config={
            "provider": "local",
            "resources_dir": str(deps.workspace / "memu" / "resources"),
        },
        database_config={
            "metadata_store": {
                "provider": "inmemory",
            },
        },
        retrieve_config={
            "method": "rag",
            "route_intention": False,
            "sufficiency_check": False,
        },
        user_config={"model": MemUScopeModel},
    )
    return MemUMemoryEngine(
        service=service,
        input_dir=deps.workspace / "memu" / "input",
    )


_MEMORY_ENGINE_WIRING: dict[str, MemoryEngineBuilder] = {
    "default": _build_default_memory_engine,
    "memu": _build_memu_memory_engine,
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


def resolve_toolset_provider(
    name: str, *, readonly_tools: dict[str, Tool] | None = None
):
    if name == "meta_common":
        return CommonMetaToolsetProvider(readonly_tools or {})
    if name not in _TOOLSET_WIRING:
        choices = ", ".join(sorted(["meta_common", *_TOOLSET_WIRING.keys()]))
        raise ValueError(f"未知 toolset wiring: {name}；可选值: {choices}")
    return _TOOLSET_WIRING[name]()
