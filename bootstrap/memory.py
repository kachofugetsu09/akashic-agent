from __future__ import annotations

from pathlib import Path
from typing import cast

from agent.config_models import Config
from agent.provider import LLMProvider
from agent.tools.meta import register_memory_meta_tools
from agent.tools.registry import ToolRegistry
from core.memory.engine import MemoryEngine
from core.memory.default_runtime_facade import DefaultMemoryRuntimeFacade
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources
from memory2.post_response_worker import PostResponseMemoryWorker


def build_memory_runtime(
    config: Config,
    workspace: Path,
    tools: ToolRegistry,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    http_resources: SharedHttpResources,
    observe_writer=None,
) -> MemoryRuntime:
    from agent.memory import MemoryStore
    from agent.skills import SkillsLoader
    from agent.tools.memorize import MemorizeTool
    from agent.tools.forget_memory import ForgetMemoryTool
    from agent.tools.filesystem import EditFileTool, WriteFileTool
    from core.memory.port import DefaultMemoryPort
    from memory2.embedder import Embedder
    from memory2.memorizer import Memorizer
    from memory2.procedure_tagger import ProcedureTagger
    from memory2.retriever import Retriever
    from memory2.store import MemoryStore2

    store = MemoryStore(workspace)
    if not config.memory_v2.enabled:
        register_memory_meta_tools(
            tools,
            write_file_tool=WriteFileTool(),
            edit_file_tool=EditFileTool(),
        )
        port = DefaultMemoryPort(store)
        facade = DefaultMemoryRuntimeFacade(
            port=port,
            profile_maint=port,
        )
        return MemoryRuntime(
            port=port,
            facade=facade,
            profile_reader=port,
            profile_maint=port,
        )

    db_path = (
        Path(config.memory_v2.db_path)
        if config.memory_v2.db_path
        else workspace / "memory" / "memory2.db"
    )
    mem2_store = MemoryStore2(db_path)
    embedder = Embedder(
        base_url=(
            config.memory_v2.base_url
            or config.light_base_url
            or config.base_url
            or ""
        ),
        api_key=(
            config.memory_v2.api_key
            or config.light_api_key
            or config.api_key
        ),
        model=config.memory_v2.embed_model,
        requester=http_resources.external_default,
    )
    memorizer = Memorizer(mem2_store, embedder)
    retriever = Retriever(
        mem2_store,
        embedder,
        top_k=config.memory_v2.retrieve_top_k,
        score_threshold=config.memory_v2.score_threshold,
        score_thresholds={
            "procedure": config.memory_v2.score_threshold_procedure,
            "preference": config.memory_v2.score_threshold_preference,
            "event": config.memory_v2.score_threshold_event,
            "profile": config.memory_v2.score_threshold_profile,
        },
        relative_delta=config.memory_v2.relative_delta,
        inject_max_chars=config.memory_v2.inject_max_chars,
        inject_max_forced=config.memory_v2.inject_max_forced,
        inject_max_procedure_preference=config.memory_v2.inject_max_procedure_preference,
        inject_max_event_profile=config.memory_v2.inject_max_event_profile,
        inject_line_max=config.memory_v2.inject_line_max,
        procedure_guard_enabled=config.memory_v2.procedure_guard_enabled,
        hotness_alpha=0.20,
    )

    port = DefaultMemoryPort(store, memorizer=memorizer, retriever=retriever)

    _skills_loader = SkillsLoader(workspace)
    tagger = ProcedureTagger(
        provider=light_provider or provider,
        model=config.light_model or config.model,
        skills_fn=lambda: [
            s["name"] for s in _skills_loader.list_skills(filter_unavailable=False)
        ],
    )

    post_mem_worker = PostResponseMemoryWorker(
        memorizer=memorizer,
        retriever=retriever,
        light_provider=light_provider or provider,
        light_model=config.light_model or config.model,
        observe_writer=observe_writer,
    )
    from bootstrap.wiring import MemoryEngineBuildDeps, resolve_memory_engine_builder

    engine_builder = resolve_memory_engine_builder(
        getattr(getattr(config, "wiring", None), "memory_engine", "default")
    )
    engine = engine_builder(
        MemoryEngineBuildDeps(
            config=config,
            workspace=workspace,
            provider=provider,
            light_provider=light_provider,
            http_resources=http_resources,
            retriever=retriever,
            memorizer=memorizer,
            tagger=tagger,
            post_response_worker=post_mem_worker,
        )
    )
    memory_engine = cast(MemoryEngine, engine)
    from agent.tools.recall_memory import RecallMemoryTool

    memorize_tool = MemorizeTool(memory_engine)
    forget_tool = ForgetMemoryTool(mem2_store)
    recall_tool = RecallMemoryTool(
        store=mem2_store,
        embedder=embedder,
        provider=light_provider or provider,
        model=config.light_model or config.model,
    )
    register_memory_meta_tools(
        tools,
        memorize_tool=memorize_tool,
        forget_tool=forget_tool,
        recall_tool=recall_tool,
        write_file_tool=WriteFileTool(),
        edit_file_tool=EditFileTool(),
    )
    facade = DefaultMemoryRuntimeFacade(
        port=port,
        engine=memory_engine,
        profile_maint=port,
    )

    return MemoryRuntime(
        port=port,
        engine=memory_engine,
        facade=facade,
        profile_reader=port,
        profile_maint=port,
        post_response_worker=post_mem_worker,
        closeables=[mem2_store, embedder],
    )
