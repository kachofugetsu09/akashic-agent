from __future__ import annotations

from pathlib import Path

from agent.config_models import Config
from agent.provider import LLMProvider
from agent.tools.meta import register_memory_meta_tools
from agent.tools.registry import ToolRegistry
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
    from agent.tools.filesystem import EditFileTool, WriteFileTool
    from core.memory.port import DefaultMemoryPort
    from memory2.dedup_decider import DedupDecider
    from memory2.embedder import Embedder
    from memory2.memorizer import Memorizer
    from memory2.procedure_tagger import ProcedureTagger
    from memory2.profile_extractor import ProfileFactExtractor
    from memory2.retriever import Retriever
    from memory2.sop_indexer import SopIndexer
    from memory2.store import MemoryStore2

    store = MemoryStore(workspace)
    if not config.memory_v2.enabled:
        register_memory_meta_tools(
            tools,
            write_file_tool=WriteFileTool(),
            edit_file_tool=EditFileTool(),
        )
        return MemoryRuntime(port=DefaultMemoryPort(store))

    db_path = (
        Path(config.memory_v2.db_path)
        if config.memory_v2.db_path
        else workspace / "memory" / "memory2.db"
    )
    mem2_store = MemoryStore2(db_path)
    embedder = Embedder(
        base_url=config.light_base_url or config.base_url or "",
        api_key=config.light_api_key or config.api_key,
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
        sop_guard_enabled=config.memory_v2.sop_guard_enabled,
        hotness_alpha=config.memory_v2.hotness_alpha,
        hotness_half_life_days=config.memory_v2.hotness_half_life_days,
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

    dedup_decider = None
    if config.memory_v2.dedup_enabled:
        dedup_decider = DedupDecider(
            store=mem2_store,
            embedder=embedder,
            provider=light_provider or provider,
            model=config.light_model or config.model,
            similarity_threshold=config.memory_v2.dedup_similarity_threshold,
            batch_dedup_threshold=config.memory_v2.batch_dedup_threshold,
        )

    post_mem_worker = PostResponseMemoryWorker(
        memorizer=memorizer,
        retriever=retriever,
        light_provider=light_provider or provider,
        light_model=config.light_model or config.model,
        tagger=tagger,
        profile_extractor=(
            ProfileFactExtractor(
                llm_client=light_provider or provider,
                model=config.light_model or config.model,
            )
            if config.memory_v2.profile_extraction_enabled
            else None
        ),
        profile_supersede_enabled=config.memory_v2.profile_supersede_enabled,
        observe_writer=observe_writer,
        dedup_decider=dedup_decider,
    )
    register_memory_meta_tools(
        tools,
        memorize_tool=MemorizeTool(port, tagger=tagger),
    )
    sop_indexer = SopIndexer(mem2_store, embedder, workspace / "sop")
    register_memory_meta_tools(
        tools,
        write_file_tool=WriteFileTool(sop_indexer=sop_indexer),
        edit_file_tool=EditFileTool(sop_indexer=sop_indexer),
    )

    return MemoryRuntime(
        port=port,
        post_response_worker=post_mem_worker,
        sop_indexer=sop_indexer,
        closeables=[mem2_store, embedder],
    )
