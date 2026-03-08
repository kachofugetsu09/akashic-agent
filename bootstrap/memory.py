from __future__ import annotations

from pathlib import Path

from agent.config_models import Config
from agent.provider import LLMProvider
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
) -> MemoryRuntime:
    from agent.memory import MemoryStore
    from agent.tools.memorize import MemorizeTool
    from agent.tools.filesystem import EditFileTool, WriteFileTool
    from core.memory.port import DefaultMemoryPort
    from memory2.embedder import Embedder
    from memory2.memorizer import Memorizer
    from memory2.retriever import Retriever
    from memory2.sop_indexer import SopIndexer
    from memory2.store import MemoryStore2

    store = MemoryStore(workspace)
    if not config.memory_v2.enabled:
        tools.register(
            WriteFileTool(),
            tags=["filesystem", "memory"],
            risk="write",
            search_keywords=["写文件", "保存文件", "创建文件", "写入文件", "新建文件"],
        )
        tools.register(
            EditFileTool(),
            tags=["filesystem", "memory"],
            risk="write",
            search_keywords=["编辑文件", "修改文件", "更新文件", "patch文件"],
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
        sop_guard_enabled=config.memory_v2.sop_guard_enabled,
    )

    port = DefaultMemoryPort(store, memorizer=memorizer, retriever=retriever)
    post_mem_worker = PostResponseMemoryWorker(
        memorizer=memorizer,
        retriever=retriever,
        light_provider=light_provider or provider,
        light_model=config.light_model or config.model,
    )
    tools.register(
        MemorizeTool(port),
        always_on=True,
        tags=["memory"],
        risk="write",
        search_keywords=["记忆", "存储知识", "记录信息", "备忘", "memorize"],
    )
    sop_indexer = SopIndexer(mem2_store, embedder, workspace / "sop")
    tools.register(
        WriteFileTool(sop_indexer=sop_indexer),
        tags=["filesystem", "memory"],
        risk="write",
        search_keywords=["写文件", "保存文件", "创建文件", "写入文件", "新建文件"],
    )
    tools.register(
        EditFileTool(sop_indexer=sop_indexer),
        tags=["filesystem", "memory"],
        risk="write",
        search_keywords=["编辑文件", "修改文件", "更新文件", "patch文件"],
    )

    return MemoryRuntime(
        port=port,
        post_response_worker=post_mem_worker,
        sop_indexer=sop_indexer,
        closeables=[mem2_store, embedder],
    )
