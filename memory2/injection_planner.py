from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.memory.port import MemoryPort
    from memory2.hyde_enhancer import HyDEAugmentResult, HyDEEnhancer

async def retrieve_procedure_items(
    memory: "MemoryPort",
    query: str = "",
    queries: list[str] | None = None,
    *,
    top_k: int,
) -> list[dict]:
    active_queries = _normalize_procedure_queries(query=query, queries=queries)
    if not active_queries:
        return []

    # 1. 并发执行多路 procedure/preference 检索，保持门控链路延迟稳定。
    tasks = [
        memory.retrieve_related(
            item_query,
            memory_types=["procedure", "preference"],
            top_k=top_k,
        )
        for item_query in active_queries
    ]
    raw_results = await asyncio.gather(*tasks)

    # 2. 同 id 命中做 max-pool，保留最高分版本。
    pooled = _max_pool_memory_items(raw_results)

    # 3. 最后按分数降序截断 top_k，兼容旧接口返回形态。
    return sorted(
        pooled,
        key=lambda item: (_item_score(item), str(item.get("id", ""))),
        reverse=True,
    )[:top_k]


async def retrieve_episodic(
    memory: "MemoryPort",
    query: str,
    *,
    memory_types: list[str] | None = None,
    top_k: int,
    context: str = "",
    hyde_enhancer: "HyDEEnhancer | None" = None,
) -> tuple[list[dict], str, str | None]:
    """Global episodic 检索（event + profile），返回 (items, scope_mode, hyde_hypothesis)。

    不含 scoped 路径——scoped 路径由 proactive 的 retrieve_history_items 负责。
    """
    # 1. 先确定这次到底查哪些记忆类型。
    #    主回复链路通常传进来的是 ["event", "profile"]；
    #    如果上层没传，就默认按 event + profile 查。
    mtypes: list[str] = memory_types if memory_types is not None else ["event", "profile"]

    # 2. 如果配置了 HyDE，就先让 light model 基于 raw query + recent context
    #    生成一个“更像记忆库表述方式”的假想查询，再用它辅助召回。
    #    这里返回的不只是增强后的 hits，还会告诉上层：
    #    - 这次是否真的用了 HyDE
    #    - 生成的假想描述是什么
    if hyde_enhancer is not None:
        result = await hyde_enhancer.augment(
            raw_query=query,
            context=context,
            retrieve_fn=memory.retrieve_related,
            top_k=top_k,
            memory_types=mtypes,
        )
        scope = "global+hyde" if result.used_hyde else "global"
        return _mark_hyde_history_paths(result.raw_hits, result.items), scope, result.hypothesis

    # 3. 如果没开 HyDE，就直接用改写后的 query 去查向量库。
    #    这里查到的就是最原始的一批 event/profile 命中结果。
    items = await memory.retrieve_related(query, memory_types=mtypes, top_k=top_k)

    # 4. 最后给命中的 item 打一个 history_path 标记，告诉上层：
    #    这批结果来自“原始历史检索”而不是 HyDE 增强路径。
    return _mark_history_path(items, "history_raw"), "global", None


async def retrieve_history_items(
    memory: "MemoryPort",
    query: str,
    *,
    memory_types: list[str],
    top_k: int,
    prefer_scoped: bool = False,
    scope_channel: str = "",
    scope_chat_id: str = "",
    allow_global: bool = True,
    context: str = "",
    hyde_enhancer: "HyDEEnhancer | None" = None,
    on_hyde_result: "Callable[[HyDEAugmentResult], None] | None" = None,
) -> tuple[list[dict], str]:
    if prefer_scoped and scope_channel and scope_chat_id:
        # 单次 embed，scoped 和 global 共用 query_vec 并发查询，省去重复的远端 embedding 调用。
        # 若 MemoryPort 实现尚未支持新接口，退化回原始串行逻辑。
        _has_vec_api = callable(getattr(memory, "embed_query", None)) and callable(
            getattr(memory, "retrieve_related_vec", None)
        )
        query_vec = await memory.embed_query(query) if _has_vec_api else []
        if query_vec:
            scoped_task = asyncio.create_task(
                memory.retrieve_related_vec(
                    query_vec,
                    memory_types=memory_types,
                    top_k=top_k,
                    scope_channel=scope_channel,
                    scope_chat_id=scope_chat_id,
                    require_scope_match=True,
                )
            )
            if allow_global:
                global_task = asyncio.create_task(
                    memory.retrieve_related_vec(
                        query_vec,
                        memory_types=memory_types,
                        top_k=top_k,
                        require_scope_match=False,
                    )
                )
                scoped_items, global_items = await asyncio.gather(
                    scoped_task, global_task
                )
                return (
                    (scoped_items, "scoped")
                    if scoped_items
                    else (global_items, "global-fallback")
                )
            else:
                scoped_items = await scoped_task
                return (scoped_items, "scoped") if scoped_items else ([], "disabled")
        else:
            # embedder 未配置，退化回原始串行逻辑
            scoped_items = await memory.retrieve_related(
                query,
                memory_types=memory_types,
                top_k=top_k,
                scope_channel=scope_channel,
                scope_chat_id=scope_chat_id,
                require_scope_match=True,
            )
            if scoped_items:
                return scoped_items, "scoped"
            if not allow_global:
                return [], "disabled"

    if not allow_global:
        return [], "disabled"

    global_kwargs: dict[str, Any] = {"memory_types": memory_types}
    if prefer_scoped:
        global_kwargs["require_scope_match"] = False

    scope_mode = "global-fallback" if prefer_scoped else "global"

    if hyde_enhancer is not None:
        hyde_result = await hyde_enhancer.augment(
            raw_query=query,
            context=context,
            retrieve_fn=memory.retrieve_related,
            top_k=top_k,
            **global_kwargs,
        )
        if on_hyde_result is not None:
            on_hyde_result(hyde_result)
        return hyde_result.items, f"{scope_mode}+hyde" if hyde_result.used_hyde else scope_mode

    items = await memory.retrieve_related(query, top_k=top_k, **global_kwargs)
    return items, scope_mode


def _dedupe_memory_items(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        item_id = str(item.get("id", "") or "")
        if item_id:
            if item_id in seen:
                continue
            seen.add(item_id)
        out.append(item)
    return out


def _normalize_procedure_queries(
    *,
    query: str = "",
    queries: list[str] | None = None,
) -> list[str]:
    raw_queries = list(queries or [])
    if not raw_queries and query:
        raw_queries = [query]

    seen: set[str] = set()
    normalized: list[str] = []
    for item in raw_queries:
        value = " ".join(str(item or "").split())
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _max_pool_memory_items(result_sets: list[list[dict]]) -> list[dict]:
    pooled_by_id: dict[str, dict] = {}
    passthrough: list[dict] = []

    for items in result_sets:
        for item in items:
            item_id = str(item.get("id", "") or "")
            if not item_id:
                passthrough.append(deepcopy(item))
                continue
            current = pooled_by_id.get(item_id)
            if current is None or _item_score(item) > _item_score(current):
                pooled_by_id[item_id] = deepcopy(item)

    return list(pooled_by_id.values()) + passthrough


def _item_score(item: dict) -> float:
    try:
        return float(item.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _mark_history_path(items: list[dict], path: str) -> list[dict]:
    marked: list[dict] = []
    for item in items:
        cloned = deepcopy(item)
        cloned["_retrieval_path"] = path
        marked.append(cloned)
    return marked


def _mark_hyde_history_paths(raw_hits: list[dict], merged_items: list[dict]) -> list[dict]:
    raw_ids = {str(item.get("id", "")) for item in raw_hits if str(item.get("id", ""))}
    marked: list[dict] = []
    for item in merged_items:
        item_id = str(item.get("id", ""))
        path = "history_raw" if item_id in raw_ids else "history_hyde"
        cloned = deepcopy(item)
        cloned["_retrieval_path"] = path
        marked.append(cloned)
    return marked
