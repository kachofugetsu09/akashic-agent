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

    # 3. 最后按分数降序截断 top_k，返回稳定的 procedure/preference 命中列表。
    return sorted(
        pooled,
        key=lambda item: (_item_score(item), str(item.get("id", ""))),
        reverse=True,
    )[:top_k]


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
        # 若当前 memory 只提供基础查询接口，则直接走 retrieve_related。
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
            # 当前 memory 未提供可复用 query_vec 的查询能力，直接走基础查询。
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
