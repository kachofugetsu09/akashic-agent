from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.memory.port import MemoryPort
    from memory2.hyde_enhancer import HyDEEnhancer


@dataclass
class MemoryInjectionResult:
    selected_items: list[dict] = field(default_factory=list)
    block: str = ""
    item_ids: list[str] = field(default_factory=list)
    procedure_hits: int = 0
    history_hits: int = 0
    history_scope_mode: str = "disabled"


async def retrieve_procedure_items(
    memory: "MemoryPort",
    query: str,
    *,
    top_k: int,
) -> list[dict]:
    return await memory.retrieve_related(
        query,
        memory_types=["procedure", "preference"],
        top_k=top_k,
    )


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
) -> tuple[list[dict], str]:
    if prefer_scoped and scope_channel and scope_chat_id:
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

    global_kwargs: dict[str, Any] = {"memory_types": memory_types}
    if prefer_scoped:
        global_kwargs["require_scope_match"] = False

    scope_mode = "global-fallback" if prefer_scoped else "global"

    if hyde_enhancer is not None:
        items, used_hyde = await hyde_enhancer.augment(
            raw_query=query,
            context=context,
            retrieve_fn=memory.retrieve_related,
            top_k=top_k,
            **global_kwargs,
        )
        return items, f"{scope_mode}+hyde" if used_hyde else scope_mode

    items = await memory.retrieve_related(query, top_k=top_k, **global_kwargs)
    return items, scope_mode


def build_memory_injection_result(
    memory: "MemoryPort",
    *,
    procedure_items: list[dict],
    history_items: list[dict],
    history_scope_mode: str = "disabled",
) -> MemoryInjectionResult:
    merged = _dedupe_memory_items(procedure_items + history_items)
    selected = memory.select_for_injection(merged)
    block, ids = memory.format_injection_with_ids(selected)
    return MemoryInjectionResult(
        selected_items=selected,
        block=block,
        item_ids=ids,
        procedure_hits=len(procedure_items),
        history_hits=len(history_items),
        history_scope_mode=history_scope_mode,
    )


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
