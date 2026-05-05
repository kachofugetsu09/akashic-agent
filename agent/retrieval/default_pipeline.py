from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agent.core.types import RetrievalTrace
from agent.looping.ports import LLMServices, MemoryConfig, MemoryServices
from agent.retrieval.protocol import (
    MemoryRetrievalPipeline,
    RetrievalRequest,
    RetrievalResult,
)
from core.memory.engine import MemoryEngineRetrieveRequest, MemoryEngineRetrieveResult, MemoryScope

if TYPE_CHECKING:
    from bus.publisher import EventPublisher


class DefaultMemoryRetrievalPipeline(MemoryRetrievalPipeline):
    def __init__(
        self,
        memory: MemoryServices,
        memory_config: MemoryConfig,
        llm: LLMServices,
        workspace: Path,
        light_model: str,
        event_publisher: "EventPublisher | None" = None,
    ) -> None:
        self._memory = memory

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        if self._memory.engine is None:
            return RetrievalResult(block="", trace=None)
        result = await self._memory.engine.retrieve(
            MemoryEngineRetrieveRequest(
                query=request.message,
                scope=MemoryScope(
                    session_key=request.session_key,
                    channel=request.channel,
                    chat_id=request.chat_id,
                ),
                context={
                    "history": request.history,
                    "session_metadata": request.session_metadata,
                },
                hints=dict(request.extra or {}),
            )
        )
        return RetrievalResult(
            block=result.text_block,
            trace=_build_retrieval_trace(result),
        )


def _build_retrieval_trace(
    result: MemoryEngineRetrieveResult,
) -> RetrievalTrace | None:
    if not result.trace and not result.hits and not result.text_block:
        return None
    return RetrievalTrace(
        gate_type=str(result.trace.get("gate_type") or "") or None,
        route_decision=str(result.trace.get("route_decision") or "") or None,
        rewritten_query=str(result.raw.get("rewritten_query") or "") or None,
        injected_count=sum(1 for hit in result.hits if hit.injected),
        raw=result.raw.get("retrieval_event"),
    )


def _build_injection_payload(
    *,
    procedure_items: list[dict],
    procedure_result: MemoryEngineRetrieveResult | None,
    history_items: list[dict],
    history_result: MemoryEngineRetrieveResult | None,
) -> tuple[list[dict], str, list[str]]:
    # TODO(memory-engine-cleanup): 旧测试改查 MemoryEngineRetrieveResult 后删除这个兼容 helper。
    procedure_ids = _engine_injected_ids(procedure_result)
    history_ids = _engine_injected_ids(history_result)
    selected = [
        item
        for item in [*procedure_items, *history_items]
        if str(item.get("id", "")) in set(procedure_ids + history_ids)
    ]
    block = "\n\n".join(
        value
        for value in [
            procedure_result.text_block if procedure_result is not None else "",
            history_result.text_block if history_result is not None else "",
        ]
        if value
    )
    return selected, block, _dedupe_ids(procedure_ids + history_ids)


def _engine_injected_ids(result: MemoryEngineRetrieveResult | None) -> list[str]:
    if result is None:
        return []
    return [hit.id for hit in result.hits if hit.injected and hit.id]


def _dedupe_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item_id in ids:
        if item_id and item_id not in seen:
            seen.add(item_id)
            out.append(item_id)
    return out
