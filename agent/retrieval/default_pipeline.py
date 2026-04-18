from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from agent.core.types import RetrievalTrace
from agent.looping.ports import LLMServices, MemoryConfig, MemoryServices
from agent.retrieval.protocol import (
    MemoryRetrievalPipeline,
    RetrievalRequest,
    RetrievalResult,
)
from core.memory.default_runtime_facade import DefaultRetrievalSemantics
from core.memory.default_runtime_facade import (
    _build_injection_payload,
    _retrieve_episodic_items,
)
from core.memory.runtime_facade import ContextRetrievalRequest, ContextRetrievalResult
from core.observe.events import RagTrace

logger = logging.getLogger("agent.retrieval")


@runtime_checkable
class _InjectedItem(Protocol):
    injected: bool


class DefaultMemoryRetrievalPipeline(MemoryRetrievalPipeline):
    def __init__(
        self,
        memory: MemoryServices,
        memory_config: MemoryConfig,
        llm: LLMServices,
        workspace: Path,
        light_model: str,
    ) -> None:
        self._memory = memory
        self._config = memory_config
        self._llm = llm
        self._workspace = workspace
        self._light_model = light_model
        self._fallback_retriever = DefaultRetrievalSemantics(
            memory=memory,
            config=memory_config,
            llm=llm,
            workspace=workspace,
            light_model=light_model,
        )
        _bind_facade_retrieval_semantics(
            memory=memory,
            retriever=self._fallback_retriever.retrieve_context,
        )

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        context_result = await _retrieve_context_with_facade(
            memory=self._memory,
            request=ContextRetrievalRequest(
                message=request.message,
                session_key=request.session_key,
                channel=request.channel,
                chat_id=request.chat_id,
                history=request.history,
                session_metadata=request.session_metadata,
                timestamp=request.timestamp,
                extra=dict(request.extra or {}),
            ),
            legacy_retriever=self._retrieve_context_via_legacy_pipeline,
        )
        block = context_result.text_block
        trace = _build_retrieval_trace(context_result)
        return RetrievalResult(block=block, trace=trace)

    async def _retrieve_context_via_legacy_pipeline(
        self, request: ContextRetrievalRequest
    ) -> ContextRetrievalResult:
        return await self._fallback_retriever.retrieve_context(request)


def _bind_facade_retrieval_semantics(
    *,
    memory: MemoryServices,
    retriever,
) -> None:
    facade = memory.facade
    if facade is None:
        return
    facade.bind_retrieval_semantics(retriever)


async def _retrieve_context_with_facade(
    *,
    memory: MemoryServices,
    request: ContextRetrievalRequest,
    legacy_retriever,
) -> ContextRetrievalResult:
    facade = getattr(memory, "facade", None)
    if facade is None:
        return await legacy_retriever(request)
    return await facade.retrieve_context(request)


def _build_retrieval_trace(
    context_result: ContextRetrievalResult,
) -> RetrievalTrace | None:
    rag_trace = context_result.raw.get("rag_trace")
    if isinstance(rag_trace, RagTrace):
        return RetrievalTrace(
            gate_type=rag_trace.gate_type,
            route_decision=rag_trace.route_decision,
            rewritten_query=rag_trace.query,
            injected_count=sum(
                1
                for item in (rag_trace.items or [])
                if isinstance(item, _InjectedItem) and item.injected
            ),
            raw=rag_trace,
        )
    if not context_result.trace and not context_result.injected_item_ids and not context_result.text_block:
        return None
    return RetrievalTrace(
        gate_type=str(context_result.trace.get("gate_type") or "") or None,
        route_decision=str(context_result.trace.get("route_decision") or "") or None,
        rewritten_query=str(context_result.raw.get("rewritten_query") or "") or None,
        injected_count=len(context_result.injected_item_ids),
        raw=context_result.trace or None,
    )
