from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from agent.postturn.protocol import PostTurnEvent
from core.memory.engine import (
    MemoryEngineRetrieveRequest,
    MemoryIngestRequest,
    MemoryIngestResult,
    MemoryScope,
    RememberRequest,
    RememberResult,
)
from core.memory.runtime_facade import (
    ConsolidationRunner,
    ContextRetrievalRequest,
    ContextRetrievalResult,
    ContextRetriever,
    InterestRetrievalRequest,
    InterestRetrievalResult,
)

if TYPE_CHECKING:
    from agent.core.types import ToolCallGroup
    from core.memory.engine import MemoryEngine
    from core.memory.port import MemoryPort
    from core.memory.profile import ProfileMaintenanceStore


InterestRetriever = Callable[[InterestRetrievalRequest], Awaitable[InterestRetrievalResult]]


class DefaultMemoryRuntimeFacade:
    def __init__(
        self,
        *,
        port: "MemoryPort",
        engine: "MemoryEngine | None" = None,
        profile_maint: "ProfileMaintenanceStore | None" = None,
        context_retriever: ContextRetriever | None = None,
        consolidation_runner: ConsolidationRunner | None = None,
        interest_retriever: InterestRetriever | None = None,
    ) -> None:
        self._port = port
        self._engine = engine
        self._profile_maint = profile_maint or port
        self._context_retriever = context_retriever
        self._consolidation_runner = consolidation_runner
        self._interest_retriever = interest_retriever

    def bind_context_retriever(self, retriever: ContextRetriever) -> None:
        self._context_retriever = retriever

    def bind_consolidation_runner(self, runner: ConsolidationRunner) -> None:
        self._consolidation_runner = runner

    async def ingest_post_turn(self, event: PostTurnEvent) -> MemoryIngestResult:
        # 1. 先保证 post-turn 入口和旧 pipeline 一样只依赖 engine。
        if self._engine is None:
            return MemoryIngestResult(
                accepted=False,
                summary="memory engine unavailable",
                raw={"reason": "engine_unavailable"},
            )

        # 2. 按旧 post-turn 语义组装标准 ingest request。
        # 2.1 tool_chain 保持旧结构，避免 worker 侧归一化漂移。
        # 2.2 source_ref 继续固定到 @post_response。
        source_ref = f"{event.session_key}@post_response"
        return await self._engine.ingest(
            MemoryIngestRequest(
                content={
                    "user_message": event.user_message,
                    "assistant_response": event.assistant_response,
                    "tool_chain": [_tool_group_to_dict(group) for group in event.tool_chain],
                    "source_ref": source_ref,
                },
                source_kind="conversation_turn",
                scope=MemoryScope(
                    session_key=event.session_key,
                    channel=event.channel,
                    chat_id=event.chat_id,
                ),
                metadata={"source_ref": source_ref},
            )
        )

    async def retrieve_context(
        self, request: ContextRetrievalRequest
    ) -> ContextRetrievalResult:
        # 1. 优先走外部注入的完整 retrieval callback。
        if self._context_retriever is not None:
            return await self._context_retriever(request)

        # 2. 没切主链前，提供一个极薄 fallback，保证 facade 可单独 contract test。
        if self._engine is None:
            return ContextRetrievalResult(
                trace={"source": "default_runtime_facade", "mode": "disabled"},
                scope_mode="disabled",
            )

        # 3. fallback 只做单次 engine.retrieve，不假装拥有旧 pipeline 全部语义。
        engine_result = await self._engine.retrieve(
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
                hints=dict(request.extra),
            )
        )
        injected_ids = [hit.id for hit in engine_result.hits if hit.injected]
        return ContextRetrievalResult(
            episodic_hits=[_memory_hit_to_item(hit) for hit in engine_result.hits],
            injected_item_ids=injected_ids,
            text_block=engine_result.text_block,
            trace={
                "source": "default_runtime_facade",
                "mode": "engine_fallback",
                **dict(engine_result.trace),
            },
            scope_mode="engine_fallback",
            raw=dict(engine_result.raw),
        )

    async def run_consolidation(
        self,
        session: object,
        *,
        archive_all: bool = False,
        await_vector_store: bool = False,
    ) -> None:
        # 1. consolidation 在 phase 1 还不迁 owner，只提供统一入口。
        if self._consolidation_runner is None:
            raise RuntimeError("consolidation_runner unavailable")
        await self._consolidation_runner(session, archive_all, await_vector_store)

    async def retrieve_interest_block(
        self, request: InterestRetrievalRequest
    ) -> InterestRetrievalResult:
        # 1. proactive 未来会切到 facade，这里先保留旧的 preference/profile recall 语义。
        if self._interest_retriever is not None:
            return await self._interest_retriever(request)

        # 2. 默认 fallback 直接转调 port.retrieve_related。
        hits = self._port.retrieve_related(
            request.query,
            memory_types=["preference", "profile"],
            top_k=request.top_k,
            scope_channel=request.scope.channel or None,
            scope_chat_id=request.scope.chat_id or None,
            require_scope_match=bool(request.scope.channel and request.scope.chat_id),
        )
        hits = await _await_if_needed(hits) or []
        texts = [str(hit.get("text", "") or "") for hit in hits if hit.get("text")]
        return InterestRetrievalResult(
            text_block="\n---\n".join(texts),
            hits=list(hits),
            trace={"source": "default_runtime_facade", "mode": "port_fallback"},
            raw={"hits": list(hits)},
        )

    async def remember_explicit(self, request: RememberRequest) -> RememberResult:
        # 1. 显式记忆仍然以 engine 为 owner，facade 只做统一入口。
        if self._engine is None:
            raise RuntimeError("memory engine unavailable")
        return await self._engine.remember(request)

    def read_long_term_context(self) -> str:
        # 1. 文件侧长期上下文读取先继续走 profile_maint/port。
        return str(self._profile_maint.read_long_term() or "")

    def read_self(self) -> str:
        # 1. proactive prompt 仍需要自我认知块，这里保持 file-side 读取入口。
        return str(self._profile_maint.read_self() or "")

    def read_recent_history(self, *, max_chars: int = 0) -> str:
        # 1. consolidation 侧仍依赖 history 文件，这里只是收一个稳定入口。
        return str(self._profile_maint.read_history(max_chars=max_chars) or "")


def _tool_group_to_dict(group: "ToolCallGroup") -> dict[str, Any]:
    return {
        "text": group.text,
        "calls": [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
                "result": call.result,
            }
            for call in group.calls
        ],
    }


def _memory_hit_to_item(hit) -> dict[str, Any]:
    metadata = dict(getattr(hit, "metadata", {}) or {})
    memory_type = str(metadata.get("memory_type", "") or "")
    item = {
        "id": str(getattr(hit, "id", "") or ""),
        "summary": str(getattr(hit, "summary", "") or ""),
        "text": str(getattr(hit, "content", "") or ""),
        "score": float(getattr(hit, "score", 0.0) or 0.0),
        "source_ref": str(getattr(hit, "source_ref", "") or ""),
        "memory_type": memory_type,
        "extra_json": metadata,
        "injected": bool(getattr(hit, "injected", False)),
    }
    return item


async def _await_if_needed(value):
    if inspect.isawaitable(value):
        return await value
    return value
