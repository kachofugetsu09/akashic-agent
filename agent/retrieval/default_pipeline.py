from __future__ import annotations

from agent.core.types import RetrievalTrace
from agent.looping.ports import MemoryServices
from agent.retrieval.protocol import (
    MemoryRetrievalPipeline,
    RetrievalRequest,
    RetrievalResult,
)
from core.memory.engine import (
    MemoryQuery,
    MemoryQueryFilters,
    MemoryQueryResult,
    MemoryScope,
)


class DefaultMemoryRetrievalPipeline(MemoryRetrievalPipeline):
    def __init__(
        self,
        memory: MemoryServices,
    ) -> None:
        self._memory = memory

    # 被动预检索入口：只转换请求形状，检索语义统一交给 MemoryEngine。
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        # 1. 没有启用记忆引擎时，主链继续无记忆回复。
        if self._memory.engine is None:
            return RetrievalResult(block="", trace=None)

        # 2. 把 agent loop 的上下文转成 engine 的稳定请求协议。
        result = await self._memory.engine.query(
            MemoryQuery(
                text=request.message,
                intent="context",
                scope=MemoryScope(
                    session_key=request.session_key,
                    channel=request.channel,
                    chat_id=request.chat_id,
                ),
                context={
                    "history": request.history,
                    "session_metadata": request.session_metadata,
                },
                filters=MemoryQueryFilters(hints=dict(request.extra or {})),
                timestamp=request.timestamp,
            )
        )

        # 3. 只返回主链需要注入的文本块和可观测 trace。
        return RetrievalResult(
            block=result.text_block,
            trace=_build_retrieval_trace(result),
        )


# 把 engine trace 收窄成 agent loop 认识的检索 trace。
def _build_retrieval_trace(
    result: MemoryQueryResult,
) -> RetrievalTrace | None:
    if not result.trace and not result.records and not result.text_block:
        return None
    return RetrievalTrace(
        gate_type=str(result.trace.get("gate_type") or "") or None,
        route_decision=str(result.trace.get("route_decision") or "") or None,
        rewritten_query=str(result.raw.get("rewritten_query") or "") or None,
        injected_count=sum(1 for record in result.records if record.injected),
        raw=result.raw.get("retrieval_event"),
    )
