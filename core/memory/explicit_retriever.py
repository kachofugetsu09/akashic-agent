from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, cast

from core.memory.runtime_facade import ExplicitRetrievalRequest, ExplicitRetrievalResult

if TYPE_CHECKING:
    from agent.provider import LLMProvider, LLMResponse
    from core.memory.port import MemoryPort

logger = logging.getLogger("memory.explicit_retriever")

_HYPOTHESIS_MAX_TOKENS = 80
_HYPOTHESIS_TIMEOUT_S = 3.0
_VECTOR_SCORE_THRESHOLD = 0.35
_VECTOR_TOP_K = 15
_ChatCall = Callable[..., Awaitable["LLMResponse"]]


class DefaultExplicitRetriever:
    def __init__(
        self,
        *,
        port: "MemoryPort",
        provider: "LLMProvider",
        model: str,
    ) -> None:
        self._port = port
        self._provider = provider
        self._model = model

    async def __call__(
        self,
        request: ExplicitRetrievalRequest,
    ) -> ExplicitRetrievalResult:
        return await self.retrieve(request)

    async def retrieve(
        self,
        request: ExplicitRetrievalRequest,
    ) -> ExplicitRetrievalResult:
        if request.search_mode == "grep":
            return self._retrieve_grep(request)
        return await self._retrieve_semantic(request)

    def _retrieve_grep(
        self,
        request: ExplicitRetrievalRequest,
    ) -> ExplicitRetrievalResult:
        if request.time_start is None or request.time_end is None:
            return ExplicitRetrievalResult(
                trace={"source": "default_explicit_retriever", "mode": "grep_missing_time"}
            )
        hits = self._port.list_events_by_time_range(
            request.time_start,
            request.time_end,
            limit=request.limit,
        )
        return ExplicitRetrievalResult(
            hits=list(hits),
            trace={
                "source": "default_explicit_retriever",
                "mode": "grep",
                "hit_count": len(hits),
            },
            raw={"hits": list(hits)},
        )

    async def _retrieve_semantic(
        self,
        request: ExplicitRetrievalRequest,
    ) -> ExplicitRetrievalResult:
        hyp1_task = asyncio.create_task(self._gen_hypothesis(request.query, style="event"))
        hyp2_task = asyncio.create_task(self._gen_hypothesis(request.query, style="general"))
        hyp1, hyp2 = await asyncio.gather(hyp1_task, hyp2_task)
        aux_queries = [text for text in (hyp1, hyp2) if text]
        types = [request.memory_type] if request.memory_type else None
        fetch_limit = max(request.limit, _VECTOR_TOP_K)
        hits = await self._port.retrieve_related(
            request.query,
            memory_types=types,
            top_k=fetch_limit,
            scope_channel=request.scope.channel or None,
            scope_chat_id=request.scope.chat_id or None,
            require_scope_match=bool(request.scope.channel and request.scope.chat_id),
            aux_queries=aux_queries,
            score_threshold=_VECTOR_SCORE_THRESHOLD,
            time_start=request.time_start,
            time_end=request.time_end,
            keyword_enabled=True,
        )
        sliced = list(hits)[: request.limit]
        logger.info(
            "explicit_retriever: mode=semantic query=%r hits=%d hyp=[%r, %r]",
            request.query[:60],
            len(sliced),
            hyp1[:50] if hyp1 else None,
            hyp2[:50] if hyp2 else None,
        )
        return ExplicitRetrievalResult(
            hits=sliced,
            trace={
                "source": "default_explicit_retriever",
                "mode": "semantic",
                "hit_count": len(sliced),
                "hyde_hypotheses": aux_queries,
            },
            raw={"hits": sliced},
        )

    async def _gen_hypothesis(self, query: str, style: str) -> str | None:
        if style == "event":
            prompt = (
                "你是个人助手的记忆系统。根据用户提问，生成一条**带具体时间**的假想记忆条目，"
                "格式如 '[2026-03-08] 用户...'\n"
                "规则：第三人称、简洁事实陈述、只输出那一条文本\n\n"
                f"用户提问：{query}\n假想记忆条目："
            )
        else:
            prompt = (
                "你是个人助手的记忆系统。根据用户提问，生成一条假想记忆条目。\n"
                "规则：始终生成肯定式、第三人称（'用户…'）、简洁事实陈述、只输出那一条文本\n\n"
                f"用户提问：{query}\n假想记忆条目："
            )
        try:
            chat = cast(_ChatCall, getattr(self._provider, "chat"))
            resp = await asyncio.wait_for(
                chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[],
                    model=self._model,
                    max_tokens=_HYPOTHESIS_MAX_TOKENS,
                ),
                timeout=_HYPOTHESIS_TIMEOUT_S,
            )
            text = (resp.content or "").strip()
            return text if text else None
        except Exception as e:
            logger.debug("explicit_retriever: hypothesis generation failed: %s", e)
            return None
