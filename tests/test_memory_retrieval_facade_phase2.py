from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.types import HistoryMessage
from agent.looping.ports import LLMServices, MemoryConfig, MemoryServices
from agent.provider import LLMResponse
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from core.memory.default_runtime_facade import DefaultRetrievalSemantics
from agent.retrieval.protocol import RetrievalRequest
from core.memory.runtime_facade import ContextRetrievalRequest, ContextRetrievalResult


class _Provider:
    async def chat(self, **kwargs):
        return LLMResponse(content="ok", tool_calls=[])


@pytest.mark.asyncio
async def test_retrieval_pipeline_prefers_facade_retrieve_context(tmp_path: Path):
    facade = MagicMock()
    facade.retrieve_context = AsyncMock(
        return_value=ContextRetrievalResult(
            text_block="facade block",
            raw={"rag_trace": None},
        )
    )
    pipeline = DefaultMemoryRetrievalPipeline(
        memory=MemoryServices(engine=MagicMock(), facade=cast(Any, facade)),
        memory_config=MemoryConfig(),
        llm=LLMServices(provider=cast(Any, _Provider()), light_provider=cast(Any, _Provider())),
        workspace=tmp_path,
        light_model="test-light",
    )

    result = await pipeline.retrieve(
        RetrievalRequest(
            message="用户提过什么",
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            history=[HistoryMessage(role="user", content="hi")],
            session_metadata={},
        )
    )

    assert result.block == "facade block"
    facade.retrieve_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_retrieval_pipeline_binds_legacy_callback_into_default_facade(tmp_path: Path):
    from core.memory.default_runtime_facade import DefaultMemoryRuntimeFacade

    engine = MagicMock()
    facade = DefaultMemoryRuntimeFacade(
        port=MagicMock(),
        engine=cast(Any, engine),
        profile_maint=MagicMock(),
    )
    pipeline = DefaultMemoryRetrievalPipeline(
        memory=MemoryServices(engine=cast(Any, engine), facade=facade),
        memory_config=MemoryConfig(),
        llm=LLMServices(provider=cast(Any, _Provider()), light_provider=cast(Any, _Provider())),
        workspace=tmp_path,
        light_model="test-light",
    )

    assert facade._retrieval_semantics is not None
    assert callable(facade._retrieval_semantics)
    assert pipeline._memory.facade is facade


@pytest.mark.asyncio
async def test_default_retrieval_semantics_returns_rich_context_result(tmp_path: Path):
    gate_resolver = MagicMock()
    gate_resolver.memory_window = 6
    gate_resolver.resolve = AsyncMock(
        return_value=(
            {
                "gate_type": "history_route",
                "episodic_query": "改写问题",
                "route_decision": "RETRIEVE",
                "route_latency_ms": 12,
                "fallback_reason": "",
                "history_memory_types": ["event", "profile"],
            },
            [{"id": "p1", "memory_type": "preference"}],
            MagicMock(),
        )
    )
    episodic_retriever = MagicMock()
    episodic_retriever.retrieve = AsyncMock(
        return_value=(
            [{"id": "e1", "memory_type": "event"}],
            "global+hyde",
            "hyde hypothesis",
            [{"id": "p1"}, {"id": "e1"}],
            "memory block",
            ["p1", "e1"],
        )
    )
    finalizer = MagicMock()
    finalizer.finalize.return_value = MagicMock()
    semantics = DefaultRetrievalSemantics(
        memory=MemoryServices(engine=MagicMock(), facade=None),
        config=MemoryConfig(),
        llm=LLMServices(provider=cast(Any, _Provider()), light_provider=cast(Any, _Provider())),
        workspace=tmp_path,
        light_model="test-light",
    )
    semantics._gate_resolver = gate_resolver
    semantics._episodic_retriever = episodic_retriever
    semantics._finalizer = finalizer

    result = await semantics.retrieve_context(
        ContextRetrievalRequest(
            message="用户提过什么",
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            history=[HistoryMessage(role="user", content="hi")],
            session_metadata={},
        )
    )

    assert result.normative_hits == [{"id": "p1", "memory_type": "preference"}]
    assert result.episodic_hits == [{"id": "e1", "memory_type": "event"}]
    assert result.injected_item_ids == ["p1", "e1"]
    assert result.text_block == "memory block"
    assert result.hyde_hypothesis == "hyde hypothesis"
    assert result.scope_mode == "global+hyde"
    assert result.trace["mode"] == "default_semantics"
    assert result.trace["route_decision"] == "RETRIEVE"
    assert result.trace["gate_type"] == "history_route"
    assert result.raw["rewritten_query"] == "改写问题"
    assert finalizer.finalize.call_args.kwargs["config"] is semantics._config


@pytest.mark.asyncio
async def test_retrieval_pipeline_builds_trace_from_structured_context_without_rag_trace(
    tmp_path: Path,
):
    facade = MagicMock()
    facade.retrieve_context = AsyncMock(
        return_value=ContextRetrievalResult(
            text_block="facade block",
            injected_item_ids=["p1", "e1"],
            trace={
                "source": "facade",
                "mode": "semantics_owner",
                "gate_type": "history_route",
                "route_decision": "RETRIEVE",
            },
            raw={"rewritten_query": "改写问题"},
        )
    )
    pipeline = DefaultMemoryRetrievalPipeline(
        memory=MemoryServices(engine=MagicMock(), facade=cast(Any, facade)),
        memory_config=MemoryConfig(),
        llm=LLMServices(provider=cast(Any, _Provider()), light_provider=cast(Any, _Provider())),
        workspace=tmp_path,
        light_model="test-light",
    )

    result = await pipeline.retrieve(
        RetrievalRequest(
            message="用户提过什么",
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            history=[HistoryMessage(role="user", content="hi")],
            session_metadata={},
        )
    )

    assert result.block == "facade block"
    assert result.trace is not None
    assert result.trace.gate_type == "history_route"
    assert result.trace.route_decision == "RETRIEVE"
    assert result.trace.rewritten_query == "改写问题"
    assert result.trace.injected_count == 2
