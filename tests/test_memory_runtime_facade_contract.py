from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.postturn.protocol import PostTurnEvent
from core.memory.default_runtime_facade import DefaultMemoryRuntimeFacade
from core.memory.engine import (
    MemoryEngineRetrieveResult,
    MemoryHit,
    MemoryIngestResult,
    MemoryScope,
    RememberRequest,
    RememberResult,
)
from core.memory.runtime_facade import (
    ContextRetrievalRequest,
    ContextRetrievalResult,
    InterestRetrievalRequest,
    MemoryRuntimeFacade,
)


@dataclass(frozen=True)
class _ToolCall:
    call_id: str
    name: str
    arguments: dict
    result: str


@dataclass(frozen=True)
class _ToolGroup:
    text: str
    calls: list[_ToolCall]


@pytest.mark.asyncio
async def test_default_runtime_facade_ingest_post_turn_delegates_to_engine():
    engine = SimpleNamespace(
        ingest=AsyncMock(return_value=MemoryIngestResult(accepted=True, raw={"ok": True}))
    )
    facade = DefaultMemoryRuntimeFacade(
        port=MagicMock(),
        engine=engine,
        profile_maint=MagicMock(),
    )

    result = await facade.ingest_post_turn(
        PostTurnEvent(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            user_message="以后用中文",
            assistant_response="好的",
            tools_used=["memorize"],
            tool_chain=[
                _ToolGroup(
                    text="memo",
                    calls=[
                        _ToolCall(
                            call_id="call-1",
                            name="memorize",
                            arguments={"summary": "以后用中文"},
                            result="ok",
                        )
                    ],
                )
            ],
            session=object(),
        )
    )

    assert result.accepted is True
    request = engine.ingest.await_args.args[0]
    assert request.scope.session_key == "cli:1"
    assert request.metadata["source_ref"] == "cli:1@post_response"
    assert request.content["tool_chain"][0]["calls"][0]["name"] == "memorize"


@pytest.mark.asyncio
async def test_default_runtime_facade_retrieve_context_uses_callback_result():
    port = MagicMock()
    callback = AsyncMock(
        return_value=ContextRetrievalResult(
            normative_hits=[{"id": "p1"}],
            episodic_hits=[{"id": "e1"}],
            injected_item_ids=["p1", "e1"],
            text_block="记忆块",
            trace={"mode": "callback"},
            hyde_hypothesis="hyde",
            scope_mode="global",
            sufficiency_trace={"retry": True},
            raw={"owner": "callback"},
        )
    )
    facade = DefaultMemoryRuntimeFacade(
        port=port,
        engine=None,
        profile_maint=MagicMock(),
        context_retriever=callback,
    )

    result = await facade.retrieve_context(
        ContextRetrievalRequest(
            message="Fitbit 型号",
            session_key="telegram:1",
            channel="telegram",
            chat_id="1",
            history=[],
            session_metadata={},
        )
    )

    assert result.text_block == "记忆块"
    assert result.normative_hits == [{"id": "p1"}]
    assert result.episodic_hits == [{"id": "e1"}]
    assert result.injected_item_ids == ["p1", "e1"]
    assert result.trace["mode"] == "callback"
    assert result.hyde_hypothesis == "hyde"
    assert result.scope_mode == "global"
    assert result.sufficiency_trace["retry"] is True
    port.reinforce_items_batch.assert_called_once_with(["p1", "e1"])


def test_default_runtime_facade_satisfies_runtime_protocol():
    facade = DefaultMemoryRuntimeFacade(
        port=MagicMock(),
        engine=None,
        profile_maint=MagicMock(),
    )

    assert isinstance(facade, MemoryRuntimeFacade)


@pytest.mark.asyncio
async def test_default_runtime_facade_retrieve_context_fallback_keeps_hits_and_injected():
    port = MagicMock()
    engine = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=MemoryEngineRetrieveResult(
                text_block="历史块",
                hits=[
                    MemoryHit(
                        id="e1",
                        summary="用户昨天提过 FitBit",
                        content="用户昨天提过 FitBit",
                        score=0.81,
                        source_ref="telegram:1@seed",
                        engine_kind="default",
                        metadata={"memory_type": "event"},
                        injected=True,
                    )
                ],
                trace={"engine": "default"},
                raw={"items": [{"id": "e1"}]},
            )
        )
    )
    facade = DefaultMemoryRuntimeFacade(
        port=port,
        engine=engine,
        profile_maint=MagicMock(),
    )

    result = await facade.retrieve_context(
        ContextRetrievalRequest(
            message="Fitbit 型号",
            session_key="telegram:1",
            channel="telegram",
            chat_id="1",
            history=[],
            session_metadata={},
        )
    )

    assert result.text_block == "历史块"
    assert result.injected_item_ids == ["e1"]
    assert result.episodic_hits[0]["id"] == "e1"
    assert result.scope_mode == "engine_fallback"
    assert result.trace["engine"] == "default"
    port.reinforce_items_batch.assert_called_once_with(["e1"])


@pytest.mark.asyncio
async def test_default_runtime_facade_retrieve_interest_block_uses_preference_profile():
    port = MagicMock()
    port.retrieve_related = AsyncMock(
        return_value=[
            {"id": "p1", "text": "用户偏好中文回复"},
            {"id": "u1", "text": "用户常用 Telegram"},
        ]
    )
    facade = DefaultMemoryRuntimeFacade(
        port=port,
        engine=None,
        profile_maint=MagicMock(),
    )

    result = await facade.retrieve_interest_block(
        InterestRetrievalRequest(
            query="如果这条内容对用户有价值，用户为什么会关心它",
            scope=MemoryScope(session_key="telegram:1", channel="telegram", chat_id="1"),
            top_k=2,
        )
    )

    assert "用户偏好中文回复" in result.text_block
    assert "用户常用 Telegram" in result.text_block
    kwargs = port.retrieve_related.await_args.kwargs
    assert kwargs["memory_types"] == ["preference", "profile"]
    assert kwargs["require_scope_match"] is True


@pytest.mark.asyncio
async def test_default_runtime_facade_remember_explicit_delegates_to_engine():
    engine = SimpleNamespace(
        remember=AsyncMock(
            return_value=RememberResult(item_id="mem-1", actual_type="preference")
        )
    )
    facade = DefaultMemoryRuntimeFacade(
        port=MagicMock(),
        engine=engine,
        profile_maint=MagicMock(),
    )

    result = await facade.remember_explicit(
        RememberRequest(
            summary="以后用中文回复",
            memory_type="preference",
            scope=MemoryScope(session_key="cli:1"),
        )
    )

    assert result.item_id == "mem-1"
    engine.remember.assert_awaited_once()


@pytest.mark.asyncio
async def test_default_runtime_facade_run_consolidation_delegates_to_runner():
    runner = AsyncMock()
    facade = DefaultMemoryRuntimeFacade(
        port=MagicMock(),
        engine=None,
        profile_maint=MagicMock(),
        consolidation_runner=runner,
    )
    session = object()

    await facade.run_consolidation(
        session,
        archive_all=True,
    )

    runner.assert_awaited_once_with(session, True)


def test_default_runtime_facade_binds_context_retriever():
    facade = DefaultMemoryRuntimeFacade(
        port=MagicMock(),
        engine=None,
        profile_maint=MagicMock(),
    )
    retriever = AsyncMock()

    facade.bind_context_retriever(retriever)

    assert facade._context_retriever is retriever


def test_default_runtime_facade_binds_retrieval_semantics():
    facade = DefaultMemoryRuntimeFacade(
        port=MagicMock(),
        engine=None,
        profile_maint=MagicMock(),
    )
    retriever = AsyncMock()

    facade.bind_retrieval_semantics(retriever)

    assert facade._retrieval_semantics is retriever


def test_default_runtime_facade_reads_file_side_context_from_profile_maint():
    profile_maint = SimpleNamespace(
        read_long_term=MagicMock(return_value="MEMORY"),
        read_self=MagicMock(return_value="SELF"),
        read_history=MagicMock(return_value="HISTORY"),
    )
    facade = DefaultMemoryRuntimeFacade(
        port=MagicMock(),
        engine=None,
        profile_maint=profile_maint,
    )

    assert facade.read_long_term_context() == "MEMORY"
    assert facade.read_self() == "SELF"
    assert facade.read_recent_history(max_chars=123) == "HISTORY"
    profile_maint.read_history.assert_called_once_with(max_chars=123)
