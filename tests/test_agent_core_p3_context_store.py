from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.context_store import DefaultContextStore
from agent.looping.handlers import ConversationTurnHandler
from agent.looping.turn_types import RetrievalTrace
from agent.retrieval.protocol import RetrievalResult
from bus.events import InboundMessage, OutboundMessage


class _DummySession:
    def __init__(self) -> None:
        self.key = "cli:1"
        self.metadata: dict[str, object] = {"mode": "test"}
        self.messages = [
            {
                "role": "user",
                "content": "hello",
                "tools_used": ["read_file"],
                "tool_chain": [
                    {
                        "text": "tool run",
                        "calls": [
                            {
                                "call_id": "call-1",
                                "name": "read_file",
                                "arguments": {"path": "/tmp/a.txt"},
                                "result": "ok",
                            }
                        ],
                    }
                ],
            },
            {"role": "assistant", "content": "world"},
        ]
        self.last_consolidated = 0

    def get_history(self, max_messages: int = 500) -> list[dict]:
        return self.messages[-max_messages:]


@pytest.mark.asyncio
async def test_default_context_store_prepare_returns_bundle_with_legacy_metadata():
    retrieval = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=RetrievalResult(
                block="remembered",
                trace=RetrievalTrace(raw={"route": "RETRIEVE"}),
                metadata={"source": "memory2"},
            )
        )
    )
    context = SimpleNamespace(
        skills=SimpleNamespace(
            list_skills=MagicMock(
                return_value=[{"name": "refactor"}, {"name": "known"}]
            )
        )
    )
    store = DefaultContextStore(retrieval=retrieval, context=context)
    session = _DummySession()
    msg = InboundMessage(
        channel="cli",
        sender="hua",
        chat_id="1",
        content="请用 $refactor 再来一次 $known $refactor",
        timestamp=datetime(2026, 4, 4, 20, 0, 0),
    )

    bundle = await store.prepare(msg=msg, session_key="cli:1", session=session)

    assert [item.content for item in bundle.history] == ["hello", "world"]
    assert bundle.memory_blocks == ["remembered"]
    assert bundle.metadata["skill_mentions"] == ["refactor", "known"]
    assert bundle.metadata["retrieved_memory_block"] == "remembered"
    assert bundle.metadata["retrieval_trace_raw"] == {"route": "RETRIEVE"}
    assert bundle.metadata["retrieval_metadata"] == {"source": "memory2"}
    assert bundle.metadata["history_messages"][0].tool_chain[0].calls[0].name == "read_file"
    request = retrieval.retrieve.await_args.args[0]
    assert request.session_key == "cli:1"
    assert request.history[0].tools_used == ["read_file"]


@pytest.mark.asyncio
async def test_conversation_turn_handler_process_reads_prepare_from_context_store():
    turn_runner = SimpleNamespace(
        run=AsyncMock(return_value=("final", ["shell"], [{"text": "done", "calls": []}], None)),
        last_retry_trace={"retries": 0},
    )
    orchestrator = SimpleNamespace(
        handle_turn=AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="1", content="final")
        )
    )
    deps = SimpleNamespace(
        llm=object(),
        llm_config=object(),
        turn_runner=turn_runner,
        retrieval=SimpleNamespace(
            retrieve=AsyncMock(
                return_value=RetrievalResult(
                    block="remembered",
                    trace=RetrievalTrace(raw={"route": "RETRIEVE"}),
                )
            )
        ),
        orchestrator=orchestrator,
        session=SimpleNamespace(
            session_manager=SimpleNamespace(get_or_create=MagicMock(return_value=_DummySession()))
        ),
        tools=SimpleNamespace(set_context=MagicMock()),
        context=SimpleNamespace(
            skills=SimpleNamespace(list_skills=MagicMock(return_value=[{"name": "refactor"}]))
        ),
    )
    handler = ConversationTurnHandler(deps)
    msg = InboundMessage(
        channel="cli",
        sender="hua",
        chat_id="1",
        content="请用 $refactor 处理一下",
    )

    out = await handler.process(msg, "cli:1")

    assert out.content == "final"
    deps.tools.set_context.assert_called_once_with(channel="cli", chat_id="1")
    turn_runner.run.assert_awaited_once()
    assert turn_runner.run.await_args.kwargs["skill_names"] == ["refactor"]
    assert turn_runner.run.await_args.kwargs["retrieved_memory_block"] == "remembered"
    trace = orchestrator.handle_turn.await_args.kwargs["result"].trace
    assert trace.retrieval == {"raw": {"route": "RETRIEVE"}}
    request = deps.retrieval.retrieve.await_args.args[0]
    assert request.session_key == "cli:1"


@pytest.mark.asyncio
async def test_conversation_turn_handler_process_uses_explicit_session_key_for_retrieval():
    turn_runner = SimpleNamespace(
        run=AsyncMock(return_value=("final", [], [], None)),
        last_retry_trace={},
    )
    orchestrator = SimpleNamespace(
        handle_turn=AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="1", content="final")
        )
    )
    retrieval = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=RetrievalResult(
                block="remembered",
                trace=RetrievalTrace(raw={"route": "RETRIEVE"}),
            )
        )
    )
    deps = SimpleNamespace(
        llm=object(),
        llm_config=object(),
        turn_runner=turn_runner,
        retrieval=retrieval,
        orchestrator=orchestrator,
        session=SimpleNamespace(
            session_manager=SimpleNamespace(
                get_or_create=MagicMock(return_value=_DummySession())
            )
        ),
        tools=SimpleNamespace(set_context=MagicMock()),
        context=SimpleNamespace(
            skills=SimpleNamespace(list_skills=MagicMock(return_value=[]))
        ),
    )
    handler = ConversationTurnHandler(deps)
    msg = InboundMessage(
        channel="telegram",
        sender="hua",
        chat_id="7674283004",
        content="定时任务执行一下",
    )

    await handler.process(msg, "scheduler:job-123")

    request = retrieval.retrieve.await_args.args[0]
    assert request.session_key == "scheduler:job-123"
    assert request.session_key != msg.session_key
