from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.context_store import (
    DefaultContextStore,
    _build_post_reply_context_budget,
    _estimate_history_budget,
)
from agent.core.types import RetrievalTrace
from agent.retrieval.protocol import RetrievalResult
from bus.events import InboundMessage


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
    assert bundle.skill_mentions == ["refactor", "known"]
    assert bundle.retrieved_memory_block == "remembered"
    assert bundle.retrieval_trace_raw == {"route": "RETRIEVE"}
    assert bundle.retrieval_metadata == {"source": "memory2"}
    assert bundle.history_messages[0].tool_chain[0].calls[0].name == "read_file"
    assert bundle.metadata == {}
    request = retrieval.retrieve.await_args.args[0]
    assert request.session_key == "cli:1"
    assert request.history[0].tools_used == ["read_file"]


@pytest.mark.asyncio
async def test_default_context_store_prepare_uses_explicit_session_key_for_retrieval():
    retrieval = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=RetrievalResult(
                block="remembered",
                trace=RetrievalTrace(raw={"route": "RETRIEVE"}),
            )
        )
    )
    context = SimpleNamespace(
        skills=SimpleNamespace(list_skills=MagicMock(return_value=[]))
    )
    store = DefaultContextStore(retrieval=retrieval, context=context)
    session = _DummySession()
    msg = InboundMessage(
        channel="telegram",
        sender="hua",
        chat_id="7674283004",
        content="定时任务执行一下",
    )

    await store.prepare(msg=msg, session_key="scheduler:job-123", session=session)

    request = retrieval.retrieve.await_args.args[0]
    assert request.session_key == "scheduler:job-123"
    assert request.session_key != msg.session_key


def test_estimate_history_budget_returns_serialized_history_size():
    stats = _estimate_history_budget(
        [
            {"role": "user", "content": "你好"},
            {
                "role": "assistant",
                "content": "收到",
                "tool_calls": [{"id": "call-1", "name": "read_file"}],
            },
        ]
    )

    assert stats["messages"] == 2
    assert stats["chars"] > 0
    assert stats["tokens"] == max(1, stats["chars"] // 3)


def test_build_post_reply_context_budget_combines_history_and_prompt():
    context = SimpleNamespace(
        last_debug_breakdown=[
            SimpleNamespace(est_tokens=100),
            SimpleNamespace(est_tokens=250),
        ]
    )

    budget = _build_post_reply_context_budget(
        context=context,
        history=[{"role": "user", "content": "你好"}],
        history_window=40,
    )

    assert budget["history_window"] == 40
    assert budget["history_messages"] == 1
    assert budget["history_chars"] > 0
    assert budget["history_tokens"] == max(1, budget["history_chars"] // 3)
    assert budget["prompt_tokens"] == 350
    assert budget["next_turn_baseline_tokens"] == (
        budget["history_tokens"] + 350
    )
