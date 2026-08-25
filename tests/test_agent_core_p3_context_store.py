from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from agent.core.passive_support import (
    build_post_reply_context_budget,
    estimate_history_budget,
)
from agent.core.passive_turn import DefaultContextStore
from bus.events import InboundMessage


class _DummySession:
    def __init__(self) -> None:
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

    def get_history(self, max_messages: int = 500) -> list[dict]:
        return self.messages[-max_messages:]


@pytest.mark.asyncio
async def test_default_context_store_prepares_only_history_and_skill_mentions() -> None:
    context = SimpleNamespace(
        skills=SimpleNamespace(
            list_skill_records=MagicMock(
                return_value=[
                    SimpleNamespace(name="refactor"),
                    SimpleNamespace(name="known"),
                ]
            )
        )
    )
    bundle = await DefaultContextStore(context=cast(Any, context)).prepare(
        msg=InboundMessage(
            channel="cli",
            sender="hua",
            chat_id="1",
            content="请用 $refactor 再来一次 $known $refactor",
            timestamp=datetime(2026, 4, 4, 20, 0, 0),
        ),
        session_key="cli:1",
        session=cast(Any, _DummySession()),
    )

    assert bundle.skill_mentions == ["refactor", "known"]
    assert bundle.history_messages[0].tool_chain[0].calls[0].name == "read_file"


@pytest.mark.asyncio
async def test_default_context_store_can_omit_session_history() -> None:
    context = SimpleNamespace(
        skills=SimpleNamespace(list_skill_records=MagicMock(return_value=[]))
    )
    bundle = await DefaultContextStore(context=cast(Any, context)).prepare(
        msg=InboundMessage(
            channel="scheduler",
            sender="scheduler",
            chat_id="job-1",
            content="查询北京天气",
            metadata={"skip_session_history": True},
        ),
        session_key="scheduler:job-1",
        session=cast(Any, _DummySession()),
    )

    assert bundle.history_messages == []


def test_estimate_history_budget_returns_serialized_history_size() -> None:
    stats = estimate_history_budget(
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


def test_build_post_reply_context_budget_combines_history_and_prompt() -> None:
    context = SimpleNamespace(
        last_debug_breakdown=[
            SimpleNamespace(est_tokens=100),
            SimpleNamespace(est_tokens=250),
        ]
    )
    budget = build_post_reply_context_budget(
        context=cast(Any, context),
        history=[{"role": "user", "content": "你好"}],
    )
    assert "history_window" not in budget
    assert budget["history_messages"] == 1
    assert budget["history_chars"] > 0
    assert budget["history_tokens"] == max(1, budget["history_chars"] // 3)
    assert budget["prompt_tokens"] == 350
    assert budget["next_turn_baseline_tokens"] == budget["history_tokens"] + 350
