from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.passive_turn import AgentCore, AgentCoreDeps
from agent.core.response_parser import parse_response
from agent.core.runtime_support import TurnRunResult
from agent.core.types import ContextBundle
from agent.lifecycle.facade import TurnLifecycle
from agent.lifecycle.types import AfterReasoningCtx
from bootstrap.wiring import wire_turn_lifecycle
from bus.event_bus import EventBus
from bus.events import InboundMessage
from bus.events_lifecycle import TurnCommitted


class _DummySession:
    def __init__(self, key: str) -> None:
        self.key = key
        self.messages: list[dict[str, object]] = []
        self.metadata: dict[str, object] = {}
        self.last_consolidated = 0

    def get_history(self, max_messages: int = 500) -> list[dict[str, object]]:
        return self.messages[-max_messages:]

    def add_message(self, role: str, content: str, media=None, **kwargs) -> None:
        msg: dict[str, object] = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if media:
            msg["media"] = list(media)
        msg.update(kwargs)
        self.messages.append(msg)


@pytest.mark.asyncio
async def test_context_store_commit_persists_commits_and_dispatches():
    order: list[str] = []
    session = _DummySession("telegram:123")
    presence = SimpleNamespace(record_user_message=MagicMock(side_effect=lambda _key: None))
    session_manager = SimpleNamespace(
        get_or_create=MagicMock(return_value=session),
        append_messages=AsyncMock(side_effect=lambda *_args, **_kwargs: order.append("persist")),
    )
    outbound = SimpleNamespace(dispatch=AsyncMock(side_effect=lambda *_args, **_kwargs: order.append("dispatch") or True))
    event_bus = EventBus()
    committed_events: list[TurnCommitted] = []

    event_bus.on(
        TurnCommitted,
        lambda event: order.append("committed") or committed_events.append(event),
    )

    context_store = SimpleNamespace(
        prepare=AsyncMock(
            return_value=ContextBundle(
                skill_mentions=["refactor"],
                retrieved_memory_block="remembered",
            )
        )
    )
    context = SimpleNamespace(
        render=MagicMock(
            return_value=SimpleNamespace(system_prompt="p", messages=[]),
        )
    )
    reasoner = SimpleNamespace(
        run_turn=AsyncMock(
            return_value=TurnRunResult(
                reply="整理好了",
                tools_used=["noop"],
                tool_chain=[{"text": "", "calls": []}],
                thinking="思考",
                streamed=True,
                context_retry={
                    "selected_plan": "full",
                    "react_stats": {
                        "iteration_count": 3,
                        "turn_input_sum_tokens": 42100,
                        "turn_input_peak_tokens": 18800,
                        "final_call_input_tokens": 17500,
                    },
                },
            )
        )
    )
    tools = SimpleNamespace(
        set_context=MagicMock()
    )
    agent_core = AgentCore(
        AgentCoreDeps(
            session=cast(
                Any,
                SimpleNamespace(
                    session_manager=session_manager,
                    presence=presence,
                ),
            ),
            context_store=cast(Any, context_store),
            context=cast(Any, context),
            tools=cast(Any, tools),
            reasoner=cast(Any, reasoner),
            event_bus=event_bus,
            outbound_port=cast(Any, outbound),
            history_window=500,
        )
    )

    out = await agent_core.process(
        InboundMessage(
            channel="telegram",
            sender="hua",
            chat_id="123",
            content="你好",
            metadata={"req_id": "r1"},
        ),
        "telegram:123",
        dispatch_outbound=True,
    )
    await event_bus.drain()

    assert out.content == "整理好了"
    assert out.media == []
    assert out.metadata["req_id"] == "r1"
    assert out.metadata["tools_used"] == ["noop"]
    assert out.metadata["streamed_reply"] is True
    assert order == ["persist", "committed", "dispatch"]
    presence.record_user_message.assert_called_once_with("telegram:123")
    session_manager.append_messages.assert_awaited_once()
    assert session.messages[-1]["content"] == "整理好了"
    assert session.messages[-1]["reasoning_content"] == "思考"
    assert session.messages[-1].get("cited_memory_ids", []) == []
    assert len(committed_events) == 1
    tc = committed_events[0]
    assert tc.persisted_user_message == "你好"
    assert tc.assistant_response == "整理好了"
    assert tc.meme_media_count == 0
    assert tc.raw_reply == "整理好了"
    assert tc.post_reply_budget["history_window"] == 500
    assert tc.post_reply_budget["history_messages"] == 2
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_turn_committed_omits_user_message_when_user_turn_not_persisted():
    session = _DummySession("cli:direct")
    session_manager = SimpleNamespace(
        get_or_create=MagicMock(return_value=session),
        append_messages=AsyncMock(),
    )
    event_bus = EventBus()
    committed_events: list[TurnCommitted] = []
    event_bus.on(TurnCommitted, lambda event: committed_events.append(event))

    context_store = SimpleNamespace(
        prepare=AsyncMock(
            return_value=ContextBundle(
                skill_mentions=[],
                retrieved_memory_block="",
            )
        )
    )
    context = SimpleNamespace(
        render=MagicMock(
            return_value=SimpleNamespace(system_prompt="p", messages=[]),
        )
    )
    reasoner = SimpleNamespace(
        run_turn=AsyncMock(
            return_value=TurnRunResult(
                reply="完成",
                tools_used=[],
                tool_chain=[],
                thinking=None,
                streamed=False,
                context_retry={},
            )
        )
    )
    agent_core = AgentCore(
        AgentCoreDeps(
            session=cast(
                Any,
                SimpleNamespace(
                    session_manager=session_manager,
                    presence=None,
                ),
            ),
            context_store=cast(Any, context_store),
            context=cast(Any, context),
            tools=cast(
                Any,
                SimpleNamespace(set_context=MagicMock()),
            ),
            reasoner=cast(Any, reasoner),
            event_bus=event_bus,
            outbound_port=cast(
                Any,
                SimpleNamespace(dispatch=AsyncMock(return_value=True)),
            ),
            history_window=500,
        )
    )

    await agent_core.process(
        InboundMessage(
            channel="cli",
            sender="hua",
            chat_id="direct",
            content="内部提示词",
            metadata={"omit_user_turn": True},
        ),
        "cli:direct",
        dispatch_outbound=False,
    )
    await event_bus.drain()

    assert committed_events[0].persisted_user_message is None
    assert committed_events[0].assistant_response == "完成"
    assert [msg["role"] for msg in session.messages] == ["assistant"]
    session_manager.append_messages.assert_awaited_once()
    await event_bus.aclose()


def test_response_parser_keeps_reply_protocols_for_plugins():
    text = "答复正文\n§cited:[mem_1]§ <meme:shy>"

    parsed = parse_response(text, tool_chain=[])

    assert parsed.clean_text == text
    assert parsed.metadata.raw_text == text


# ── 新链 (AfterReasoning + AfterTurn) 端到端测试 ──

