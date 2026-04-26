from __future__ import annotations
import asyncio
from typing import Any, cast

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.context_store import DefaultContextStore
from agent.core.response_parser import ResponseMetadata, parse_response
from agent.looping.lifecycle_consumers import (
    register_observe_trace_consumers,
    register_turn_committed_consumers,
)
from agent.retrieval.protocol import RetrievalResult
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
async def test_commit_fanout_enqueues_observe_and_recent_before_return():
    session = _DummySession("telegram:trace")
    session_manager = SimpleNamespace(
        get_or_create=MagicMock(return_value=session),
        append_messages=AsyncMock(),
    )
    event_bus = EventBus()
    release_recent = asyncio.Event()
    recent_started = asyncio.Event()

    class _Writer:
        def __init__(self) -> None:
            self.events: list[object] = []

        def emit(self, event: object) -> None:
            self.events.append(event)

    async def _refresh_recent_turns(*, session) -> None:
        recent_started.set()
        await release_recent.wait()

    writer = _Writer()
    consolidation = SimpleNamespace(
        refresh_recent_turns=AsyncMock(side_effect=_refresh_recent_turns)
    )
    register_observe_trace_consumers(
        event_bus=event_bus,
        trace=cast(Any, SimpleNamespace(observe_writer=writer)),
    )
    register_turn_committed_consumers(
        event_bus=event_bus,
        consolidation=cast(Any, consolidation),
        session_manager=cast(Any, session_manager),
        scheduler=cast(Any, SimpleNamespace(schedule_consolidation=MagicMock())),
        memory_engine=None,
    )
    store = DefaultContextStore(
        retrieval=cast(Any, SimpleNamespace(retrieve=AsyncMock(return_value=RetrievalResult(block="")))),
        context=cast(Any, SimpleNamespace(skills=SimpleNamespace(list_skills=MagicMock(return_value=[])))),
        session=cast(Any, SimpleNamespace(session_manager=session_manager, presence=None)),
        trace=cast(Any, SimpleNamespace(workspace=Path("."), observe_writer=writer)),
        outbound=cast(Any, SimpleNamespace(dispatch=AsyncMock())),
        event_bus=event_bus,
    )

    await store.commit(
        msg=InboundMessage(
            channel="telegram",
            sender="hua",
            chat_id="trace",
            content="你好",
        ),
        session_key="telegram:trace",
        reply="收到",
        response_metadata=ResponseMetadata(raw_text="收到"),
        tools_used=[],
        tool_chain=[],
        thinking=None,
        streamed_reply=False,
        retrieval_raw=None,
        context_retry={},
    )

    recent_tasks = [
        task
        for task in asyncio.all_tasks()
        if task.get_name() == "recent_context:telegram:trace"
    ]
    assert writer.events
    assert cast(Any, writer.events[0]).source == "agent"
    assert recent_tasks

    release_recent.set()
    await asyncio.gather(*recent_tasks)
    assert recent_started.is_set()
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_context_store_commit_persists_commits_and_dispatches():
    order: list[str] = []
    session = _DummySession("telegram:123")
    presence = SimpleNamespace(record_user_message=MagicMock(side_effect=lambda _key: None))
    session_manager = SimpleNamespace(
        get_or_create=MagicMock(return_value=session),
        append_messages=AsyncMock(side_effect=lambda *_args, **_kwargs: order.append("persist")),
    )
    writer = SimpleNamespace(
        events=[],
        emit=lambda event: order.append("observe") or writer.events.append(event),
    )
    outbound = SimpleNamespace(
        dispatch=AsyncMock(side_effect=lambda *_args, **_kwargs: order.append("dispatch") or True)
    )
    event_bus = EventBus()
    committed_events: list[TurnCommitted] = []
    event_bus.on(
        TurnCommitted,
        lambda event: order.append("committed") or committed_events.append(event),
    )
    decorator = SimpleNamespace(
        decorate=MagicMock(
            return_value=SimpleNamespace(
                content="整理好了",
                media=["/tmp/meme.png"],
                tag="shy",
            )
        )
    )
    store = DefaultContextStore(
        retrieval=cast(Any, SimpleNamespace(retrieve=AsyncMock(return_value=RetrievalResult(block="")))),
        context=cast(Any, SimpleNamespace(skills=SimpleNamespace(list_skills=MagicMock(return_value=[])))),
        session=cast(Any, SimpleNamespace(session_manager=session_manager, presence=presence)),
        trace=cast(Any, SimpleNamespace(workspace=Path("."), observe_writer=writer)),
        outbound=cast(Any, outbound),
        meme_decorator=cast(Any, decorator),
        event_bus=event_bus,
    )
    msg = InboundMessage(
        channel="telegram",
        sender="hua",
        chat_id="123",
        content="你好",
        metadata={"req_id": "r1"},
    )
    out = await store.commit(
        msg=msg,
        session_key="telegram:123",
        reply="整理好了",
        response_metadata=ResponseMetadata(
            raw_text="<meme:shy> 整理好了\n§cited:[mem_1]§",
            cited_memory_ids=["mem_1"],
            meme_tag="shy",
        ),
        tools_used=["noop"],
        tool_chain=[{"text": "", "calls": []}],
        thinking="思考",
        streamed_reply=True,
        retrieval_raw={"route": "RETRIEVE"},
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
    await event_bus.drain()

    assert out.content == "整理好了"
    assert out.media == ["/tmp/meme.png"]
    assert out.metadata["req_id"] == "r1"
    assert out.metadata["tools_used"] == ["noop"]
    assert out.metadata["streamed_reply"] is True
    presence.record_user_message.assert_called_once_with("telegram:123")
    session_manager.append_messages.assert_awaited_once()
    assert writer.events == []
    assert out.metadata["tool_chain"][0]["text"] == ""
    outbound.dispatch.assert_awaited_once()
    assert order == [
        "persist",
        "committed",
        "dispatch",
    ]
    assert committed_events[0].session_key == "telegram:123"
    assert committed_events[0].input_message == "你好"
    assert committed_events[0].persisted_user_message == "你好"
    assert committed_events[0].assistant_response == "整理好了"
    assert committed_events[0].tools_used == ["noop"]
    assert committed_events[0].thinking == "思考"
    assert committed_events[0].raw_reply == "<meme:shy> 整理好了\n§cited:[mem_1]§"
    assert committed_events[0].meme_tag == "shy"
    assert committed_events[0].meme_media_count == 1
    assert committed_events[0].retrieval_raw == {"route": "RETRIEVE"}
    assert committed_events[0].tool_chain_raw[0]["text"] == ""
    assert committed_events[0].tool_call_groups[0].text == ""
    assert committed_events[0].post_reply_budget["history_window"] == 500
    assert committed_events[0].post_reply_budget["history_messages"] == 2
    assert committed_events[0].post_reply_budget["history_chars"] > 0
    assert committed_events[0].post_reply_budget["history_tokens"] == max(
        1,
        committed_events[0].post_reply_budget["history_chars"] // 3,
    )
    assert committed_events[0].react_stats["iteration_count"] == 3
    assert committed_events[0].react_stats["turn_input_sum_tokens"] == 42100
    assert committed_events[0].react_stats["turn_input_peak_tokens"] == 18800
    assert committed_events[0].react_stats["final_call_input_tokens"] == 17500
    assert committed_events[0].extra == {}
    assert session.messages[-1]["content"] == "整理好了"
    assert session.messages[-1]["reasoning_content"] == "思考"
    assert session.messages[-1]["cited_memory_ids"] == ["mem_1"]
    decorator.decorate.assert_called_once_with("整理好了", meme_tag="shy")
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
    store = DefaultContextStore(
        retrieval=cast(Any, SimpleNamespace(retrieve=AsyncMock(return_value=RetrievalResult(block="")))),
        context=cast(Any, SimpleNamespace(skills=SimpleNamespace(list_skills=MagicMock(return_value=[])))),
        session=cast(Any, SimpleNamespace(session_manager=session_manager, presence=None)),
        trace=cast(Any, SimpleNamespace(workspace=Path("."), observe_writer=None)),
        outbound=cast(Any, SimpleNamespace(dispatch=AsyncMock())),
        event_bus=event_bus,
    )

    await store.commit(
        msg=InboundMessage(
            channel="cli",
            sender="hua",
            chat_id="direct",
            content="内部提示词",
            metadata={"omit_user_turn": True},
        ),
        session_key="cli:direct",
        reply="完成",
        response_metadata=ResponseMetadata(raw_text="完成"),
        tools_used=[],
        tool_chain=[],
        thinking=None,
        streamed_reply=False,
        retrieval_raw=None,
        context_retry={},
    )
    await event_bus.drain()

    assert committed_events[0].input_message == "内部提示词"
    assert committed_events[0].persisted_user_message is None
    assert committed_events[0].assistant_response == "完成"
    assert [msg["role"] for msg in session.messages] == ["assistant"]
    session_manager.append_messages.assert_awaited_once_with(session, session.messages[-1:])
    await event_bus.aclose()


def test_response_parser_strips_ascii_marker_only_at_end():
    parsed = parse_response("答复正文\n§cited:[mem_1,mem-2]§", tool_chain=[])

    assert parsed.clean_text == "答复正文"
    assert parsed.metadata.cited_memory_ids == ["mem_1", "mem-2"]


def test_response_parser_strips_marker_with_spaces_after_commas():
    parsed = parse_response("答复正文\n§cited:[mem_1, mem-2]§", tool_chain=[])

    assert parsed.clean_text == "答复正文"
    assert parsed.metadata.cited_memory_ids == ["mem_1", "mem-2"]


def test_response_parser_keeps_body_text_when_marker_not_at_end():
    text = "正文里提到 §cited:[mem_1]§ 这串文本，但不是协议行。\n后面还有内容"

    parsed = parse_response(text, tool_chain=[])

    assert parsed.clean_text == text
    assert parsed.metadata.cited_memory_ids == []


def test_response_parser_strips_meme_tags_and_keeps_first_tag():
    parsed = parse_response("好的 <meme:HAPPY> 收到 <meme:agree>", tool_chain=[])

    assert parsed.clean_text == "好的  收到"
    assert parsed.metadata.meme_tag == "happy"


def test_response_parser_tool_chain_fallback_uses_recall_memory_cited_item_ids():
    tool_chain = [
        {
            "text": "thinking",
            "calls": [
                {
                    "name": "recall_memory",
                    "result": "{\"count\":2,\"cited_item_ids\":[\"mem_1\",\"mem_2\"]}",
                }
            ],
        }
    ]

    parsed = parse_response("答复正文", tool_chain=tool_chain)

    assert parsed.clean_text == "答复正文"
    assert parsed.metadata.cited_memory_ids == ["mem_1", "mem_2"]


def test_response_parser_tool_chain_fallback_uses_item_ids():
    tool_chain = [
        {
            "text": "thinking",
            "calls": [
                {
                    "name": "recall_memory",
                    "result": (
                        "{\"count\":2,\"items\":["
                        "{\"id\":\"mem_1\"},"
                        "{\"id\":\"mem_2\"}"
                        "]}"
                    ),
                }
            ],
        }
    ]

    parsed = parse_response("答复正文", tool_chain=tool_chain)

    assert parsed.metadata.cited_memory_ids == ["mem_1", "mem_2"]
