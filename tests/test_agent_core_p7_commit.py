from __future__ import annotations
from typing import Any, cast

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.context_store import DefaultContextStore
from agent.core.response_parser import ResponseMetadata, parse_response
from agent.retrieval.protocol import RetrievalResult
from bus.event_bus import EventBus
from bus.events import InboundMessage
from bus.events_lifecycle import TurnPersisted


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
async def test_context_store_commit_persists_observes_schedules_and_dispatches():
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
    post_turn = SimpleNamespace(
        events=[],
        schedule=lambda event: order.append("post_turn") or post_turn.events.append(event),
    )
    outbound = SimpleNamespace(
        dispatch=AsyncMock(side_effect=lambda *_args, **_kwargs: order.append("dispatch") or True)
    )
    event_bus = EventBus()
    persisted_events: list[TurnPersisted] = []
    persisted_snapshots: list[dict[str, object]] = []

    def _capture_and_mutate_persisted(event: TurnPersisted) -> None:
        order.append("persisted")
        persisted_events.append(event)
        persisted_snapshots.append(
            {
                "history_window": event.post_reply_budget["history_window"],
                "iteration_count": event.react_stats["iteration_count"],
                "tool_text": event.tool_chain[0]["text"],
            }
        )
        event.tool_chain[0]["text"] = "mutated"
        event.post_reply_budget["history_window"] = 1
        event.react_stats["iteration_count"] = 99

    event_bus.on(
        TurnPersisted,
        _capture_and_mutate_persisted,
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
        post_turn=cast(Any, post_turn),
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
    post_turn_action = SimpleNamespace(
        run=AsyncMock(side_effect=lambda: order.append("post_turn_action"))
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
        post_turn_actions=[post_turn_action],
    )

    assert out.content == "整理好了"
    assert out.media == ["/tmp/meme.png"]
    assert out.metadata["req_id"] == "r1"
    assert out.metadata["tools_used"] == ["noop"]
    assert out.metadata["streamed_reply"] is True
    presence.record_user_message.assert_called_once_with("telegram:123")
    session_manager.append_messages.assert_awaited_once()
    assert writer.events == []
    assert post_turn.events[0].assistant_response == "整理好了"
    assert post_turn.events[0].tool_chain[0].text == ""
    assert out.metadata["tool_chain"][0]["text"] == ""
    outbound.dispatch.assert_awaited_once()
    post_turn_action.run.assert_awaited_once()
    assert order == [
        "persist",
        "persisted",
        "post_turn",
        "post_turn_action",
        "dispatch",
    ]
    assert persisted_events[0].session_key == "telegram:123"
    assert persisted_events[0].user_message == "你好"
    assert persisted_events[0].assistant_response == "整理好了"
    assert persisted_events[0].tools_used == ["noop"]
    assert persisted_events[0].thinking == "思考"
    assert persisted_events[0].raw_reply == "<meme:shy> 整理好了\n§cited:[mem_1]§"
    assert persisted_events[0].meme_tag == "shy"
    assert persisted_events[0].meme_media_count == 1
    assert persisted_events[0].retrieval_raw == {"route": "RETRIEVE"}
    assert persisted_snapshots[0]["tool_text"] == ""
    assert persisted_snapshots[0]["history_window"] == 500
    assert persisted_snapshots[0]["iteration_count"] == 3
    assert persisted_events[0].post_reply_budget["history_window"] == 1
    assert persisted_events[0].react_stats["iteration_count"] == 99
    assert persisted_events[0].post_reply_budget["history_messages"] == 2
    assert persisted_events[0].post_reply_budget["history_chars"] > 0
    assert persisted_events[0].post_reply_budget["history_tokens"] == max(
        1,
        persisted_events[0].post_reply_budget["history_chars"] // 3,
    )
    assert persisted_events[0].react_stats["turn_input_sum_tokens"] == 42100
    assert persisted_events[0].react_stats["turn_input_peak_tokens"] == 18800
    assert persisted_events[0].react_stats["final_call_input_tokens"] == 17500
    assert session.messages[-1]["content"] == "整理好了"
    assert session.messages[-1]["reasoning_content"] == "思考"
    assert session.messages[-1]["cited_memory_ids"] == ["mem_1"]
    decorator.decorate.assert_called_once_with("整理好了", meme_tag="shy")


@pytest.mark.asyncio
async def test_turn_persisted_omits_user_message_when_user_turn_not_persisted():
    session = _DummySession("cli:direct")
    session_manager = SimpleNamespace(
        get_or_create=MagicMock(return_value=session),
        append_messages=AsyncMock(),
    )
    event_bus = EventBus()
    persisted_events: list[TurnPersisted] = []
    event_bus.on(TurnPersisted, lambda event: persisted_events.append(event))
    store = DefaultContextStore(
        retrieval=cast(Any, SimpleNamespace(retrieve=AsyncMock(return_value=RetrievalResult(block="")))),
        context=cast(Any, SimpleNamespace(skills=SimpleNamespace(list_skills=MagicMock(return_value=[])))),
        session=cast(Any, SimpleNamespace(session_manager=session_manager, presence=None)),
        trace=cast(Any, SimpleNamespace(workspace=Path("."), observe_writer=None)),
        post_turn=cast(Any, SimpleNamespace(schedule=MagicMock())),
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

    assert persisted_events[0].user_message is None
    assert persisted_events[0].assistant_response == "完成"
    assert [msg["role"] for msg in session.messages] == ["assistant"]
    session_manager.append_messages.assert_awaited_once_with(session, session.messages[-1:])


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
