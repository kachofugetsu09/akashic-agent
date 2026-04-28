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
from agent.looping.lifecycle_consumers import (
    register_observe_trace_consumers,
    register_turn_committed_consumers,
)
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
async def test_commit_fanout_enqueues_observe_and_recent_before_return():
    session = _DummySession("telegram:trace")
    session_manager = SimpleNamespace(get_or_create=MagicMock(return_value=session))
    event_bus = EventBus()
    release_recent = asyncio.Event()

    class _Writer:
        def __init__(self) -> None:
            self.events: list[object] = []

        def emit(self, event: object) -> None:
            self.events.append(event)

    async def _refresh_recent_turns(*, session) -> None:
        await release_recent.wait()

    writer = _Writer()
    register_observe_trace_consumers(
        event_bus=event_bus,
        trace=cast(Any, SimpleNamespace(workspace=None, observe_writer=writer)),
    )
    register_turn_committed_consumers(
        event_bus=event_bus,
        consolidation=cast(Any, SimpleNamespace(refresh_recent_turns=AsyncMock(side_effect=_refresh_recent_turns))),
        session_manager=cast(Any, session_manager),
        scheduler=cast(Any, SimpleNamespace(schedule_consolidation=MagicMock())),
        memory_engine=None,
    )

    await event_bus.emit(
        TurnCommitted(
            session_key="telegram:trace",
            channel="telegram",
            chat_id="trace",
            input_message="你好",
            persisted_user_message="你好",
            assistant_response="收到",
            tools_used=[],
            thinking=None,
            raw_reply="收到",
            meme_tag=None,
            meme_media_count=0,
            tool_chain_raw=[],
            tool_call_groups=[],
            retrieval_raw=None,
            post_reply_budget={
                "history_window": 500,
                "history_messages": 1,
                "history_chars": 12,
                "history_tokens": 4,
                "prompt_tokens": 3,
                "next_turn_baseline_tokens": 1,
            },
            react_stats={},
        )
    )

    recent_tasks = [
        task
        for task in asyncio.all_tasks()
        if task.get_name() == "recent_context:telegram:trace"
    ]
    assert writer.events
    assert recent_tasks
    release_recent.set()
    await asyncio.gather(*recent_tasks)
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
    writer = SimpleNamespace(events=[], emit=lambda event: order.append("observe") or writer.events.append(event))
    outbound = SimpleNamespace(dispatch=AsyncMock(side_effect=lambda *_args, **_kwargs: order.append("dispatch") or True))
    event_bus = EventBus()
    committed_events: list[TurnCommitted] = []

    event_bus.on(
        TurnCommitted,
        lambda event: order.append("committed") or committed_events.append(event),
    )
    register_observe_trace_consumers(
        event_bus=event_bus,
        trace=cast(Any, SimpleNamespace(workspace=None, observe_writer=writer)),
    )
    register_turn_committed_consumers(
        event_bus=event_bus,
        consolidation=cast(Any, SimpleNamespace(refresh_recent_turns=AsyncMock())),
        session_manager=cast(Any, session_manager),
        scheduler=cast(Any, SimpleNamespace(schedule_consolidation=MagicMock())),
        memory_engine=None,
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
    assert writer.events
    assert order == ["persist", "committed", "observe", "dispatch"]
    presence.record_user_message.assert_called_once_with("telegram:123")
    session_manager.append_messages.assert_awaited_once()
    assert session.messages[-1]["content"] == "整理好了"
    assert session.messages[-1]["reasoning_content"] == "思考"
    assert session.messages[-1].get("cited_memory_ids", []) == []
    assert len(committed_events) == 1
    tc = committed_events[0]
    assert tc.persisted_user_message == "你好"
    assert tc.assistant_response == "整理好了"
    assert tc.retrieval_raw is None
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


# ── 新链 (AfterReasoning + AfterTurn) 端到端测试 ──


@pytest.mark.asyncio
async def test_new_chain_after_reasoning_persists_meme_and_fires_turn_committed():
    from agent.core.passive_turn import ContextStore

    order: list[str] = []
    session = _DummySession("telegram:456")
    presence = SimpleNamespace(
        record_user_message=MagicMock(side_effect=lambda _key: order.append("presence")),
    )
    session_manager = SimpleNamespace(
        get_or_create=MagicMock(return_value=session),
        peek_next_message_id=MagicMock(return_value="telegram:456:0"),
        append_messages=AsyncMock(side_effect=lambda *a, **kw: order.append("persist")),
    )
    event_bus = EventBus()
    committed_events: list[TurnCommitted] = []
    event_bus.on(
        TurnCommitted,
        lambda event: order.append("committed") or committed_events.append(event),
    )
    dispatch_port = SimpleNamespace(
        dispatch=AsyncMock(side_effect=lambda *a, **kw: order.append("dispatch")),
    )
    decorator = SimpleNamespace(
        decorate=MagicMock(
            return_value=SimpleNamespace(
                content="装饰后内容",
                media=["/tmp/meme.png"],
                tag="shy",
            )
        )
    )
    context_store = SimpleNamespace(
        prepare=AsyncMock(
            return_value=ContextBundle(
                skill_mentions=[],
                retrieved_memory_block="",
            )
        ),
    )
    context = SimpleNamespace(
        render=MagicMock(return_value=SimpleNamespace(system_prompt="p", messages=[])),
    )
    tools = SimpleNamespace(set_context=MagicMock())
    reasoner = SimpleNamespace(
        run_turn=AsyncMock(
            return_value=TurnRunResult(
                reply="原始回复 <meme:shy>\n§cited:[mem_1]§",
                tools_used=["noop"],
                tool_chain=[{"text": "done", "calls": []}],
                thinking="思考",
                streamed=True,
                context_retry={
                    "selected_plan": "full",
                    "react_stats": {
                        "iteration_count": 2,
                        "turn_input_sum_tokens": 5000,
                    },
                },
            )
        ),
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
            context_store=cast(ContextStore, context_store),
            context=cast(Any, context),
            tools=cast(Any, tools),
            reasoner=cast(Any, reasoner),
            event_bus=event_bus,
            outbound_port=cast(Any, dispatch_port),
            history_window=100,
        )
    )
    wire_turn_lifecycle(
        lifecycle=TurnLifecycle(event_bus),
        meme_decorator=cast(Any, decorator),
        active_turn_states={},
    )
    msg = InboundMessage(
        channel="telegram",
        sender="hua",
        chat_id="456",
        content="你好",
        metadata={"req_id": "r2"},
    )

    out = await agent_core.process(msg, "telegram:456")
    await event_bus.drain()

    # 1. meme handler 经过 AfterReasoning chain 被调用
    decorator.decorate.assert_called_once_with("原始回复", meme_tag="shy")

    # 2. outbound 内容来自 meme 装饰后
    assert out.content == "装饰后内容"
    assert out.media == ["/tmp/meme.png"]
    assert out.metadata["req_id"] == "r2"
    assert out.metadata["streamed_reply"] is True

    # 3. persist 写入 session
    assert len(session.messages) == 2
    assert session.messages[0]["role"] == "user"
    assert session.messages[1]["role"] == "assistant"
    assert session.messages[1]["content"] == "装饰后内容"
    assert session.messages[1]["reasoning_content"] == "思考"
    assert session.messages[1]["cited_memory_ids"] == ["mem_1"]
    presence.record_user_message.assert_called_once_with("telegram:456")
    session_manager.append_messages.assert_awaited_once()

    # 4. TurnCommitted 字段正确
    assert len(committed_events) == 1
    tc = committed_events[0]
    assert tc.session_key == "telegram:456"
    assert tc.input_message == "你好"
    assert tc.persisted_user_message == "你好"
    assert tc.assistant_response == "装饰后内容"
    assert tc.tools_used == ["noop"]
    assert tc.thinking == "思考"
    assert tc.raw_reply == "原始回复 <meme:shy>\n§cited:[mem_1]§"
    assert tc.meme_tag == "shy"
    assert tc.meme_media_count == 1
    assert tc.retrieval_raw is None
    assert tc.post_reply_budget["history_window"] == 100
    assert tc.post_reply_budget["history_messages"] == 2
    assert tc.react_stats["iteration_count"] == 2
    assert tc.react_stats["turn_input_sum_tokens"] == 5000

    # 5. 执行顺序: presence → persist → committed → dispatch
    assert order == ["presence", "persist", "committed", "dispatch"]

    # 6. dispatch 实际发送
    dispatch_port.dispatch.assert_awaited_once()

    await event_bus.aclose()
