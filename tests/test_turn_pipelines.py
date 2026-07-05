import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.runtime_support import SessionLike, TurnRunResult
from agent.looping.core import AgentLoop, _supports_stream_events
from agent.looping.interrupt import TurnInterruptState
from agent.lifecycle.facade import TurnLifecycle
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, MemoryServices
from agent.provider import LLMResponse
from agent.retrieval.protocol import (
    MemoryRetrievalPipeline,
    RetrievalRequest,
    RetrievalResult,
)
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import TurnCommitted
from core.memory.engine import MemoryQueryResult
from bootstrap.wiring import wire_turn_lifecycle


class _NoopTool(Tool):
    @property
    def name(self) -> str:
        return "noop"

    @property
    def description(self) -> str:
        return "noop"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs) -> str:
        return "ok"


class _Provider:
    async def chat(self, **kwargs):
        return LLMResponse(content="ok", tool_calls=[])


class _PendingTask:
    def __init__(self) -> None:
        self.cancelled = False

    def done(self) -> bool:
        return False

    def cancel(self) -> None:
        self.cancelled = True


class _CustomRetrieval(MemoryRetrievalPipeline):
    def __init__(self, block: str) -> None:
        self._block = block
        self.requests: list[RetrievalRequest] = []

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.requests.append(request)
        return RetrievalResult(block=self._block)


class _FakeMemoryEngine:
    def read_self(self) -> str:
        return ""

    def read_recent_context(self) -> str:
        return ""

    def get_memory_context(self) -> str:
        return ""

    def has_long_term_memory(self) -> bool:
        return False

    async def query(self, request) -> MemoryQueryResult:
        return MemoryQueryResult(text_block="", records=[], raw={})

    async def refresh_recent_turns(self, request) -> None:
        return None

    async def consolidate(self, request) -> None:
        return None


def test_stream_events_only_support_telegram_private_chat():
    assert _supports_stream_events("telegram", "123")
    assert not _supports_stream_events("telegram", "-1001")
    assert not _supports_stream_events("telegram", "@alice")
    assert not _supports_stream_events("qq", "123")
    assert not _supports_stream_events("cli", "direct")


def test_stream_event_sink_respects_suppression_flag():
    loop = object.__new__(AgentLoop)
    loop._event_bus = EventBus()
    msg = InboundMessage(
        channel="telegram",
        sender="u",
        chat_id="123",
        content="hello",
        metadata={"suppress_stream_events": True},
    )

    assert AgentLoop._build_stream_event_sink(loop, msg) is None


@pytest.mark.asyncio
async def test_process_direct_suppresses_stream_and_memory_when_requested():
    loop = object.__new__(AgentLoop)
    loop._passive_runtime_lock = asyncio.Lock()
    loop._process = AsyncMock(
        return_value=OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="ok",
        )
    )

    result = await AgentLoop.process_direct(
        loop,
        content="天气",
        session_key="scheduler:job",
        channel="telegram",
        chat_id="123",
        omit_user_turn=True,
        skip_post_memory=True,
        skip_memory_retrieval=True,
        disabled_tools=["message_push"],
    )

    msg = loop._process.await_args.args[0]
    assert result == "ok"
    assert msg.metadata == {
        "omit_user_turn": True,
        "skip_post_memory": True,
        "skip_memory_retrieval": True,
        "suppress_stream_events": True,
        "disabled_tools": ["message_push"],
    }
    assert loop._process.await_args.kwargs["dispatch_outbound"] is False


@pytest.mark.asyncio
async def test_process_direct_waits_for_passive_runtime_admission():
    loop = object.__new__(AgentLoop)
    loop._passive_runtime_lock = asyncio.Lock()
    events: list[str] = []

    async def _process(
        msg: InboundMessage,
        session_key: str | None = None,
        busy_session_key: str | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        key = session_key or msg.session_key
        events.append(f"start:{key}")
        if key == "cli:1":
            await asyncio.sleep(0.02)
        events.append(f"end:{key}")
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=key,
        )

    loop._process = _process
    passive_msg = InboundMessage(
        channel="cli",
        sender="u",
        chat_id="1",
        content="hello",
    )
    passive_task = asyncio.create_task(
        AgentLoop._process_with_runtime_admission(loop, passive_msg)
    )
    await asyncio.sleep(0)
    direct_task = asyncio.create_task(
        AgentLoop.process_direct(
            loop,
            content="天气",
            session_key="scheduler:job",
            channel="telegram",
            chat_id="123",
        )
    )

    await asyncio.gather(passive_task, direct_task)

    assert events == [
        "start:cli:1",
        "end:cli:1",
        "start:scheduler:job",
        "end:scheduler:job",
    ]


@pytest.mark.asyncio
async def test_process_uses_busy_session_key_for_processing_state(tmp_path: Path):
    loop = _make_loop(tmp_path)
    state = MagicMock()
    loop._processing_state = state  # type: ignore[attr-defined]
    loop._core_runner.process = AsyncMock(  # type: ignore[attr-defined]
        return_value=OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="ok",
        )
    )
    msg = InboundMessage(
        channel="telegram",
        sender="user",
        chat_id="123",
        content="天气",
    )

    outbound = await loop._process(
        msg,
        session_key="scheduler:job",
        busy_session_key="telegram:123",
        dispatch_outbound=False,
    )

    assert outbound.content == "ok"
    state.enter.assert_called_once_with("telegram:123")
    state.exit.assert_called_once_with("telegram:123")
    loop._core_runner.process.assert_awaited_once_with(  # type: ignore[attr-defined]
        msg,
        "scheduler:job",
        dispatch_outbound=False,
    )


def _make_loop(
    tmp_path: Path,
    *,
    retrieval_pipeline: MemoryRetrievalPipeline | None = None,
) -> AgentLoop:
    tools = ToolRegistry()
    tools.register(_NoopTool())
    return AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=cast(Any, _Provider()),
            light_provider=cast(Any, _Provider()),
            tools=tools,
            session_manager=MagicMock(),
            workspace=tmp_path,
            memory_services=MemoryServices(engine=cast(Any, _FakeMemoryEngine())),
            retrieval_pipeline=retrieval_pipeline,
        ),
        AgentLoopConfig(),
    )


def test_agent_loop_uses_custom_retrieval_pipeline(tmp_path: Path):
    custom_retrieval = _CustomRetrieval(block="MEM_BLOCK")
    loop = _make_loop(
        tmp_path,
        retrieval_pipeline=custom_retrieval,
    )
    session = MagicMock()
    session.key = "cli:1"
    session.messages = []
    session.metadata = {}
    session.get_history = MagicMock(
        return_value=[{"role": "user", "content": f"m{i}"} for i in range(200)]
    )
    session.add_message = MagicMock()
    loop.session_manager.get_or_create.return_value = session
    loop.session_manager.append_messages = AsyncMock(return_value=None)
    loop._reasoner.run_turn = AsyncMock(return_value=TurnRunResult(reply="ok"))

    msg = InboundMessage(channel="cli", sender="u", chat_id="1", content="hello")
    asyncio.run(loop._core_runner.process(msg, msg.session_key))

    assert custom_retrieval.requests
    assert custom_retrieval.requests[0].message == "hello"
    run_kwargs = loop._reasoner.run_turn.await_args.kwargs
    assert "base_history" in run_kwargs
    assert run_kwargs["base_history"] is None


def test_agent_loop_fanouts_turn_committed_from_passive_turn(tmp_path: Path):
    loop = _make_loop(
        tmp_path,
        retrieval_pipeline=_CustomRetrieval(block="MEM_BLOCK"),
    )
    turn_events: list[TurnCommitted] = []
    loop._event_bus.on(TurnCommitted, lambda event: turn_events.append(event))
    session = MagicMock()
    session.key = "cli:1"
    session.messages = []
    session.metadata = {}
    session.get_history = MagicMock(return_value=[])
    session.add_message = MagicMock(
        side_effect=lambda role, content, **kwargs: session.messages.append(
            {"role": role, "content": content, **kwargs}
        )
    )
    loop.session_manager.get_or_create.return_value = session
    loop.session_manager.append_messages = AsyncMock(return_value=None)
    loop._reasoner.run_turn = AsyncMock(
        return_value=TurnRunResult(
            reply="ok",
            tool_chain=[
                {
                    "text": "",
                    "calls": [
                        {
                            "name": "noop",
                            "arguments": {"x": 1},
                            "result": "done",
                        }
                    ],
                }
            ],
            context_retry={
                "react_stats": {
                    "iteration_count": 1,
                    "turn_input_sum_tokens": 100,
                }
            },
        )
    )

    msg = InboundMessage(channel="cli", sender="u", chat_id="1", content="hello")

    async def _process_and_drain() -> None:
        await loop._core_runner.process(msg, msg.session_key)
        await loop._event_bus.drain()
        await loop._event_bus.aclose()

    asyncio.run(_process_and_drain())

    assert turn_events
    turn_event = turn_events[0]
    assert turn_event.session_key == "cli:1"
    assert turn_event.persisted_user_message == "hello"
    assert turn_event.assistant_response == "ok"
    assert turn_event.tool_chain_raw[0]["calls"][0]["name"] == "noop"
    assert turn_event.react_stats["iteration_count"] == 1
    assert turn_event.react_stats["turn_input_sum_tokens"] == 100


def test_request_interrupt_uses_active_turn_state_snapshot(tmp_path: Path):
    loop = _make_loop(tmp_path)
    session_key = "telegram:123"
    pending = _PendingTask()
    loop._active_tasks[session_key] = pending  # type: ignore[attr-defined]
    loop._active_turn_states[session_key] = TurnInterruptState(  # type: ignore[attr-defined]
        session_key=session_key,
        original_user_message="原始消息 A",
    )

    result = loop.request_interrupt(session_key, sender="1", command="/stop")

    assert result.status == "interrupted"
    assert pending.cancelled is True
    assert loop._interrupt_states[session_key].original_user_message == "原始消息 A"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_resumed_interrupt_state_completes_normally(tmp_path: Path):
    loop = _make_loop(tmp_path)
    session_key = "telegram:123"
    loop._interrupt_states[session_key] = TurnInterruptState(  # type: ignore[attr-defined]
        session_key=session_key,
        original_user_message="原始消息 A",
        partial_reply="半截回答",
        tools_used=["noop"],
        tool_chain_partial=[{"text": "", "calls": []}],
    )
    session_messages: list[dict[str, Any]] = []

    def _add_message(role: str, content: str, **kwargs: Any) -> None:
        session_messages.append({"role": role, "content": content, **kwargs})

    session = SimpleNamespace(
        key=session_key,
        messages=session_messages,
        add_message=_add_message,
    )
    loop.session_manager.get_or_create.return_value = session
    loop.session_manager.append_messages = AsyncMock(return_value=None)

    async def _slow_process(*args, **kwargs):
        await asyncio.sleep(0.05)
        return MagicMock(content="ok")

    loop._core_runner.process = AsyncMock(side_effect=_slow_process)  # type: ignore[attr-defined]

    msg = InboundMessage(
        channel="telegram",
        sender="1",
        chat_id="123",
        content="补充 B",
    )
    outbound = await loop._process(msg)

    assert outbound.content == "ok"
    assert session_key not in loop._interrupt_states  # type: ignore[attr-defined]
    processed_msg = loop._core_runner.process.await_args.args[0]  # type: ignore[attr-defined]
    assert processed_msg.content == "补充 B"
    assert "【上一轮任务" not in processed_msg.content
    assert session.messages[0]["content"] == "原始消息 A"
    assert session.messages[1]["content"] == "[interrupted]"
    assert session.messages[1]["tools_used"] == ["noop"]
    loop.session_manager.append_messages.assert_awaited_once_with(
        session,
        session.messages,
    )


@pytest.mark.asyncio
async def test_agent_loop_afterstep_fires_with_turn_lifecycle_wiring(tmp_path: Path):
    loop = _make_loop(tmp_path)
    session_key = "cli:123"
    loop._active_turn_states[session_key] = TurnInterruptState(
        session_key=session_key,
        original_user_message="hello",
    )
    wire_turn_lifecycle(
        lifecycle=TurnLifecycle(loop._event_bus),
        active_turn_states=loop.active_turn_states,
    )
    msg = InboundMessage(channel="cli", sender="u", chat_id="123", content="你好")
    session = SimpleNamespace(
        key=session_key,
        messages=[],
        metadata={},
        last_consolidated=0,
        get_history=MagicMock(return_value=[]),
        add_message=MagicMock(),
    )
    loop.session_manager.get_or_create.return_value = session

    await loop._reasoner.run_turn(
        msg=msg,
        session=cast(SessionLike, session),
        base_history=[],
    )

    state = loop._active_turn_states[session_key]
    assert state.partial_reply == "ok"
    assert state.tools_used == []
    assert state.tool_chain_partial == []
