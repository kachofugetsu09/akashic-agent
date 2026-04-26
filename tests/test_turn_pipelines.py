import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.types import ToolCall, ToolCallGroup
from agent.core.runtime_support import TurnRunResult
from agent.looping.core import AgentLoop, _supports_stream_events
from agent.looping.interrupt import TurnInterruptState
from agent.looping.lifecycle_consumers import register_turn_committed_consumers
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps
from agent.memory import MemoryStore
from agent.provider import LLMResponse
from agent.retrieval.protocol import (
    MemoryRetrievalPipeline,
    RetrievalRequest,
    RetrievalResult,
)
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from bus.events import InboundMessage
from bus.events_lifecycle import TurnCommitted
from core.memory.engine import MemoryIngestRequest
from core.memory.port import DefaultMemoryPort


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


def test_stream_events_only_support_telegram_private_chat():
    assert _supports_stream_events("telegram", "123")
    assert not _supports_stream_events("telegram", "-1001")
    assert not _supports_stream_events("telegram", "@alice")
    assert not _supports_stream_events("qq", "123")
    assert not _supports_stream_events("cli", "direct")


def _make_loop(
    tmp_path: Path,
    *,
    retrieval_pipeline: MemoryRetrievalPipeline | None = None,
    observe_writer: object | None = None,
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
            memory_port=DefaultMemoryPort(MemoryStore(tmp_path)),
            retrieval_pipeline=retrieval_pipeline,
            observe_writer=observe_writer,
        ),
        AgentLoopConfig(),
    )


@pytest.mark.asyncio
async def test_turn_committed_recent_context_consumer_serializes_same_session():
    event_bus = EventBus()
    session = MagicMock()
    session_manager = SimpleNamespace(get_or_create=MagicMock(return_value=session))
    active = 0
    max_active = 0
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    finished: list[str] = []

    async def _refresh_recent_turns(*, session) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if len(finished) == 0:
            first_started.set()
            await release_first.wait()
        finished.append(str(session.key))
        active -= 1

    session.key = "cli:1"
    consolidation = SimpleNamespace(
        refresh_recent_turns=AsyncMock(side_effect=_refresh_recent_turns)
    )
    scheduler = SimpleNamespace(schedule_consolidation=MagicMock())
    register_turn_committed_consumers(
        event_bus=event_bus,
        consolidation=cast(Any, consolidation),
        session_manager=cast(Any, session_manager),
        scheduler=cast(Any, scheduler),
        memory_engine=None,
    )
    event_a = TurnCommitted(
        session_key="cli:1",
        channel="cli",
        chat_id="1",
        input_message="a",
        persisted_user_message="a",
        assistant_response="ok-a",
        tools_used=[],
        timestamp=datetime.now(),
    )
    event_b = TurnCommitted(
        session_key="cli:1",
        channel="cli",
        chat_id="1",
        input_message="b",
        persisted_user_message="b",
        assistant_response="ok-b",
        tools_used=[],
        timestamp=datetime.now(),
    )

    event_bus.enqueue(event_a)
    await event_bus.drain()
    await first_started.wait()
    event_bus.enqueue(event_b)
    await event_bus.drain()
    await asyncio.sleep(0)

    assert max_active == 1

    release_first.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert max_active == 1
    assert consolidation.refresh_recent_turns.await_count == 2
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_turn_committed_consumers_schedule_consolidation_and_ingest_memory():
    event_bus = EventBus()
    session = MagicMock()
    session.key = "cli:1"
    session_manager = SimpleNamespace(get_or_create=MagicMock(return_value=session))
    scheduler = SimpleNamespace(schedule_consolidation=MagicMock())
    consolidation = SimpleNamespace(refresh_recent_turns=AsyncMock())
    engine = MagicMock()
    engine.ingest = AsyncMock(return_value=MagicMock())
    register_turn_committed_consumers(
        event_bus=event_bus,
        consolidation=cast(Any, consolidation),
        session_manager=cast(Any, session_manager),
        scheduler=cast(Any, scheduler),
        memory_engine=cast(Any, engine),
    )

    event_bus.enqueue(
        TurnCommitted(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            input_message="hello",
            persisted_user_message="hello",
            assistant_response="ok",
            tools_used=["tool_a"],
            tool_call_groups=[
                ToolCallGroup(
                    text="t",
                    calls=[
                        ToolCall(
                            call_id="c1",
                            name="tool_a",
                            arguments={"x": 1},
                            result="done",
                        )
                    ],
                )
            ],
            timestamp=datetime.now(),
        )
    )
    await event_bus.drain()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    scheduler.schedule_consolidation.assert_called_once_with(session, "cli:1")
    engine.ingest.assert_awaited_once()
    request = engine.ingest.await_args.args[0]
    assert isinstance(request, MemoryIngestRequest)
    assert request.source_kind == "conversation_turn"
    assert request.scope.session_key == "cli:1"
    assert request.metadata["source_ref"] == "cli:1@post_response"
    content = cast(dict[str, object], request.content)
    tool_chain = cast(list[dict[str, object]], content["tool_chain"])
    calls = cast(list[dict[str, object]], tool_chain[0]["calls"])
    assert calls[0]["name"] == "tool_a"
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_turn_committed_consumers_skip_post_memory():
    event_bus = EventBus()
    session = MagicMock()
    session.key = "cli:1"
    session_manager = SimpleNamespace(get_or_create=MagicMock(return_value=session))
    scheduler = SimpleNamespace(schedule_consolidation=MagicMock())
    consolidation = SimpleNamespace(refresh_recent_turns=AsyncMock())
    engine = MagicMock()
    engine.ingest = AsyncMock(return_value=MagicMock())
    register_turn_committed_consumers(
        event_bus=event_bus,
        consolidation=cast(Any, consolidation),
        session_manager=cast(Any, session_manager),
        scheduler=cast(Any, scheduler),
        memory_engine=cast(Any, engine),
    )

    event_bus.enqueue(
        TurnCommitted(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            input_message="hello",
            persisted_user_message="hello",
            assistant_response="ok",
            tools_used=[],
            timestamp=datetime.now(),
            extra={"skip_post_memory": True},
        )
    )
    await event_bus.drain()
    await asyncio.sleep(0)

    engine.ingest.assert_not_awaited()
    scheduler.schedule_consolidation.assert_called_once()
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_turn_committed_consumers_serializes_same_session_post_mem():
    started: list[str] = []
    finished: list[str] = []
    active = 0
    max_active = 0
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def _ingest(request: MemoryIngestRequest):
        nonlocal active, max_active
        content = cast(dict[str, object], request.content)
        label = str(content["user_message"])
        started.append(label)
        active += 1
        max_active = max(max_active, active)
        if label == "a":
            first_started.set()
            await release_first.wait()
        finished.append(label)
        active -= 1
        return MagicMock()

    engine = MagicMock()
    engine.ingest = AsyncMock(side_effect=_ingest)
    session = MagicMock()
    session.key = "cli:1"
    session_manager = SimpleNamespace(get_or_create=MagicMock(return_value=session))
    scheduler = SimpleNamespace(schedule_consolidation=MagicMock())
    consolidation = SimpleNamespace(refresh_recent_turns=AsyncMock())
    event_bus = EventBus()
    register_turn_committed_consumers(
        event_bus=event_bus,
        consolidation=cast(Any, consolidation),
        session_manager=cast(Any, session_manager),
        scheduler=cast(Any, scheduler),
        memory_engine=cast(Any, engine),
    )
    event_a = TurnCommitted(
        session_key="cli:1",
        channel="cli",
        chat_id="1",
        input_message="a",
        persisted_user_message="a",
        assistant_response="ok-a",
        tools_used=[],
        timestamp=datetime.now(),
    )
    event_b = TurnCommitted(
        session_key="cli:1",
        channel="cli",
        chat_id="1",
        input_message="b",
        persisted_user_message="b",
        assistant_response="ok-b",
        tools_used=[],
        timestamp=datetime.now(),
    )

    event_bus.enqueue(event_a)
    await event_bus.drain()
    await first_started.wait()
    event_bus.enqueue(event_b)
    await event_bus.drain()
    await asyncio.sleep(0)

    assert started == ["a"]
    assert finished == []
    assert max_active == 1

    release_first.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert started == ["a", "b"]
    assert finished == ["a", "b"]
    assert max_active == 1
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_turn_committed_consumers_keep_cross_session_parallelism():
    active = 0
    max_active = 0
    ready = asyncio.Event()
    release = asyncio.Event()

    async def _ingest(request: MemoryIngestRequest):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if max_active >= 2:
            ready.set()
        await release.wait()
        active -= 1
        return MagicMock()

    engine = MagicMock()
    engine.ingest = AsyncMock(side_effect=_ingest)
    session_manager = SimpleNamespace(
        get_or_create=MagicMock(side_effect=lambda key: SimpleNamespace(key=key))
    )
    scheduler = SimpleNamespace(schedule_consolidation=MagicMock())
    consolidation = SimpleNamespace(refresh_recent_turns=AsyncMock())
    event_bus = EventBus()
    register_turn_committed_consumers(
        event_bus=event_bus,
        consolidation=cast(Any, consolidation),
        session_manager=cast(Any, session_manager),
        scheduler=cast(Any, scheduler),
        memory_engine=cast(Any, engine),
    )

    event_bus.enqueue(
        TurnCommitted(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            input_message="a",
            persisted_user_message="a",
            assistant_response="ok-a",
            tools_used=[],
            timestamp=datetime.now(),
        )
    )
    event_bus.enqueue(
        TurnCommitted(
            session_key="cli:2",
            channel="cli",
            chat_id="2",
            input_message="b",
            persisted_user_message="b",
            assistant_response="ok-b",
            tools_used=[],
            timestamp=datetime.now(),
        )
    )
    await event_bus.drain()

    await ready.wait()
    assert max_active >= 2

    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await event_bus.aclose()


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


def test_agent_loop_observes_passive_turn_trace_from_event(tmp_path: Path):
    class _Writer:
        def __init__(self) -> None:
            self.events: list[object] = []

        def emit(self, event: object) -> None:
            self.events.append(event)

    writer = _Writer()
    loop = _make_loop(
        tmp_path,
        retrieval_pipeline=_CustomRetrieval(block="MEM_BLOCK"),
        observe_writer=writer,
    )
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

    assert writer.events
    turn_event = cast(Any, writer.events[0])
    assert turn_event.source == "agent"
    assert turn_event.session_key == "cli:1"
    assert turn_event.user_msg == "hello"
    assert turn_event.llm_output == "ok"
    assert turn_event.tool_calls[0]["name"] == "noop"
    assert turn_event.react_iteration_count == 1
    assert turn_event.react_input_sum_tokens == 100


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
async def test_resumed_interrupt_state_survives_timeout(tmp_path: Path):
    loop = _make_loop(tmp_path)
    session_key = "telegram:123"
    loop._interrupt_states[session_key] = TurnInterruptState(  # type: ignore[attr-defined]
        session_key=session_key,
        original_user_message="原始消息 A",
        partial_reply="半截回答",
    )
    loop._MESSAGE_TIMEOUT_S = 0.01  # type: ignore[attr-defined]

    async def _slow_process(*args, **kwargs):
        await asyncio.sleep(0.05)
        return MagicMock(content="ok")

    loop._core_runner.process = _slow_process  # type: ignore[attr-defined]

    msg = InboundMessage(
        channel="telegram",
        sender="1",
        chat_id="123",
        content="补充 B",
    )
    outbound = await loop._process(msg)

    assert "超时" in outbound.content
    assert session_key in loop._interrupt_states  # type: ignore[attr-defined]
    assert loop._interrupt_states[session_key].original_user_message == "原始消息 A"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_interrupt_state_carries_partial_progress(tmp_path: Path):
    loop = _make_loop(tmp_path)
    session_key = "telegram:123"
    msg = InboundMessage(
        channel="telegram",
        sender="1",
        chat_id="123",
        content="原始消息 A",
    )
    loop._active_turn_states[session_key] = TurnInterruptState(  # type: ignore[attr-defined]
        session_key=session_key,
        original_user_message=msg.content,
    )
    progress_sink = loop._build_progress_sink(msg)  # type: ignore[attr-defined]
    await progress_sink(
        {
            "partial_reply": "工具阶段说明",
            "partial_thinking": "思考片段",
            "tools_used": ["shell"],
            "tool_chain_partial": [{"text": "tool", "calls": []}],
        }
    )
    loop._append_partial_reply(session_key, " + 流式增量")  # type: ignore[attr-defined]
    pending = _PendingTask()
    loop._active_tasks[session_key] = pending  # type: ignore[attr-defined]

    loop.request_interrupt(session_key)
    state = loop._interrupt_states[session_key]  # type: ignore[attr-defined]

    assert state.partial_reply == "工具阶段说明 + 流式增量"
    assert state.partial_thinking == "思考片段"
    assert state.tools_used == ["shell"]
    assert state.tool_chain_partial == [{"text": "tool", "calls": []}]


def test_agent_loop_configures_progress_sink_without_stream_factory(tmp_path: Path):
    loop = _make_loop(tmp_path)
    progress_factory = getattr(loop._reasoner, "_progress_sink_factory", None)

    assert callable(progress_factory)
