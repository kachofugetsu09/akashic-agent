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
from bus.events import InboundMessage
from bus.events_lifecycle import TurnCommitted
from core.memory.engine import MemoryEngineRetrieveResult
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

    async def retrieve(self, request) -> MemoryEngineRetrieveResult:
        return MemoryEngineRetrieveResult(text_block="", hits=[], raw={})

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
    )
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

    assert outbound.content == "ok"
    assert session_key not in loop._interrupt_states  # type: ignore[attr-defined]


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
