from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from bus.event_bus import EventBus
from agent.core.response_parser import ResponseMetadata
from agent.lifecycle.facade import TurnLifecycle
from agent.lifecycle.phase import Phase, PhaseFrame
from agent.lifecycle.types import (
    AfterReasoningCtx,
    AfterStepCtx,
    AfterTurnCtx,
    BeforeReasoningCtx,
    BeforeStepCtx,
    BeforeTurnCtx,
)


@dataclass
class _TextFrame(PhaseFrame[str, str]):
    pass


class _SetupModule:
    produces = ("text:value",)

    async def run(self, frame: _TextFrame) -> _TextFrame:
        frame.slots["text:value"] = f"setup_{frame.input}"
        return frame


class _MutateModule:
    requires = ("text:value",)
    produces = ("text:value",)

    async def run(self, frame: _TextFrame) -> _TextFrame:
        frame.slots["text:value"] = f"{frame.slots['text:value']}_mutated"
        return frame


class _FinalizeModule:
    requires = ("text:value",)

    async def run(self, frame: _TextFrame) -> _TextFrame:
        frame.output = f"{frame.slots['text:value']}_finalized"
        return frame


class _FailingModule:
    async def run(self, frame: _TextFrame) -> _TextFrame:
        raise RuntimeError("setup failed")


class _NoOutputModule:
    async def run(self, frame: _TextFrame) -> _TextFrame:
        return frame


class _NeedsMissingSlotModule:
    requires = ("missing:value",)

    async def run(self, frame: _TextFrame) -> _TextFrame:
        frame.output = str(frame.slots["missing:value"])
        return frame


@pytest.mark.asyncio
async def test_phase_modules_run_in_order():
    phase = Phase[str, str, _TextFrame](
        [_SetupModule(), _MutateModule(), _FinalizeModule()],
        frame_factory=_TextFrame,
    )
    result = await phase.run("hello")
    assert result == "setup_hello_mutated_finalized"


@pytest.mark.asyncio
async def test_phase_modules_can_passthrough():
    phase = Phase[str, str, _TextFrame](
        [_SetupModule(), _FinalizeModule()],
        frame_factory=_TextFrame,
    )
    result = await phase.run("hello")
    assert result == "setup_hello_finalized"


@pytest.mark.asyncio
async def test_phase_module_exception_propagates():
    phase = Phase[str, str, _TextFrame]([_FailingModule()], frame_factory=_TextFrame)
    with pytest.raises(RuntimeError, match="setup failed"):
        await phase.run("x")


@pytest.mark.asyncio
async def test_phase_requires_output():
    phase = Phase[str, str, _TextFrame]([_NoOutputModule()], frame_factory=_TextFrame)
    with pytest.raises(RuntimeError, match="Phase 模块链未产生 output"):
        await phase.run("x")


def test_phase_warns_when_slot_not_closed(caplog: pytest.LogCaptureFixture):
    with caplog.at_level("WARNING", logger="agent.lifecycle.phase"):
        Phase[str, str, _TextFrame](
            [_NeedsMissingSlotModule()],
            frame_factory=_TextFrame,
        )
    assert "Phase slot 未闭合" in caplog.text


_now = datetime.now()


def _before_turn_ctx(**kwargs: object) -> BeforeTurnCtx:
    return BeforeTurnCtx(
        session_key="k", channel="c", chat_id="ch", content="hello",
        timestamp=_now, retrieved_memory_block="", retrieval_trace_raw=None,
        history_messages=(),
    )


@pytest.mark.asyncio
async def test_lifecycle_on_before_turn():
    bus = EventBus()
    lifecycle = TurnLifecycle(bus)
    handler = AsyncMock(return_value=None)
    lifecycle.on_before_turn(handler)
    await bus.emit(_before_turn_ctx())
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifecycle_on_before_reasoning():
    bus = EventBus()
    lifecycle = TurnLifecycle(bus)
    handler = AsyncMock(return_value=None)
    lifecycle.on_before_reasoning(handler)
    await bus.emit(BeforeReasoningCtx(
        session_key="k", channel="c", chat_id="ch", content="hello",
        timestamp=_now, skill_names=[], retrieved_memory_block="",
    ))
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifecycle_on_before_step():
    bus = EventBus()
    lifecycle = TurnLifecycle(bus)
    handler = AsyncMock(return_value=None)
    lifecycle.on_before_step(handler)
    await bus.emit(BeforeStepCtx(
        session_key="k", channel="c", chat_id="ch", iteration=0,
        input_tokens_estimate=100, visible_tool_names=None,
    ))
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifecycle_on_after_reasoning():
    bus = EventBus()
    lifecycle = TurnLifecycle(bus)
    handler = AsyncMock(return_value=None)
    lifecycle.on_after_reasoning(handler)
    await bus.emit(AfterReasoningCtx(
        session_key="k", channel="c", chat_id="ch",
        tools_used=(), thinking=None,
        response_metadata=ResponseMetadata(raw_text=""),
        streamed=False, tool_chain=(), context_retry={}, reply="hi",
    ))
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifecycle_on_after_step():
    bus = EventBus()
    lifecycle = TurnLifecycle(bus)
    handler = AsyncMock(return_value=None)
    lifecycle.on_after_step(handler)
    await bus.fanout(AfterStepCtx(
        session_key="k", channel="c", chat_id="ch", iteration=0,
        tools_called=(), partial_reply="",
        tools_used_so_far=(), tool_chain_partial=(),
        partial_thinking=None, has_more=True,
    ))
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifecycle_on_after_turn():
    bus = EventBus()
    lifecycle = TurnLifecycle(bus)
    handler = AsyncMock(return_value=None)
    lifecycle.on_after_turn(handler)
    await bus.fanout(AfterTurnCtx(
        session_key="k", channel="c", chat_id="ch", reply="hi",
        tools_used=(), thinking=None, will_dispatch=True,
    ))
    handler.assert_awaited_once()
