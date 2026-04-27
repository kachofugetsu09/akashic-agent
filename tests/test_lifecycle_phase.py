from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from bus.event_bus import EventBus
from agent.core.response_parser import ResponseMetadata
from agent.lifecycle.facade import TurnLifecycle
from agent.lifecycle.phase import GatePhase, TapPhase
from agent.lifecycle.types import (
    AfterReasoningCtx,
    AfterStepCtx,
    AfterTurnCtx,
    BeforeReasoningCtx,
    BeforeStepCtx,
    BeforeTurnCtx,
)


# ── stub contexts ──

@dataclass
class _GateCtx:
    value: str = ""


@dataclass(frozen=True)
class _TapCtx:
    value: str = ""


# ── GatePhase: setup produces ctx, chain mutates it, finalize produces output ──

class _MutateGate(GatePhase[str, _GateCtx, str]):
    async def _setup(self, input: str) -> _GateCtx:
        return _GateCtx(value=f"setup_{input}")

    async def _finalize(self, ctx: _GateCtx, input: str) -> str:
        return f"{ctx.value}_finalized"


@pytest.mark.asyncio
async def test_gate_phase_setup_chain_finalize():
    bus = EventBus()

    async def handler(ctx: _GateCtx) -> _GateCtx:
        ctx.value = f"{ctx.value}_mutated"
        return ctx

    bus.on(_GateCtx, handler)

    phase = _MutateGate(bus)
    result = await phase.run("hello")
    assert result == "setup_hello_mutated_finalized"


@pytest.mark.asyncio
async def test_gate_phase_no_handler_chain_passthrough():
    bus = EventBus()

    phase = _MutateGate(bus)
    result = await phase.run("hello")
    assert result == "setup_hello_finalized"


# ── TapPhase: setup produces ctx, chain is read-only, finalize produces output ──

class _ReadonlyTap(TapPhase[str, _TapCtx, str]):
    async def _setup(self, input: str) -> _TapCtx:
        return _TapCtx(value=f"setup_{input}")

    async def _finalize(self, ctx: _TapCtx, input: str) -> str:
        return f"{ctx.value}_finalized"


@pytest.mark.asyncio
async def test_tap_phase_chain_does_not_mutate():
    bus = EventBus()

    side_effect: list[str] = []

    async def handler(ctx: _TapCtx) -> None:
        side_effect.append(ctx.value)

    bus.on(_TapCtx, handler)

    phase = _ReadonlyTap(bus)
    result = await phase.run("hello")
    assert result == "setup_hello_finalized"
    assert side_effect == ["setup_hello"]


@pytest.mark.asyncio
async def test_tap_handler_exception_does_not_kill_turn():
    bus = EventBus()

    call_order: list[str] = []

    async def broken_handler(ctx: _TapCtx) -> None:
        call_order.append("broken_start")
        msg = 1 / 0
        _ = msg

    async def good_handler(ctx: _TapCtx) -> None:
        call_order.append("good")

    bus.on(_TapCtx, broken_handler)
    bus.on(_TapCtx, good_handler)

    phase = _ReadonlyTap(bus)
    result = await phase.run("hello")
    assert result == "setup_hello_finalized"
    assert "good" in call_order


# ── GatePhase exception propagates ──

class _FailingGate(GatePhase[str, _GateCtx, str]):
    async def _setup(self, input: str) -> _GateCtx:
        raise RuntimeError("setup failed")

    async def _finalize(self, ctx: _GateCtx, input: str) -> str:
        return "never_reached"


@pytest.mark.asyncio
async def test_gate_phase_setup_exception_propagates():
    bus = EventBus()
    phase = _FailingGate(bus)
    with pytest.raises(RuntimeError, match="setup failed"):
        await phase.run("x")


# ── TurnLifecycle: 每个 on_xxx 单独验证 mapping 正确 ──

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
        tools_used=(), thinking=None, dispatch_outbound=True,
    ))
    handler.assert_awaited_once()
