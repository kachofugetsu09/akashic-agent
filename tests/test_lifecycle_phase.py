from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from bus.event_bus import EventBus
from agent.lifecycle.facade import TurnLifecycle
from agent.lifecycle.phase import (
    Phase,
    PhaseFrame,
    append_string_exports,
    topo_sort_modules,
)
from agent.lifecycle.types import (
    AfterStepCtx,
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


class _NeedsMissingModuleSlotModule:
    slot = "plugin.consumer"
    requires = ("plugin.provider",)

    async def run(self, frame: _TextFrame) -> _TextFrame:
        frame.output = "disabled module ran"
        return frame


class _BuiltinNeedsMissingModuleSlot:
    slot = "before_turn.consumer"
    requires = ("before_turn.provider",)

    async def run(self, frame: _TextFrame) -> _TextFrame:
        return frame


class _PassThroughFinalizeModule:
    async def run(self, frame: _TextFrame) -> _TextFrame:
        frame.output = frame.input
        return frame


class _NeedsDisabledModuleSlotModule:
    slot = "plugin.after_consumer"
    requires = ("plugin.consumer",)

    async def run(self, frame: _TextFrame) -> _TextFrame:
        frame.output = "dependent module ran"
        return frame


class _PluginProviderModule:
    slot = "plugin.provider"

    async def run(self, frame: _TextFrame) -> _TextFrame:
        return frame


def test_string_exports_reject_invalid_value_without_partial_append() -> None:
    target = ["existing"]

    with pytest.raises(
        TypeError,
        match=r"key=outbound:media:image index=1 type=NoneType",
    ):
        append_string_exports(
            target,
            {"outbound:media:image": ["/tmp/a.png", None]},
        )

    assert target == ["existing"]


def test_string_exports_reject_later_key_without_partial_append() -> None:
    target = ["existing"]

    with pytest.raises(TypeError, match=r"key=second type=NoneType"):
        append_string_exports(target, {"first": "ok", "second": None})

    assert target == ["existing"]


def test_string_exports_reject_non_list_value() -> None:
    with pytest.raises(TypeError, match=r"key=prompt:extra_hint:test type=dict"):
        append_string_exports([], {"prompt:extra_hint:test": {"text": "hint"}})


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


def test_phase_rejects_unclosed_slot():
    with pytest.raises(RuntimeError, match="Phase slot 未闭合"):
        Phase[str, str, _TextFrame](
            [_NeedsMissingSlotModule()],
            frame_factory=_TextFrame,
        )


def test_phase_rejects_missing_builtin_module_dependency():
    with pytest.raises(RuntimeError, match="Phase 模块依赖不存在"):
        Phase[str, str, _TextFrame](
            [_BuiltinNeedsMissingModuleSlot()],
            frame_factory=_TextFrame,
        )


def test_phase_rejects_module_dependency_after_consumer():
    with pytest.raises(RuntimeError, match="Phase 模块依赖未满足"):
        Phase[str, str, _TextFrame](
            [_NeedsMissingModuleSlotModule(), _PluginProviderModule()],
            frame_factory=_TextFrame,
        )


def test_phase_warns_when_module_dependency_missing(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level("WARNING", logger="agent.lifecycle.phase"):
        Phase[str, str, _TextFrame](
            [_NeedsMissingModuleSlotModule()],
            frame_factory=_TextFrame,
        )
    assert "Phase 模块依赖不存在" in caplog.text
    assert "Phase slot 未闭合" not in caplog.text


@pytest.mark.asyncio
async def test_phase_disables_module_with_missing_module_dependency(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level("WARNING", logger="agent.lifecycle.phase"):
        phase = Phase[str, str, _TextFrame](
            [_NeedsMissingModuleSlotModule(), _PassThroughFinalizeModule()],
            frame_factory=_TextFrame,
        )
    result = await phase.run("hello")
    assert result == "hello"
    assert "已禁用模块" in caplog.text


def test_topo_sort_disables_missing_module_dependency_recursively(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level("WARNING", logger="agent.lifecycle.phase"):
        modules = topo_sort_modules(
            [
                _NeedsMissingModuleSlotModule(),
                _NeedsDisabledModuleSlotModule(),
            ]
        )
    assert modules == []
    assert "plugin.consumer" in caplog.text
    assert "plugin.after_consumer" in caplog.text


_now = datetime.now()


def test_before_turn_ctx_preserves_positional_plugin_constructor_abi() -> None:
    skills = ["existing-skill"]
    ctx = BeforeTurnCtx(
        "k",
        "c",
        "ch",
        "hello",
        _now,
        (),
        skills,
        turn_id="turn:durable",
    )

    assert ctx.skill_names is skills
    assert ctx.turn_id == "turn:durable"
    assert (
        inspect.signature(BeforeTurnCtx).parameters["turn_id"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


@pytest.mark.asyncio
async def test_lifecycle_on_after_step():
    bus = EventBus()
    lifecycle = TurnLifecycle(bus)
    handler = AsyncMock(return_value=None)
    subscription = lifecycle.on_after_step(handler)
    await bus.fanout(
        AfterStepCtx(
            session_key="k",
            channel="c",
            chat_id="ch",
            iteration=0,
            context_tokens_estimate=0,
            tools_called=(),
            partial_reply="",
            tools_used_so_far=(),
            tool_chain_partial=(),
            partial_thinking=None,
            has_more=True,
        )
    )
    handler.assert_awaited_once()
    assert subscription.active is True

    subscription.close()
    await bus.fanout(
        AfterStepCtx(
            session_key="k",
            channel="c",
            chat_id="ch",
            iteration=1,
            context_tokens_estimate=0,
            tools_called=(),
            partial_reply="",
            tools_used_so_far=(),
            tool_chain_partial=(),
            partial_thinking=None,
            has_more=False,
        )
    )
    assert subscription.active is False
    assert handler.await_count == 1
    assert bus.handler_count() == 0
