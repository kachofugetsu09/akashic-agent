from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
import subprocess
import sys
from typing import Any, AsyncIterator, cast

import pytest

from agent.core.response_parser import ResponseMetadata
from agent.lifecycle.composition import run_turn_stage_event
from agent.lifecycle.phases.after_reasoning import (
    AfterReasoningFrame,
    default_after_reasoning_modules,
)
from agent.lifecycle.phases.prompt_render import (
    PromptRenderFrame,
    default_prompt_render_modules,
)
from agent.lifecycle.types import AfterReasoningCtx, PromptRenderCtx
from agent.plugin_composition import Bail, CompositionError, CompositionRoot
from agent.plugins.snapshot import (
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.turn_events.after_reasoning import (
    AFTER_REASONING_BEFORE_EVENT_BUS,
    AFTER_REASONING_BEFORE_PERSIST,
)
from agent.turn_events.prompt_render import PROMPT_RENDER_AFTER_EVENT_BUS
from bus.event_bus import EventBus


@asynccontextmanager
async def _bound_root(root: CompositionRoot) -> AsyncIterator[None]:
    store = RuntimeSnapshotStore()
    store.install(RuntimeSnapshotCompiler().compile({}, composition_root=root))
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        yield
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
        await store.close()
        await root.dispose()


def _prompt_ctx() -> PromptRenderCtx:
    return PromptRenderCtx(
        session_key="session",
        channel="test",
        chat_id="chat",
        content="hello",
        media=None,
        timestamp=datetime.now(),
        history=[],
        skill_names=[],
        retrieved_memory_block="",
        disabled_sections=set(),
        turn_injection_prompt="",
    )


def _answer_ctx() -> AfterReasoningCtx:
    return AfterReasoningCtx(
        session_key="session",
        channel="test",
        chat_id="chat",
        tools_used=(),
        thinking=None,
        response_metadata=ResponseMetadata(raw_text="hello"),
        streamed=False,
        tool_chain=(),
        context_retry={},
        reply="hello",
    )


@pytest.mark.asyncio
async def test_prompt_seam_runs_before_legacy_phase_modules() -> None:
    order: list[str] = []
    root = CompositionRoot("prompt-seam")

    async def plugin(ctx) -> None:
        await ctx.on(
            PROMPT_RENDER_AFTER_EVENT_BUS,
            lambda _: order.append("composition"),
        )

    class LegacyModule:
        slot = "legacy.prompt"
        requires = ("prompt_render.emit", "prompt:ctx")

        async def run(self, frame):
            order.append("legacy-phase")
            return frame

    await root.mount(plugin, name="prompt-plugin")
    bus = EventBus()
    bus.on(PromptRenderCtx, lambda _: order.append("event-bus"))
    modules = default_prompt_render_modules(
        bus,
        cast(Any, object()),
        plugin_modules=cast(Any, [LegacyModule()]),
    )
    slots = [cast(str, getattr(module, "slot")) for module in modules]
    frame = PromptRenderFrame(
        input=cast(Any, None),
        slots={"prompt:ctx": _prompt_ctx()},
    )

    async with _bound_root(root):
        for module in modules[
            slots.index("prompt_render.emit") : slots.index("legacy.prompt") + 1
        ]:
            frame = await module.run(frame)

    assert order == ["event-bus", "composition", "legacy-phase"]
    assert slots.index("legacy.prompt") < slots.index("prompt_render.collect_exports")


@pytest.mark.asyncio
async def test_answer_seams_preserve_legacy_module_positions() -> None:
    order: list[str] = []
    root = CompositionRoot("answer-seam")

    def preprocess(ctx: AfterReasoningCtx) -> None:
        order.append("preprocess")
        ctx.persist_assistant_metadata["cited_memory_ids"] = ["mem_1"]

    async def plugin(ctx) -> None:
        await ctx.on(
            AFTER_REASONING_BEFORE_EVENT_BUS,
            preprocess,
        )
        await ctx.on(
            AFTER_REASONING_BEFORE_PERSIST,
            lambda _: order.append("cleanup"),
        )

    class LegacyPre:
        slot = "legacy.answer_pre"
        requires = ("after_reasoning.build_ctx", "reasoning:ctx")

        async def run(self, frame):
            order.append("legacy-pre")
            return frame

    class LegacyPost:
        slot = "legacy.answer_post"
        requires = ("after_reasoning.emit", "reasoning:ctx")

        async def run(self, frame):
            order.append("legacy-post")
            return frame

    await root.mount(plugin, name="answer-plugin")
    bus = EventBus()
    bus.on(AfterReasoningCtx, lambda _: order.append("event-bus"))
    modules = default_after_reasoning_modules(
        bus,
        cast(Any, object()),
        plugin_modules=cast(Any, [LegacyPre(), LegacyPost()]),
    )
    slots = [cast(str, getattr(module, "slot")) for module in modules]
    answer_ctx = _answer_ctx()
    frame = AfterReasoningFrame(
        input=cast(Any, None),
        slots={"reasoning:ctx": answer_ctx},
    )

    async with _bound_root(root):
        for module in modules[
            slots.index("legacy.answer_pre") : slots.index(
                "after_reasoning.composition_cleanup"
            )
            + 1
        ]:
            frame = await module.run(frame)

    assert order == [
        "legacy-pre",
        "preprocess",
        "event-bus",
        "legacy-post",
        "cleanup",
    ]
    assert answer_ctx.persist_assistant_metadata == {"cited_memory_ids": ["mem_1"]}


@pytest.mark.asyncio
async def test_lifecycle_seam_rejects_bail() -> None:
    root = CompositionRoot("lifecycle-bail")

    async def plugin(ctx) -> None:
        await ctx.on(PROMPT_RENDER_AFTER_EVENT_BUS, lambda _: Bail("blocked"))

    await root.mount(plugin, name="bailing-plugin")
    async with _bound_root(root):
        with pytest.raises(CompositionError) as caught:
            await run_turn_stage_event(PROMPT_RENDER_AFTER_EVENT_BUS, _prompt_ctx())

    assert caught.value.code == "TURN_STAGE_BAIL_NOT_ALLOWED"


def test_turn_stage_event_names_encode_their_exact_owner_position() -> None:
    assert PROMPT_RENDER_AFTER_EVENT_BUS.name == ("turn.prompt_render.after_event_bus")
    assert AFTER_REASONING_BEFORE_EVENT_BUS.name == (
        "turn.after_reasoning.before_event_bus"
    )
    assert AFTER_REASONING_BEFORE_PERSIST.name == (
        "turn.after_reasoning.before_persist"
    )


def test_turn_event_contracts_import_without_phase_runtime() -> None:
    code = (
        "from agent.turn_events.after_reasoning import "
        "AFTER_REASONING_BEFORE_EVENT_BUS; "
        "from agent.turn_events.prompt_render import "
        "PROMPT_RENDER_AFTER_EVENT_BUS; "
        "import sys; "
        "assert 'agent.lifecycle.phases.after_reasoning' not in sys.modules; "
        "assert AFTER_REASONING_BEFORE_EVENT_BUS.name.endswith('before_event_bus'); "
        "assert PROMPT_RENDER_AFTER_EVENT_BUS.name.endswith('after_event_bus')"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_turn_stage_event_propagates_listener_failure() -> None:
    root = CompositionRoot("turn-stage-failure")

    def fail(_: PromptRenderCtx) -> None:
        raise RuntimeError("stage failed")

    async def plugin(ctx) -> None:
        await ctx.on(PROMPT_RENDER_AFTER_EVENT_BUS, fail)

    await root.mount(plugin, name="failing-plugin")
    async with _bound_root(root):
        with pytest.raises(RuntimeError, match="stage failed"):
            await run_turn_stage_event(PROMPT_RENDER_AFTER_EVENT_BUS, _prompt_ctx())
