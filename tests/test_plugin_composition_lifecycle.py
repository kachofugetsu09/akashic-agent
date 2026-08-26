from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import subprocess
import sys
from typing import Any, AsyncIterator, cast

import pytest

from agent.core.response_parser import ResponseMetadata
from agent.lifecycle.composition import (
    AFTER_REASONING_CLEANUP_EVENT,
    AFTER_REASONING_PREPROCESS_EVENT,
    CONTEXT_PREPARED_EVENT,
    PROMPT_RENDER_EVENT,
    observe_composition_event,
    run_composition_lifecycle,
)
from agent.lifecycle.phases.before_turn import (
    BeforeTurnFrame,
    default_before_turn_modules,
)
from agent.lifecycle.phases.after_turn import (
    AfterTurnFrame,
    default_after_turn_modules,
)
from agent.lifecycle.phases.after_reasoning import (
    AfterReasoningFrame,
    default_after_reasoning_modules,
)
from agent.lifecycle.phases.prompt_render import (
    PromptRenderFrame,
    default_prompt_render_modules,
)
from agent.lifecycle.types import AfterReasoningCtx, BeforeTurnCtx, PromptRenderCtx
from agent.lifecycle.composition import observe_composition_domain_event
from agent.plugin_composition import (
    Bail,
    CompositionError,
    CompositionRoot,
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
)
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.turn_events.after_turn import AFTER_TURN_COMMITTED
from agent.turn_events.observe import (
    MEMORY_WRITTEN_EVENT,
    RETRIEVAL_COMPLETED_EVENT,
)
from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted
from core.memory.engine import MemoryQueryResult, MemoryRecord
from core.memory.events import MemoryWritten, RetrievalCompleted
from agent.retrieval.events import build_retrieval_completed
from agent.retrieval.protocol import RetrievalRequest


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
        disabled_sections=set(),
        turn_injection_prompt="",
    )


def _before_turn_ctx() -> BeforeTurnCtx:
    return BeforeTurnCtx(
        session_key="session",
        channel="test",
        chat_id="chat",
        content="hello",
        timestamp=datetime.now(),
        history_messages=(),
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
        await ctx.on(PROMPT_RENDER_EVENT, lambda _: order.append("composition"))

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
async def test_context_prepared_seam_runs_after_legacy_before_turn_modules() -> None:
    order: list[str] = []
    observed: list[BeforeTurnCtx] = []
    root = CompositionRoot("context-prepared-seam")

    async def plugin(ctx) -> None:
        def observe(payload: BeforeTurnCtx) -> None:
            order.append("composition")
            assert payload.extra_hints == ["legacy hint"]
            observed.append(payload)

        await ctx.on(CONTEXT_PREPARED_EVENT, observe)

    class LegacyModule:
        slot = "legacy.before_turn"
        requires = ("before_turn.emit", "session:ctx")

        async def run(self, frame):
            order.append("legacy-phase")
            frame.slots["session:extra_hint:legacy"] = "legacy hint"
            return frame

    await root.mount(plugin, name="context-plugin")
    bus = EventBus()
    bus.on(BeforeTurnCtx, lambda _: order.append("event-bus"))
    modules = default_before_turn_modules(
        bus,
        cast(Any, object()),
        cast(Any, object()),
        plugin_modules=cast(Any, [LegacyModule()]),
    )
    slots = [cast(str, getattr(module, "slot")) for module in modules]
    payload = _before_turn_ctx()
    frame = BeforeTurnFrame(
        input=cast(Any, None),
        slots={"session:ctx": payload},
    )

    async with _bound_root(root):
        for module in modules[
            slots.index("before_turn.emit") : slots.index(
                "before_turn.composition_context_prepared"
            )
            + 1
        ]:
            frame = await module.run(frame)

    assert order == ["event-bus", "legacy-phase", "composition"]
    assert observed == [payload]
    assert (
        slots.index("before_turn.collect_exports")
        < slots.index("before_turn.composition_context_prepared")
        < slots.index("before_turn.return")
    )


@pytest.mark.asyncio
async def test_context_prepared_seam_is_noop_without_composition_root() -> None:
    store = RuntimeSnapshotStore()
    store.install(RuntimeSnapshotCompiler().compile({}))
    lease = store.lease()
    token = bind_runtime_snapshot(lease)

    try:
        await run_composition_lifecycle(CONTEXT_PREPARED_EVENT, _before_turn_ctx())
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
        await store.close()


@pytest.mark.asyncio
async def test_lifecycle_seam_is_noop_without_runtime_binding() -> None:
    await run_composition_lifecycle(CONTEXT_PREPARED_EVENT, _before_turn_ctx())


@pytest.mark.asyncio
async def test_lifecycle_seam_rejects_inherited_wrong_task_binding() -> None:
    observed: list[str] = []
    root = CompositionRoot("wrong-task-lifecycle")

    async def plugin(ctx) -> None:
        await ctx.on(CONTEXT_PREPARED_EVENT, lambda _: observed.append("called"))

    await root.mount(plugin, name="observer")
    async with _bound_root(root):
        task = asyncio.create_task(
            run_composition_lifecycle(
                CONTEXT_PREPARED_EVENT,
                _before_turn_ctx(),
            )
        )
        with pytest.raises(CompositionError) as caught:
            await task

    assert caught.value.code == "RUNTIME_SNAPSHOT_BINDING_MISMATCH"
    assert observed == []


@pytest.mark.asyncio
async def test_lifecycle_seam_rejects_released_owner_lease() -> None:
    root = CompositionRoot("inactive-lifecycle")
    store = RuntimeSnapshotStore()
    store.install(RuntimeSnapshotCompiler().compile({}, composition_root=root))
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    await lease.release()

    try:
        with pytest.raises(CompositionError) as caught:
            await run_composition_lifecycle(
                CONTEXT_PREPARED_EVENT,
                _before_turn_ctx(),
            )
    finally:
        reset_runtime_snapshot(token)
        await store.close()
        await root.dispose()

    assert caught.value.code == "RUNTIME_SNAPSHOT_BINDING_INACTIVE"


@pytest.mark.asyncio
async def test_answer_seams_preserve_legacy_module_positions() -> None:
    order: list[str] = []
    root = CompositionRoot("answer-seam")

    async def plugin(ctx) -> None:
        await ctx.on(
            AFTER_REASONING_PREPROCESS_EVENT,
            lambda _: order.append("preprocess"),
        )
        await ctx.on(
            AFTER_REASONING_CLEANUP_EVENT,
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
    frame = AfterReasoningFrame(
        input=cast(Any, None),
        slots={"reasoning:ctx": _answer_ctx()},
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


@pytest.mark.asyncio
async def test_lifecycle_seam_rejects_bail() -> None:
    root = CompositionRoot("lifecycle-bail")

    async def plugin(ctx) -> None:
        await ctx.on(PROMPT_RENDER_EVENT, lambda _: Bail("blocked"))

    await root.mount(plugin, name="bailing-plugin")
    async with _bound_root(root):
        with pytest.raises(CompositionError) as caught:
            await run_composition_lifecycle(PROMPT_RENDER_EVENT, _prompt_ctx())

    assert caught.value.code == "LIFECYCLE_BAIL_NOT_ALLOWED"


@pytest.mark.asyncio
async def test_runtime_lifecycle_bail_fails_loud(tmp_path) -> None:
    calls: list[str] = []
    root = CompositionRoot("runtime-lifecycle-bail")

    async def first(ctx) -> None:
        await ctx.on(
            RUNTIME_STARTED,
            lambda _: (calls.append("bail"), Bail("blocked"))[1],
        )

    async def second(ctx) -> None:
        await ctx.on(RUNTIME_STARTED, lambda _: calls.append("second"))

    await root.mount(first, name="bailing-plugin")
    await root.mount(second, name="later-plugin")
    manager = PluginManager([], event_bus=EventBus(), workspace=tmp_path)
    snapshot = RuntimeSnapshot("runtime-bail", {}, None, composition_root=root)

    with pytest.raises(CompositionError) as caught:
        await cast(Any, manager)._start_runtime_snapshot(snapshot)

    assert caught.value.code == "RUNTIME_LIFECYCLE_BAIL_NOT_ALLOWED"
    assert calls == ["bail"]
    await root.dispose()


@pytest.mark.asyncio
async def test_runtime_stop_failure_remains_retryable(tmp_path) -> None:
    stop_calls: list[str] = []
    root = CompositionRoot("runtime-stop-retry")

    async def plugin(ctx) -> None:
        async def stop(_event: object) -> None:
            stop_calls.append("stop")
            if len(stop_calls) == 1:
                raise RuntimeError("fixture stop failure")

        await ctx.on(RUNTIME_STOPPING, stop)

    await root.mount(plugin, name="retrying-plugin")
    manager = PluginManager([], event_bus=EventBus(), workspace=tmp_path)
    snapshot = RuntimeSnapshot("runtime-stop", {}, None, composition_root=root)
    await cast(Any, manager)._start_runtime_snapshot(snapshot)

    with pytest.raises(RuntimeError, match="fixture stop failure"):
        await cast(Any, manager)._stop_runtime_snapshot(snapshot)
    await cast(Any, manager)._stop_runtime_snapshot(snapshot)
    await cast(Any, manager)._stop_runtime_snapshot(snapshot)

    assert stop_calls == ["stop", "stop"]
    await root.dispose()


@pytest.mark.asyncio
async def test_after_turn_committed_event_runs_after_legacy_fanout() -> None:
    order: list[str] = []
    observed: list[TurnCommitted] = []
    root = CompositionRoot("after-turn-committed")

    def on_committed(event: TurnCommitted) -> None:
        order.append("composition")
        observed.append(event)

    async def plugin(ctx) -> None:
        await ctx.on(AFTER_TURN_COMMITTED, on_committed)

    await root.mount(plugin, name="observe-plugin")
    bus = EventBus()
    bus.on(TurnCommitted, lambda _: order.append("event-bus"))
    module = _fanout_committed_module(bus)
    committed = _committed_event()
    frame = AfterTurnFrame(
        input=cast(Any, None),
        slots={"turn:committed": committed},
    )

    async with _bound_root(root):
        result = await module.run(frame)

    assert result is frame
    assert order == ["event-bus", "composition"]
    assert observed == [committed]


@pytest.mark.asyncio
async def test_after_turn_committed_event_propagates_listener_failure() -> None:
    root = CompositionRoot("after-turn-committed-failure")

    def fail(_: TurnCommitted) -> None:
        raise RuntimeError("observe failed")

    async def plugin(ctx) -> None:
        await ctx.on(AFTER_TURN_COMMITTED, fail)

    await root.mount(plugin, name="failing-observe-plugin")
    frame = AfterTurnFrame(
        input=cast(Any, None),
        slots={"turn:committed": _committed_event()},
    )

    async with _bound_root(root):
        with pytest.raises(RuntimeError, match="observe failed"):
            await _fanout_committed_module(EventBus()).run(frame)


def test_after_turn_event_contract_imports_without_phase_runtime() -> None:
    code = (
        "from agent.turn_events.after_turn import AFTER_TURN_COMMITTED; "
        "import sys; "
        "assert 'agent.lifecycle.phases.after_turn' not in sys.modules; "
        "assert AFTER_TURN_COMMITTED.name == 'turn.after_turn.committed'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_after_turn_committed_event_keeps_legacy_path_without_root() -> None:
    observed: list[TurnCommitted] = []
    bus = EventBus()
    bus.on(TurnCommitted, observed.append)
    frame = AfterTurnFrame(
        input=cast(Any, None),
        slots={"turn:committed": _committed_event()},
    )

    result = await _fanout_committed_module(bus).run(frame)

    assert result is frame
    assert observed == [frame.slots["turn:committed"]]


@pytest.mark.asyncio
async def test_domain_observe_event_runs_after_legacy_fanout() -> None:
    order: list[str] = []
    observed: dict[str, object] = {}
    root = CompositionRoot("domain-observe-order")

    def observe(name: str):
        def callback(event: object) -> None:
            order.append(f"composition.{name}")
            observed[name] = event

        return callback

    async def plugin(ctx) -> None:
        await ctx.on(RETRIEVAL_COMPLETED_EVENT, observe("retrieval"))
        await ctx.on(MEMORY_WRITTEN_EVENT, observe("memory"))

    await root.mount(plugin, name="domain-observer")
    bus = EventBus()
    bus.on(RetrievalCompleted, lambda event: order.append("legacy.retrieval"))
    bus.on(MemoryWritten, lambda event: order.append("legacy.memory"))
    retrieval = _retrieval_completed_event()
    memory = _memory_written_event()

    async with _bound_root(root):
        await bus.fanout(retrieval)
        await bus.fanout(memory)

    assert order == [
        "legacy.retrieval",
        "composition.retrieval",
        "legacy.memory",
        "composition.memory",
    ]
    assert observed == {
        "retrieval": retrieval,
        "memory": memory,
    }


@pytest.mark.asyncio
async def test_domain_observe_event_is_not_skipped_without_legacy_handlers() -> None:
    observed: list[MemoryWritten] = []
    root = CompositionRoot("domain-observe-no-legacy")

    async def plugin(ctx) -> None:
        await ctx.on(MEMORY_WRITTEN_EVENT, observed.append)

    await root.mount(plugin, name="domain-observer")
    event = _memory_written_event()
    async with _bound_root(root):
        await EventBus().fanout(event)

    assert observed == [event]


@pytest.mark.asyncio
async def test_domain_observe_event_uses_bound_candidate_root() -> None:
    observed: list[str] = []
    first = CompositionRoot("domain-observe-first")
    second = CompositionRoot("domain-observe-second")

    async def first_plugin(ctx) -> None:
        await ctx.on(MEMORY_WRITTEN_EVENT, lambda _: observed.append("first"))

    async def second_plugin(ctx) -> None:
        await ctx.on(MEMORY_WRITTEN_EVENT, lambda _: observed.append("second"))

    await first.mount(first_plugin, name="first-observer")
    await second.mount(second_plugin, name="second-observer")
    event = _memory_written_event()
    bus = EventBus()

    async with _bound_root(first):
        await bus.fanout(event)
    async with _bound_root(second):
        await bus.fanout(event)

    assert observed == ["first", "second"]


@pytest.mark.asyncio
async def test_domain_observe_event_rejects_inherited_wrong_task_binding() -> None:
    root = CompositionRoot("domain-observe-wrong-task")

    async def plugin(ctx) -> None:
        await ctx.on(MEMORY_WRITTEN_EVENT, lambda _: None)

    await root.mount(plugin, name="domain-observer")
    async with _bound_root(root):
        task = asyncio.create_task(
            observe_composition_event(
                MEMORY_WRITTEN_EVENT,
                _memory_written_event(),
            )
        )
        with pytest.raises(CompositionError) as caught:
            await task

    assert caught.value.code == "RUNTIME_SNAPSHOT_BINDING_MISMATCH"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["fanout", "enqueue"])
async def test_event_bus_rejects_inherited_wrong_task_binding(
    operation: str,
) -> None:
    root = CompositionRoot(f"event-bus-wrong-task-{operation}")

    async def plugin(ctx) -> None:
        await ctx.on(MEMORY_WRITTEN_EVENT, lambda _: None)

    await root.mount(plugin, name="domain-observer")
    store = RuntimeSnapshotStore()
    store.install(RuntimeSnapshotCompiler().compile({}, composition_root=root))
    bus = EventBus()
    bus.bind_runtime_snapshot_store(store)
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        if operation == "fanout":
            task = asyncio.create_task(bus.fanout(_memory_written_event()))
        else:

            async def enqueue() -> None:
                bus.enqueue(_memory_written_event())

            task = asyncio.create_task(enqueue())
        with pytest.raises(CompositionError) as caught:
            await task
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
        await bus.aclose()
        await store.close()
        await root.dispose()

    assert caught.value.code == "RUNTIME_SNAPSHOT_BINDING_MISMATCH"


@pytest.mark.asyncio
async def test_retrieval_completed_event_payload() -> None:
    observed: list[RetrievalCompleted] = []
    root = CompositionRoot("retrieval-completed")

    async def plugin(ctx) -> None:
        await ctx.on(RETRIEVAL_COMPLETED_EVENT, lambda event: observed.append(event))

    await root.mount(plugin, name="retrieval-observer")

    request = RetrievalRequest(
        message="original",
        session_key="session",
        channel="test",
        chat_id="chat",
        history=[],
        session_metadata={},
    )
    result = MemoryQueryResult(
        text_block="memory block",
        records=[
            MemoryRecord(
                id="memory-1",
                kind="event",
                summary="a long enough memory summary",
                score=0.91,
                engine_kind="fake",
                signals={"confidence_label": "certain", "forced": True},
                injected=True,
            )
        ],
        trace={"route_decision": "RETRIEVE", "hyde_hypotheses": ["aux query"]},
        raw={"rewritten_query": "rewritten"},
    )

    async with _bound_root(root):
        await observe_composition_domain_event(
            build_retrieval_completed(request, result)
        )

    assert len(observed) == 1
    event = observed[0]
    assert event.query == "rewritten"
    assert event.orig_query == "original"
    assert event.route_decision == "RETRIEVE"
    assert event.aux_queries == ["aux query"]
    assert event.injected_count == 1
    assert event.hits[0].item_id == "memory-1"
    assert event.hits[0].confidence_label == "certain"
    assert event.hits[0].forced is True
    assert event.hits[0].metadata["forced"] is True


def test_domain_event_contract_imports_without_phase_runtime() -> None:
    code = (
        "from agent.turn_events.observe import ("
        "RETRIEVAL_COMPLETED_EVENT, MEMORY_WRITTEN_EVENT); "
        "import sys; "
        "assert 'agent.lifecycle.phases.after_turn' not in sys.modules; "
        "assert RETRIEVAL_COMPLETED_EVENT.name == 'memory.retrieval.completed'; "
        "assert MEMORY_WRITTEN_EVENT.name == 'memory.written'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _fanout_committed_module(bus: EventBus) -> Any:
    modules = default_after_turn_modules(
        bus,
        cast(Any, object()),
        cast(Any, object()),
    )
    return next(
        item
        for item in modules
        if getattr(item, "slot", "") == "after_turn.fanout_committed"
    )


def _committed_event() -> TurnCommitted:
    return TurnCommitted(
        session_key="session",
        channel="test",
        chat_id="chat",
        input_message="hello",
        persisted_user_message="hello",
        assistant_response="world",
        tools_used=[],
    )


def _retrieval_completed_event() -> RetrievalCompleted:
    return RetrievalCompleted(
        session_key="session",
        channel="test",
        chat_id="chat",
        query="query",
        orig_query=None,
        hits=[],
        injected_count=0,
        route_decision=None,
    )


def _memory_written_event() -> MemoryWritten:
    return MemoryWritten(
        session_key="session",
        channel="test",
        chat_id="chat",
        action="supersede",
        source_ref="session@post_response",
        superseded_ids=["memory-1"],
    )
