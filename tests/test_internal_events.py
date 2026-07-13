import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.policies.delegation import SpawnDecision, SpawnDecisionMeta
from bus.event_bus import EventBus
from bus.events import SpawnCompletionItem
from bus.internal_events import SpawnCompletionEvent


@dataclass
class _FakeLifecycleEvent:
    session_key: str
    channel: str
    chat_id: str
    content: str
    thinking: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


def test_spawn_completion_item_carries_typed_payload():
    decision = SpawnDecision(
        should_spawn=True,
        label="job",
        meta=SpawnDecisionMeta(
            source="heuristic",
            confidence="high",
            reason_code="long_running",
        ),
    )
    event = SpawnCompletionEvent(
        job_id="abcd1234",
        label="job",
        task="do work",
        status="incomplete",
        exit_reason="forced_summary",
        result="partial",
    )

    item = SpawnCompletionItem(
        channel="telegram",
        chat_id="123",
        event=event,
        decision=decision,
    )

    assert item.session_key == "telegram:123"
    assert item.event == event
    assert item.decision == decision


@pytest.mark.asyncio
async def test_event_bus_observe_and_intercept_are_ordered():
    event_bus = EventBus()
    observed: list[str] = []

    event_bus.on(
        _FakeLifecycleEvent,
        lambda event: observed.append(event.content),
    )
    event_bus.on(
        _FakeLifecycleEvent,
        lambda event: _FakeLifecycleEvent(
            session_key=event.session_key,
            channel=event.channel,
            chat_id=event.chat_id,
            content=event.content + "!",
            thinking=event.thinking,
            media=list(event.media),
            metadata=dict(event.metadata),
        ),
    )

    await event_bus.observe(
        _FakeLifecycleEvent(
            session_key="telegram:123",
            channel="telegram",
            chat_id="123",
            content="ok",
        )
    )
    dispatch = await event_bus.emit(
        _FakeLifecycleEvent(session_key="telegram:123", channel="telegram", chat_id="123", content="ok")
    )

    assert observed == ["ok", "ok"]
    assert dispatch.content == "ok!"


@pytest.mark.asyncio
async def test_event_bus_fanout_keeps_other_observers_when_one_fails(caplog):
    event_bus = EventBus()
    observed: list[str] = []

    def _bad(_event: _FakeLifecycleEvent) -> None:
        raise RuntimeError("boom")

    event_bus.on(_FakeLifecycleEvent, _bad)
    event_bus.on(_FakeLifecycleEvent, lambda event: observed.append(event.content))

    await event_bus.fanout(
        _FakeLifecycleEvent(
            session_key="telegram:123",
            channel="telegram",
            chat_id="123",
            content="ok",
        )
    )

    assert observed == ["ok"]
    assert "fanout completed with observer errors" in caplog.text


@pytest.mark.asyncio
async def test_event_bus_enqueue_runs_observers_in_background():
    event_bus = EventBus()
    observed: list[str] = []

    event_bus.on(_FakeLifecycleEvent, lambda event: observed.append(event.content))
    event_bus.enqueue(
        _FakeLifecycleEvent(
            session_key="telegram:123",
            channel="telegram",
            chat_id="123",
            content="ok",
        )
    )
    await event_bus.drain()

    assert observed == ["ok"]
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_event_bus_reports_admission_task_failure(caplog):
    class _FailingSnapshotStore:
        current = SimpleNamespace(accepting_leases=False)

        async def acquire(self):
            raise RuntimeError("snapshot admission failed")

    event_bus = EventBus()
    event_bus.bind_runtime_snapshot_store(cast(Any, _FailingSnapshotStore()))
    event_bus.enqueue(
        _FakeLifecycleEvent(
            session_key="telegram:123",
            channel="telegram",
            chat_id="123",
            content="ok",
        )
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert "event enqueue admission failed" in caplog.text
    assert "snapshot admission failed" in caplog.text
    assert not event_bus._pending_enqueue_tasks
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_event_bus_close_preserves_inflight_admission_failure():
    admission_started = asyncio.Event()

    class _AdmissionFailsDuringCancellation:
        current = SimpleNamespace(accepting_leases=False)

        async def acquire(self):
            admission_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise RuntimeError("admission cleanup failed")

    event_bus = EventBus()
    event_bus.bind_runtime_snapshot_store(
        cast(Any, _AdmissionFailsDuringCancellation())
    )
    event_bus.enqueue(
        _FakeLifecycleEvent(
            session_key="telegram:123",
            channel="telegram",
            chat_id="123",
            content="ok",
        )
    )
    await admission_started.wait()

    with pytest.raises(RuntimeError, match="admission cleanup failed"):
        await event_bus.aclose()

    assert not event_bus._pending_enqueue_tasks
    assert event_bus._observe_task is None


@pytest.mark.asyncio
async def test_event_bus_drain_preserves_dispatcher_failure():
    dispatcher_started = asyncio.Event()

    class _FailingDispatcher(EventBus):
        async def _fanout_queued(self, envelope):
            dispatcher_started.set()
            raise RuntimeError(f"dispatcher failed: {envelope.event.content}")

    event_bus = _FailingDispatcher()
    event_bus.enqueue(
        _FakeLifecycleEvent(
            session_key="telegram:123",
            channel="telegram",
            chat_id="123",
            content="fault",
        )
    )
    await dispatcher_started.wait()

    with pytest.raises(RuntimeError, match="dispatcher failed: fault"):
        await event_bus.drain()

    await event_bus.aclose()


@pytest.mark.asyncio
async def test_event_bus_self_cancelled_observer_does_not_stall_close():
    event_bus = EventBus()
    observed: list[str] = []

    async def cancel_self(event: _FakeLifecycleEvent) -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await asyncio.sleep(0)

    event_bus.on(_FakeLifecycleEvent, cancel_self)
    event_bus.on(_FakeLifecycleEvent, lambda event: observed.append(event.content))
    for content in ("first", "second"):
        event_bus.enqueue(
            _FakeLifecycleEvent(
                session_key="telegram:123",
                channel="telegram",
                chat_id="123",
                content=content,
            )
        )

    await asyncio.wait_for(event_bus.aclose(), timeout=1)

    assert observed == ["first", "second"]
