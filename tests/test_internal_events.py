from dataclasses import dataclass, field

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
