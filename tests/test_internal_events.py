import pytest

from agent.policies.delegation import SpawnDecision, SpawnDecisionMeta
from bus.event_bus import EventBus
from bus.events import SpawnCompletionItem
from bus.events_lifecycle import BeforeDispatch, TurnCompleted
from bus.internal_events import SpawnCompletionEvent


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
        TurnCompleted,
        lambda event: observed.append(event.reply),
    )
    event_bus.on(
        BeforeDispatch,
        lambda event: BeforeDispatch(
            channel=event.channel,
            chat_id=event.chat_id,
            content=event.content + "!",
            thinking=event.thinking,
            media=event.media,
            metadata=event.metadata,
        ),
    )

    await event_bus.observe(
        TurnCompleted(
            session_key="telegram:123",
            channel="telegram",
            chat_id="123",
            reply="ok",
            tools_used=[],
        )
    )
    dispatch = await event_bus.emit(
        BeforeDispatch(channel="telegram", chat_id="123", content="ok")
    )

    assert observed == ["ok"]
    assert dispatch.content == "ok!"
