from __future__ import annotations
from typing import Any, cast

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.looping.ports import ObservabilityServices, SessionServices
from agent.turns.orchestrator import TurnOrchestrator, TurnOrchestratorDeps
from agent.turns.outbound import OutboundDispatch
from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted


class _DummySession:
    def __init__(self, key: str) -> None:
        self.key = key
        self.messages: list[dict[str, object]] = []
        self.metadata: dict[str, object] = {}
        self.last_consolidated = 0

    def add_message(self, role: str, content: str, media=None, **kwargs) -> None:
        msg: dict[str, object] = {
            "role": role,
            "content": content,
        }
        if media:
            msg["media"] = list(media)
        msg.update(kwargs)
        self.messages.append(msg)


@pytest.mark.asyncio
async def test_orchestrator_skip_runs_side_effects_without_dispatch():
    order: list[str] = []

    class _Effect:
        async def run(self) -> None:
            order.append("side_effect")

    class _Outbound:
        async def dispatch(self, outbound: OutboundDispatch) -> bool:
            order.append("dispatch")
            return True

    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=SessionServices(
                session_manager=cast(Any, SimpleNamespace(get_or_create=lambda _key: _DummySession("telegram:123"))),
                presence=None,
            ),
            trace=ObservabilityServices(workspace=Path("."), observe_writer=None),
            post_turn=cast(Any, SimpleNamespace(schedule=lambda event: order.append("post_turn"))),
            outbound=_Outbound(),
        )
    )

    sent = await orchestrator.handle_proactive_turn(
        result=TurnResult(
            decision="skip",
            outbound=None,
            trace=TurnTrace(source="proactive", extra={"skip_reason": "quiet_hours"}),
            side_effects=[_Effect()],
        ),
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
    )

    assert sent is False
    assert order == ["side_effect"]


@pytest.mark.asyncio
async def test_orchestrator_proactive_reply_persists_dispatches_and_runs_success_effects():
    order: list[str] = []
    session = _DummySession("telegram:123")
    post_turn_events: list[object] = []
    committed_events: list[TurnCommitted] = []
    event_bus = EventBus()
    event_bus.on(TurnCommitted, lambda event: committed_events.append(event))

    class _Effect:
        def __init__(self, name: str) -> None:
            self._name = name

        async def run(self) -> None:
            order.append(self._name)

    class _Outbound:
        async def dispatch(self, outbound: OutboundDispatch) -> bool:
            order.append("dispatch")
            assert outbound.content == "hello"
            return True

    class _Writer:
        def __init__(self) -> None:
            self.events: list[object] = []

        def emit(self, event: object) -> None:
            order.append("observe")
            self.events.append(event)

    presence = SimpleNamespace(record_proactive_sent=lambda _key: order.append("presence"))
    session_manager = SimpleNamespace(
        get_or_create=lambda _key: session,
        append_messages=AsyncMock(side_effect=lambda *_args, **_kwargs: order.append("persist")),
    )
    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=SessionServices(session_manager=cast(Any, session_manager), presence=cast(Any, presence)),
            trace=ObservabilityServices(workspace=Path("."), observe_writer=_Writer()),
            post_turn=cast(Any, SimpleNamespace(schedule=lambda event: post_turn_events.append(event))),
            outbound=_Outbound(),
            event_bus=event_bus,
        )
    )

    sent = await orchestrator.handle_proactive_turn(
        result=TurnResult(
            decision="reply",
            outbound=TurnOutbound(session_key="telegram:123", content="hello"),
            evidence=["feed:1"],
            trace=TurnTrace(
                source="proactive",
                extra={
                    "tools_used": ["web_search"],
                    "tool_chain": [{"text": "", "calls": []}],
                    "steps_taken": 2,
                },
            ),
            side_effects=[_Effect("side_effect")],
            success_side_effects=[_Effect("success_effect")],
            failure_side_effects=[_Effect("failure_effect")],
        ),
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
    )
    await event_bus.drain()

    assert sent is True
    assert session.messages[0]["proactive"] is True
    assert session.messages[0]["content"] == "hello"
    assert post_turn_events
    assert committed_events[0].session_key == "telegram:123"
    assert committed_events[0].persisted_user_message is None
    assert committed_events[0].assistant_response == "hello"
    assert committed_events[0].extra == {"skip_observe_trace": True}
    assert order == ["persist", "side_effect", "dispatch", "presence", "success_effect", "observe"]
    await event_bus.aclose()
