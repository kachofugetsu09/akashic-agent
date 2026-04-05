from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.looping.ports import ObservabilityServices, SessionServices
from agent.turns.orchestrator import TurnOrchestrator, TurnOrchestratorDeps
from agent.turns.outbound import OutboundDispatch
from agent.turns.result import TurnOutbound, TurnResult, TurnTrace


class _DummySession:
    def __init__(self, key: str) -> None:
        self.key = key
        self.messages: list[dict] = []
        self.metadata: dict[str, object] = {}
        self.last_consolidated = 0

    def add_message(self, role: str, content: str, media=None, **kwargs) -> None:
        msg = {
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
                session_manager=SimpleNamespace(get_or_create=lambda _key: _DummySession("telegram:123")),
                presence=None,
            ),
            trace=ObservabilityServices(workspace=Path("."), observe_writer=None),
            post_turn=SimpleNamespace(schedule=lambda event: order.append("post_turn")),
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
            session=SessionServices(session_manager=session_manager, presence=presence),
            trace=ObservabilityServices(workspace=Path("."), observe_writer=_Writer()),
            post_turn=SimpleNamespace(schedule=lambda event: post_turn_events.append(event)),
            outbound=_Outbound(),
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

    assert sent is True
    assert session.messages[0]["proactive"] is True
    assert session.messages[0]["content"] == "hello"
    assert post_turn_events
    assert order == ["persist", "side_effect", "dispatch", "presence", "success_effect", "observe"]
