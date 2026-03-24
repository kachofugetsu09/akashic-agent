from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.looping.ports import ObservabilityServices, SessionServices
from agent.looping.turn_types import RetrievalTrace
from agent.turns.outbound import OutboundDispatch
from agent.turns.orchestrator import TurnOrchestrator, TurnOrchestratorDeps
from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from bus.events import InboundMessage


class _DummySession:
    def __init__(self, key: str) -> None:
        self.key = key
        self.messages: list[dict] = []
        self.metadata: dict[str, object] = {}
        self.last_consolidated = 0

    def get_history(self, max_messages: int = 500) -> list[dict]:
        return self.messages[-max_messages:]

    def add_message(self, role: str, content: str, media=None, **kwargs) -> None:
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if media:
            msg["media"] = list(media)
        msg.update(kwargs)
        self.messages.append(msg)


@pytest.mark.asyncio
async def test_orchestrator_passive_order_and_outbound_metadata():
    order: list[str] = []

    async def _append_messages(_session, _msgs):
        order.append("persist")

    presence = SimpleNamespace(record_user_message=lambda _key: None)
    session_manager = SimpleNamespace(append_messages=AsyncMock(side_effect=_append_messages))
    session_svc = SessionServices(session_manager=session_manager, presence=presence)

    class _Writer:
        def __init__(self) -> None:
            self.events: list[object] = []

        def emit(self, event: object) -> None:
            order.append("observe")
            self.events.append(event)

    writer = _Writer()
    trace_svc = ObservabilityServices(workspace=Path("."), observe_writer=writer)

    class _PostTurn:
        def __init__(self) -> None:
            self.events: list[object] = []

        def schedule(self, event) -> None:
            order.append("post_turn")
            self.events.append(event)

    post_turn = _PostTurn()
    dispatched: list[OutboundDispatch] = []

    class _Outbound:
        async def dispatch(self, outbound: OutboundDispatch) -> bool:
            order.append("dispatch")
            dispatched.append(outbound)
            return True

    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=session_svc,
            trace=trace_svc,
            post_turn=post_turn,
            outbound=_Outbound(),
        )
    )

    class _Effect:
        async def run(self) -> None:
            order.append("side_effect")

    msg = InboundMessage(
        channel="telegram",
        sender="user",
        chat_id="123",
        content="你好",
        metadata={"req_id": "r1"},
    )
    session = _DummySession("telegram:123")
    session_manager.get_or_create = lambda _key: session
    result = TurnResult(
        decision="reply",
        outbound=TurnOutbound(session_key="telegram:123", content="收到"),
        trace=TurnTrace(
            source="passive",
            retrieval={"raw": RetrievalTrace(raw={"rag": 1}).raw},
            extra={
                "tools_used": ["noop"],
                "tool_chain": [{"text": "", "calls": []}],
                "thinking": "思考",
            },
        ),
        side_effects=[_Effect()],
    )
    out = await orchestrator.handle_turn(msg=msg, result=result)

    assert out.channel == "telegram"
    assert out.chat_id == "123"
    assert out.content == "收到"
    assert out.thinking == "思考"
    assert out.metadata["req_id"] == "r1"
    assert out.metadata["tools_used"] == ["noop"]
    assert len(writer.events) == 2
    assert dispatched[0].content == "收到"
    assert order == ["persist", "observe", "observe", "post_turn", "side_effect", "dispatch"]
