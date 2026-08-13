from __future__ import annotations
from typing import Any, cast

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.looping.ports import SessionServices
from agent.turns.orchestrator import TurnOrchestrator, TurnOrchestratorDeps
from agent.turns.outbound import OutboundDispatch, PushToolOutboundPort
from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from bus.events import DeliveryReceipt, DeliveryStatus


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
        async def dispatch(self, outbound: OutboundDispatch) -> DeliveryReceipt:
            order.append("dispatch")
            return DeliveryReceipt(DeliveryStatus.SUCCESS)

    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=SessionServices(
                session_manager=cast(Any, SimpleNamespace(get_or_create=lambda _key: _DummySession("telegram:123"))),
                presence=None,
            ),
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
    dispatched_delivery_ids: list[str] = []

    class _Effect:
        def __init__(self, name: str) -> None:
            self._name = name

        async def run(self) -> None:
            order.append(self._name)

    class _Outbound:
        async def dispatch(self, outbound: OutboundDispatch) -> DeliveryReceipt:
            order.append("dispatch")
            assert outbound.content == "hello"
            delivery_id = outbound.metadata["delivery_id"]
            assert isinstance(delivery_id, str)
            assert len(delivery_id) == 32
            assert outbound.control_turn_id is not None
            assert outbound.control_turn_id.startswith("turn:")
            dispatched_delivery_ids.append(delivery_id)
            return DeliveryReceipt(DeliveryStatus.SUCCESS)

    presence = SimpleNamespace(record_proactive_sent=lambda _key: order.append("presence"))
    session_manager = SimpleNamespace(
        get_or_create=lambda _key: session,
        append_messages=AsyncMock(side_effect=lambda *_args, **_kwargs: order.append("persist")),
    )
    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=SessionServices(session_manager=cast(Any, session_manager), presence=cast(Any, presence)),
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
    assert session.messages[-1]["control_turn_id"].startswith("turn:")
    assert session.messages[0]["proactive"] is True
    assert session.messages[0]["content"] == "hello"
    assert session.messages[0]["delivery_id"] == dispatched_delivery_ids[0]
    assert order == ["side_effect", "dispatch", "persist", "presence", "success_effect"]


@pytest.mark.asyncio
async def test_orchestrator_failed_dispatch_does_not_persist_proactive_message():
    session = _DummySession("telegram:123")
    session_manager = SimpleNamespace(
        get_or_create=lambda _key: session,
        append_messages=AsyncMock(return_value=None),
    )
    outbound = SimpleNamespace(
        dispatch=AsyncMock(return_value=DeliveryReceipt(DeliveryStatus.FAILED))
    )
    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=SessionServices(
                session_manager=cast(Any, session_manager),
                presence=None,
            ),
            outbound=cast(Any, outbound),
        )
    )

    sent = await orchestrator.handle_proactive_turn(
        result=TurnResult(
            decision="reply",
            outbound=TurnOutbound(session_key="telegram:123", content="not delivered"),
        ),
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
    )

    assert sent is False
    assert session.messages == []
    session_manager.append_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_partial_dispatch_does_not_persist_proactive_message():
    session = _DummySession("telegram:123")
    session_manager = SimpleNamespace(
        get_or_create=lambda _key: session,
        append_messages=AsyncMock(return_value=None),
    )
    outbound = SimpleNamespace(
        dispatch=AsyncMock(
            return_value=DeliveryReceipt(
                DeliveryStatus.PARTIAL,
                detail="attachment unavailable",
            )
        )
    )
    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=SessionServices(
                session_manager=cast(Any, session_manager),
                presence=None,
            ),
            outbound=cast(Any, outbound),
        )
    )

    sent = await orchestrator.handle_proactive_turn(
        result=TurnResult(
            decision="reply",
            outbound=TurnOutbound(
                session_key="telegram:123",
                content="正文",
                media=["/tmp/report.pdf"],
            ),
        ),
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
    )

    assert sent is False
    assert session.messages == []
    session_manager.append_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_push_outbound_port_propagates_unexpected_tool_error():
    push_tool = SimpleNamespace(
        dispatch=AsyncMock(side_effect=RuntimeError("channel disconnected"))
    )
    port = PushToolOutboundPort(push_tool)

    with pytest.raises(RuntimeError, match="channel disconnected"):
        await port.dispatch(
            OutboundDispatch(
                channel="telegram",
                chat_id="123",
                content="hello",
            )
        )


@pytest.mark.asyncio
async def test_push_outbound_port_forwards_internal_metadata():
    push_tool = SimpleNamespace(
        dispatch=AsyncMock(
            return_value=DeliveryReceipt(DeliveryStatus.SUCCESS)
        )
    )
    port = PushToolOutboundPort(push_tool)

    sent = await port.dispatch(
        OutboundDispatch(
            channel="mobile",
            chat_id="123",
            content="hello",
            metadata={"delivery_id": "delivery-1"},
        )
    )

    assert sent.status is DeliveryStatus.SUCCESS
    pushed = push_tool.dispatch.await_args.args[0]
    assert pushed.channel == "mobile"
    assert pushed.chat_id == "123"
    assert pushed.content == "hello"
    assert pushed.metadata == {"delivery_id": "delivery-1"}


@pytest.mark.asyncio
async def test_orchestrator_logs_dispatch_error_and_runs_failure_effect(
    caplog: pytest.LogCaptureFixture,
):
    session = _DummySession("telegram:123")
    session_manager = SimpleNamespace(
        get_or_create=lambda _key: session,
        append_messages=AsyncMock(return_value=None),
    )
    outbound = SimpleNamespace(
        dispatch=AsyncMock(side_effect=RuntimeError("channel disconnected"))
    )
    failures: list[str] = []

    class _FailureEffect:
        async def run(self) -> None:
            failures.append("failed")

    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=SessionServices(
                session_manager=cast(Any, session_manager),
                presence=None,
            ),
            outbound=cast(Any, outbound),
        )
    )

    with caplog.at_level("ERROR", logger="agent.turn_orchestrator"):
        sent = await orchestrator.handle_proactive_turn(
            result=TurnResult(
                decision="reply",
                outbound=TurnOutbound(
                    session_key="telegram:123",
                    content="not delivered",
                ),
                failure_side_effects=[_FailureEffect()],
            ),
            session_key="telegram:123",
            channel="telegram",
            chat_id="123",
        )

    assert sent is False
    assert failures == ["failed"]
    assert session.messages == []
    session_manager.append_messages.assert_not_awaited()
    assert "channel disconnected" in caplog.text


@pytest.mark.asyncio
async def test_orchestrator_proactive_reply_dispatches_media():
    session = _DummySession("telegram:123")
    dispatched: list[OutboundDispatch] = []

    class _Outbound:
        async def dispatch(self, outbound: OutboundDispatch) -> DeliveryReceipt:
            dispatched.append(outbound)
            return DeliveryReceipt(
                DeliveryStatus.SUCCESS,
                canonical_media=("/stable/meme.png",),
            )

    session_manager = SimpleNamespace(
        get_or_create=lambda _key: session,
        append_messages=AsyncMock(return_value=None),
    )
    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=SessionServices(session_manager=cast(Any, session_manager), presence=None),
            outbound=_Outbound(),
        )
    )

    sent = await orchestrator.handle_proactive_turn(
        result=TurnResult(
            decision="reply",
            outbound=TurnOutbound(
                session_key="telegram:123",
                content="新表情来啦",
                media=["/tmp/meme.png"],
            ),
        ),
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
    )

    assert sent is True
    assert dispatched[0].media == ["/tmp/meme.png"]
    assert session.messages[0]["media"] == ["/stable/meme.png"]
