from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TypedDict, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.context import ContextBuilder
from agent.core.passive_turn import ContextStore, Reasoner, _NoopOutboundPort
from agent.core.runtime_support import SessionLike, TurnRunResult
from agent.core.passive_support import predict_current_user_source_ref
from agent.core.passive_turn import PassiveTurnDeps, PassiveTurnPipeline
from agent.core.types import ContextBundle
from agent.plugin_composition.channels import (
    ChannelDeliveryReceipt,
    DeliveryStatus as ChannelDeliveryStatus,
)
from agent.model_runtime.context_compaction import CommittedContextUnit
from agent.plugin_composition import ChatModels
from agent.looping.ports import SessionServices
from agent.tools.registry import ToolRegistry
from agent.turns.outbound import OutboundDispatch, OutboundPort
from bus.event_bus import EventBus
from bus.events import (
    InboundMessage,
    OutboundMessage,
    TurnDisposition,
)
from agent.lifecycle.types import BeforeReasoningCtx, BeforeTurnCtx
from session.manager import SessionManager
from tests.model_plugin_fakes import build_test_chat_models


class _ChatModelsRunArgs(TypedDict):
    chat_models: ChatModels


def _chat_models() -> _ChatModelsRunArgs:
    return {"chat_models": build_test_chat_models(object())}


class _DummySession:
    def __init__(self, key: str) -> None:
        self.key = key
        self._created_at = datetime(2026, 1, 1, tzinfo=UTC)
        self.messages: list[dict] = []
        self.metadata: dict[str, object] = {}
        self.last_consolidated = 0

    @property
    def created_at(self) -> datetime:
        return self._created_at

    def get_history(self, max_messages: int = 500) -> list[dict]:
        return self.messages[-max_messages:]

    def add_message(
        self,
        role: str,
        content: str,
        media=None,
        **kwargs: object,
    ) -> dict[str, object]:
        if media is not None:
            kwargs["media"] = media
        message = {"role": role, "content": content, **kwargs}
        self.messages.append(message)
        return message

    def history_units(self) -> tuple[CommittedContextUnit, ...]:
        """Render the fake's persisted rows as complete immutable history units."""

        units: list[CommittedContextUnit] = []
        for index, message in enumerate(self.messages):
            message_id = message.get("id")
            if not isinstance(message_id, str) or not message_id:
                message_id = f"{self.key}:{index}"
            raw_seq = message.get("seq")
            seq = (
                raw_seq
                if isinstance(raw_seq, int) and not isinstance(raw_seq, bool)
                else index
            )
            units.append(
                CommittedContextUnit(
                    source_from_seq=seq,
                    consolidated_through_seq=seq,
                    source_message_ids=(message_id,),
                    messages=(dict(message),),
                    message_refs=((message_id, seq),),
                )
            )
        return tuple(units)


class _DeliveringOutboundPort:
    async def dispatch(self, _outbound: OutboundDispatch) -> ChannelDeliveryReceipt:
        return ChannelDeliveryReceipt(
            delivery_id="test-delivery",
            status=ChannelDeliveryStatus.DELIVERED,
        )


@pytest.mark.asyncio
async def test_noop_outbound_port_fails_loud_without_committed_dispatcher() -> None:
    with pytest.raises(RuntimeError, match="committed Channel outbound port 未绑定"):
        await _NoopOutboundPort().dispatch(
            OutboundDispatch(channel="mobile", chat_id="device", content="hello")
        )


@pytest.mark.asyncio
async def test_passive_turn_runs_prepare_prompt_run_commit_in_order():
    order: list[str] = []
    session = _DummySession("telegram:123")
    context_store = SimpleNamespace(
        prepare=AsyncMock(
            side_effect=lambda **kwargs: order.append("prepare")
            or ContextBundle(
                skill_mentions=["refactor"],
            )
        )
    )
    context = SimpleNamespace(
        render=MagicMock(
            side_effect=lambda request: order.append("render")
            or SimpleNamespace(
                system_prompt="system prompt",
                messages=[],
            )
        )
    )
    tools = SimpleNamespace(
        set_context=MagicMock(side_effect=lambda **kwargs: order.append("tool_context"))
    )
    reasoner = SimpleNamespace(
        run_turn=AsyncMock(
            side_effect=lambda *args, **kwargs: order.append("run")
            or TurnRunResult(
                reply="final <meme:shy>\n§cited:[mem_1]§",
                tools_used=["shell"],
                tool_chain=[{"text": "done", "calls": []}],
                thinking="think",
                context_retry={"selected_plan": "full"},
                mobile_attention="confirmation",
            )
        ),
    )
    pipeline = PassiveTurnPipeline(
        PassiveTurnDeps(
            session=cast(
                SessionServices,
                SimpleNamespace(
                    session_manager=SimpleNamespace(
                        get_or_create=MagicMock(return_value=session),
                        peek_next_message_id=MagicMock(return_value="telegram:123:0"),
                        append_messages=AsyncMock(),
                    ),
                    presence=None,
                ),
            ),
            context_store=cast(ContextStore, context_store),
            context=cast(ContextBuilder, context),
            tools=cast(ToolRegistry, tools),
            reasoner=cast(Reasoner, reasoner),
            outbound_port=cast(OutboundPort, _DeliveringOutboundPort()),
        )
    )
    msg = InboundMessage(
        channel="telegram",
        sender="hua",
        chat_id="123",
        content="你好",
        timestamp=datetime(2026, 4, 4, 22, 0, 0),
    )

    out = await pipeline.run(msg, "telegram:123", **_chat_models())

    assert out.content == "final <meme:shy>\n§cited:[mem_1]§"
    assert out.metadata["mobile_attention"] == "confirmation"
    assert order == ["prepare", "tool_context", "render", "run"]
    assert context_store.prepare.await_args.kwargs["session_key"] == "telegram:123"
    render_request = context.render.call_args.args[0]
    assert render_request.current_message == ""
    assert render_request.skill_names == ["refactor"]
    tools.set_context.assert_called_once_with(
        channel="telegram",
        chat_id="123",
        session_key="telegram:123",
        turn_id="",
        current_user_source_ref="telegram:123:0",
        current_timestamp="2026-04-04T22:00:00",
    )
    assert reasoner.run_turn.await_args.kwargs["skill_names"] == ["refactor"]
    # AfterReasoning persists user+assistant messages to session
    assert len(session.messages) == 2
    assert session.messages[0]["role"] == "user"
    assert session.messages[1]["role"] == "assistant"
    assert session.messages[1]["content"] == "final <meme:shy>\n§cited:[mem_1]§"


@pytest.mark.asyncio
async def test_passive_turn_coerces_empty_reply_before_commit():
    session = _DummySession("cli:1")
    context_store = SimpleNamespace(
        prepare=AsyncMock(return_value=ContextBundle()),
    )
    pipeline = PassiveTurnPipeline(
        PassiveTurnDeps(
            session=cast(
                SessionServices,
                SimpleNamespace(
                    session_manager=SimpleNamespace(
                        get_or_create=MagicMock(return_value=session),
                        peek_next_message_id=MagicMock(return_value="cli:1:0"),
                        append_messages=AsyncMock(),
                    ),
                    presence=None,
                ),
            ),
            context_store=cast(ContextStore, context_store),
            context=cast(
                ContextBuilder,
                SimpleNamespace(
                    render=MagicMock(
                        return_value=SimpleNamespace(
                            system_prompt="prompt", messages=[]
                        )
                    ),
                ),
            ),
            tools=cast(
                ToolRegistry,
                SimpleNamespace(set_context=MagicMock()),
            ),
            reasoner=cast(
                Reasoner,
                SimpleNamespace(
                    run_turn=AsyncMock(return_value=TurnRunResult(reply=None)),
                ),
            ),
            outbound_port=cast(OutboundPort, _DeliveringOutboundPort()),
        )
    )
    msg = InboundMessage(
        channel="cli",
        sender="hua",
        chat_id="1",
        content="hi",
        metadata={"mobile_attention": "confirmation"},
    )

    out = await pipeline.run(msg, "cli:1", **_chat_models())

    assert "no response to give" in out.content
    assert "mobile_attention" not in out.metadata


@pytest.mark.asyncio
async def test_passive_turn_before_reasoning_can_patch_context():
    session = _DummySession("telegram:123")
    context_store = SimpleNamespace(
        prepare=AsyncMock(
            return_value=ContextBundle(
                skill_mentions=["old"],
            )
        ),
    )
    context = SimpleNamespace(
        render=MagicMock(
            return_value=SimpleNamespace(system_prompt="prompt", messages=[])
        )
    )
    tools = SimpleNamespace(set_context=MagicMock())
    reasoner = SimpleNamespace(
        run_turn=AsyncMock(return_value=TurnRunResult(reply="ok")),
    )
    event_bus = EventBus()

    event_bus.on(
        BeforeReasoningCtx,
        lambda ctx: BeforeReasoningCtx(
            session_key=ctx.session_key,
            channel=ctx.channel,
            chat_id=ctx.chat_id,
            content=ctx.content,
            timestamp=ctx.timestamp,
            skill_names=["new"],
        ),
    )
    pipeline = PassiveTurnPipeline(
        PassiveTurnDeps(
            session=cast(
                SessionServices,
                SimpleNamespace(
                    session_manager=SimpleNamespace(
                        get_or_create=MagicMock(return_value=session),
                        peek_next_message_id=MagicMock(return_value="telegram:123:0"),
                        append_messages=AsyncMock(),
                    ),
                    presence=None,
                ),
            ),
            context_store=cast(ContextStore, context_store),
            context=cast(ContextBuilder, context),
            tools=cast(ToolRegistry, tools),
            reasoner=cast(Reasoner, reasoner),
            event_bus=event_bus,
            outbound_port=cast(OutboundPort, _DeliveringOutboundPort()),
        )
    )
    msg = InboundMessage(channel="telegram", sender="hua", chat_id="123", content="hi")

    await pipeline.run(msg, "telegram:123", **_chat_models())

    render_request = context.render.call_args.args[0]
    assert render_request.skill_names == ["new"]
    assert reasoner.run_turn.await_args.kwargs["skill_names"] == ["new"]


def test_predict_current_user_source_ref_falls_back_to_last_session_message():
    session = _DummySession("telegram:123")
    session.messages.append({"id": "telegram:123:41"})

    value = predict_current_user_source_ref(
        session_manager=cast(SessionManager, SimpleNamespace()),
        session=cast(SessionLike, session),
    )

    assert value == "telegram:123:41"


@pytest.mark.asyncio
async def test_before_turn_abort_skips_reasoner_and_commit_and_dispatches():
    session = _DummySession("telegram:123")
    context_store = SimpleNamespace(
        prepare=AsyncMock(return_value=ContextBundle()),
    )
    context = SimpleNamespace(
        render=MagicMock(return_value=SimpleNamespace(system_prompt="p", messages=[])),
    )
    tools = SimpleNamespace(set_context=MagicMock())
    reasoner = SimpleNamespace(run_turn=AsyncMock())
    event_bus = EventBus()
    dispatch_port = AsyncMock(return_value=True)

    async def abort_handler(ctx):
        ctx.abort = True
        ctx.abort_reply = "blocked by policy"
        return ctx

    event_bus.on(BeforeTurnCtx, abort_handler)

    pipeline = PassiveTurnPipeline(
        PassiveTurnDeps(
            session=cast(
                SessionServices,
                SimpleNamespace(
                    session_manager=SimpleNamespace(
                        get_or_create=MagicMock(return_value=session),
                    )
                ),
            ),
            context_store=cast(ContextStore, context_store),
            context=cast(ContextBuilder, context),
            tools=cast(ToolRegistry, tools),
            reasoner=cast(Reasoner, reasoner),
            event_bus=event_bus,
            outbound_port=cast(OutboundPort, dispatch_port),
        )
    )
    msg = InboundMessage(channel="telegram", sender="hua", chat_id="123", content="hi")

    out = await pipeline.run(
        msg,
        "telegram:123",
        dispatch_outbound=True,
        **_chat_models(),
    )

    assert out.content == "blocked by policy"
    assert out.turn_disposition is TurnDisposition.SHORT_CIRCUITED
    # 不经过 reasoner 和持久化
    reasoner.run_turn.assert_not_called()
    # 通过 outbound_port 实际 dispatch
    dispatch_port.dispatch.assert_awaited_once()
    dispatched = dispatch_port.dispatch.await_args.args[0]
    assert dispatched.content == "blocked by policy"


@pytest.mark.asyncio
async def test_before_reasoning_abort_skips_reasoner_and_commit_and_dispatches():
    session = _DummySession("telegram:123")
    context_store = SimpleNamespace(
        prepare=AsyncMock(return_value=ContextBundle()),
    )
    context = SimpleNamespace(
        render=MagicMock(return_value=SimpleNamespace(system_prompt="p", messages=[])),
    )
    tools = SimpleNamespace(set_context=MagicMock())
    reasoner = SimpleNamespace(run_turn=AsyncMock())
    event_bus = EventBus()
    dispatch_port = AsyncMock(return_value=True)

    async def abort_handler(ctx):
        ctx.abort = True
        ctx.abort_reply = "rate limited"
        return ctx

    event_bus.on(BeforeReasoningCtx, abort_handler)

    pipeline = PassiveTurnPipeline(
        PassiveTurnDeps(
            session=cast(
                SessionServices,
                SimpleNamespace(
                    session_manager=SimpleNamespace(
                        get_or_create=MagicMock(return_value=session),
                    )
                ),
            ),
            context_store=cast(ContextStore, context_store),
            context=cast(ContextBuilder, context),
            tools=cast(ToolRegistry, tools),
            reasoner=cast(Reasoner, reasoner),
            event_bus=event_bus,
            outbound_port=cast(OutboundPort, dispatch_port),
        )
    )
    msg = InboundMessage(channel="telegram", sender="hua", chat_id="123", content="hi")

    out = await pipeline.run(
        msg,
        "telegram:123",
        dispatch_outbound=True,
        **_chat_models(),
    )

    assert out.content == "rate limited"
    assert out.turn_disposition is TurnDisposition.SHORT_CIRCUITED
    reasoner.run_turn.assert_not_called()
    dispatch_port.dispatch.assert_awaited_once()
    dispatched = dispatch_port.dispatch.await_args.args[0]
    assert dispatched.content == "rate limited"


@pytest.mark.asyncio
async def test_abort_does_not_dispatch_when_dispatch_outbound_false():
    session = _DummySession("telegram:123")
    context_store = SimpleNamespace(
        prepare=AsyncMock(return_value=ContextBundle()),
    )
    context = SimpleNamespace(
        render=MagicMock(return_value=SimpleNamespace(system_prompt="p", messages=[])),
    )
    tools = SimpleNamespace(set_context=MagicMock())
    reasoner = SimpleNamespace(run_turn=AsyncMock())
    event_bus = EventBus()
    dispatch_port = AsyncMock(return_value=True)

    async def abort_handler(ctx):
        ctx.abort = True
        ctx.abort_reply = "quiet abort"
        return ctx

    event_bus.on(BeforeTurnCtx, abort_handler)

    pipeline = PassiveTurnPipeline(
        PassiveTurnDeps(
            session=cast(
                SessionServices,
                SimpleNamespace(
                    session_manager=SimpleNamespace(
                        get_or_create=MagicMock(return_value=session),
                    )
                ),
            ),
            context_store=cast(ContextStore, context_store),
            context=cast(ContextBuilder, context),
            tools=cast(ToolRegistry, tools),
            reasoner=cast(Reasoner, reasoner),
            event_bus=event_bus,
            outbound_port=cast(OutboundPort, dispatch_port),
        )
    )
    msg = InboundMessage(channel="telegram", sender="hua", chat_id="123", content="hi")

    out = await pipeline.run(msg, "telegram:123", dispatch_outbound=False)

    assert out.content == "quiet abort"
    reasoner.run_turn.assert_not_called()
    dispatch_port.dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_reasoner_exception_turn_returns_control_outbound():
    session = _DummySession("telegram:123")
    context_store = SimpleNamespace(
        prepare=AsyncMock(return_value=ContextBundle()),
    )
    context = SimpleNamespace(
        render=MagicMock(return_value=SimpleNamespace(system_prompt="p", messages=[])),
    )
    tools = SimpleNamespace(set_context=MagicMock())
    reasoner = SimpleNamespace(
        run_turn=AsyncMock(side_effect=RuntimeError("budget guard")),
    )
    dispatch_port = AsyncMock(return_value=True)

    pipeline = PassiveTurnPipeline(
        PassiveTurnDeps(
            session=cast(
                SessionServices,
                SimpleNamespace(
                    session_manager=SimpleNamespace(
                        get_or_create=MagicMock(return_value=session),
                        peek_next_message_id=MagicMock(return_value="telegram:123:0"),
                        append_messages=AsyncMock(),
                    ),
                    presence=None,
                ),
            ),
            context_store=cast(ContextStore, context_store),
            context=cast(ContextBuilder, context),
            tools=cast(ToolRegistry, tools),
            reasoner=cast(Reasoner, reasoner),
            outbound_port=cast(OutboundPort, dispatch_port),
        )
    )
    msg = InboundMessage(channel="telegram", sender="hua", chat_id="123", content="hi")

    out = await pipeline.run(
        msg,
        "telegram:123",
        dispatch_outbound=True,
        **_chat_models(),
    )

    assert out.content == "处理消息时出错，请稍后再试。"
    dispatch_port.dispatch.assert_awaited_once()
    dispatched = dispatch_port.dispatch.await_args.args[0]
    assert dispatched.content == "处理消息时出错，请稍后再试。"

    with pytest.raises(RuntimeError, match="budget guard"):
        _ = await pipeline.run(
            msg,
            "telegram:123",
            dispatch_outbound=False,
            **_chat_models(),
        )


@pytest.mark.asyncio
async def test_after_turn_dispatch_exception_is_not_wrapped_by_control_outbound():
    session = _DummySession("telegram:123")
    context_store = SimpleNamespace(
        prepare=AsyncMock(return_value=ContextBundle()),
    )
    context = SimpleNamespace(
        render=MagicMock(return_value=SimpleNamespace(system_prompt="p", messages=[])),
    )
    tools = SimpleNamespace(set_context=MagicMock())
    reasoner = SimpleNamespace(
        run_turn=AsyncMock(
            return_value=TurnRunResult(
                reply="ok",
                tools_used=[],
                tool_chain=[],
                thinking=None,
                context_retry={},
            )
        )
    )
    dispatch_port = SimpleNamespace(
        dispatch=AsyncMock(side_effect=RuntimeError("dispatch failed"))
    )

    pipeline = PassiveTurnPipeline(
        PassiveTurnDeps(
            session=cast(
                SessionServices,
                SimpleNamespace(
                    session_manager=SimpleNamespace(
                        get_or_create=MagicMock(return_value=session),
                        peek_next_message_id=MagicMock(return_value="telegram:123:0"),
                        append_messages=AsyncMock(),
                    ),
                    presence=None,
                ),
            ),
            context_store=cast(ContextStore, context_store),
            context=cast(ContextBuilder, context),
            tools=cast(ToolRegistry, tools),
            reasoner=cast(Reasoner, reasoner),
            outbound_port=cast(OutboundPort, dispatch_port),
        )
    )
    msg = InboundMessage(channel="telegram", sender="hua", chat_id="123", content="hi")

    with pytest.raises(RuntimeError, match="dispatch failed"):
        await pipeline.run(
            msg,
            "telegram:123",
            dispatch_outbound=True,
            **_chat_models(),
        )

    assert len(session.messages) == 2
    assert session.messages[0]["role"] == "user"
    assert session.messages[1]["role"] == "assistant"
    assert session.messages[1]["content"] == "ok"
    dispatch_port.dispatch.assert_awaited_once()
