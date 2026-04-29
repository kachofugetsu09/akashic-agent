from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from agent.context import ContextBuilder
from agent.core.passive_support import build_context_hint_message
from agent.core.passive_turn import ContextStore
from agent.core.types import ContextBundle
from agent.lifecycle.phase import Phase
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from bus.events import InboundMessage
from agent.lifecycle.types import (
    AfterStepCtx,
    BeforeReasoningCtx,
    BeforeReasoningInput,
    BeforeStepCtx,
    BeforeStepInput,
    BeforeTurnCtx,
    TurnState,
)
from agent.lifecycle.phases.after_step import (
    AfterStepFrame,
    default_after_step_modules,
)
from agent.lifecycle.phases.before_reasoning import (
    BeforeReasoningFrame,
    default_before_reasoning_modules,
)
from agent.lifecycle.phases.before_step import (
    BeforeStepFrame,
    default_before_step_modules,
)
from agent.lifecycle.phases.before_turn import (
    BeforeTurnFrame,
    default_before_turn_modules,
)
from session.manager import SessionManager

_now = datetime.now()


def _inbound() -> InboundMessage:
    return InboundMessage(
        channel="telegram", sender="user", chat_id="123",
        content="hello", timestamp=_now,
    )


class _DummySession:
    def __init__(self, key: str) -> None:
        self.key = key
        self.messages: list[dict[str, object]] = []
        self.metadata: dict[str, object] = {}
        self.last_consolidated = 0

    def get_history(self, max_messages: int = 500, *, start_index: int | None = None) -> list[dict[str, object]]:
        return list(self.messages)

    def add_message(self, role: str, content: str, media=None, **kwargs: object) -> None:
        msg: dict[str, object] = {"role": role, "content": content}
        if media:
            msg["media"] = list(media)
        msg.update(kwargs)
        self.messages.append(msg)


# ── BeforeTurn ──


@pytest.mark.asyncio
async def test_before_turn_setup_fills_turn_state():
    bus = EventBus()
    session = _DummySession("telegram:123")

    session_mgr = SimpleNamespace(
        get_or_create=lambda key: session,
    )

    bundle = ContextBundle(
        skill_mentions=["search"],
        retrieved_memory_block="block_text",
        retrieval_trace_raw={"trace": 1},
        history_messages=[{"role": "user", "content": "prev"}],
    )
    ctx_store = SimpleNamespace(
        prepare=AsyncMock(return_value=bundle),
    )

    phase = Phase(
        default_before_turn_modules(
            bus,
            cast(SessionManager, session_mgr),
            cast(ContextStore, ctx_store),
        ),
        frame_factory=BeforeTurnFrame,
    )
    msg = _inbound()
    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)

    ctx = await phase.run(state)

    assert state.session is session
    assert state.retrieval_raw == {"trace": 1}
    assert ctx.skill_names == ["search"]
    assert ctx.retrieved_memory_block == "block_text"
    assert ctx.retrieval_trace_raw == {"trace": 1}
    assert ctx.history_messages == ({"role": "user", "content": "prev"},)
    assert ctx.abort is False


@pytest.mark.asyncio
async def test_before_turn_chain_can_abort():
    bus = EventBus()
    session = _DummySession("telegram:123")

    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    bundle = ContextBundle()
    ctx_store = SimpleNamespace(prepare=AsyncMock(return_value=bundle))

    async def abort_handler(ctx):
        ctx.abort = True
        ctx.abort_reply = "rate limited"
        return ctx

    bus.on(BeforeTurnCtx, abort_handler)

    phase = Phase(
        default_before_turn_modules(
            bus,
            cast(SessionManager, session_mgr),
            cast(ContextStore, ctx_store),
        ),
        frame_factory=BeforeTurnFrame,
    )
    msg = _inbound()
    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)

    ctx = await phase.run(state)
    assert ctx.abort is True
    assert ctx.abort_reply == "rate limited"


@pytest.mark.asyncio
async def test_before_turn_memory_status_command_aborts_without_context_prepare():
    bus = EventBus()
    session = _DummySession("telegram:123")
    session.messages = [
        {
            "role": "user",
            "content": '<system-reminder data-system-context-frame="true">内部</system-reminder>',
        },
        {"role": "user", "content": "帮我看看 Telegram 流式消息为什么重复发送"},
        {"role": "assistant", "content": "已修复"},
        {"role": "user", "content": "再看一下超时问题"},
    ]
    session.last_consolidated = 3
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    ctx_store = SimpleNamespace(prepare=AsyncMock())

    phase = Phase(
        default_before_turn_modules(
            bus,
            cast(SessionManager, session_mgr),
            cast(ContextStore, ctx_store),
        ),
        frame_factory=BeforeTurnFrame,
    )
    msg = InboundMessage(
        channel="telegram",
        sender="user",
        chat_id="123",
        content="/memory_status",
        timestamp=_now,
    )
    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)

    ctx = await phase.run(state)

    assert ctx.abort is True
    assert "上次整理到 1 条用户消息之前。" in ctx.abort_reply
    assert "帮我看看 Telegram 流式消息为什么重复发送" in ctx.abort_reply
    assert "尚未整理的用户消息数：1" in ctx.abort_reply
    assert "当前会话消息数：4" in ctx.abort_reply
    assert "内部" not in ctx.abort_reply
    ctx_store.prepare.assert_not_called()


@pytest.mark.asyncio
async def test_before_turn_chain_can_modify_skill_names():
    bus = EventBus()
    session = _DummySession("telegram:123")

    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    bundle = ContextBundle(skill_mentions=["search"])
    ctx_store = SimpleNamespace(prepare=AsyncMock(return_value=bundle))

    async def add_skill(ctx):
        ctx.skill_names.append("added_skill")
        return ctx

    bus.on(BeforeTurnCtx, add_skill)

    phase = Phase(
        default_before_turn_modules(
            bus,
            cast(SessionManager, session_mgr),
            cast(ContextStore, ctx_store),
        ),
        frame_factory=BeforeTurnFrame,
    )
    msg = _inbound()
    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)

    ctx = await phase.run(state)
    assert ctx.skill_names == ["search", "added_skill"]


# ── BeforeReasoning ──


@pytest.mark.asyncio
async def test_before_reasoning_setup_calls_tools_set_context():
    bus = EventBus()
    tools = Mock()
    tools.set_context = Mock()

    session = _DummySession("telegram:123")
    session.messages.append({"role": "user", "content": "prev", "id": "msg_42"})
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)

    context_builder = Mock()
    context_builder.render = Mock(return_value=None)

    phase = Phase(
        default_before_reasoning_modules(
            bus,
            cast(ToolRegistry, tools),
            cast(SessionManager, session_mgr),
            cast(ContextBuilder, context_builder),
        ),
        frame_factory=BeforeReasoningFrame,
    )
    msg = _inbound()

    before_turn = BeforeTurnCtx(
        session_key="telegram:123", channel=msg.channel, chat_id=msg.chat_id,
        content=msg.content, timestamp=msg.timestamp,
        retrieved_memory_block="block", retrieval_trace_raw=None,
        history_messages=(),
        skill_names=["search"],
    )

    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)
    state.session = session

    ctx = await phase.run(BeforeReasoningInput(state=state, before_turn=before_turn))

    tools.set_context.assert_called_once()
    call_kwargs = tools.set_context.call_args[1]
    assert call_kwargs["channel"] == "telegram"
    assert call_kwargs["chat_id"] == "123"
    assert "current_user_source_ref" in call_kwargs

    assert ctx.skill_names == ["search"]
    assert ctx.retrieved_memory_block == "block"
    assert ctx.extra_hints == []


@pytest.mark.asyncio
async def test_before_reasoning_requires_session():
    bus = EventBus()
    tools = Mock()
    session_mgr = Mock()
    context_builder = Mock()

    phase = Phase(
        default_before_reasoning_modules(
            bus,
            cast(ToolRegistry, tools),
            cast(SessionManager, session_mgr),
            cast(ContextBuilder, context_builder),
        ),
        frame_factory=BeforeReasoningFrame,
    )
    msg = _inbound()

    before_turn = BeforeTurnCtx(
        session_key="telegram:123", channel=msg.channel, chat_id=msg.chat_id,
        content=msg.content, timestamp=msg.timestamp,
        retrieved_memory_block="", retrieval_trace_raw=None,
        history_messages=(),
    )

    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)
    # session is None

    with pytest.raises(RuntimeError, match="BeforeReasoning requires TurnState.session"):
        await phase.run(BeforeReasoningInput(state=state, before_turn=before_turn))


@pytest.mark.asyncio
async def test_before_reasoning_finalize_calls_render():
    bus = EventBus()
    tools = Mock()
    tools.set_context = Mock()

    session = _DummySession("telegram:123")
    session.my_meta = {"a": 1}
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    session_mgr.peek_next_message_id = None

    context_builder = Mock()
    context_builder.render = Mock(return_value=None)

    phase = Phase(
        default_before_reasoning_modules(
            bus,
            cast(ToolRegistry, tools),
            cast(SessionManager, session_mgr),
            cast(ContextBuilder, context_builder),
        ),
        frame_factory=BeforeReasoningFrame,
    )
    msg = _inbound()

    before_turn = BeforeTurnCtx(
        session_key="telegram:123", channel=msg.channel, chat_id=msg.chat_id,
        content=msg.content, timestamp=msg.timestamp,
        retrieved_memory_block="block", retrieval_trace_raw=None,
        history_messages=(),
        skill_names=["search"],
    )

    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)
    state.session = session

    ctx = await phase.run(BeforeReasoningInput(state=state, before_turn=before_turn))

    context_builder.render.assert_called_once()
    call_args = context_builder.render.call_args[0][0]
    assert call_args.skill_names == ["search"]
    assert call_args.retrieved_memory_block == "block"
    assert call_args.channel == msg.channel
    assert call_args.chat_id == msg.chat_id


@pytest.mark.asyncio
async def test_before_reasoning_chain_can_add_extra_hints():
    bus = EventBus()
    tools = Mock()
    tools.set_context = Mock()

    session = _DummySession("telegram:123")
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)

    context_builder = Mock()
    context_builder.render = Mock(return_value=None)

    async def hint_handler(ctx):
        ctx.extra_hints.append("hint from plugin")
        return ctx

    bus.on(BeforeReasoningCtx, hint_handler)

    phase = Phase(
        default_before_reasoning_modules(
            bus,
            cast(ToolRegistry, tools),
            cast(SessionManager, session_mgr),
            cast(ContextBuilder, context_builder),
        ),
        frame_factory=BeforeReasoningFrame,
    )
    msg = _inbound()

    before_turn = BeforeTurnCtx(
        session_key="telegram:123", channel=msg.channel, chat_id=msg.chat_id,
        content=msg.content, timestamp=msg.timestamp,
        retrieved_memory_block="", retrieval_trace_raw=None,
        history_messages=(),
    )

    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)
    state.session = session

    ctx = await phase.run(BeforeReasoningInput(state=state, before_turn=before_turn))
    assert ctx.extra_hints == ["hint from plugin"]


@pytest.mark.asyncio
async def test_before_reasoning_chain_modify_skill_names_used_in_finalize_render():
    bus = EventBus()
    tools = Mock()
    tools.set_context = Mock()

    session = _DummySession("telegram:123")
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)

    context_builder = Mock()
    context_builder.render = Mock(return_value=None)

    async def modify_chain(ctx: BeforeReasoningCtx) -> BeforeReasoningCtx:
        ctx.skill_names.append("chain_added_skill")
        ctx.retrieved_memory_block = "chain_modified_block"
        return ctx

    bus.on(BeforeReasoningCtx, modify_chain)

    phase = Phase(
        default_before_reasoning_modules(
            bus,
            cast(ToolRegistry, tools),
            cast(SessionManager, session_mgr),
            cast(ContextBuilder, context_builder),
        ),
        frame_factory=BeforeReasoningFrame,
    )
    msg = _inbound()

    before_turn = BeforeTurnCtx(
        session_key="telegram:123", channel=msg.channel, chat_id=msg.chat_id,
        content=msg.content, timestamp=msg.timestamp,
        retrieved_memory_block="original_block", retrieval_trace_raw=None,
        history_messages=(),
        skill_names=["base_skill"],
    )

    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)
    state.session = session

    _ = await phase.run(BeforeReasoningInput(state=state, before_turn=before_turn))

    # finalize 必须用 chain 修改后的值 render
    call_args = context_builder.render.call_args[0][0]
    assert "chain_added_skill" in call_args.skill_names
    assert call_args.retrieved_memory_block == "chain_modified_block"


@pytest.mark.asyncio
async def test_before_step_setup_records_token_estimate():
    bus = EventBus()
    phase = Phase(default_before_step_modules(bus), frame_factory=BeforeStepFrame)
    messages = [{"role": "user", "content": "hello"}]

    ctx = await phase.run(
        BeforeStepInput(
            session_key="k",
            channel="c",
            chat_id="ch",
            iteration=1,
            messages=messages,
            visible_names=None,
        )
    )

    assert ctx.input_tokens_estimate > 0


@pytest.mark.asyncio
async def test_before_step_finalize_injects_extra_hints():
    bus = EventBus()

    async def append_hint(ctx: BeforeStepCtx) -> BeforeStepCtx:
        ctx.extra_hints.append("hints from plugin")
        return ctx

    bus.on(BeforeStepCtx, append_hint)
    phase = Phase(default_before_step_modules(bus), frame_factory=BeforeStepFrame)
    messages = [{"role": "user", "content": "hello"}]

    await phase.run(
        BeforeStepInput(
            session_key="k",
            channel="c",
            chat_id="ch",
            iteration=1,
            messages=messages,
            visible_names=None,
        )
    )

    expected = build_context_hint_message("plugin_hints", "hints from plugin")
    assert messages == [{"role": "user", "content": "hello"}, expected]


@pytest.mark.asyncio
async def test_before_step_finalize_early_stop():
    bus = EventBus()

    async def stop_early(ctx: BeforeStepCtx) -> BeforeStepCtx:
        ctx.early_stop = True
        ctx.early_stop_reply = "预算不足"
        return ctx

    bus.on(BeforeStepCtx, stop_early)
    phase = Phase(default_before_step_modules(bus), frame_factory=BeforeStepFrame)
    messages = [{"role": "user", "content": "hello"}]

    ctx = await phase.run(
        BeforeStepInput(
            session_key="k",
            channel="c",
            chat_id="ch",
            iteration=1,
            messages=messages,
            visible_names=None,
        )
    )

    assert ctx.early_stop is True
    assert ctx.early_stop_reply == "预算不足"


@pytest.mark.asyncio
async def test_after_step_phase_runs_observers():
    bus = EventBus()
    side_effect: list[str] = []

    async def handler(ctx: AfterStepCtx) -> None:
        side_effect.append(ctx.partial_reply)

    bus.on(AfterStepCtx, handler)
    phase = Phase(default_after_step_modules(bus), frame_factory=AfterStepFrame)
    await phase.run(
        AfterStepCtx(
            session_key="k",
            channel="c",
            chat_id="ch",
            iteration=0,
            tools_called=(),
            partial_reply="ok",
            tools_used_so_far=(),
            tool_chain_partial=(),
            partial_thinking=None,
            has_more=True,
        )
    )

    assert side_effect == ["ok"]
