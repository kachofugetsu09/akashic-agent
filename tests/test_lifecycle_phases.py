from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from agent.context import ContextBuilder
from agent.core.passive_support import build_context_hint_message
from agent.core.passive_turn import ContextStore
from agent.core.types import ContextBundle
from agent.lifecycle.phase import Phase
from agent.tools.registry import ToolRegistry
from core.observe.db import open_db as open_observe_db
from bus.event_bus import EventBus
from bus.events import InboundMessage
from agent.lifecycle.types import (
    AfterStepCtx,
    BeforeReasoningCtx,
    BeforeReasoningInput,
    BeforeStepCtx,
    BeforeStepInput,
    BeforeTurnCtx,
    PromptRenderCtx,
    PromptRenderInput,
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
from agent.lifecycle.phases.prompt_render import (
    PromptRenderFrame,
    default_prompt_render_modules,
)
from agent.prompting import PromptSectionRender
from session.manager import SessionManager

_now = datetime.now()


class _MemoryStatusPluginModule:
    requires = ("session:session",)
    produces = ("session:ctx",)

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        if "session:ctx" in frame.slots:
            return frame
        state = frame.input
        if state.msg.content != "/memory_status":
            return frame
        session = state.session
        if session is None:
            return frame
        messages = list(getattr(session, "messages", []))
        last = max(0, int(getattr(session, "last_consolidated", 0)))
        last = min(last, len(messages))
        frame.slots["session:ctx"] = BeforeTurnCtx(
            session_key=state.session_key,
            channel=state.msg.channel,
            chat_id=state.msg.chat_id,
            content=state.msg.content,
            timestamp=state.msg.timestamp,
            skill_names=[],
            retrieved_memory_block="",
            retrieval_trace_raw=None,
            history_messages=(),
            abort=True,
            abort_reply=_format_memory_status_reply(messages, last),
        )
        return frame


class _KVCachePluginModule:
    requires = ("session:session",)
    produces = ("session:ctx",)

    def __init__(self, observe_db_path) -> None:
        self._observe_db_path = observe_db_path

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        if "session:ctx" in frame.slots:
            return frame
        state = frame.input
        if state.msg.content != "/kvcache":
            return frame
        frame.slots["session:ctx"] = BeforeTurnCtx(
            session_key=state.session_key,
            channel=state.msg.channel,
            chat_id=state.msg.chat_id,
            content=state.msg.content,
            timestamp=state.msg.timestamp,
            skill_names=[],
            retrieved_memory_block="",
            retrieval_trace_raw=None,
            history_messages=(),
            abort=True,
            abort_reply=_build_kvcache_reply(state, self._observe_db_path),
        )
        return frame


def _format_memory_status_reply(messages: list[dict[str, object]], last_consolidated: int) -> str:
    consolidated_user = _count_real_user_messages(messages[:last_consolidated])
    total_user = _count_real_user_messages(messages)
    pending_user = max(0, total_user - consolidated_user)
    last_user_message = _latest_real_user_content(messages[:last_consolidated])

    lines = ["记忆整理状态："]
    if last_consolidated <= 0 or not last_user_message:
        lines.append("当前会话还没有完成过记忆整理。")
    elif pending_user == 0:
        lines.append("当前会话已经整理到最新的用户消息。")
    else:
        lines.append(f"上次整理到 {pending_user} 条用户消息之前。")
    if last_user_message:
        lines.extend(["", "最后已整理的用户消息：", f"“{_preview_text(last_user_message)}”"])
    lines.extend(
        [
            "",
            f"尚未整理的用户消息数：{pending_user}",
            f"当前会话消息数：{len(messages)}",
        ]
    )
    return "\n".join(lines)


def _build_kvcache_reply(state: TurnState, observe_db_path) -> str:
    if not observe_db_path or not observe_db_path.exists():
        return "暂无 KVCache 数据（observe 数据库不存在）。"
    conn = open_observe_db(observe_db_path)
    try:
        rows = conn.execute(
            """SELECT llm_output, ts, react_cache_prompt_tokens, react_cache_hit_tokens
               FROM turns WHERE session_key=? AND source='agent'
               ORDER BY id DESC LIMIT ?""",
            [state.session_key, 5],
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return "暂无 KVCache 数据。"
    overall_prompt = sum(r[2] or 0 for r in rows)
    overall_hit = sum(r[3] or 0 for r in rows)
    overall_pct = (overall_hit / overall_prompt * 100) if overall_prompt > 0 else 0.0
    lines = [f"最近 {len(rows)} 轮 KVCache 状态（总命中率 {overall_pct:.2f}%）", ""]
    for llm_output, ts, prompt_tokens, hit_tokens in rows:
        content = str(llm_output or "").strip()
        preview = _preview_text(content.replace("\n", " "), limit=80)
        hit = hit_tokens or 0
        prompt = prompt_tokens or 0
        pct = (hit / prompt * 100) if prompt > 0 else 0.0
        lines.append(preview or "（无内容）")
        lines.append(_format_ts(str(ts)))
        lines.append(f"{hit:,} / {prompt:,}")
        lines.append(f"{pct:.2f}%")
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def _count_real_user_messages(messages: list[dict[str, object]]) -> int:
    return sum(1 for item in messages if _is_real_user_message(item))


def _latest_real_user_content(messages: list[dict[str, object]]) -> str:
    for item in reversed(messages):
        if _is_real_user_message(item):
            return str(item.get("content", "")).strip()
    return ""


def _is_real_user_message(item: dict[str, object]) -> bool:
    content = str(item.get("content", "")).strip()
    return item.get("role") == "user" and bool(content) and "data-system-context-frame" not in content


def _preview_text(text: str, limit: int = 80) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def _format_ts(ts: str) -> str:
    match = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return f"{match.month}-{match.day} {match:%H:%M}"


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
            plugin_modules_early=[],
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
            plugin_modules_early=[_MemoryStatusPluginModule()],
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
            plugin_modules_early=[_MemoryStatusPluginModule()],
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
async def test_before_turn_accepts_custom_command_module():
    bus = EventBus()
    session = _DummySession("telegram:123")
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    ctx_store = SimpleNamespace(prepare=AsyncMock())

    class CustomCommandModule:
        requires = ("session:session",)
        produces = ("session:ctx",)

        async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
            state = frame.input
            if state.msg.content != "/debug":
                return frame
            frame.slots["session:ctx"] = BeforeTurnCtx(
                session_key=state.session_key,
                channel=state.msg.channel,
                chat_id=state.msg.chat_id,
                content=state.msg.content,
                timestamp=state.msg.timestamp,
                skill_names=[],
                retrieved_memory_block="",
                retrieval_trace_raw=None,
                history_messages=(),
                abort=True,
                abort_reply="debug ok",
            )
            return frame

    phase = Phase(
        default_before_turn_modules(
            bus,
            cast(SessionManager, session_mgr),
            cast(ContextStore, ctx_store),
            plugin_modules_early=[_MemoryStatusPluginModule(), CustomCommandModule()],
        ),
        frame_factory=BeforeTurnFrame,
    )
    msg = InboundMessage(
        channel="telegram",
        sender="user",
        chat_id="123",
        content="/debug",
        timestamp=_now,
    )
    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)

    ctx = await phase.run(state)

    assert ctx.abort is True
    assert ctx.abort_reply == "debug ok"
    ctx_store.prepare.assert_not_called()


@pytest.mark.asyncio
async def test_before_turn_accepts_plugin_modules_early_and_late():
    bus = EventBus()
    session = _DummySession("telegram:123")
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    bundle = ContextBundle(
        skill_mentions=["memo"],
        retrieved_memory_block="retrieved block",
        retrieval_trace_raw={"trace": 1},
        history_messages=[{"role": "user", "content": "prev"}],
    )
    ctx_store = SimpleNamespace(prepare=AsyncMock(return_value=bundle))
    seen: list[str] = []

    class EarlyPluginModule:
        requires = ("session:session",)

        async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
            seen.append("early")
            frame.input.msg.metadata["early_seen"] = True
            return frame

    class LatePluginModule:
        requires = ("session:ctx",)
        produces = ("session:ctx",)

        async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
            seen.append("late")
            ctx = cast(BeforeTurnCtx, frame.slots["session:ctx"])
            ctx.extra_metadata["late_seen"] = ctx.retrieved_memory_block
            frame.slots["session:ctx"] = ctx
            return frame

    phase = Phase(
        default_before_turn_modules(
            bus,
            cast(SessionManager, session_mgr),
            cast(ContextStore, ctx_store),
            plugin_modules_early=[EarlyPluginModule()],
            plugin_modules_late=[LatePluginModule()],
        ),
        frame_factory=BeforeTurnFrame,
    )
    msg = _inbound()
    msg.metadata["seed"] = "x"
    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)

    ctx = await phase.run(state)

    assert seen == ["early", "late"]
    assert state.msg.metadata["early_seen"] is True
    assert ctx.extra_metadata["late_seen"] == "retrieved block"
    ctx_store.prepare.assert_called_once()


@pytest.mark.asyncio
async def test_before_turn_kvcache_command(tmp_path):
    bus = EventBus()
    session = _DummySession("telegram:123")
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    ctx_store = SimpleNamespace(prepare=AsyncMock())

    db_path = tmp_path / "observe" / "observe.db"
    conn = open_observe_db(db_path)
    conn.execute(
        """INSERT INTO turns (source, session_key, user_msg, llm_output, ts,
           react_cache_prompt_tokens, react_cache_hit_tokens)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            "agent",
            "telegram:123",
            "之前的问题",
            "这是之前的回答",
            "2026-04-29T16:14:00.123456+00:00",
            52564,
            50560,
        ],
    )
    conn.execute(
        """INSERT INTO turns (source, session_key, user_msg, llm_output, ts,
           react_cache_prompt_tokens, react_cache_hit_tokens)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            "agent",
            "telegram:123",
            "新的问题",
            "这是新的回答\n有多行",
            "2026-04-29T16:15:00+00:00",
            50000,
            40000,
        ],
    )
    conn.commit()
    conn.close()

    phase = Phase(
        default_before_turn_modules(
            bus,
            cast(SessionManager, session_mgr),
            cast(ContextStore, ctx_store),
            plugin_modules_early=[_KVCachePluginModule(db_path)],
        ),
        frame_factory=BeforeTurnFrame,
    )
    msg = InboundMessage(
        channel="telegram",
        sender="user",
        chat_id="123",
        content="/kvcache",
        timestamp=_now,
    )
    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)

    ctx = await phase.run(state)

    assert ctx.abort is True
    assert "最近 2 轮 KVCache 状态" in ctx.abort_reply
    assert "总命中率" in ctx.abort_reply
    assert "这是之前的回答" in ctx.abort_reply
    assert "这是新的回答" in ctx.abort_reply
    assert "4-29 16:14" in ctx.abort_reply
    assert "4-29 16:15" in ctx.abort_reply
    assert "50,560 / 52,564" in ctx.abort_reply
    assert "96.19%" in ctx.abort_reply
    assert "80.00%" in ctx.abort_reply
    assert ctx.abort_reply.count("\n\n") <= 2
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
async def test_prompt_render_chain_appends_bottom_section(tmp_path):
    bus = EventBus()

    async def append_section(ctx: PromptRenderCtx) -> PromptRenderCtx:
        ctx.system_sections_bottom.append(
            PromptSectionRender(
                name="plugin_protocol",
                content="# Plugin Protocol\n\n稳定协议",
                is_static=False,
            )
        )
        return ctx

    bus.on(PromptRenderCtx, append_section)
    memory = SimpleNamespace(
        read_self=lambda: "",
        read_profile=lambda: "",
        read_recent_context=lambda: "",
    )
    context = ContextBuilder(tmp_path, memory=cast(Any, memory))
    phase = Phase(
        default_prompt_render_modules(bus, context),
        frame_factory=PromptRenderFrame,
    )

    result = await phase.run(
        PromptRenderInput(
            session_key="k",
            channel="cli",
            chat_id="ch",
            content="hello",
            media=None,
            timestamp=_now,
            history=[],
            skill_names=None,
            retrieved_memory_block="",
            disabled_sections=set(),
            turn_injection_prompt="",
        )
    )

    assert "Plugin Protocol" in str(result.messages[0]["content"])


@pytest.mark.asyncio
async def test_prompt_render_chain_respects_disabled_sections(tmp_path):
    class BottomModule:
        async def run(self, frame: PromptRenderFrame) -> PromptRenderFrame:
            ctx = cast(PromptRenderCtx, frame.slots["prompt:ctx"])
            ctx.system_sections_bottom.append(
                PromptSectionRender(
                    name="memes",
                    content="# Memes\n\n<meme:happy>",
                    is_static=False,
                )
            )
            return frame

    memory = SimpleNamespace(
        read_self=lambda: "",
        read_profile=lambda: "",
        read_recent_context=lambda: "",
    )
    context = ContextBuilder(tmp_path, memory=cast(Any, memory))
    phase = Phase(
        default_prompt_render_modules(
            EventBus(),
            context,
            plugin_modules_bottom=[BottomModule()],
        ),
        frame_factory=PromptRenderFrame,
    )

    result = await phase.run(
        PromptRenderInput(
            session_key="k",
            channel="cli",
            chat_id="ch",
            content="hello",
            media=None,
            timestamp=_now,
            history=[],
            skill_names=None,
            retrieved_memory_block="",
            disabled_sections={"memes"},
            turn_injection_prompt="",
        )
    )

    assert "<meme:happy>" not in str(result.messages[0]["content"])


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
