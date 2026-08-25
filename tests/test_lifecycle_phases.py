from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from agent.context import ContextBuilder
from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentRef,
    ChannelDeliveryReceipt,
    DeliveryStatus as ChannelDeliveryStatus,
)
from agent.control.context import running_turn_id
from agent.core.passive_support import build_context_hint_message
from agent.core.passive_turn import (
    ContextStore,
    PassiveTurnDeps,
    PassiveTurnPipeline,
    Reasoner,
)
from agent.core.response_parser import ResponseMetadata
from agent.core.runtime_support import TurnRunResult
from agent.control.ports import TurnUserInput
from agent.core.types import ContextBundle
from agent.lifecycle.phase import Phase
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from bus.events import (
    InboundMessage,
    OutboundMessage,
)
from bus.events_lifecycle import TurnCommitted
from core.error_context import current_client_message_id, current_session_key
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from bootstrap.channel_attachment_import import ChannelOutboundAttachmentImporter
from agent.lifecycle.types import (
    AfterReasoningCtx,
    AfterReasoningInput,
    AfterStepCtx,
    AfterTurnCtx,
    BeforeReasoningCtx,
    BeforeReasoningInput,
    BeforeStepCtx,
    BeforeStepInput,
    BeforeTurnCtx,
    PromptRenderCtx,
    PromptRenderInput,
    TurnSnapshot,
    TurnState,
)
from agent.lifecycle.phases.after_reasoning import (
    AfterReasoningFrame,
    _collect_persist_assistant_metadata,
    _collect_persist_user_metadata,
    default_after_reasoning_modules,
)
from agent.lifecycle.phases.after_step import (
    AfterStepFrame,
    default_after_step_modules,
)
from agent.lifecycle.phases.after_turn import (
    AfterTurnFrame,
    default_after_turn_modules,
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
from agent.persona import reset_veda
from agent.turns.outbound import OutboundDispatch, OutboundPort
from session.manager import SessionManager, logical_history_unit_ranges

_now = datetime.now()


def open_observe_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            source TEXT NOT NULL,
            session_key TEXT NOT NULL,
            user_msg TEXT,
            llm_output TEXT NOT NULL DEFAULT '',
            raw_llm_output TEXT,
            meme_tag TEXT,
            meme_media_count INTEGER,
            tool_calls TEXT,
            tool_chain_json TEXT,
            history_window INTEGER,
            history_messages INTEGER,
            history_chars INTEGER,
            history_tokens INTEGER,
            prompt_tokens INTEGER,
            next_turn_baseline_tokens INTEGER,
            error TEXT,
            react_iteration_count INTEGER,
            react_input_sum_tokens INTEGER,
            react_input_peak_tokens INTEGER,
            react_final_input_tokens INTEGER,
            react_cache_prompt_tokens INTEGER,
            react_cache_hit_tokens INTEGER
        )
        """)
    return conn


class _MemoryStatusPluginModule:
    slot = "test.memory_status"
    requires = ("before_turn.acquire_session", "session:session")
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
            history_messages=(),
            abort=True,
            abort_reply=_format_memory_status_reply(messages, last),
        )
        return frame


class _DummyOutbound:
    async def dispatch(self, outbound: OutboundDispatch) -> ChannelDeliveryReceipt:
        return ChannelDeliveryReceipt(
            delivery_id="test-delivery",
            status=ChannelDeliveryStatus.DELIVERED,
        )


class _KVCachePluginModule:
    slot = "test.kvcache"
    requires = ("before_turn.acquire_session", "session:session")
    produces = ("session:ctx",)

    def __init__(self, db_path) -> None:
        self._db_path = db_path

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
            history_messages=(),
            abort=True,
            abort_reply=_build_kvcache_reply(state, self._db_path),
        )
        return frame


def _format_memory_status_reply(
    messages: list[dict[str, object]], last_consolidated: int
) -> str:
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
        lines.extend(
            ["", "最后已整理的用户消息：", f"“{_preview_text(last_user_message)}”"]
        )
    lines.extend(
        [
            "",
            f"尚未整理的用户消息数：{pending_user}",
            f"当前会话消息数：{len(messages)}",
        ]
    )
    return "\n".join(lines)


def _build_kvcache_reply(state: TurnState, db_path) -> str:
    if not db_path or not db_path.exists():
        return "暂无 KVCache 数据（observe 数据库不存在）。"
    conn = open_observe_db(db_path)
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
    return (
        item.get("role") == "user"
        and bool(content)
        and "data-system-context-frame" not in content
    )


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
        channel="telegram",
        sender="user",
        chat_id="123",
        content="hello",
        timestamp=_now,
    )


class _DummySession:
    def __init__(self, key: str) -> None:
        self.key = key
        self.messages: list[dict[str, object]] = []
        self.metadata: dict[str, object] = {}
        self.last_consolidated = 0

    def get_history(self, max_messages: int = 500) -> list[dict[str, object]]:
        return list(self.messages)

    def history_units(self, *, after_seq: int = -1) -> tuple[SimpleNamespace, ...]:
        return (SimpleNamespace(messages=tuple(self.messages)),)

    def add_message(
        self, role: str, content: str, media=None, **kwargs: object
    ) -> dict[str, object]:
        msg: dict[str, object] = {"role": role, "content": content}
        if media:
            msg["media"] = list(media)
        msg.update(kwargs)
        self.messages.append(msg)
        return msg


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
    assert ctx.skill_names == ["search"]
    assert ctx.channel == "telegram"
    assert ctx.chat_id == "123"
    assert ctx.history_messages == ({"role": "user", "content": "prev"},)
    assert ctx.abort is False


@pytest.mark.asyncio
async def test_before_turn_existing_admission_never_creates_deleted_session():
    bus = EventBus()
    session = _DummySession("mobile:deleted")
    get_existing = Mock(return_value=session)
    get_or_create = Mock(side_effect=AssertionError("不得重建已删除会话"))
    session_mgr = SimpleNamespace(
        get_existing=get_existing, get_or_create=get_or_create
    )
    ctx_store = SimpleNamespace(prepare=AsyncMock(return_value=ContextBundle()))
    phase = Phase(
        default_before_turn_modules(
            bus,
            cast(SessionManager, session_mgr),
            cast(ContextStore, ctx_store),
        ),
        frame_factory=BeforeTurnFrame,
    )
    msg = _inbound()
    msg.metadata = {"require_existing_session": True}
    state = TurnState(msg=msg, session_key="mobile:deleted", dispatch_outbound=True)

    await phase.run(state)

    get_existing.assert_called_once_with("mobile:deleted")
    get_or_create.assert_not_called()
    assert "require_existing_session" not in msg.metadata


@pytest.mark.asyncio
async def test_before_turn_uses_cli_session_override_context():
    bus = EventBus()
    session = _DummySession("telegram:7674283004")
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    ctx_store = SimpleNamespace(prepare=AsyncMock(return_value=ContextBundle()))
    phase = Phase(
        default_before_turn_modules(
            bus,
            cast(SessionManager, session_mgr),
            cast(ContextStore, ctx_store),
        ),
        frame_factory=BeforeTurnFrame,
    )
    msg = InboundMessage(
        channel="cli",
        sender="user",
        chat_id="cli-1",
        content="hello",
        timestamp=_now,
        metadata={
            "session_key_override": "telegram:7674283004",
            "context_channel": "telegram",
            "context_chat_id": "7674283004",
        },
    )
    state = TurnState(msg=msg, session_key=msg.session_key, dispatch_outbound=True)

    ctx = await phase.run(state)

    assert state.session is session
    assert ctx.session_key == "telegram:7674283004"
    assert ctx.channel == "telegram"
    assert ctx.chat_id == "7674283004"


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
            plugin_modules=[_MemoryStatusPluginModule()],
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
            plugin_modules=[_MemoryStatusPluginModule()],
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
async def test_before_turn_context_prepare_counts_multi_input_turn_once():
    bus = EventBus()
    session = _DummySession("telegram:123")
    session.messages = [
        {
            "role": "user" if index < 29 else "assistant",
            "content": f"message-{index}",
            "control_turn_id": "turn-one",
        }
        for index in range(30)
    ]
    session.last_consolidated = 0
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    ctx_store = SimpleNamespace(
        prepare=AsyncMock(return_value=ContextBundle(history_messages=[]))
    )
    phase = Phase(
        default_before_turn_modules(
            bus,
            cast(SessionManager, session_mgr),
            cast(ContextStore, ctx_store),
        ),
        frame_factory=BeforeTurnFrame,
    )

    ctx = await phase.run(
        TurnState(
            msg=_inbound(),
            session_key="telegram:123",
            dispatch_outbound=True,
        )
    )

    assert ctx.abort is False
    ctx_store.prepare.assert_awaited_once()


@pytest.mark.asyncio
async def test_before_turn_preserves_generic_turn_effect_metadata():
    bus = EventBus()
    session = _DummySession("telegram:123")
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    ctx_store = SimpleNamespace(
        prepare=AsyncMock(return_value=ContextBundle(history_messages=[]))
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
    msg.metadata["effects"] = {"post_commit": "suppress"}
    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)

    await phase.run(state)

    assert msg.metadata["effects"] == {"post_commit": "suppress"}


@pytest.mark.asyncio
async def test_before_turn_accepts_custom_command_module():
    bus = EventBus()
    session = _DummySession("telegram:123")
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    ctx_store = SimpleNamespace(prepare=AsyncMock())

    class CustomCommandModule:
        slot = "test.custom_command"
        requires = ("before_turn.acquire_session", "session:session")
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
            plugin_modules=[_MemoryStatusPluginModule(), CustomCommandModule()],
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
async def test_before_turn_accepts_plugin_modules():
    bus = EventBus()
    session = _DummySession("telegram:123")
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    bundle = ContextBundle(
        skill_mentions=["memo"],
        history_messages=[{"role": "user", "content": "prev"}],
    )
    ctx_store = SimpleNamespace(prepare=AsyncMock(return_value=bundle))
    seen: list[str] = []

    class EarlyPluginModule:
        slot = "test.before_turn.early"
        requires = ("before_turn.acquire_session", "session:session")

        async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
            seen.append("early")
            frame.input.msg.metadata["early_seen"] = True
            return frame

    class LatePluginModule:
        slot = "test.before_turn.late"
        requires = ("before_turn.emit", "session:ctx")
        produces = ("session:ctx",)

        async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
            seen.append("late")
            ctx = cast(BeforeTurnCtx, frame.slots["session:ctx"])
            ctx.extra_metadata["late_seen"] = ",".join(ctx.skill_names)
            frame.slots["session:ctx"] = ctx
            frame.slots["session:extra_hint:late"] = "hint from before turn"
            return frame

    phase = Phase(
        default_before_turn_modules(
            bus,
            cast(SessionManager, session_mgr),
            cast(ContextStore, ctx_store),
            plugin_modules=[EarlyPluginModule(), LatePluginModule()],
        ),
        frame_factory=BeforeTurnFrame,
    )
    msg = _inbound()
    msg.metadata["seed"] = "x"
    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)

    ctx = await phase.run(state)

    assert seen == ["early", "late"]
    assert state.msg.metadata["early_seen"] is True
    assert ctx.extra_metadata["late_seen"] == "memo"
    assert ctx.extra_hints == ["hint from before turn"]
    ctx_store.prepare.assert_called_once()


@pytest.mark.asyncio
async def test_before_turn_projects_durable_execution_turn_id():
    bus = EventBus()
    session = _DummySession("programmatic:ctx")
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    ctx_store = SimpleNamespace(
        prepare=AsyncMock(
            return_value=ContextBundle(
                skill_mentions=[],
                history_messages=[],
            )
        )
    )
    phase = Phase(
        default_before_turn_modules(
            bus,
            cast(SessionManager, session_mgr),
            cast(ContextStore, ctx_store),
        ),
        frame_factory=BeforeTurnFrame,
    )
    msg = InboundMessage(
        channel="programmatic",
        sender="owner",
        chat_id="ctx",
        content="hello",
        timestamp=_now,
        metadata={"_control_execution_turn_id": "turn:durable"},
    )

    ctx = await phase.run(
        TurnState(
            msg=msg,
            session_key="programmatic:ctx",
            dispatch_outbound=False,
        )
    )

    assert ctx.turn_id == "turn:durable"


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
            plugin_modules=[_KVCachePluginModule(db_path)],
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
        session_key="telegram:123",
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=msg.content,
        timestamp=msg.timestamp,
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
        session_key="telegram:123",
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=msg.content,
        timestamp=msg.timestamp,
        history_messages=(),
    )

    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)
    # session is None

    with pytest.raises(
        RuntimeError, match="BeforeReasoning requires TurnState.session"
    ):
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
        session_key="telegram:123",
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=msg.content,
        timestamp=msg.timestamp,
        history_messages=(),
        skill_names=["search"],
    )

    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)
    state.session = session

    ctx = await phase.run(BeforeReasoningInput(state=state, before_turn=before_turn))

    context_builder.render.assert_called_once()
    call_args = context_builder.render.call_args[0][0]
    assert call_args.skill_names == ["search"]
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
        session_key="telegram:123",
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=msg.content,
        timestamp=msg.timestamp,
        history_messages=(),
        extra_hints=["hint from before turn"],
    )

    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)
    state.session = session

    ctx = await phase.run(BeforeReasoningInput(state=state, before_turn=before_turn))
    assert ctx.extra_hints == ["hint from before turn", "hint from plugin"]


@pytest.mark.asyncio
async def test_before_reasoning_collects_export_slots():
    bus = EventBus()
    tools = Mock()
    tools.set_context = Mock()
    session = _DummySession("telegram:123")
    session_mgr = SimpleNamespace(get_or_create=lambda key: session)
    context_builder = Mock()
    context_builder.render = Mock(return_value=None)

    class SlotModule:
        slot = "test.before_reasoning.slot"
        requires = ("before_reasoning.emit", "reasoning:ctx")

        async def run(self, frame: BeforeReasoningFrame) -> BeforeReasoningFrame:
            frame.slots["reasoning:extra_hint:test"] = "slot hint"
            frame.slots["reasoning:abort_reply"] = "slot abort"
            return frame

    phase = Phase(
        default_before_reasoning_modules(
            bus,
            cast(ToolRegistry, tools),
            cast(SessionManager, session_mgr),
            cast(ContextBuilder, context_builder),
            plugin_modules=[SlotModule()],
        ),
        frame_factory=BeforeReasoningFrame,
    )
    msg = _inbound()
    before_turn = BeforeTurnCtx(
        session_key="telegram:123",
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=msg.content,
        timestamp=msg.timestamp,
        history_messages=(),
    )
    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)
    state.session = session

    ctx = await phase.run(BeforeReasoningInput(state=state, before_turn=before_turn))

    assert ctx.extra_hints == ["slot hint"]
    assert ctx.abort is True
    assert ctx.abort_reply == "slot abort"
    context_builder.render.assert_not_called()


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
        session_key="telegram:123",
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=msg.content,
        timestamp=msg.timestamp,
        history_messages=(),
        skill_names=["base_skill"],
    )

    state = TurnState(msg=msg, session_key="telegram:123", dispatch_outbound=True)
    state.session = session

    _ = await phase.run(BeforeReasoningInput(state=state, before_turn=before_turn))

    # finalize 必须用 chain 修改后的值 render
    call_args = context_builder.render.call_args[0][0]
    assert "chain_added_skill" in call_args.skill_names


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
    _ = reset_veda(tmp_path)

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
        get_memory_context=lambda: "",
    )
    context = ContextBuilder(tmp_path, cast(Any, memory))
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
            disabled_sections=set(),
            turn_injection_prompt="",
        )
    )

    assert "Plugin Protocol" in str(result.messages[0]["content"])


@pytest.mark.asyncio
async def test_prompt_render_chain_respects_disabled_sections(tmp_path):
    _ = reset_veda(tmp_path)

    class BottomModule:
        slot = "test.prompt.bottom"
        requires = ("prompt_render.emit", "prompt:ctx")
        produces = ("prompt:ctx",)

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
        get_memory_context=lambda: "",
    )
    context = ContextBuilder(tmp_path, cast(Any, memory))
    phase = Phase(
        default_prompt_render_modules(
            EventBus(),
            context,
            plugin_modules=[BottomModule()],
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
            disabled_sections={"memes"},
            turn_injection_prompt="",
        )
    )

    assert "<meme:happy>" not in str(result.messages[0]["content"])


@pytest.mark.asyncio
async def test_prompt_render_collects_export_slots(tmp_path):
    _ = reset_veda(tmp_path)

    class SlotModule:
        slot = "test.prompt.slot"
        requires = ("prompt_render.emit", "prompt:ctx")

        async def run(self, frame: PromptRenderFrame) -> PromptRenderFrame:
            frame.slots["prompt:section_top:top_slot"] = "top content"
            frame.slots["prompt:section_bottom:bottom_slot"] = PromptSectionRender(
                name="bottom_slot",
                content="bottom content",
                is_static=False,
            )
            frame.slots["prompt:extra_hint:test"] = "hint content"
            return frame

    memory = SimpleNamespace(
        read_self=lambda: "",
        read_profile=lambda: "",
        get_memory_context=lambda: "",
    )
    context = ContextBuilder(tmp_path, cast(Any, memory))
    phase = Phase(
        default_prompt_render_modules(
            EventBus(),
            context,
            plugin_modules=[SlotModule()],
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
            disabled_sections=set(),
            turn_injection_prompt="",
        )
    )
    rendered = str(result.messages)

    assert "top content" in rendered
    assert "bottom content" in rendered
    assert "hint content" in rendered


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
async def test_before_step_collects_export_slots():
    class SlotModule:
        slot = "test.before_step.slot"
        requires = ("before_step.emit", "step:ctx")

        async def run(self, frame: BeforeStepFrame) -> BeforeStepFrame:
            frame.slots["step:extra_hint:test"] = "slot step hint"
            frame.slots["step:abort_reply"] = "slot stop"
            return frame

    phase = Phase(
        default_before_step_modules(
            EventBus(),
            plugin_modules=[SlotModule()],
        ),
        frame_factory=BeforeStepFrame,
    )
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

    assert ctx.extra_hints == ["slot step hint"]
    assert ctx.early_stop is True
    assert ctx.early_stop_reply == "slot stop"


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
            context_tokens_estimate=0,
            tools_called=(),
            partial_reply="ok",
            tools_used_so_far=(),
            tool_chain_partial=(),
            partial_thinking=None,
            has_more=True,
        )
    )

    assert side_effect == ["ok"]


@pytest.mark.asyncio
async def test_after_step_collects_telemetry_slots_before_fanout():
    bus = EventBus()
    seen: list[dict[str, Any]] = []

    class SlotModule:
        slot = "test.after_step.pre"
        requires = ("after_step.copy_input", "step:ctx")

        async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
            frame.slots["step:telemetry:test"] = {"ok": True}
            return frame

    class AfterFanoutSlotModule:
        slot = "test.after_step.post"
        requires = ("after_step.fanout", "step:ctx")

        async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
            frame.slots["step:telemetry:after"] = "done"
            frame.slots["step:telemetry:test"] = "overwritten"
            return frame

    async def handler(ctx: AfterStepCtx) -> None:
        seen.append(dict(ctx.extra_metadata))

    bus.on(AfterStepCtx, handler)
    phase = Phase(
        default_after_step_modules(
            bus,
            plugin_modules=[SlotModule(), AfterFanoutSlotModule()],
        ),
        frame_factory=AfterStepFrame,
    )
    ctx = await phase.run(
        AfterStepCtx(
            session_key="k",
            channel="c",
            chat_id="ch",
            iteration=0,
            context_tokens_estimate=0,
            tools_called=(),
            partial_reply="ok",
            tools_used_so_far=(),
            tool_chain_partial=(),
            partial_thinking=None,
            has_more=True,
        )
    )

    assert seen == [{"test": {"ok": True}}]
    assert ctx.extra_metadata == {"test": {"ok": True}, "after": "done"}


@pytest.mark.asyncio
async def test_after_reasoning_collects_v3_metadata_and_outbound_slots():
    class SlotModule:
        slot = "test.after_reasoning.slot"
        requires = ("after_reasoning.emit", "reasoning:ctx")

        async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
            ctx = cast(AfterReasoningCtx, frame.slots["reasoning:ctx"])
            ctx.persist_user_metadata["akasha_reinforce"] = {
                "target_message_ids": ["message-1"]
            }
            ctx.persist_assistant_metadata["citation_ids"] = ["mem_1"]
            frame.slots["outbound:metadata:plugin_flag"] = "m"
            frame.slots["outbound:media:image"] = ["/tmp/a.png", None, 1]
            return frame

    session = _DummySession("telegram:123")
    msg = _inbound()
    msg.metadata["client_message_id"] = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    state = TurnState(msg=msg, session_key=session.key, dispatch_outbound=True)
    state.session = session
    state.extra_metadata["before_turn_flag"] = "bt"

    async def append_messages(
        current: _DummySession,
        messages: list[dict[str, object]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for index, persisted in enumerate(messages):
            persisted["id"] = f"{current.key}:{index}"

    class Importer:
        async def import_media(
            self,
            media: tuple[str, ...],
        ) -> tuple[AttachmentRef, ...]:
            return tuple(
                AttachmentRef(
                    artifact_id=f"artifact-{index}",
                    kind=AttachmentKind.IMAGE,
                    filename=Path(source).name,
                    media_type="image/png",
                    size_bytes=index + 1,
                    sha256=f"{index + 1:064x}",
                )
                for index, source in enumerate(media)
            )

    services = SimpleNamespace(
        presence=Mock(),
        session_manager=SimpleNamespace(append_messages=append_messages),
        outbound_attachment_importer=Importer(),
    )
    turn_result = TurnRunResult(
        reply="reply",
        tool_chain=[],
        tools_used=[],
        media=["/tmp/from-turn.png"],
        thinking=None,
        streamed=False,
        context_retry={},
    )
    phase = Phase(
        default_after_reasoning_modules(
            EventBus(),
            cast(Any, services),
            plugin_modules=[SlotModule()],
        ),
        frame_factory=AfterReasoningFrame,
    )

    result = await phase.run(AfterReasoningInput(state=state, turn_result=turn_result))

    assert session.messages[0]["akasha_reinforce"] == {
        "target_message_ids": ["message-1"]
    }
    assert session.messages[0]["client_message_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert session.messages[1]["citation_ids"] == ["mem_1"]
    assert session.messages[1]["attachment_ids"] == ["artifact-0", "artifact-1"]
    assert result.outbound.metadata["before_turn_flag"] == "bt"
    assert result.outbound.metadata["plugin_flag"] == "m"
    assert [ref.artifact_id for ref in result.outbound.attachment_refs] == [
        "artifact-0",
        "artifact-1",
    ]
    assert result.outbound.media == []
    assert result.outbound.session_message_id == "telegram:123:1"


@pytest.mark.asyncio
async def test_after_reasoning_commits_outbound_attachment_binding_atomically(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    manager = SessionManager(workspace)
    artifact_store = ChannelAttachmentArtifactStore(
        workspace=workspace,
        session_store=manager.control_store,
    )
    source = tmp_path / "generated.png"
    source.write_bytes(b"generated image bytes")
    session = manager.get_or_create("telegram:123")
    state = TurnState(
        msg=_inbound(),
        session_key=session.key,
        dispatch_outbound=True,
    )
    state.session = session
    services = SimpleNamespace(
        presence=None,
        session_manager=manager,
        outbound_attachment_importer=ChannelOutboundAttachmentImporter(artifact_store),
    )
    phase = Phase(
        default_after_reasoning_modules(EventBus(), cast(Any, services)),
        frame_factory=AfterReasoningFrame,
    )

    result = await phase.run(
        AfterReasoningInput(
            state=state,
            turn_result=TurnRunResult(
                reply="reply",
                media=[str(source)],
            ),
        )
    )

    assistant = session.messages[-1]
    attachment_ids = assistant.get("attachment_ids")
    assert isinstance(attachment_ids, list) and len(attachment_ids) == 1
    assert manager.control_store.message_attachment_ids(
        cast(str, assistant["id"])
    ) == tuple(attachment_ids)
    assert result.outbound.attachment_refs[0].artifact_id == attachment_ids[0]
    assert result.outbound.media == []
    manager.close()


def _assistant_metadata_ctx() -> AfterReasoningCtx:
    return AfterReasoningCtx(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        tools_used=(),
        thinking=None,
        response_metadata=ResponseMetadata(raw_text="reply"),
        streamed=False,
        tool_chain=(),
        context_retry={},
        reply="reply",
    )


def test_after_reasoning_rejects_fixed_assistant_metadata_field() -> None:
    ctx = _assistant_metadata_ctx()
    ctx.persist_assistant_metadata["tools_used"] = ["spoof"]

    with pytest.raises(ValueError, match="metadata 字段不可写: tools_used"):
        _ = _collect_persist_assistant_metadata(ctx)


def test_after_reasoning_rejects_core_owned_user_metadata_field() -> None:
    ctx = _assistant_metadata_ctx()
    ctx.persist_user_metadata["control_turn_id"] = "spoof"

    with pytest.raises(ValueError, match="user plugin metadata 字段不可写"):
        _ = _collect_persist_user_metadata(ctx)


@pytest.mark.parametrize(
    "field",
    [
        "control_turn_id",
        "turn_terminal",
        "turn_input_count",
        "skip_post_memory",
        "turn_duration_ms",
    ],
)
def test_after_reasoning_rejects_core_owned_assistant_metadata(field: str) -> None:
    ctx = _assistant_metadata_ctx()
    ctx.persist_assistant_metadata[field] = "spoof"

    with pytest.raises(ValueError, match=f"metadata 字段不可写: {field}"):
        _ = _collect_persist_assistant_metadata(ctx)


@pytest.mark.asyncio
async def test_session_manager_adopts_pending_rows_before_post_commit_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session = manager.get_or_create("telegram:commit-cancel")
    pending: list[dict[str, object]] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "reply"},
    ]
    pending_metadata = {
        "sentinel": "committed",
        "last_turn_tool_calls_count": 1,
    }
    original_persist = manager._persist_session

    def persist_then_cancel(*args: Any, **kwargs: Any) -> int:
        count = original_persist(*args, **kwargs)
        task = asyncio.current_task()
        assert task is not None
        _ = task.cancel()
        return count

    monkeypatch.setattr(manager, "_persist_session", persist_then_cancel)

    async def append_then_checkpoint() -> None:
        await manager.append_messages(
            session,
            pending,
            metadata=pending_metadata,
        )
        await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.create_task(append_then_checkpoint())

    assert session.messages == pending
    assert [message["id"] for message in session.messages] == [
        "telegram:commit-cancel:0",
        "telegram:commit-cancel:1",
    ]
    assert session.metadata == pending_metadata
    manager.close()
    reloaded = SessionManager(tmp_path / "workspace")
    persisted = reloaded.get_or_create(session.key)
    assert persisted.messages == session.messages
    assert persisted.metadata == pending_metadata
    reloaded.close()


@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.asyncio
async def test_session_metadata_stays_unchanged_when_pending_append_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[BaseException],
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session = manager.get_or_create("telegram:metadata-rollback")
    session.metadata = {
        "sentinel": "old",
        "last_turn_tool_calls_count": 7,
        "last_turn_ts": "old-ts",
    }
    manager.save(session)
    original_metadata = dict(session.metadata)

    def fail_before_commit(*args: Any, **kwargs: Any) -> int:
        raise failure_type("injected append failure")

    monkeypatch.setattr(manager, "_persist_session", fail_before_commit)
    phase = Phase(
        default_after_reasoning_modules(
            EventBus(),
            cast(Any, SimpleNamespace(presence=None, session_manager=manager)),
        ),
        frame_factory=AfterReasoningFrame,
    )

    with pytest.raises(failure_type, match="injected append failure"):
        _ = await phase.run(
            AfterReasoningInput(
                state=TurnState(
                    msg=_inbound(),
                    session_key=session.key,
                    dispatch_outbound=True,
                    session=session,
                ),
                turn_result=TurnRunResult(
                    reply="reply",
                    tool_chain=[{"calls": [{"name": "shell"}]}],
                ),
            )
        )

    assert session.messages == []
    assert session.metadata == original_metadata
    manager.close()
    reloaded = SessionManager(tmp_path / "workspace")
    persisted = reloaded.get_or_create(session.key)
    assert persisted.messages == []
    assert persisted.metadata == original_metadata
    reloaded.close()


def test_late_legacy_observer_keeps_existing_phase_dag_contract() -> None:
    class LateObserverModule:
        slot = "test.after_reasoning.late_observer"
        requires = ("after_reasoning.persist_user", "reasoning:persisted_user")

        async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
            return frame

    modules = default_after_reasoning_modules(
        EventBus(),
        cast(Any, SimpleNamespace(presence=None, session_manager=object())),
        plugin_modules=[LateObserverModule()],
    )
    slots = [module.slot for module in modules]

    assert slots.index("after_reasoning.persist_user") < slots.index(
        "test.after_reasoning.late_observer"
    )
    assert slots.index("test.after_reasoning.late_observer") < slots.index(
        "after_reasoning.seal_metadata"
    )


@pytest.mark.asyncio
async def test_after_reasoning_persists_mobile_canonical_ids(tmp_path: Path):
    class SpoofMetadataModule:
        slot = "test.after_reasoning.spoof_client_message_id"
        requires = ("after_reasoning.emit", "reasoning:ctx")

        async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
            frame.slots["outbound:metadata:client_message_id"] = (
                "01ARZ3NDEKTSV4RRFFQ69G5FAW"
            )
            return frame

    manager = SessionManager(tmp_path / "workspace")
    artifact_store = ChannelAttachmentArtifactStore(
        workspace=tmp_path / "workspace",
        session_store=manager.control_store,
    )
    attachment = await artifact_store.import_bytes(
        b"mobile-user-attachment",
        kind=AttachmentKind.FILE,
        filename="note.txt",
        media_type="text/plain",
    )
    session = manager.get_or_create("mobile:00000000-0000-0000-0000-000000000001")
    msg = InboundMessage(
        channel="mobile",
        sender="device:test",
        chat_id="00000000-0000-0000-0000-000000000001",
        content="hello",
        media=["/proc/self/fd/999"],
        metadata={
            "client_message_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "attachment_ids": [attachment.artifact_id],
        },
    )
    state = TurnState(msg=msg, session_key=session.key, dispatch_outbound=True)
    state.session = session
    phase = Phase(
        default_after_reasoning_modules(
            EventBus(),
            cast(Any, SimpleNamespace(presence=None, session_manager=manager)),
            plugin_modules=[SpoofMetadataModule()],
        ),
        frame_factory=AfterReasoningFrame,
    )

    result = await phase.run(
        AfterReasoningInput(
            state=state,
            turn_result=TurnRunResult(reply="reply"),
        )
    )
    manager.close()
    reloaded = SessionManager(tmp_path / "workspace")
    messages = reloaded.get_or_create(session.key).messages

    assert messages[0]["client_message_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert result.outbound.metadata["persisted_user_message_id"] == messages[0]["id"]
    assert (
        result.outbound.metadata["client_message_id"]
        == messages[0]["client_message_id"]
    )
    assert result.outbound.session_message_id == messages[1]["id"]
    assert messages[0]["attachment_ids"] == [attachment.artifact_id]
    assert messages[0].get("media") in (None, [])
    assert "/proc/" not in json.dumps(messages[0], ensure_ascii=False)
    assert reloaded.control_store.message_attachment_ids(
        cast(str, messages[0]["id"])
    ) == (attachment.artifact_id,)
    reloaded.close()


@pytest.mark.asyncio
async def test_after_reasoning_commits_all_same_turn_users_before_final_assistant(
    tmp_path: Path,
):
    """保持已送达 proactive 与随后提交的 interaction 各自成单元。"""

    class _Source:
        def used_inputs(self) -> tuple[TurnUserInput, ...]:
            return (
                TurnUserInput(
                    "i1",
                    0,
                    "u1",
                    (),
                    {"client_message_id": "client:previous-attempt"},
                    _now,
                ),
                TurnUserInput(
                    "i2",
                    1,
                    "u2",
                    (),
                    {
                        "client_message_id": "client:current-attempt",
                        "effects": {"post_commit": "suppress"},
                    },
                    _now,
                ),
            )

    # 1. 先提交交错送达且已经结束的 proactive 单元。
    manager = SessionManager(tmp_path / "workspace")
    session = manager.get_or_create("telegram:same-turn")
    proactive = session.add_message(
        "assistant",
        "proactive",
        proactive=True,
        delivery_id="delivery-1",
    )
    await manager.append_messages(session, [proactive])
    msg = InboundMessage(
        channel="telegram",
        sender="user",
        chat_id="same-turn",
        content="u1",
        metadata={
            "control_turn_id": "turn-1",
            "_control_turn_input_source": _Source(),
        },
    )
    state = TurnState(msg=msg, session_key=session.key, dispatch_outbound=True)
    state.session = session
    phase = Phase(
        default_after_reasoning_modules(
            EventBus(),
            cast(Any, SimpleNamespace(presence=None, session_manager=manager)),
        ),
        frame_factory=AfterReasoningFrame,
    )

    # 2. 最终 attempt 一次性提交此前累积的全部 U 和唯一 A。
    result = await phase.run(
        AfterReasoningInput(
            state=state,
            turn_result=TurnRunResult(reply="final"),
        )
    )
    manager.close()
    reloaded = SessionManager(tmp_path / "workspace")
    messages = reloaded.get_or_create(session.key).messages

    # 3. 单元切分、interaction 删除都不得吞掉 proactive。
    assert [(item["role"], item["content"]) for item in messages] == [
        ("assistant", "proactive"),
        ("user", "u1"),
        ("user", "u2"),
        ("assistant", "final"),
    ]
    assert logical_history_unit_ranges(messages) == [(0, 1), (1, 4)]
    assert [item["turn_input_ordinal"] for item in messages[1:3]] == [0, 1]
    assert [item["timestamp"] for item in messages[1:3]] == [
        _now.isoformat(),
        _now.isoformat(),
    ]
    assert all(item["control_turn_id"] == "turn-1" for item in messages[1:])
    assert messages[3]["turn_terminal"] is True
    assert messages[3]["turn_input_count"] == 2
    assert messages[2]["effects"] == {"post_commit": "suppress"}
    assert messages[3]["effects"] == {"post_commit": "suppress"}
    assert result.outbound.metadata["persisted_user_message_ids"] == [
        messages[1]["id"],
        messages[2]["id"],
    ]
    assert result.outbound.metadata["persisted_user_message_id"] == messages[2]["id"]
    assert result.outbound.metadata["client_message_id"] == "client:current-attempt"
    deletion = reloaded.control_store.delete_interaction("turn-1")
    assert deletion is not None
    assert deletion.message_ids == tuple(item["id"] for item in messages[1:])
    assert [
        item["content"]
        for item in reloaded.control_store.fetch_session_messages(session.key)
    ] == ["proactive"]
    reloaded.close()


@pytest.mark.asyncio
async def test_after_reasoning_persists_clean_mobile_reply_projection(tmp_path: Path):
    manager = SessionManager(tmp_path / "workspace")
    session = manager.get_or_create("mobile:00000000-0000-0000-0000-000000000001")
    merged = "【你正在回复一条历史消息】\n被回复消息：旧回答\n\n【你当前新消息】\n继续"
    server_received_at = datetime.fromisoformat("2026-07-16T04:04:52+00:00")
    msg = InboundMessage(
        channel="mobile",
        sender="device:test",
        chat_id="00000000-0000-0000-0000-000000000001",
        content=merged,
        timestamp=server_received_at,
        metadata={
            "client_message_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "client_created_at": "2026-07-16T04:05:06+00:00",
            "display_content": "继续",
            "reply_to_message_id": "mobile:00000000-0000-0000-0000-000000000001:0",
            "reply_role": "assistant",
            "reply_preview": "旧回答",
        },
    )
    state = TurnState(msg=msg, session_key=session.key, dispatch_outbound=True)
    state.session = session
    phase = Phase(
        default_after_reasoning_modules(
            EventBus(),
            cast(Any, SimpleNamespace(presence=None, session_manager=manager)),
        ),
        frame_factory=AfterReasoningFrame,
    )

    await phase.run(
        AfterReasoningInput(
            state=state,
            turn_result=TurnRunResult(
                reply="reply",
                context_retry={"llm_user_content": merged},
            ),
        )
    )
    manager.close()
    reloaded = SessionManager(tmp_path / "workspace")
    user = reloaded.get_or_create(session.key).messages[0]

    assert user["content"] == "继续"
    assert user["timestamp"] == server_received_at.isoformat()
    assert user["client_created_at"] == "2026-07-16T04:05:06+00:00"
    assert user["llm_user_content"] == merged
    assert user["reply_to_message_id"].endswith(":0")
    assert user["reply_role"] == "assistant"
    assert user["reply_preview"] == "旧回答"
    reloaded.close()


@pytest.mark.asyncio
async def test_after_turn_collects_extra_and_telemetry_slots():
    committed_extra: list[dict[str, object]] = []
    committed_events: list[TurnCommitted] = []
    after_turn_metadata: list[dict[str, object]] = []
    bus = EventBus()

    class ExtraModule:
        slot = "test.after_turn.extra"
        requires = ("after_turn.build_work", "turn:extra")

        async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
            frame.slots["turn:extra:plugin_flag"] = "extra"
            return frame

    class TelemetryModule:
        slot = "test.after_turn.telemetry"
        requires = ("after_turn.build_ctx", "turn:ctx")

        async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
            frame.slots["turn:telemetry:plugin_flag"] = "telemetry"
            return frame

    async def committed_handler(event: TurnCommitted) -> None:
        committed_events.append(event)
        committed_extra.append(dict(event.extra))

    async def after_turn_handler(ctx: AfterTurnCtx) -> None:
        after_turn_metadata.append(dict(ctx.extra_metadata))

    bus.on(AfterTurnCtx, after_turn_handler)
    bus.on(TurnCommitted, committed_handler)
    session = _DummySession("telegram:123")
    msg = _inbound()
    state = TurnState(msg=msg, session_key=session.key, dispatch_outbound=False)
    state.session = session
    ctx = AfterReasoningCtx(
        session_key=session.key,
        channel=msg.channel,
        chat_id=msg.chat_id,
        tools_used=(),
        thinking=None,
        response_metadata=ResponseMetadata(raw_text="reply"),
        streamed=False,
        tool_chain=(),
        context_retry={},
        reply="reply",
    )
    context = Mock()
    context.render = Mock(return_value=SimpleNamespace(messages=[]))
    context.last_debug_breakdown = []
    phase = Phase(
        default_after_turn_modules(
            bus,
            _DummyOutbound(),
            cast(ContextBuilder, context),
            plugin_modules=[ExtraModule(), TelemetryModule()],
        ),
        frame_factory=AfterTurnFrame,
    )

    await phase.run(
        TurnSnapshot(
            state=state,
            outbound=OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="reply",
                metadata={
                    "persisted_user_message_id": "telegram:123:0",
                    "persisted_user_message_ids": [
                        "telegram:123:0",
                        "telegram:123:1",
                    ],
                },
                session_message_id="telegram:123:2",
            ),
            ctx=ctx,
        )
    )

    assert committed_extra[0]["plugin_flag"] == "extra"
    assert committed_events[0].persisted_user_message_id == "telegram:123:0"
    assert committed_events[0].persisted_user_message_ids == (
        "telegram:123:0",
        "telegram:123:1",
    )
    assert committed_events[0].assistant_message_id == "telegram:123:2"
    assert after_turn_metadata == [{"plugin_flag": "telemetry"}]


@contextmanager
def _turn_identity(
    *,
    session_key: str,
    turn_id: str,
    client_message_id: str,
) -> Iterator[None]:
    """对齐真实 turn 边界：session_key 来自 TurnState.session_key、
    turn_id 是 loop owner 建立的 running_turn_id、
    client_message_id 来自真实 inbound metadata。"""
    session_token = current_session_key.set(session_key)
    turn_token = running_turn_id.set(turn_id)
    client_token = current_client_message_id.set(client_message_id)
    try:
        yield
    finally:
        current_client_message_id.reset(client_token)
        running_turn_id.reset(turn_token)
        current_session_key.reset(session_token)


def _identity_inbound(
    *,
    client_message_id: str,
    control_turn_id: str,
) -> InboundMessage:
    """真实 inbound 身份：client_message_id 与 control_turn_id 都在入站 metadata
    （loop owner 在 turn 边界写入，control_turn_id 恒等于 running_turn_id）。"""
    msg = _inbound()
    msg.metadata["client_message_id"] = client_message_id
    msg.metadata["control_turn_id"] = control_turn_id
    return msg


def _milestone_records(
    caplog: pytest.LogCaptureFixture,
    *events: str,
) -> list[Any]:
    return [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("event") in events
    ]


@pytest.mark.asyncio
async def test_after_reasoning_append_records_success_milestones(
    caplog: pytest.LogCaptureFixture,
) -> None:
    appended: list[tuple[str, list[dict[str, object]]]] = []

    async def append_messages(
        current: _DummySession,
        messages: list[dict[str, object]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        appended.append((current.key, messages))

    turn_id = "turn:final"
    client_message_id = "cm:01"
    session = _DummySession("telegram:123")
    msg = _identity_inbound(
        client_message_id=client_message_id,
        control_turn_id=turn_id,
    )
    state = TurnState(msg=msg, session_key=session.key, dispatch_outbound=True)
    state.session = session
    phase = Phase(
        default_after_reasoning_modules(
            EventBus(),
            cast(
                Any,
                SimpleNamespace(
                    presence=None,
                    session_manager=SimpleNamespace(append_messages=append_messages),
                ),
            ),
        ),
        frame_factory=AfterReasoningFrame,
    )

    with _turn_identity(
        session_key=state.session_key,
        turn_id=turn_id,
        client_message_id=client_message_id,
    ):
        with caplog.at_level(
            logging.INFO, logger="agent.lifecycle.phases.after_reasoning"
        ):
            result = await phase.run(
                AfterReasoningInput(
                    state=state,
                    turn_result=TurnRunResult(reply="reply"),
                )
            )

    assert [key for key, _ in appended] == [state.session_key]
    assert [item["role"] for item in appended[0][1]] == ["user", "assistant"]
    # DB append 与 milestone 三元 identity 相同：client_message_id / control_turn_id
    # 写进持久化 user 消息，append 的 session 与里程碑 session_id 一致。
    persisted_user = appended[0][1][0]
    assert persisted_user["client_message_id"] == client_message_id
    assert persisted_user["control_turn_id"] == turn_id
    # 正常 final 的 OutboundMessage 直接携带 running turn id，不再依赖 channel fallback。
    assert result.outbound.control_turn_id == turn_id
    records = _milestone_records(
        caplog, "after_reasoning.append.start", "after_reasoning.append.done"
    )
    assert [record.akashic_fields["event"] for record in records] == [
        "after_reasoning.append.start",
        "after_reasoning.append.done",
    ]
    assert {record.akashic_fields["session_id"] for record in records} == {
        state.session_key
    }
    assert {record.akashic_fields["turn_id"] for record in records} == {turn_id}
    assert {record.akashic_fields["client_message_id"] for record in records} == {
        client_message_id
    }
    start, done = records
    assert start.akashic_fields["duration_ms"] is None
    assert start.akashic_fields["origin"] == "missing"
    assert done.akashic_fields["duration_ms"] is not None
    assert done.akashic_fields["outcome"] == "done"


@pytest.mark.asyncio
async def test_after_reasoning_append_records_error_milestones(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def append_messages(
        current: _DummySession,
        messages: list[dict[str, object]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise RuntimeError("append exploded")

    turn_id = "turn:final"
    client_message_id = "cm:01"
    session = _DummySession("telegram:123")
    msg = _identity_inbound(
        client_message_id=client_message_id,
        control_turn_id=turn_id,
    )
    state = TurnState(msg=msg, session_key=session.key, dispatch_outbound=True)
    state.session = session
    phase = Phase(
        default_after_reasoning_modules(
            EventBus(),
            cast(
                Any,
                SimpleNamespace(
                    presence=None,
                    session_manager=SimpleNamespace(append_messages=append_messages),
                ),
            ),
        ),
        frame_factory=AfterReasoningFrame,
    )

    with _turn_identity(
        session_key=state.session_key,
        turn_id=turn_id,
        client_message_id=client_message_id,
    ):
        with caplog.at_level(
            logging.INFO, logger="agent.lifecycle.phases.after_reasoning"
        ):
            with pytest.raises(RuntimeError, match="append exploded"):
                await phase.run(
                    AfterReasoningInput(
                        state=state,
                        turn_result=TurnRunResult(reply="reply"),
                    )
                )

    records = _milestone_records(
        caplog, "after_reasoning.append.start", "after_reasoning.append.error"
    )
    assert [record.akashic_fields["event"] for record in records] == [
        "after_reasoning.append.start",
        "after_reasoning.append.error",
    ]
    assert {record.akashic_fields["session_id"] for record in records} == {
        state.session_key
    }
    assert {record.akashic_fields["turn_id"] for record in records} == {turn_id}
    assert {record.akashic_fields["client_message_id"] for record in records} == {
        client_message_id
    }
    start, error = records
    assert start.akashic_fields["duration_ms"] is None
    assert error.akashic_fields["duration_ms"] is not None
    assert error.akashic_fields["outcome"] == "error"
    assert error.levelno == logging.ERROR
    assert not _milestone_records(caplog, "after_reasoning.append.done")


@pytest.mark.asyncio
async def test_after_reasoning_append_records_cancelled_milestone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def append_messages(
        current: _DummySession,
        messages: list[dict[str, object]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise asyncio.CancelledError()

    turn_id = "turn:final"
    client_message_id = "cm:01"
    session = _DummySession("telegram:123")
    msg = _identity_inbound(
        client_message_id=client_message_id,
        control_turn_id=turn_id,
    )
    state = TurnState(msg=msg, session_key=session.key, dispatch_outbound=True)
    state.session = session
    phase = Phase(
        default_after_reasoning_modules(
            EventBus(),
            cast(
                Any,
                SimpleNamespace(
                    presence=None,
                    session_manager=SimpleNamespace(append_messages=append_messages),
                ),
            ),
        ),
        frame_factory=AfterReasoningFrame,
    )

    with _turn_identity(
        session_key=state.session_key,
        turn_id=turn_id,
        client_message_id=client_message_id,
    ):
        with caplog.at_level(
            logging.INFO, logger="agent.lifecycle.phases.after_reasoning"
        ):
            with pytest.raises(asyncio.CancelledError):
                await phase.run(
                    AfterReasoningInput(
                        state=state,
                        turn_result=TurnRunResult(reply="reply"),
                    )
                )

    records = _milestone_records(
        caplog, "after_reasoning.append.start", "after_reasoning.append.cancelled"
    )
    assert [record.akashic_fields["event"] for record in records] == [
        "after_reasoning.append.start",
        "after_reasoning.append.cancelled",
    ]
    assert {record.akashic_fields["session_id"] for record in records} == {
        state.session_key
    }
    assert {record.akashic_fields["turn_id"] for record in records} == {turn_id}
    assert {record.akashic_fields["client_message_id"] for record in records} == {
        client_message_id
    }
    start, cancelled = records
    assert start.akashic_fields["duration_ms"] is None
    assert cancelled.akashic_fields["duration_ms"] is not None
    assert cancelled.akashic_fields["outcome"] == "cancelled"
    assert cancelled.levelno == logging.WARNING
    assert not _milestone_records(caplog, "after_reasoning.append.done")


def _after_turn_phase(
    bus: EventBus,
    *,
    turn_id: str = "turn:final",
    client_message_id: str = "cm:01",
) -> tuple[Phase, TurnState, _DummySession]:
    session = _DummySession("telegram:123")
    msg = _identity_inbound(
        client_message_id=client_message_id,
        control_turn_id=turn_id,
    )
    state = TurnState(msg=msg, session_key=session.key, dispatch_outbound=False)
    state.session = session
    ctx = AfterReasoningCtx(
        session_key=session.key,
        channel=msg.channel,
        chat_id=msg.chat_id,
        tools_used=(),
        thinking=None,
        response_metadata=ResponseMetadata(raw_text="reply"),
        streamed=False,
        tool_chain=(),
        context_retry={},
        reply="reply",
    )
    context = Mock()
    context.render = Mock(return_value=SimpleNamespace(messages=[]))
    context.last_debug_breakdown = []
    phase = Phase(
        default_after_turn_modules(
            bus,
            _DummyOutbound(),
            cast(ContextBuilder, context),
        ),
        frame_factory=AfterTurnFrame,
    )
    return phase, state, session


async def _run_after_turn(
    phase: Phase,
    state: TurnState,
    *,
    reply_to: str | None = None,
    media: list[str] | None = None,
    session_message_id: str | None = None,
) -> None:
    session = cast(_DummySession, state.session)
    msg = state.msg
    await phase.run(
        TurnSnapshot(
            state=state,
            outbound=OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="reply",
                reply_to=reply_to,
                media=list(media or []),
                session_message_id=session_message_id,
                control_turn_id=str(msg.metadata.get("control_turn_id") or ""),
            ),
            ctx=AfterReasoningCtx(
                session_key=session.key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                tools_used=(),
                thinking=None,
                response_metadata=ResponseMetadata(raw_text="reply"),
                streamed=False,
                tool_chain=(),
                context_retry={},
                reply="reply",
            ),
        )
    )


@pytest.mark.asyncio
async def test_after_turn_fanout_records_returned_milestone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    delivered: list[TurnCommitted] = []
    bus = EventBus()
    bus.on(TurnCommitted, lambda event: delivered.append(event))
    phase, state, _ = _after_turn_phase(bus)
    turn_id = "turn:final"
    client_message_id = "cm:01"

    with _turn_identity(
        session_key=state.session_key,
        turn_id=turn_id,
        client_message_id=client_message_id,
    ):
        with caplog.at_level(logging.INFO, logger="agent.lifecycle.phases.after_turn"):
            await _run_after_turn(phase, state)

    assert [item.turn_id for item in delivered] == [turn_id]
    assert [item.client_message_id for item in delivered] == [client_message_id]
    records = _milestone_records(
        caplog,
        "after_turn.turn_committed_fanout.start",
        "after_turn.turn_committed_fanout.returned",
    )
    assert [record.akashic_fields["event"] for record in records] == [
        "after_turn.turn_committed_fanout.start",
        "after_turn.turn_committed_fanout.returned",
    ]
    assert {record.akashic_fields["session_id"] for record in records} == {
        state.session_key
    }
    assert {record.akashic_fields["turn_id"] for record in records} == {turn_id}
    assert {record.akashic_fields["client_message_id"] for record in records} == {
        client_message_id
    }
    start, returned = records
    assert start.akashic_fields["duration_ms"] is None
    assert returned.akashic_fields["duration_ms"] is not None
    assert returned.akashic_fields["outcome"] == "returned"


class _ExplodingFanoutBus(EventBus):
    async def fanout(self, event: object) -> None:
        raise RuntimeError("fanout exploded")


@pytest.mark.asyncio
async def test_after_turn_fanout_records_error_milestone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    phase, state, _ = _after_turn_phase(_ExplodingFanoutBus())
    turn_id = "turn:final"
    client_message_id = "cm:01"

    with _turn_identity(
        session_key=state.session_key,
        turn_id=turn_id,
        client_message_id=client_message_id,
    ):
        with caplog.at_level(logging.INFO, logger="agent.lifecycle.phases.after_turn"):
            with pytest.raises(RuntimeError, match="fanout exploded"):
                await _run_after_turn(phase, state)

    records = _milestone_records(
        caplog,
        "after_turn.turn_committed_fanout.start",
        "after_turn.turn_committed_fanout.error",
    )
    assert [record.akashic_fields["event"] for record in records] == [
        "after_turn.turn_committed_fanout.start",
        "after_turn.turn_committed_fanout.error",
    ]
    assert {record.akashic_fields["session_id"] for record in records} == {
        state.session_key
    }
    assert {record.akashic_fields["turn_id"] for record in records} == {turn_id}
    assert {record.akashic_fields["client_message_id"] for record in records} == {
        client_message_id
    }
    start, error = records
    assert start.akashic_fields["duration_ms"] is None
    assert error.akashic_fields["duration_ms"] is not None
    assert error.akashic_fields["outcome"] == "error"
    assert error.levelno == logging.ERROR
    assert not _milestone_records(caplog, "after_turn.turn_committed_fanout.returned")


@pytest.mark.asyncio
async def test_after_turn_committed_event_carries_client_message_id_identity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TurnCommitted 与 milestone 的 session/turn/client_message_id 三元身份相同，
    全部来自真实 TurnState.session_key、inbound metadata 与 running_turn_id。"""

    delivered: list[TurnCommitted] = []
    bus = EventBus()
    bus.on(TurnCommitted, lambda event: delivered.append(event))
    phase, state, _ = _after_turn_phase(bus)
    turn_id = "turn:final"
    client_message_id = "cm:01"

    # session contextvar 与 turn state 对齐，模拟 turn 边界三件套一起写入。
    with _turn_identity(
        session_key=state.session_key,
        turn_id=turn_id,
        client_message_id=client_message_id,
    ):
        with caplog.at_level(logging.INFO, logger="agent.lifecycle.phases.after_turn"):
            await _run_after_turn(phase, state)

    assert delivered
    committed = delivered[0]
    assert committed.session_key == state.session_key
    assert committed.turn_id == turn_id
    assert committed.client_message_id == client_message_id
    # contextvar 是唯一写入点；事件身份与 turn 里程碑完全一致。
    records = _milestone_records(
        caplog,
        "after_turn.turn_committed_fanout.start",
        "after_turn.turn_committed_fanout.returned",
    )
    assert records
    assert {record.akashic_fields["session_id"] for record in records} == {
        state.session_key
    }
    assert {record.akashic_fields["turn_id"] for record in records} == {turn_id}
    assert {record.akashic_fields["client_message_id"] for record in records} == {
        client_message_id
    }


@pytest.mark.asyncio
async def test_after_turn_fanout_returns_after_observer_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """真实 EventBus handler 抛 RuntimeError 时，错误由 EventBus 自身观测并记录，
    fanout 仍正常返回：returned 只表示 EventBus await 返回，不宣称所有 handler 成功。"""

    async def exploding_handler(event: TurnCommitted) -> None:
        raise RuntimeError("observer exploded")

    bus = EventBus()
    bus.on(TurnCommitted, exploding_handler)
    phase, state, _ = _after_turn_phase(bus)
    turn_id = "turn:final"
    client_message_id = "cm:01"

    with _turn_identity(
        session_key=state.session_key,
        turn_id=turn_id,
        client_message_id=client_message_id,
    ):
        with caplog.at_level(logging.INFO, logger="agent.lifecycle.phases.after_turn"):
            await _run_after_turn(phase, state)

    # EventBus 隔离观察者并记录异常：observer error + fanout 失败计数。
    observer_errors = [
        record
        for record in caplog.records
        if record.name == "bus.event_bus"
        and "observer error for TurnCommitted" in record.getMessage()
    ]
    assert observer_errors
    assert "exploding_handler" in observer_errors[0].getMessage()
    failure_summary = [
        record
        for record in caplog.records
        if record.name == "bus.event_bus"
        and record.getMessage().startswith("fanout completed with observer errors:")
    ]
    assert failure_summary
    assert "failed=1 total=1" in failure_summary[0].getMessage()
    # fanout 自身正常返回，记录 returned；不冒充 error，也不吞掉 EventBus 的观测。
    records = _milestone_records(
        caplog,
        "after_turn.turn_committed_fanout.start",
        "after_turn.turn_committed_fanout.returned",
    )
    assert [record.akashic_fields["event"] for record in records] == [
        "after_turn.turn_committed_fanout.start",
        "after_turn.turn_committed_fanout.returned",
    ]
    assert records[1].akashic_fields["outcome"] == "returned"
    assert not _milestone_records(caplog, "after_turn.turn_committed_fanout.error")


@pytest.mark.asyncio
async def test_after_turn_dispatch_forwards_typed_identity_to_channel_port() -> None:
    """after_turn 将 control/reply/session/media 身份完整交给 typed Channel port。"""

    bus = EventBus()
    dispatched: list[OutboundDispatch] = []

    class _RecordingOutbound:
        async def dispatch(self, outbound: OutboundDispatch) -> ChannelDeliveryReceipt:
            dispatched.append(outbound)
            return ChannelDeliveryReceipt(
                delivery_id="delivery-final",
                status=ChannelDeliveryStatus.DELIVERED,
            )

    outbound_port = _RecordingOutbound()
    turn_id = "turn:final"
    client_message_id = "cm:01"
    session = _DummySession("telegram:123")
    msg = _identity_inbound(
        client_message_id=client_message_id,
        control_turn_id=turn_id,
    )
    state = TurnState(msg=msg, session_key=session.key, dispatch_outbound=True)
    state.session = session
    context = Mock()
    context.render = Mock(return_value=SimpleNamespace(messages=[]))
    context.last_debug_breakdown = []
    phase = Phase(
        default_after_turn_modules(
            EventBus(),
            outbound_port,
            cast(ContextBuilder, context),
        ),
        frame_factory=AfterTurnFrame,
    )

    with _turn_identity(
        session_key=state.session_key,
        turn_id=turn_id,
        client_message_id=client_message_id,
    ):
        await _run_after_turn(
            phase,
            state,
            reply_to="message-1",
            media=["/tmp/image.png"],
            session_message_id="telegram:123:2",
        )

    assert len(dispatched) == 1
    outbound_message = dispatched[0]
    assert outbound_message.control_turn_id == turn_id
    assert outbound_message.reply_to == "message-1"
    assert outbound_message.session_message_id == "telegram:123:2"
    assert outbound_message.media == ["/tmp/image.png"]
    assert outbound_message.content == "reply"
    assert outbound_message.channel == msg.channel
    assert outbound_message.chat_id == msg.chat_id


def _control_outbound_pipeline(
    session: _DummySession,
    *,
    reasoner_error: RuntimeError | None = None,
) -> Any:
    reasoner = SimpleNamespace(
        run_turn=AsyncMock(
            side_effect=(
                reasoner_error if reasoner_error is not None else lambda **_: None
            )
        ),
    )
    dispatch_port = AsyncMock(return_value=True)
    context_store = SimpleNamespace(
        prepare=AsyncMock(return_value=ContextBundle()),
    )
    context = SimpleNamespace(
        render=MagicMock(return_value=SimpleNamespace(system_prompt="p", messages=[])),
    )
    pipeline = PassiveTurnPipeline(
        PassiveTurnDeps(
            session=cast(
                Any,
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
            tools=cast(Any, SimpleNamespace(set_context=MagicMock())),
            reasoner=cast(Reasoner, reasoner),
            outbound_port=cast(OutboundPort, dispatch_port),
        )
    )
    return pipeline, dispatch_port


@pytest.mark.asyncio
async def test_control_outbound_forwards_current_turn_id_under_turn_context() -> None:
    """abort/error 的 _control_outbound 在当前 turn context 下把 running_turn_id
    传入 dispatch；返回对象身份一致，不因 dispatch 而被替换。"""

    session = _DummySession("telegram:123")
    pipeline, dispatch_port = _control_outbound_pipeline(
        session,
        reasoner_error=RuntimeError("budget guard"),
    )
    msg = _inbound()
    turn_id = "turn:control"
    with _turn_identity(
        session_key="telegram:123",
        turn_id=turn_id,
        client_message_id="cm:01",
    ):
        out = await pipeline.run(msg, "telegram:123", dispatch_outbound=True)

    assert out.content == "处理消息时出错，请稍后再试。"
    # 返回对象身份一致：没有被 dispatch 改写或替换。
    assert out.control_turn_id is None
    dispatch_port.dispatch.assert_awaited_once()
    dispatched = dispatch_port.dispatch.await_args.args[0]
    assert isinstance(dispatched, OutboundDispatch)
    assert dispatched.control_turn_id == turn_id


@pytest.mark.asyncio
async def test_control_outbound_does_not_fabricate_turn_id_without_turn() -> None:
    """proactive 无 turn 的 abort/error 消息不伪造 control_turn_id。"""

    session = _DummySession("telegram:123")
    pipeline, dispatch_port = _control_outbound_pipeline(
        session,
        reasoner_error=RuntimeError("budget guard"),
    )
    msg = _inbound()

    out = await pipeline.run(msg, "telegram:123", dispatch_outbound=True)

    assert out.content == "处理消息时出错，请稍后再试。"
    dispatch_port.dispatch.assert_awaited_once()
    dispatched = dispatch_port.dispatch.await_args.args[0]
    assert isinstance(dispatched, OutboundDispatch)
    assert dispatched.control_turn_id is None
