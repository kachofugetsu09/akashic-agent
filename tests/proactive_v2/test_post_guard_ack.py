"""
TDD — Phase 6: proactive_v2/agent_tick.py — Post-guard + ACK

测试分两层：
  A. 模块级纯函数（build_delivery_key、ack_* helpers）— 无需 tick()
  B. tick() 集成测试 — 验证 flow 分支正确触发 guard 和 ACK
"""
from __future__ import annotations
from typing import Any, cast

import json
from hashlib import sha1
from unittest.mock import AsyncMock, MagicMock

import pytest

from proactive_v2.agent_tick import (
    ack_discarded,
    ack_on_success,
    ack_post_guard_fail,
    build_delivery_key,
)
from proactive_v2.context import AgentTickContext
from proactive_v2.gateway import GatewayDeps
from proactive_v2.tools import ToolDeps
from tests.proactive_v2.conftest import (
    FakeAckSink,
    FakeAlertAckSink,
    FakeLLM,
    FakeStateStore,
    cfg_with,
    make_agent_tick,
)


# ═══════════════════════════════════════════════════════════════════════════
# A. 模块级纯函数测试
# ═══════════════════════════════════════════════════════════════════════════

# ── build_delivery_key ────────────────────────────────────────────────────

def _make_ctx(cited: list[str] = cast(Any, ()), message: str = "") -> AgentTickContext:
    ctx = AgentTickContext()
    ctx.cited_item_ids = list(cited)
    ctx.final_message = message
    return ctx


def test_delivery_key_from_cited_ids():
    cited = ["feed-mcp:1", "alert-mcp:2"]
    expected = sha1(json.dumps(sorted(cited)).encode()).hexdigest()[:16]
    assert build_delivery_key(_make_ctx(cited=cited)) == expected


def test_delivery_key_order_independent():
    k1 = build_delivery_key(_make_ctx(cited=["feed-mcp:1", "feed-mcp:2"], message="A"))
    k2 = build_delivery_key(_make_ctx(cited=["feed-mcp:2", "feed-mcp:1"], message="B"))
    assert k1 == k2


def test_delivery_key_different_item_sets_differ():
    k1 = build_delivery_key(_make_ctx(cited=["feed-mcp:1"]))
    k2 = build_delivery_key(_make_ctx(cited=["feed-mcp:2"]))
    assert k1 != k2


def test_delivery_key_fallback_to_message_when_no_cited():
    msg = "context only message"
    expected = sha1(msg[:500].encode()).hexdigest()[:16]
    assert build_delivery_key(_make_ctx(message=msg)) == expected


def test_delivery_key_message_truncated_to_500():
    msg_long = "x" * 600
    msg_short = "x" * 500
    k1 = build_delivery_key(_make_ctx(message=msg_long))
    k2 = build_delivery_key(_make_ctx(message=msg_short))
    assert k1 == k2


def test_delivery_key_is_16_chars():
    assert len(build_delivery_key(_make_ctx(cited=["feed-mcp:1"]))) == 16


def test_delivery_key_prefers_content_url_over_event_id():
    ctx1 = _make_ctx(cited=["feed-mcp:1"])
    ctx1.fetched_contents = [
        {"id": "1", "event_id": "1", "ack_server": "feed-mcp", "url": "https://example.com/a", "title": "A"}
    ]
    ctx2 = _make_ctx(cited=["feed-mcp:2"])
    ctx2.fetched_contents = [
        {"id": "2", "event_id": "2", "ack_server": "feed-mcp", "url": "https://example.com/a", "title": "A"}
    ]
    assert build_delivery_key(ctx1) == build_delivery_key(ctx2)


def test_delivery_key_falls_back_to_source_and_title_when_url_missing():
    ctx1 = _make_ctx(cited=["feed-mcp:1"])
    ctx1.fetched_contents = [
        {"id": "1", "event_id": "1", "ack_server": "feed-mcp", "source": "HLTV", "title": "Same title"}
    ]
    ctx2 = _make_ctx(cited=["feed-mcp:2"])
    ctx2.fetched_contents = [
        {"id": "2", "event_id": "2", "ack_server": "feed-mcp", "source": "HLTV", "title": "Same title"}
    ]
    assert build_delivery_key(ctx1) == build_delivery_key(ctx2)


# ── ack_discarded ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ack_discarded_720h():
    ctx = AgentTickContext()
    ctx.discarded_item_ids = {"feed-mcp:1", "feed-mcp:2"}
    sink = FakeAckSink()
    await ack_discarded(ctx, sink)
    assert sink.acked("feed-mcp:1", 720)
    assert sink.acked("feed-mcp:2", 720)


@pytest.mark.asyncio
async def test_ack_discarded_does_not_touch_cited():
    ctx = AgentTickContext()
    ctx.discarded_item_ids = {"feed-mcp:1"}
    ctx.cited_item_ids = ["feed-mcp:2"]
    sink = FakeAckSink()
    await ack_discarded(ctx, sink)
    assert sink.not_acked("feed-mcp:2")


@pytest.mark.asyncio
async def test_ack_discarded_empty_set_no_calls():
    ctx = AgentTickContext()
    sink = FakeAckSink()
    await ack_discarded(ctx, sink)
    assert sink.calls == []


@pytest.mark.asyncio
async def test_ack_discarded_none_ack_fn_no_error():
    ctx = AgentTickContext()
    ctx.discarded_item_ids = {"feed-mcp:1"}
    await ack_discarded(ctx, None)  # should not raise


# ── ack_post_guard_fail ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ack_post_guard_fail_cited_24h():
    ctx = AgentTickContext()
    ctx.cited_item_ids = ["feed-mcp:1"]
    ctx.interesting_item_ids = {"feed-mcp:1"}
    sink = FakeAckSink()
    await ack_post_guard_fail(ctx, sink)
    assert sink.acked("feed-mcp:1", 24)


@pytest.mark.asyncio
async def test_ack_post_guard_fail_uncited_interesting_24h():
    ctx = AgentTickContext()
    ctx.cited_item_ids = ["feed-mcp:1"]
    ctx.interesting_item_ids = {"feed-mcp:1", "feed-mcp:2"}  # feed-mcp:2 uncited
    sink = FakeAckSink()
    await ack_post_guard_fail(ctx, sink)
    assert sink.acked("feed-mcp:2", 24)


@pytest.mark.asyncio
async def test_ack_post_guard_fail_discarded_720h():
    ctx = AgentTickContext()
    ctx.cited_item_ids = []
    ctx.discarded_item_ids = {"feed-mcp:3"}
    sink = FakeAckSink()
    await ack_post_guard_fail(ctx, sink)
    assert sink.acked("feed-mcp:3", 720)


@pytest.mark.asyncio
async def test_ack_post_guard_fail_all_three_buckets():
    ctx = AgentTickContext()
    ctx.cited_item_ids = ["feed-mcp:1"]
    ctx.interesting_item_ids = {"feed-mcp:1", "feed-mcp:2"}
    ctx.discarded_item_ids = {"feed-mcp:3"}
    sink = FakeAckSink()
    await ack_post_guard_fail(ctx, sink)
    assert sink.acked("feed-mcp:1", 24)   # cited
    assert sink.acked("feed-mcp:2", 24)   # uncited interesting
    assert sink.acked("feed-mcp:3", 720)  # discarded


# ── ack_on_success ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ack_on_success_content_cited_168h():
    ctx = AgentTickContext()
    ctx.fetched_contents = [{"id": "c1", "ack_server": "feed-mcp"}]
    ctx.fetched_alerts = []
    ctx.cited_item_ids = ["feed-mcp:c1"]
    ctx.interesting_item_ids = {"feed-mcp:c1"}
    sink = FakeAckSink()
    await ack_on_success(ctx, sink)
    assert sink.acked("feed-mcp:c1", 168)


@pytest.mark.asyncio
async def test_ack_on_success_alert_cited_168h():
    ctx = AgentTickContext()
    ctx.fetched_alerts = [{"id": "a1", "ack_server": "alert-mcp"}]
    ctx.fetched_contents = []
    ctx.cited_item_ids = ["alert-mcp:a1"]
    ctx.interesting_item_ids = {"alert-mcp:a1"}
    sink = FakeAckSink()
    await ack_on_success(ctx, sink)
    assert sink.acked("alert-mcp:a1", 168)


@pytest.mark.asyncio
async def test_ack_on_success_uncited_interesting_24h():
    ctx = AgentTickContext()
    ctx.fetched_contents = [
        {"id": "c1", "ack_server": "feed-mcp"},
        {"id": "c2", "ack_server": "feed-mcp"},
    ]
    ctx.fetched_alerts = []
    ctx.cited_item_ids = ["feed-mcp:c1"]
    ctx.interesting_item_ids = {"feed-mcp:c1", "feed-mcp:c2"}  # c2 uncited
    sink = FakeAckSink()
    await ack_on_success(ctx, sink)
    assert sink.acked("feed-mcp:c1", 168)   # cited → 168h
    assert sink.acked("feed-mcp:c2", 24)    # uncited interesting → 24h


@pytest.mark.asyncio
async def test_ack_on_success_discarded_720h():
    ctx = AgentTickContext()
    ctx.fetched_contents = [{"id": "c1", "ack_server": "feed-mcp"}]
    ctx.fetched_alerts = []
    ctx.cited_item_ids = ["feed-mcp:c1"]
    ctx.interesting_item_ids = {"feed-mcp:c1"}
    ctx.discarded_item_ids = {"feed-mcp:c99"}
    sink = FakeAckSink()
    await ack_on_success(ctx, sink)
    assert sink.acked("feed-mcp:c99", 720)


@pytest.mark.asyncio
async def test_ack_on_success_split_alert_content_by_compound_key():
    """alert-mcp:42 和 feed-mcp:42 即使 id 相同，复合键不碰撞，都被正确 ACK"""
    ctx = AgentTickContext()
    ctx.fetched_alerts = [{"id": "42", "ack_server": "alert-mcp"}]
    ctx.fetched_contents = [{"id": "42", "ack_server": "feed-mcp"}]
    ctx.cited_item_ids = ["alert-mcp:42", "feed-mcp:42"]
    ctx.interesting_item_ids = {"alert-mcp:42", "feed-mcp:42"}
    sink = FakeAckSink()
    await ack_on_success(ctx, sink)
    assert sink.acked("alert-mcp:42", 168)
    assert sink.acked("feed-mcp:42", 168)


@pytest.mark.asyncio
async def test_ack_on_success_cited_not_in_fetched_not_acked():
    """cited 但不在 fetched_alerts/contents 中的 key 不 ACK（外部来源）"""
    ctx = AgentTickContext()
    ctx.fetched_alerts = []
    ctx.fetched_contents = []
    ctx.cited_item_ids = ["unknown-mcp:x"]
    ctx.interesting_item_ids = {"unknown-mcp:x"}
    sink = FakeAckSink()
    await ack_on_success(ctx, sink)
    assert sink.not_acked("unknown-mcp:x")


# ═══════════════════════════════════════════════════════════════════════════
# B. tick() 集成测试
# ═══════════════════════════════════════════════════════════════════════════

def _make_tick_with_sink(llm, *, state=None, sender=None, deduper=None,
                          tool_deps_extra=None, cfg=None):
    sink = FakeAckSink()
    if state is None:
        state = FakeStateStore()
    deps = ToolDeps(
        recent_chat_fn=AsyncMock(return_value=[]),
        ack_fn=sink,
    )
    gateway = GatewayDeps(
        alert_fn=AsyncMock(return_value=[]),
        feed_fn=AsyncMock(return_value=[]),
    )
    if tool_deps_extra:
        for k, v in tool_deps_extra.items():
            if hasattr(gateway, k):
                setattr(gateway, k, v)
            else:
                setattr(deps, k, v)
    if sender is None:
        sender = AsyncMock()
        sender.send.return_value = True
    if deduper is None:
        deduper = AsyncMock()
        deduper.is_duplicate = AsyncMock(return_value=(False, ""))
    tick = make_agent_tick(
        llm_fn=llm,
        state_store=state,
        sender=sender,
        deduper=deduper,
        tool_deps=deps,
        gateway_deps=gateway,
        cfg=cfg,
    )
    return tick, sink


# ── delivery_dedupe ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delivery_dedupe_hit_prevents_send():
    state = FakeStateStore()
    state.set_is_duplicate(True)
    sender = AsyncMock()
    sender.send.return_value = True

    llm = FakeLLM([
        ("message_push", {"message": "hello", "evidence": ["feed-mcp:1"]}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(llm, state=state, sender=sender)
    await tick.tick()

    sender.send.assert_not_called()


@pytest.mark.asyncio
async def test_delivery_dedupe_hit_acks_cited_24h():
    state = FakeStateStore()
    state.set_is_duplicate(True)
    event = {"id": "1", "ack_server": "feed-mcp"}

    llm = FakeLLM([
        ("message_push", {"message": "hello", "evidence": ["feed-mcp:1"]}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(
        llm, state=state, tool_deps_extra={"feed_fn": AsyncMock(return_value=[event])}
    )
    await tick.tick()

    assert sink.acked("feed-mcp:1", 24)


@pytest.mark.asyncio
async def test_delivery_dedupe_hit_no_mark_delivery():
    state = FakeStateStore()
    state.set_is_duplicate(True)

    llm = FakeLLM([
        ("message_push", {"message": "hi", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(llm, state=state)
    await tick.tick()

    assert state._deliveries == []


# ── message_dedupe ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_message_dedupe_hit_prevents_send():
    deduper = AsyncMock()
    deduper.is_duplicate = AsyncMock(return_value=(True, "similar to prev"))
    sender = AsyncMock()
    sender.send.return_value = True

    llm = FakeLLM([
        ("message_push", {"message": "hello", "evidence": ["feed-mcp:1"]}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(llm, sender=sender, deduper=deduper)
    await tick.tick()

    sender.send.assert_not_called()


@pytest.mark.asyncio
async def test_message_dedupe_hit_acks_cited_24h():
    deduper = AsyncMock()
    deduper.is_duplicate = AsyncMock(return_value=(True, "dup"))
    event = {"id": "1", "ack_server": "feed-mcp"}

    llm = FakeLLM([
        ("message_push", {"message": "hello", "evidence": ["feed-mcp:1"]}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(
        llm,
        deduper=deduper,
        tool_deps_extra={"feed_fn": AsyncMock(return_value=[event])},
    )
    await tick.tick()

    assert sink.acked("feed-mcp:1", 24)


@pytest.mark.asyncio
async def test_message_dedupe_disabled_skips_check():
    deduper = AsyncMock()
    deduper.is_duplicate = AsyncMock(return_value=(True, "would dup"))
    sender = AsyncMock()
    sender.send.return_value = True

    llm = FakeLLM([
        ("message_push", {"message": "hi", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(
        llm, sender=sender, deduper=deduper,
        cfg=cfg_with(message_dedupe_enabled=False),
    )
    await tick.tick()

    deduper.is_duplicate.assert_not_called()
    sender.send.assert_called_once()


@pytest.mark.asyncio
async def test_message_dedupe_called_with_correct_message():
    deduper = AsyncMock()
    deduper.is_duplicate = AsyncMock(return_value=(False, ""))

    llm = FakeLLM([
        ("message_push", {"message": "the message", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(llm, deduper=deduper)
    await tick.tick()

    call_kwargs = deduper.is_duplicate.call_args[1]
    assert call_kwargs["new_message"] == "the message"
    assert call_kwargs["new_state_summary_tag"] == "none"


# ── send success ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_success_calls_sender():
    sender = AsyncMock()
    sender.send.return_value = True
    llm = FakeLLM([
        ("message_push", {"message": "hi", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(llm, sender=sender)
    await tick.tick()
    sender.send.assert_called_once_with("hi")


@pytest.mark.asyncio
async def test_send_success_marks_delivery():
    state = FakeStateStore()
    llm = FakeLLM([
        ("message_push", {"message": "hi", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(llm, state=state)
    await tick.tick()
    assert len(state._deliveries) == 1


@pytest.mark.asyncio
async def test_send_success_acks_content_168h():
    event = {"id": "c1", "ack_server": "feed-mcp"}
    llm = FakeLLM([
        ("get_content_events", {}),
        ("message_push", {"message": "hi", "evidence": ["feed-mcp:c1"]}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(
        llm, tool_deps_extra={"feed_fn": AsyncMock(return_value=[event])}
    )
    await tick.tick()
    assert sink.acked("feed-mcp:c1", 168)


@pytest.mark.asyncio
async def test_send_success_acks_discarded_720h():
    llm = FakeLLM([
        ("mark_not_interesting", {"item_ids": ["feed-mcp:bad"]}),
        ("message_push", {"message": "hi", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(llm)
    await tick.tick()
    assert sink.acked("feed-mcp:bad", 720)


# ── send failure ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_failure_no_mark_delivery():
    state = FakeStateStore()
    sender = AsyncMock()
    sender.send.return_value = False

    llm = FakeLLM([
        ("message_push", {"message": "hi", "evidence": ["feed-mcp:1"]}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(llm, state=state, sender=sender)
    await tick.tick()

    assert state._deliveries == []


@pytest.mark.asyncio
async def test_send_failure_no_ack_cited():
    sender = AsyncMock()
    sender.send.return_value = False

    llm = FakeLLM([
        ("message_push", {"message": "hi", "evidence": ["feed-mcp:1"]}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(llm, sender=sender)
    await tick.tick()

    assert sink.not_acked("feed-mcp:1")


@pytest.mark.asyncio
async def test_send_failure_acks_discarded_720h():
    sender = AsyncMock()
    sender.send.return_value = False

    llm = FakeLLM([
        ("mark_not_interesting", {"item_ids": ["feed-mcp:bad"]}),
        ("message_push", {"message": "hi", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(llm, sender=sender)
    await tick.tick()

    assert sink.acked("feed-mcp:bad", 720)


# ── skip → only discarded ACK ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skip_acks_discarded_720h():
    llm = FakeLLM([
        ("mark_not_interesting", {"item_ids": ["feed-mcp:x"]}),
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
    ])
    tick, sink = _make_tick_with_sink(llm)
    await tick.tick()

    assert sink.acked("feed-mcp:x", 720)


@pytest.mark.asyncio
async def test_skip_no_ack_interesting():
    """skip 路径：interesting_set 中的条目不 ACK"""
    llm = FakeLLM([("finish_turn", {"decision": "skip", "reason": "user_busy"})])
    tick, sink = _make_tick_with_sink(llm)
    # interesting_set 为空，sink 应无调用
    await tick.tick()
    assert sink.calls == []


@pytest.mark.asyncio
async def test_no_terminal_action_no_ack_cited():
    """LLM 返回 None，无 terminal → 不 ACK cited"""
    llm = FakeLLM([])
    tick, sink = _make_tick_with_sink(llm)
    tick.last_ctx = None
    await tick.tick()
    assert sink.calls == []


# ── context-only ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_only_marks_state_on_success():
    from tests.proactive_v2.conftest import FakeRng
    state = FakeStateStore()
    state.set_context_only_count(0)
    state.set_last_context_only_at(None)

    llm = FakeLLM([
        ("message_push", {"message": "steam update", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    sink = FakeAckSink()
    deps = ToolDeps(
        recent_chat_fn=AsyncMock(return_value=[]),
        ack_fn=sink,
    )
    gateway = GatewayDeps(
        alert_fn=AsyncMock(return_value=[]),
        feed_fn=AsyncMock(return_value=[]),
    )
    sender = AsyncMock()
    sender.send.return_value = True
    deduper = AsyncMock()
    deduper.is_duplicate = AsyncMock(return_value=(False, ""))

    tick = make_agent_tick(
        llm_fn=llm,
        state_store=state,
        sender=sender,
        deduper=deduper,
        tool_deps=deps,
        gateway_deps=gateway,
        cfg=cfg_with(agent_tick_context_prob=1.0, context_only_daily_max=3),
        rng=FakeRng(value=0.0),  # random() < 1.0 → gate 开
    )
    await tick.tick()

    assert state.context_only_send_marked is True


@pytest.mark.asyncio
async def test_context_only_not_marked_when_cited_ids_present():
    """cited_ids 非空说明是 content/alert 路径，不算 context-only"""
    from tests.proactive_v2.conftest import FakeRng
    state = FakeStateStore()
    state.set_context_only_count(0)
    state.set_last_context_only_at(None)

    llm = FakeLLM([
        ("message_push", {"message": "msg", "evidence": ["feed-mcp:1"]}),
        ("finish_turn", {"decision": "reply"}),
    ])
    sink = FakeAckSink()
    deps = ToolDeps(
        recent_chat_fn=AsyncMock(return_value=[]),
        ack_fn=sink,
    )
    gateway = GatewayDeps(
        alert_fn=AsyncMock(return_value=[]),
        feed_fn=AsyncMock(return_value=[]),
    )
    sender = AsyncMock()
    sender.send.return_value = True
    deduper = AsyncMock()
    deduper.is_duplicate = AsyncMock(return_value=(False, ""))

    tick = make_agent_tick(
        llm_fn=llm,
        state_store=state,
        sender=sender,
        deduper=deduper,
        tool_deps=deps,
        gateway_deps=gateway,
        cfg=cfg_with(agent_tick_context_prob=1.0, context_only_daily_max=3),
        rng=FakeRng(value=0.0),
    )
    await tick.tick()

    assert state.context_only_send_marked is False


# ── cited vs discarded conflict ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_cited_wins_over_discarded_gets_168h_not_720h():
    """先 mark_not_interesting 再 send_message → cited 优先，168h 而非 720h"""
    event = {"id": "c1", "ack_server": "feed-mcp"}
    llm = FakeLLM([
        ("get_content_events", {}),
        ("mark_not_interesting", {"item_ids": ["feed-mcp:c1"]}),  # discarded
        ("message_push", {"message": "actually good", "evidence": ["feed-mcp:c1"]}),  # cited wins
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(
        llm, tool_deps_extra={"feed_fn": AsyncMock(return_value=[event])}
    )
    await tick.tick()

    # cited 优先 → 168h；不应有 720h
    assert sink.acked("feed-mcp:c1", 168)
    assert 720 not in sink.ttls_for("feed-mcp:c1")


# ── alert ACK 语义（§20：post-guard 失败时 alert 不 ACK）────────────────────

@pytest.mark.asyncio
async def test_ack_post_guard_fail_alert_cited_uses_alert_ack_fn():
    """post-guard 失败时，alert cited key → alert_ack_fn（独立通道，无 TTL），不走普通 ack_fn。"""
    ctx = AgentTickContext()
    ctx.fetched_alerts = [{"id": "a1", "ack_server": "alert-mcp"}]
    ctx.cited_item_ids = ["alert-mcp:a1"]
    ctx.interesting_item_ids = {"alert-mcp:a1"}
    sink = FakeAckSink()
    alert_sink = FakeAlertAckSink()
    await ack_post_guard_fail(ctx, sink, alert_ack_fn=alert_sink)
    assert alert_sink.called_with("alert-mcp:a1")  # alert 走独立通道
    assert sink.not_acked("alert-mcp:a1")           # 不走普通 ack_fn


@pytest.mark.asyncio
async def test_ack_post_guard_fail_alert_cited_fallback_to_ack_fn_when_no_alert_ack_fn():
    """post-guard 失败，无 alert_ack_fn 时，alert cited → ack_fn 24h（回退）。"""
    ctx = AgentTickContext()
    ctx.fetched_alerts = [{"id": "a1", "ack_server": "alert-mcp"}]
    ctx.cited_item_ids = ["alert-mcp:a1"]
    ctx.interesting_item_ids = {"alert-mcp:a1"}
    sink = FakeAckSink()
    await ack_post_guard_fail(ctx, sink)  # alert_ack_fn=None
    assert sink.acked("alert-mcp:a1", 24)


@pytest.mark.asyncio
async def test_ack_post_guard_fail_content_cited_and_alert_cited_separate_channels():
    """post-guard 失败：content cited → ack_fn 24h；alert cited → alert_ack_fn。"""
    ctx = AgentTickContext()
    ctx.fetched_alerts = [{"id": "a1", "ack_server": "alert-mcp"}]
    ctx.fetched_contents = [{"id": "c1", "ack_server": "feed-mcp"}]
    ctx.cited_item_ids = ["alert-mcp:a1", "feed-mcp:c1"]
    ctx.interesting_item_ids = {"alert-mcp:a1", "feed-mcp:c1"}
    sink = FakeAckSink()
    alert_sink = FakeAlertAckSink()
    await ack_post_guard_fail(ctx, sink, alert_ack_fn=alert_sink)
    assert sink.acked("feed-mcp:c1", 24)           # content → 24h
    assert alert_sink.called_with("alert-mcp:a1")  # alert → 独立通道
    assert sink.not_acked("alert-mcp:a1")           # alert 不重复走 ack_fn


@pytest.mark.asyncio
async def test_ack_post_guard_fail_uncited_fetched_alert_also_acked():
    """post-guard 失败：uncited fetched alert 也应被 ACK，一次性清空本批 alert，
    防止逐条 tick 循环（每 tick 换一条 alert 反复被 dedupe 拦截）。"""
    ctx = AgentTickContext()
    ctx.fetched_alerts = [
        {"id": "a1", "ack_server": "alert-mcp"},
        {"id": "a2", "ack_server": "alert-mcp"},
    ]
    ctx.cited_item_ids = ["alert-mcp:a1"]  # 只 cite 了 a1
    ctx.interesting_item_ids = {"alert-mcp:a1"}
    sink = FakeAckSink()
    alert_sink = FakeAlertAckSink()
    await ack_post_guard_fail(ctx, sink, alert_ack_fn=alert_sink)
    assert alert_sink.called_with("alert-mcp:a1")  # cited alert → ack
    assert alert_sink.called_with("alert-mcp:a2")  # uncited alert → 也 ack，不留给下一 tick


# ── message_dedupe 传真实 recent_proactive（§17）──────────────────────────

@pytest.mark.asyncio
async def test_message_dedupe_receives_recent_proactive_list():
    """_post_loop 应将 recent_proactive_fn() 的结果传给 deduper.is_duplicate（同步函数）"""
    deduper = AsyncMock()
    deduper.is_duplicate = AsyncMock(return_value=(False, ""))

    recent_msgs = [{"role": "assistant", "text": "上次发的消息"}]
    recent_proactive_fn = lambda: recent_msgs  # 同步函数，不能 await

    llm = FakeLLM([
        ("message_push", {"message": "new message", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    sink = FakeAckSink()
    deps = ToolDeps(
        recent_chat_fn=AsyncMock(return_value=[]),
        ack_fn=sink,
    )
    gateway = GatewayDeps(
        alert_fn=AsyncMock(return_value=[]),
        feed_fn=AsyncMock(return_value=[]),
    )
    sender = AsyncMock()
    sender.send.return_value = True

    tick = make_agent_tick(
        llm_fn=llm,
        deduper=deduper,
        sender=sender,
        tool_deps=deps,
        gateway_deps=gateway,
        recent_proactive_fn=recent_proactive_fn,
    )
    await tick.tick()

    call_kwargs = deduper.is_duplicate.call_args[1]
    assert call_kwargs["recent_proactive"] == recent_msgs


@pytest.mark.asyncio
async def test_message_dedupe_empty_list_when_no_fn():
    """未传 recent_proactive_fn 时，recent_proactive 为空列表"""
    deduper = AsyncMock()
    deduper.is_duplicate = AsyncMock(return_value=(False, ""))

    llm = FakeLLM([
        ("message_push", {"message": "msg", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(llm, deduper=deduper)
    await tick.tick()

    call_kwargs = deduper.is_duplicate.call_args[1]
    assert call_kwargs["recent_proactive"] == []


# ── Fix 3: interesting_item_ids 语义（fetched content - discarded）─────────

@pytest.mark.asyncio
async def test_discarded_content_not_in_interesting():
    """mark_not_interesting 后未 cite → discarded 720h，不应出现 24h"""
    event = {"id": "c1", "ack_server": "feed-mcp"}
    llm = FakeLLM([
        ("get_content_events", {}),
        ("mark_not_interesting", {"item_ids": ["feed-mcp:c1"]}),
        ("message_push", {"message": "hi", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(
        llm, tool_deps_extra={"feed_fn": AsyncMock(return_value=[event])}
    )
    await tick.tick()
    assert sink.acked("feed-mcp:c1", 720)
    assert 24 not in sink.ttls_for("feed-mcp:c1")


# ── Fix 4: alert_ack_fn 独立通道（§20：成功时 alert cited 走独立 ack_fn）──

@pytest.mark.asyncio
async def test_ack_on_success_alert_cited_calls_alert_ack_fn():
    """发送成功：cited alert 调用 alert_ack_fn（独立通道），不调用普通 ack_fn"""
    from proactive_v2.agent_tick import ack_on_success
    ctx = AgentTickContext()
    ctx.fetched_alerts = [{"id": "a1", "ack_server": "alert-mcp"}]
    ctx.fetched_contents = []
    ctx.cited_item_ids = ["alert-mcp:a1"]
    ctx.interesting_item_ids = set()
    ctx.discarded_item_ids = set()

    regular_sink = FakeAckSink()
    alert_sink = FakeAlertAckSink()
    await ack_on_success(ctx, regular_sink, alert_ack_fn=alert_sink)

    assert alert_sink.all_keys() == {"alert-mcp:a1"}   # alert 走独立通道
    assert regular_sink.not_acked("alert-mcp:a1")       # 普通 ack_fn 不调用


@pytest.mark.asyncio
async def test_ack_on_success_alert_ack_fn_none_falls_back_to_regular():
    """alert_ack_fn=None 时，cited alert 回退到普通 ack_fn（168h）"""
    from proactive_v2.agent_tick import ack_on_success
    ctx = AgentTickContext()
    ctx.fetched_alerts = [{"id": "a1", "ack_server": "alert-mcp"}]
    ctx.fetched_contents = []
    ctx.cited_item_ids = ["alert-mcp:a1"]
    ctx.interesting_item_ids = set()
    ctx.discarded_item_ids = set()

    regular_sink = FakeAckSink()
    await ack_on_success(ctx, regular_sink, alert_ack_fn=None)

    assert regular_sink.acked("alert-mcp:a1", 168)  # 回退到 168h


@pytest.mark.asyncio
async def test_ack_on_success_content_unaffected_by_alert_ack_fn():
    """alert_ack_fn 独立时，content cited 仍走普通 ack_fn（168h）"""
    from proactive_v2.agent_tick import ack_on_success
    ctx = AgentTickContext()
    ctx.fetched_alerts = []
    ctx.fetched_contents = [{"id": "c1", "ack_server": "feed-mcp"}]
    ctx.cited_item_ids = ["feed-mcp:c1"]
    ctx.interesting_item_ids = set()
    ctx.discarded_item_ids = set()

    regular_sink = FakeAckSink()
    alert_sink = FakeAlertAckSink()
    await ack_on_success(ctx, regular_sink, alert_ack_fn=alert_sink)

    assert regular_sink.acked("feed-mcp:c1", 168)
    assert alert_sink.keys == []  # alert_ack_fn 未被调用


# ── Fix 1: TOOL_SCHEMAS 应使用 OpenAI 格式 ────────────────────────────────

def test_tool_schemas_have_openai_format():
    """每个 schema 必须是 {"type":"function","function":{name,description,parameters}} 格式"""
    from proactive_v2.tools import TOOL_SCHEMAS
    for schema in TOOL_SCHEMAS:
        assert schema.get("type") == "function", f"missing type=function: {schema.get('name', schema)}"
        fn = schema.get("function", {})
        assert "name" in fn, f"function missing name: {schema}"
        assert "description" in fn, f"function missing description: {schema}"
        assert "parameters" in fn, f"function missing parameters: {schema}"


def test_tool_schemas_no_input_schema_key():
    """不应有 Anthropic 风格的 input_schema 顶层 key"""
    from proactive_v2.tools import TOOL_SCHEMAS
    for schema in TOOL_SCHEMAS:
        assert "input_schema" not in schema, f"Anthropic-style input_schema found: {schema}"
        assert "name" not in schema or schema.get("type") == "function", \
            f"bare name at top level (Anthropic style): {schema}"


# ── Fix 1: _run_loop 应追加 OpenAI 格式消息 ───────────────────────────────

@pytest.mark.asyncio
async def test_run_loop_appends_openai_format_tool_messages():
    """_run_loop 工具调用消息：assistant.tool_calls + role=tool（非 Anthropic content=[tool_use]）"""
    llm = FakeLLM([
        ("get_content_events", {}),      # step 1 → produces messages for step 2
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),  # step 2 → terminates
    ])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()

    # 第二次 LLM 调用收到的 messages 包含 step 1 的工具调用和结果
    assert len(llm.calls) >= 2, "Expected at least 2 LLM calls"
    msgs = llm.calls[1]  # second call

    # assistant message: 必须有 tool_calls，不能是 Anthropic content=[{type:tool_use}]
    assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    asst = assistant_msgs[0]
    assert "tool_calls" in asst, "assistant message must have tool_calls (OpenAI format)"
    assert asst["tool_calls"][0].get("type") == "function"
    assert "function" in asst["tool_calls"][0]
    # 不应有 Anthropic 风格的 content 列表
    content = asst.get("content")
    assert not isinstance(content, list), "assistant content must not be list (Anthropic style)"

    # tool result: role=tool，有 tool_call_id（非 Anthropic role=user content=[tool_result]）
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "tool_call_id" in tool_msgs[0], "tool result must have tool_call_id (OpenAI format)"


# ── Fix 2: mark_interesting 工具（§10 显式分类）─────────────────────────

def test_mark_interesting_tool_in_schema():
    """TOOL_SCHEMAS 必须包含 mark_interesting 工具"""
    from proactive_v2.tools import TOOL_SCHEMAS
    names = {s["function"]["name"] for s in TOOL_SCHEMAS if s.get("type") == "function"}
    assert "mark_interesting" in names, "mark_interesting tool missing from TOOL_SCHEMAS"


@pytest.mark.asyncio
async def test_mark_interesting_uncited_acks_24h_on_success():
    """mark_interesting 后未引用 → 发送成功后 24h ACK（uncited interesting）"""
    event = {"id": "c1", "ack_server": "feed-mcp"}
    llm = FakeLLM([
        ("get_content_events", {}),
        ("mark_interesting", {"item_ids": ["feed-mcp:c1"]}),  # 显式标记感兴趣
        ("message_push", {"message": "hi", "evidence": []}),     # 未引用
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(
        llm, tool_deps_extra={"feed_fn": AsyncMock(return_value=[event])}
    )
    await tick.tick()
    assert sink.acked("feed-mcp:c1", 24)


@pytest.mark.asyncio
async def test_fetched_but_unclassified_not_acked_on_skip():
    """拉取了 content metadata 但未分类 → skip 时不 ACK（unclassified）"""
    event = {"id": "c1", "ack_server": "feed-mcp"}
    llm = FakeLLM([
        ("get_content_events", {}),
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
    ])
    tick, sink = _make_tick_with_sink(
        llm, tool_deps_extra={"feed_fn": AsyncMock(return_value=[event])}
    )
    await tick.tick()
    assert sink.not_acked("feed-mcp:c1")


@pytest.mark.asyncio
async def test_fetched_but_unclassified_not_acked_on_send():
    """拉取了 content 但未分类 → send_message 时也不 ACK（未经 mark_interesting 的不算 interesting）"""
    event = {"id": "c1", "ack_server": "feed-mcp"}
    llm = FakeLLM([
        ("get_content_events", {}),
        # 未调用 mark_interesting，也未调用 mark_not_interesting
        ("message_push", {"message": "other topic", "evidence": []}),  # c1 未 cite
        ("finish_turn", {"decision": "reply"}),
    ])
    tick, sink = _make_tick_with_sink(
        llm, tool_deps_extra={"feed_fn": AsyncMock(return_value=[event])}
    )
    await tick.tick()
    assert sink.not_acked("feed-mcp:c1")
