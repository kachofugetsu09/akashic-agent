"""
TDD — Phase 1: proactive_v2/context.py

测试覆盖：
  - AgentTickContext 默认值
  - 复合键格式（ack_server:id）
  - interesting/discarded 集合语义
  - context_as_fallback_open 字段用途
  - _fetch 保护字段
"""

from datetime import datetime, timezone

import pytest

from proactive_v2.context import AgentTickContext


# ── 默认值 ────────────────────────────────────────────────────────────────

def test_tick_id_is_8_chars():
    ctx = AgentTickContext()
    assert len(ctx.tick_id) == 8


def test_tick_id_is_alphanumeric():
    ctx = AgentTickContext()
    assert ctx.tick_id.isalnum()


def test_tick_ids_are_unique():
    ids = {AgentTickContext().tick_id for _ in range(100)}
    assert len(ids) == 100


def test_now_utc_is_timezone_aware():
    ctx = AgentTickContext()
    assert ctx.now_utc.tzinfo is not None


def test_now_utc_is_utc():
    ctx = AgentTickContext()
    offset = ctx.now_utc.utcoffset().total_seconds()
    assert offset == 0.0


def test_session_key_defaults_empty():
    ctx = AgentTickContext()
    assert ctx.session_key == ""


def test_context_as_fallback_open_defaults_false():
    ctx = AgentTickContext()
    assert ctx.context_as_fallback_open is False


def test_terminal_action_defaults_none():
    ctx = AgentTickContext()
    assert ctx.terminal_action is None


def test_skip_reason_defaults_empty():
    ctx = AgentTickContext()
    assert ctx.skip_reason == ""


def test_skip_note_defaults_empty():
    ctx = AgentTickContext()
    assert ctx.skip_note == ""


def test_final_message_defaults_empty():
    ctx = AgentTickContext()
    assert ctx.final_message == ""


def test_cited_item_ids_defaults_empty_list():
    ctx = AgentTickContext()
    assert ctx.cited_item_ids == []


def test_steps_taken_defaults_zero():
    ctx = AgentTickContext()
    assert ctx.steps_taken == 0


# ── 集合字段默认值 ────────────────────────────────────────────────────────

def test_interesting_item_ids_defaults_empty_set():
    ctx = AgentTickContext()
    assert ctx.interesting_item_ids == set()


def test_discarded_item_ids_defaults_empty_set():
    ctx = AgentTickContext()
    assert ctx.discarded_item_ids == set()


def test_fetched_alerts_defaults_empty_list():
    ctx = AgentTickContext()
    assert ctx.fetched_alerts == []


def test_fetched_contents_defaults_empty_list():
    ctx = AgentTickContext()
    assert ctx.fetched_contents == []


def test_fetched_context_defaults_empty_list():
    ctx = AgentTickContext()
    assert ctx.fetched_context == []


# ── _fetch 保护字段 ───────────────────────────────────────────────────────

def test_alerts_fetched_defaults_false():
    ctx = AgentTickContext()
    assert ctx._alerts_fetched is False


def test_contents_fetched_defaults_false():
    ctx = AgentTickContext()
    assert ctx._contents_fetched is False


def test_context_fetched_defaults_false():
    ctx = AgentTickContext()
    assert ctx._context_fetched is False


# ── 实例隔离（shared default mutable 不能共享） ────────────────────────────

def test_interesting_set_is_independent_per_instance():
    ctx1 = AgentTickContext()
    ctx2 = AgentTickContext()
    ctx1.interesting_item_ids.add("feed-mcp:1")
    assert "feed-mcp:1" not in ctx2.interesting_item_ids


def test_discarded_set_is_independent_per_instance():
    ctx1 = AgentTickContext()
    ctx2 = AgentTickContext()
    ctx1.discarded_item_ids.add("feed-mcp:2")
    assert "feed-mcp:2" not in ctx2.discarded_item_ids


def test_cited_item_ids_is_independent_per_instance():
    ctx1 = AgentTickContext()
    ctx2 = AgentTickContext()
    ctx1.cited_item_ids.append("feed-mcp:3")
    assert ctx2.cited_item_ids == []


def test_fetched_alerts_is_independent_per_instance():
    ctx1 = AgentTickContext()
    ctx2 = AgentTickContext()
    ctx1.fetched_alerts.append({"id": "a1"})
    assert ctx2.fetched_alerts == []


# ── 复合键格式语义 ────────────────────────────────────────────────────────

def test_compound_key_format_from_event():
    event = {"id": "abc123", "ack_server": "feed-mcp"}
    key = f"{event['ack_server']}:{event['id']}"
    assert key == "feed-mcp:abc123"


def test_alert_content_same_id_no_collision():
    """alert-mcp 和 feed-mcp 即使 id 相同，复合键不碰撞"""
    alert_key = f"alert-mcp:42"
    content_key = f"feed-mcp:42"
    assert alert_key != content_key


def test_ack_split_by_set_intersection():
    """复合键可通过集合交集安全拆分 alert / content ACK"""
    fetched_alert_keys = {"alert-mcp:a1", "alert-mcp:a2"}
    fetched_content_keys = {"feed-mcp:c1", "feed-mcp:c2"}
    cited_set = {"alert-mcp:a1", "feed-mcp:c1"}

    cited_alert = cited_set & fetched_alert_keys
    cited_content = cited_set & fetched_content_keys

    assert cited_alert == {"alert-mcp:a1"}
    assert cited_content == {"feed-mcp:c1"}
    assert cited_alert.isdisjoint(cited_content)


def test_interesting_set_stores_compound_keys():
    ctx = AgentTickContext()
    ctx.interesting_item_ids.add("feed-mcp:item1")
    ctx.interesting_item_ids.add("alert-mcp:item2")
    assert "feed-mcp:item1" in ctx.interesting_item_ids
    assert "alert-mcp:item2" in ctx.interesting_item_ids


# ── terminal_action Literal 类型值 ───────────────────────────────────────

def test_terminal_action_accepts_reply():
    ctx = AgentTickContext()
    ctx.terminal_action = "reply"
    assert ctx.terminal_action == "reply"


def test_terminal_action_accepts_skip():
    ctx = AgentTickContext()
    ctx.terminal_action = "skip"
    assert ctx.terminal_action == "skip"


# ── context_as_fallback_open：ACK 阶段使用，agent 不写 ────────────────────

def test_context_as_fallback_open_can_be_set():
    ctx = AgentTickContext()
    ctx.context_as_fallback_open = True
    assert ctx.context_as_fallback_open is True


def test_context_as_fallback_open_false_by_default_for_each_new_ctx():
    for _ in range(10):
        assert AgentTickContext().context_as_fallback_open is False
