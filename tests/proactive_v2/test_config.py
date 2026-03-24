"""
TDD — Phase 2: proactive/config.py v2 新字段

测试覆盖：
  - 新增 v2 字段的默认值
  - v2-only 语义（不再存在 use_agent_tick）
  - 复用 v1 已有字段（不重复定义）
  - 字段类型正确
"""

import pytest

from proactive.config import ProactiveConfig


# ── v2-only ───────────────────────────────────────────────────────────────

def test_use_agent_tick_removed():
    assert not hasattr(ProactiveConfig(), "use_agent_tick")


# ── agent loop ───────────────────────────────────────────────────────────

def test_agent_tick_max_steps_default():
    assert ProactiveConfig().agent_tick_max_steps == 20


def test_agent_tick_max_steps_is_int():
    assert isinstance(ProactiveConfig().agent_tick_max_steps, int)


def test_agent_tick_model_default_empty():
    assert ProactiveConfig().agent_tick_model == ""


def test_agent_tick_content_limit_default():
    assert ProactiveConfig().agent_tick_content_limit == 5


def test_agent_tick_content_limit_is_int():
    assert isinstance(ProactiveConfig().agent_tick_content_limit, int)


# ── web_fetch 截断 ────────────────────────────────────────────────────────

def test_agent_tick_web_fetch_max_chars_default():
    assert ProactiveConfig().agent_tick_web_fetch_max_chars == 8_000


def test_agent_tick_web_fetch_max_chars_is_int():
    assert isinstance(ProactiveConfig().agent_tick_web_fetch_max_chars, int)


def test_web_fetch_max_chars_less_than_50000():
    """必须远小于 WebFetchTool 上限 50000，否则 token 预算失效"""
    assert ProactiveConfig().agent_tick_web_fetch_max_chars < 50_000


# ── context gate ─────────────────────────────────────────────────────────

def test_agent_tick_context_prob_default():
    assert ProactiveConfig().agent_tick_context_prob == 0.03


def test_agent_tick_context_prob_is_float():
    assert isinstance(ProactiveConfig().agent_tick_context_prob, float)


def test_agent_tick_context_prob_in_range():
    prob = ProactiveConfig().agent_tick_context_prob
    assert 0.0 <= prob <= 1.0


# ── delivery cooldown ─────────────────────────────────────────────────────

def test_agent_tick_delivery_cooldown_hours_default():
    assert ProactiveConfig().agent_tick_delivery_cooldown_hours == 1


def test_agent_tick_delivery_cooldown_hours_is_int():
    assert isinstance(ProactiveConfig().agent_tick_delivery_cooldown_hours, int)


# ── 复用 v1 已有字段（不重复定义） ────────────────────────────────────────

def test_delivery_dedupe_hours_exists():
    assert hasattr(ProactiveConfig(), "delivery_dedupe_hours")


def test_context_only_daily_max_exists():
    assert hasattr(ProactiveConfig(), "context_only_daily_max")


def test_context_only_min_interval_hours_exists():
    assert hasattr(ProactiveConfig(), "context_only_min_interval_hours")


def test_message_dedupe_enabled_exists():
    assert hasattr(ProactiveConfig(), "message_dedupe_enabled")


def test_message_dedupe_recent_n_exists():
    assert hasattr(ProactiveConfig(), "message_dedupe_recent_n")


# ── 字段可被覆盖（dataclass override） ────────────────────────────────────


def test_max_steps_can_be_overridden():
    cfg = ProactiveConfig(agent_tick_max_steps=10)
    assert cfg.agent_tick_max_steps == 10


def test_web_fetch_max_chars_can_be_overridden():
    cfg = ProactiveConfig(agent_tick_web_fetch_max_chars=4_000)
    assert cfg.agent_tick_web_fetch_max_chars == 4_000


def test_model_can_be_overridden():
    cfg = ProactiveConfig(agent_tick_model="claude-sonnet-4-6")
    assert cfg.agent_tick_model == "claude-sonnet-4-6"


# ── v2 字段不影响 v1 字段默认值 ──────────────────────────────────────────

def test_v1_enabled_still_false_by_default():
    assert ProactiveConfig().enabled is False


def test_v1_delivery_dedupe_hours_unchanged():
    assert ProactiveConfig().delivery_dedupe_hours == 24


def test_v1_context_only_daily_max_unchanged():
    assert ProactiveConfig().context_only_daily_max == 1
