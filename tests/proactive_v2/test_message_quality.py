"""
tests/proactive_v2/test_message_quality.py

验证两个消息质量 fix：
1. get_recent_chat 只返回 role=user 消息，不含 bot 自身历史推送（role=assistant）
2. 系统提示【发送要求】包含禁止脑补具体数字/比分/排名等事实的硬规则
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from proactive_v2.context import AgentTickContext
from proactive_v2.tools import _get_recent_chat


# ── Fix 1: get_recent_chat 过滤 role=assistant ────────────────────────────


@pytest.mark.asyncio
async def test_get_recent_chat_filters_out_proactive_pushes():
    """主动推送（proactive=True）不应出现在结果中，防止 bot 循环引用自己的推送。"""
    mixed = [
        {"role": "user", "content": "在吗"},
        {"role": "assistant", "content": "G2 刚赢了 FaZe，bo3 2-1", "proactive": True},
        {"role": "user", "content": "好的"},
        {"role": "assistant", "content": "心率 130 偏高", "proactive": True},
    ]
    fake_chat_fn = AsyncMock(return_value=mixed)
    ctx = AgentTickContext()

    raw = await _get_recent_chat(ctx, {"n": 20}, recent_chat_fn=fake_chat_fn)
    result = json.loads(raw)

    proactive_msgs = [m for m in result if m.get("proactive")]
    assert not proactive_msgs, (
        "get_recent_chat 不应返回 proactive=True 的消息，"
        f"实际返回了：{proactive_msgs}"
    )


@pytest.mark.asyncio
async def test_get_recent_chat_keeps_passive_assistant_replies():
    """被动回复（role=assistant, proactive 为假）应该保留，代表用户主动发起的对话上下文。"""
    mixed = [
        {"role": "user", "content": "G2 最近状态怎样"},
        {"role": "assistant", "content": "G2 最近手感不错，jks 发挥稳定"},  # 被动回复，无 proactive 字段
        {"role": "user", "content": "了解"},
        {"role": "assistant", "content": "CS2 比赛结果", "proactive": True},  # 主动推送
    ]
    fake_chat_fn = AsyncMock(return_value=mixed)
    ctx = AgentTickContext()

    raw = await _get_recent_chat(ctx, {}, recent_chat_fn=fake_chat_fn)
    result = json.loads(raw)

    contents = [m["content"] for m in result]
    assert "G2 最近手感不错，jks 发挥稳定" in contents, "被动回复应该被保留"
    assert "CS2 比赛结果" not in contents, "主动推送应该被过滤"


@pytest.mark.asyncio
async def test_get_recent_chat_empty_after_filtering_all_proactive():
    """全部是主动推送时，返回空列表。"""
    all_proactive = [
        {"role": "assistant", "content": "push 1", "proactive": True},
        {"role": "assistant", "content": "push 2", "proactive": True},
    ]
    fake_chat_fn = AsyncMock(return_value=all_proactive)
    ctx = AgentTickContext()

    raw = await _get_recent_chat(ctx, {}, recent_chat_fn=fake_chat_fn)
    result = json.loads(raw)

    assert result == []


@pytest.mark.asyncio
async def test_get_recent_chat_all_user_messages_pass_through():
    """没有 assistant 消息时，全量返回用户消息。"""
    user_only = [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "ok"},
    ]
    fake_chat_fn = AsyncMock(return_value=user_only)
    ctx = AgentTickContext()

    raw = await _get_recent_chat(ctx, {}, recent_chat_fn=fake_chat_fn)
    result = json.loads(raw)

    assert result == user_only


@pytest.mark.asyncio
async def test_get_recent_chat_mixed_passive_and_proactive():
    """被动回复和主动推送混合时，只过滤主动推送，被动回复完整保留。"""
    mixed = [
        {"role": "user", "content": "最近睡眠怎样"},
        {"role": "assistant", "content": "昨晚深睡时间偏少，可能跟熬夜有关"},  # 被动回复
        {"role": "user", "content": "嗯"},
        {"role": "assistant", "content": "心率偏高预警", "proactive": True},   # 主动推送
        {"role": "user", "content": "看到了"},
    ]
    fake_chat_fn = AsyncMock(return_value=mixed)
    ctx = AgentTickContext()

    raw = await _get_recent_chat(ctx, {}, recent_chat_fn=fake_chat_fn)
    result = json.loads(raw)

    assert len(result) == 4  # 3 user + 1 passive reply
    proactive_msgs = [m for m in result if m.get("proactive")]
    assert not proactive_msgs


# ── Fix 2: 系统提示【发送要求】包含脑补禁止规则 ──────────────────────────


def _make_system_prompt() -> str:
    from proactive_v2.agent_tick import AgentTick
    from proactive_v2.gateway import GatewayResult
    from proactive.config import ProactiveConfig
    from unittest.mock import MagicMock

    gate = MagicMock()
    gate.should_act.return_value = (True, {})

    tick = AgentTick(
        cfg=ProactiveConfig(),
        session_key="test",
        state_store=MagicMock(),
        any_action_gate=gate,
        last_user_at_fn=lambda: None,
        passive_busy_fn=None,
        deduper=MagicMock(),
        tool_deps=__import__("proactive_v2.tools", fromlist=["ToolDeps"]).ToolDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[]),
            context_fn=AsyncMock(return_value=[]),
            recent_chat_fn=AsyncMock(return_value=[]),
        ),
        llm_fn=None,
    )
    ctx = AgentTickContext(session_key="test", now_utc=datetime.now(timezone.utc))
    gw = GatewayResult()
    return tick._build_system_prompt(ctx, gw)


def test_system_prompt_contains_no_hallucination_rule():
    """【发送要求】必须有明确的脑补禁止规则，覆盖数字/比分/排名等。"""
    prompt = _make_system_prompt()
    assert "脑补" in prompt, "发送要求应包含'脑补'禁止规则"
    # 确保是在发送要求附近
    idx = prompt.find("【发送要求】")
    assert idx >= 0, "系统提示应有【发送要求】章节"
    send_section = prompt[idx:]
    assert "脑补" in send_section, "脑补禁止规则应位于【发送要求】章节内"


def test_system_prompt_no_hallucination_covers_specific_facts():
    """脑补禁止规则应明确覆盖比分、排名等可验证事实，而不是空泛的禁止。"""
    prompt = _make_system_prompt()
    idx = prompt.find("【发送要求】")
    send_section = prompt[idx:]

    # 应至少命中以下关键词之一（排名、比分、数字）
    keywords = ["排名", "比分", "数字", "结果"]
    matched = [kw for kw in keywords if kw in send_section]
    assert matched, (
        f"【发送要求】的脑补禁止规则应包含 {keywords} 中至少一个关键词，"
        f"实际发送要求内容：{send_section[:400]}"
    )


def test_system_prompt_no_hallucination_requires_data_source():
    """禁止脑补的规则应明确说明事实必须来自本轮数据。"""
    prompt = _make_system_prompt()
    idx = prompt.find("【发送要求】")
    send_section = prompt[idx:]

    assert "本轮" in send_section or "Alerts" in send_section or "Content" in send_section, (
        "脑补禁止规则应说明事实必须来自本轮 Alerts/Content 数据"
    )
