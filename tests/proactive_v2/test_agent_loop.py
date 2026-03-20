"""
TDD — Phase 5: proactive_v2/agent_tick.py — Agent Loop

测试覆盖：
  - max_steps 保护：loop 不超过 agent_tick_max_steps
  - 终止工具（send_message/skip）调用后 loop 立即结束
  - LLM 返回 None → loop 结束
  - llm_fn=None → loop 不执行任何工具
  - Alert 路径：ctx.terminal_action="send" + cited_ids
  - Content 路径：interesting_set 从 cited_ids 推断
  - mark_not_interesting 在 loop 内写 discarded_set
  - skip(user_busy) 路径
  - LLM 消息历史：工具结果追加到 messages
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from proactive_v2.tools import ToolDeps
from tests.proactive_v2.conftest import FakeLLM, cfg_with, make_agent_tick


# ── max_steps 保护 ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_loop_stops_at_max_steps():
    """LLM 一直返回非终止工具 → loop 在 max_steps 处退出"""
    llm = FakeLLM([("get_recent_chat", {})] * 25)
    tick = make_agent_tick(
        llm_fn=llm,
        cfg=cfg_with(agent_tick_max_steps=20),
        tool_deps=ToolDeps(recent_chat_fn=AsyncMock(return_value=[])),
    )
    await tick.tick()
    assert tick.last_ctx.steps_taken == 20
    assert tick.last_ctx.terminal_action is None


@pytest.mark.asyncio
async def test_loop_max_steps_configurable():
    llm = FakeLLM([("get_recent_chat", {})] * 15)
    tick = make_agent_tick(
        llm_fn=llm,
        cfg=cfg_with(agent_tick_max_steps=5),
        tool_deps=ToolDeps(recent_chat_fn=AsyncMock(return_value=[])),
    )
    await tick.tick()
    assert tick.last_ctx.steps_taken == 5


# ── LLM 返回 None → 结束 ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_loop_stops_when_llm_returns_none():
    llm = FakeLLM([])  # 空序列，第一次就返回 None
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    assert tick.last_ctx.steps_taken == 0
    assert tick.last_ctx.terminal_action is None


@pytest.mark.asyncio
async def test_loop_stops_after_partial_sequence_then_none():
    llm = FakeLLM([
        ("get_alert_events", {}),
        ("get_content_events", {}),
        # 之后 None，loop 结束
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[]),
        ),
    )
    await tick.tick()
    assert tick.last_ctx.steps_taken == 2
    assert tick.last_ctx.terminal_action is None


# ── llm_fn=None → loop 不执行任何工具 ────────────────────────────────────

@pytest.mark.asyncio
async def test_loop_with_no_llm_fn_executes_nothing():
    tick = make_agent_tick(llm_fn=None)
    await tick.tick()
    assert tick.last_ctx.steps_taken == 0


# ── 终止工具立即结束 loop ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_message_stops_loop_immediately():
    llm = FakeLLM([
        ("send_message", {"text": "Hello!", "cited_ids": []}),
        ("get_recent_chat", {}),   # 不应执行
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(recent_chat_fn=AsyncMock(return_value=[])),
    )
    await tick.tick()
    # send_message 是第 1 步，之后 loop 停止
    assert tick.last_ctx.steps_taken == 1
    assert tick.last_ctx.terminal_action == "send"


@pytest.mark.asyncio
async def test_skip_stops_loop_immediately():
    llm = FakeLLM([
        ("skip", {"reason": "no_content"}),
        ("get_recent_chat", {}),   # 不应执行
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(recent_chat_fn=AsyncMock(return_value=[])),
    )
    await tick.tick()
    assert tick.last_ctx.steps_taken == 1
    assert tick.last_ctx.terminal_action == "skip"


@pytest.mark.asyncio
async def test_only_first_terminal_counts():
    """send_message 之后即使 LLM 想再 skip，也不会被执行"""
    llm = FakeLLM([
        ("send_message", {"text": "Hi", "cited_ids": []}),
        ("skip", {"reason": "no_content"}),
    ])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    assert tick.last_ctx.terminal_action == "send"
    assert tick.last_ctx.steps_taken == 1


# ── send_message 写 ctx ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_message_writes_final_message():
    llm = FakeLLM([
        ("send_message", {"text": "Hello world!", "cited_ids": []}),
    ])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    assert tick.last_ctx.final_message == "Hello world!"


@pytest.mark.asyncio
async def test_send_message_writes_cited_ids():
    llm = FakeLLM([
        ("send_message", {"text": "msg", "cited_ids": ["feed-mcp:1", "alert-mcp:2"]}),
    ])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    assert tick.last_ctx.cited_item_ids == ["feed-mcp:1", "alert-mcp:2"]


@pytest.mark.asyncio
async def test_send_message_cited_added_to_interesting():
    llm = FakeLLM([
        ("send_message", {"text": "msg", "cited_ids": ["feed-mcp:1"]}),
    ])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    assert "feed-mcp:1" in tick.last_ctx.interesting_item_ids


# ── skip 写 ctx ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skip_writes_reason():
    llm = FakeLLM([("skip", {"reason": "user_busy"})])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    assert tick.last_ctx.skip_reason == "user_busy"


@pytest.mark.asyncio
async def test_skip_writes_note():
    llm = FakeLLM([("skip", {"reason": "other", "note": "debug"})])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    assert tick.last_ctx.skip_note == "debug"


# ── Alert 路径 ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_alert_path_send_sets_terminal():
    alert = {"id": "a1", "ack_server": "alert-mcp", "title": "CPU 告警",
              "body": "使用率 95%", "severity": "high", "triggered_at": "2026-01-01T00:00:00Z"}
    llm = FakeLLM([
        ("get_alert_events", {}),
        ("send_message", {"text": "告警：CPU 95%", "cited_ids": ["alert-mcp:a1"]}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(alert_fn=AsyncMock(return_value=[alert])),
    )
    await tick.tick()
    assert tick.last_ctx.terminal_action == "send"
    assert tick.last_ctx.cited_item_ids == ["alert-mcp:a1"]


@pytest.mark.asyncio
async def test_alert_stored_in_ctx_fetched_alerts():
    alert = {"id": "a1", "ack_server": "alert-mcp", "title": "T",
              "body": "B", "severity": "low", "triggered_at": "2026-01-01T00:00:00Z"}
    llm = FakeLLM([
        ("get_alert_events", {}),
        ("skip", {"reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(alert_fn=AsyncMock(return_value=[alert])),
    )
    await tick.tick()
    assert tick.last_ctx.fetched_alerts == [alert]


@pytest.mark.asyncio
async def test_alert_fn_called_once_even_if_llm_calls_twice():
    alert_fn = AsyncMock(return_value=[])
    llm = FakeLLM([
        ("get_alert_events", {}),
        ("get_alert_events", {}),   # 重复，应命中缓存
        ("skip", {"reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(alert_fn=alert_fn),
    )
    await tick.tick()
    assert alert_fn.call_count == 1


# ── Content 路径 ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_content_stored_in_ctx_fetched_contents():
    event = {"id": "c1", "ack_server": "feed-mcp", "url": "https://x.com",
             "title": "T", "source_name": "S", "published_at": "2026-01-01T00:00:00Z"}
    llm = FakeLLM([
        ("get_alert_events", {}),
        ("get_content_events", {}),
        ("skip", {"reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[event]),
        ),
    )
    await tick.tick()
    assert tick.last_ctx.fetched_contents == [event]


@pytest.mark.asyncio
async def test_content_fn_called_with_configured_limit():
    feed_fn = AsyncMock(return_value=[])
    llm = FakeLLM([
        ("get_content_events", {"limit": 3}),
        ("skip", {"reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        cfg=cfg_with(agent_tick_content_limit=3),
        tool_deps=ToolDeps(feed_fn=feed_fn),
    )
    await tick.tick()
    # limit 来自工具调用参数
    feed_fn.assert_called_once_with(limit=3)


@pytest.mark.asyncio
async def test_content_path_send_interesting_tracked():
    """send_message 中的 cited_ids 自动加入 interesting_set"""
    event = {"id": "c1", "ack_server": "feed-mcp", "url": "https://x.com",
             "title": "T", "source_name": "S", "published_at": "2026-01-01T00:00:00Z"}
    llm = FakeLLM([
        ("get_alert_events", {}),
        ("get_content_events", {}),
        ("send_message", {"text": "Great article", "cited_ids": ["feed-mcp:c1"]}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[event]),
        ),
    )
    await tick.tick()
    assert "feed-mcp:c1" in tick.last_ctx.interesting_item_ids
    assert tick.last_ctx.terminal_action == "send"


# ── mark_not_interesting 在 loop 内 ───────────────────────────────────────

@pytest.mark.asyncio
async def test_mark_not_interesting_in_loop_writes_discarded():
    event = {"id": "c1", "ack_server": "feed-mcp", "url": "https://x.com",
             "title": "T", "source_name": "S", "published_at": "2026-01-01T00:00:00Z"}
    llm = FakeLLM([
        ("get_content_events", {}),
        ("mark_not_interesting", {"item_ids": ["feed-mcp:c1"]}),
        ("skip", {"reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(feed_fn=AsyncMock(return_value=[event])),
    )
    await tick.tick()
    assert "feed-mcp:c1" in tick.last_ctx.discarded_item_ids


@pytest.mark.asyncio
async def test_mark_not_interesting_multiple_items():
    llm = FakeLLM([
        ("mark_not_interesting", {"item_ids": ["feed-mcp:1", "feed-mcp:2"]}),
        ("skip", {"reason": "no_content"}),
    ])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    assert "feed-mcp:1" in tick.last_ctx.discarded_item_ids
    assert "feed-mcp:2" in tick.last_ctx.discarded_item_ids


# ── get_context_data 最多调用一次 ────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_data_fn_called_only_once_in_loop():
    context_fn = AsyncMock(return_value=[])
    llm = FakeLLM([
        ("get_context_data", {}),
        ("get_context_data", {}),   # 第二次应命中缓存
        ("skip", {"reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(context_fn=context_fn),
    )
    await tick.tick()
    assert context_fn.call_count == 1


# ── recall_memory 在 loop 内 ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recall_memory_in_loop():
    from unittest.mock import MagicMock
    memory = MagicMock()
    memory.retrieve_related.return_value = [{"text": "用户喜欢 RPG"}]
    llm = FakeLLM([
        ("recall_memory", {"query": "RPG games"}),
        ("send_message", {"text": "RPG 推荐", "cited_ids": []}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(memory=memory),
    )
    await tick.tick()
    assert tick.last_ctx.terminal_action == "send"
    memory.retrieve_related.assert_called_once()


# ── user_busy skip 路径 ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_user_busy_skip():
    llm = FakeLLM([
        ("get_recent_chat", {}),
        ("skip", {"reason": "user_busy"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(recent_chat_fn=AsyncMock(return_value=[
            {"role": "user", "content": "我现在很忙"}
        ])),
    )
    await tick.tick()
    assert tick.last_ctx.terminal_action == "skip"
    assert tick.last_ctx.skip_reason == "user_busy"


# ── LLM 收到 messages 历史 ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_receives_growing_message_history():
    """每次 LLM 调用时 messages 应包含之前所有工具的结果"""
    llm = FakeLLM([
        ("get_alert_events", {}),
        ("skip", {"reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(alert_fn=AsyncMock(return_value=[])),
    )
    await tick.tick()
    # 第一次调用：messages 可能只有 system prompt（无工具历史）
    # 第二次调用：messages 应包含 get_alert_events 的 tool_use + tool_result
    assert len(llm.calls) == 2
    first_call_msg_count = len(llm.calls[0])
    second_call_msg_count = len(llm.calls[1])
    assert second_call_msg_count > first_call_msg_count


@pytest.mark.asyncio
async def test_llm_receives_system_message():
    """第一次调用时 messages 应包含 system prompt"""
    llm = FakeLLM([("skip", {"reason": "no_content"})])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    assert llm.calls  # 至少调用了一次
    first_messages = llm.calls[0]
    roles = [m.get("role") for m in first_messages]
    assert "system" in roles


# ── unknown tool 不 crash loop ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_tool_breaks_loop_gracefully():
    llm = FakeLLM([
        ("nonexistent_tool", {}),
        ("skip", {"reason": "no_content"}),  # 不应执行
    ])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    # execute() 在分发前就递增 steps_taken，所以是 1；但 terminal_action 不变
    assert tick.last_ctx.terminal_action is None
    assert tick.last_ctx.steps_taken == 1   # unknown tool 被调用了，只是分发失败


# ── steps_taken 精确计数 ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_steps_taken_counts_all_tool_calls():
    llm = FakeLLM([
        ("get_alert_events", {}),
        ("get_content_events", {}),
        ("get_recent_chat", {}),
        ("skip", {"reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[]),
            recent_chat_fn=AsyncMock(return_value=[]),
        ),
    )
    await tick.tick()
    assert tick.last_ctx.steps_taken == 4
