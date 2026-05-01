"""
TDD — Phase 5: proactive_v2/agent_tick.py — Agent Loop

测试覆盖：
  - max_steps 保护：loop 不超过 agent_tick_max_steps
  - 终止工具（message_push+finish_turn/skip）调用后 loop 立即结束
  - LLM 返回 None → loop 结束
  - llm_fn=None → loop 不执行任何工具
  - Alert 路径：ctx.terminal_action="send" + cited_ids
  - Content 路径：interesting_set 从 cited_ids 推断
  - mark_not_interesting 在 loop 内写 discarded_set
  - skip(user_busy) 路径
  - LLM 消息历史：工具结果追加到 messages
  - tool_choice="auto"：主 loop 兼容 thinking 模式供应商
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.prompting import is_context_frame
from proactive_v2.gateway import GatewayDeps
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
async def test_loop_puts_runtime_context_frame_before_kickoff():
    llm = FakeLLM([("finish_turn", {"decision": "skip", "reason": "no_content"})])
    tick = make_agent_tick(
        llm_fn=llm,
        gateway_deps=GatewayDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(
                return_value=[
                    {
                        "event_id": "c1",
                        "ack_server": "feed",
                        "title": "测试内容",
                        "source_name": "feed",
                        "url": "https://example.com/a",
                    }
                ]
            ),
            context_fn=AsyncMock(return_value=[]),
        ),
    )

    await tick.tick()

    messages = llm.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert is_context_frame(str(messages[1]["content"]))
    assert "proactive_tick_state" in str(messages[1]["content"])
    assert str(messages[2]["content"]).startswith("开始本轮 proactive 处理。")


@pytest.mark.asyncio
async def test_loop_stops_after_partial_sequence_then_none():
    llm = FakeLLM([
        ("get_alert_events", {}),
        ("get_content_events", {}),
        # 之后 None，loop 结束
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        gateway_deps=GatewayDeps(
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
    assert len(tick._state_store.tick_log_starts) == 1
    assert len(tick._state_store.tick_log_finishes) == 1
    assert tick._state_store.tick_log_finishes[0]["terminal_action"] == "skip"


# ── 终止工具立即结束 loop ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_message_stops_loop_immediately():
    llm = FakeLLM([
        ("message_push", {"message": "Hello!", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
        ("get_recent_chat", {}),   # 不应执行
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(recent_chat_fn=AsyncMock(return_value=[])),
    )
    await tick.tick()
    # message_push + finish_turn 是前 2 步，之后 loop 停止
    assert tick.last_ctx.steps_taken == 2
    assert tick.last_ctx.terminal_action == "reply"


@pytest.mark.asyncio
async def test_skip_stops_loop_immediately():
    llm = FakeLLM([
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
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
        ("message_push", {"message": "Hi", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
    ])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    assert tick.last_ctx.terminal_action == "reply"
    assert tick.last_ctx.steps_taken == 2


# ── send_message 写 ctx ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_message_writes_final_message():
    llm = FakeLLM([
        ("message_push", {"message": "Hello world!", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    assert tick.last_ctx.final_message == "Hello world!"


@pytest.mark.asyncio
async def test_tool_chain_step_logs_capture_args_and_results():
    llm = FakeLLM([
        ("message_push", {"message": "Hello world!", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    assert len(tick._state_store.tick_step_logs) == 2
    first = tick._state_store.tick_step_logs[0]
    second = tick._state_store.tick_step_logs[1]
    assert first["tool_name"] == "message_push"
    assert first["tool_args"]["message"] == "Hello world!"
    assert "\"ok\": true" in first["tool_result_text"]
    assert first["final_message_after"] == ""
    assert second["tool_name"] == "finish_turn"
    assert second["terminal_action_after"] == "reply"
    assert second["final_message_after"] == "Hello world!"


@pytest.mark.asyncio
async def test_send_message_writes_cited_ids():
    llm = FakeLLM([
        ("message_push", {"message": "msg", "evidence": ["feed-mcp:1", "alert-mcp:2"]}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        gateway_deps=GatewayDeps(
            alert_fn=AsyncMock(return_value=[{"ack_server": "alert-mcp", "event_id": "2", "title": "a"}]),
            feed_fn=AsyncMock(return_value=[{"id": "1", "ack_server": "feed-mcp", "title": "t"}]),
        ),
    )
    await tick.tick()
    assert tick.last_ctx.cited_item_ids == ["feed-mcp:1", "alert-mcp:2"]


@pytest.mark.asyncio
async def test_send_message_cited_added_to_interesting():
    llm = FakeLLM([
        ("message_push", {"message": "msg", "evidence": ["feed-mcp:1"]}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        gateway_deps=GatewayDeps(
            feed_fn=AsyncMock(return_value=[{"id": "1", "ack_server": "feed-mcp", "title": "t"}]),
        ),
    )
    await tick.tick()
    assert "feed-mcp:1" in tick.last_ctx.interesting_item_ids


# ── skip 写 ctx ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skip_writes_reason():
    llm = FakeLLM([("finish_turn", {"decision": "skip", "reason": "user_busy"})])
    tick = make_agent_tick(llm_fn=llm)
    await tick.tick()
    assert tick.last_ctx.skip_reason == "user_busy"


@pytest.mark.asyncio
async def test_skip_writes_note():
    llm = FakeLLM([("finish_turn", {"decision": "skip", "reason": "other", "note": "debug"})])
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
        ("message_push", {"message": "告警：CPU 95%", "evidence": ["alert-mcp:a1"]}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        gateway_deps=GatewayDeps(alert_fn=AsyncMock(return_value=[alert])),
    )
    await tick.tick()
    assert tick.last_ctx.terminal_action == "reply"
    assert tick.last_ctx.cited_item_ids == ["alert-mcp:a1"]


@pytest.mark.asyncio
async def test_alert_stored_in_ctx_fetched_alerts():
    alert = {"id": "a1", "ack_server": "alert-mcp", "title": "T",
              "body": "B", "severity": "low", "triggered_at": "2026-01-01T00:00:00Z"}
    llm = FakeLLM([
        ("get_alert_events", {}),
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        gateway_deps=GatewayDeps(alert_fn=AsyncMock(return_value=[alert])),
    )
    await tick.tick()
    assert tick.last_ctx.fetched_alerts == [alert]


@pytest.mark.asyncio
async def test_alert_fn_called_once_even_if_llm_calls_twice():
    alert_fn = AsyncMock(return_value=[])
    llm = FakeLLM([
        ("get_alert_events", {}),
        ("get_alert_events", {}),   # 重复，应命中缓存
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        gateway_deps=GatewayDeps(alert_fn=alert_fn),
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
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        gateway_deps=GatewayDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[event]),
        ),
    )
    await tick.tick()
    assert tick.last_ctx.fetched_contents == [{
        "id": "c1",
        "event_id": "c1",
        "ack_server": "feed-mcp",
        "url": "https://x.com",
        "title": "T",
        "source": "S",
        "published_at": "2026-01-01T00:00:00Z",
    }]


@pytest.mark.asyncio
async def test_content_fn_called_with_configured_limit():
    feed_fn = AsyncMock(return_value=[])
    llm = FakeLLM([
        ("get_content_events", {"limit": 3}),
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        cfg=cfg_with(agent_tick_content_limit=3),
        gateway_deps=GatewayDeps(feed_fn=feed_fn, content_limit=3),
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
        ("message_push", {"message": "Great article", "evidence": ["feed-mcp:c1"]}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        gateway_deps=GatewayDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[event]),
        ),
    )
    await tick.tick()
    assert "feed-mcp:c1" in tick.last_ctx.interesting_item_ids
    assert tick.last_ctx.terminal_action == "reply"


# ── mark_not_interesting 在 loop 内 ───────────────────────────────────────

@pytest.mark.asyncio
async def test_mark_not_interesting_in_loop_writes_discarded():
    event = {"id": "c1", "ack_server": "feed-mcp", "url": "https://x.com",
             "title": "T", "source_name": "S", "published_at": "2026-01-01T00:00:00Z"}
    llm = FakeLLM([
        ("get_content_events", {}),
        ("mark_not_interesting", {"item_ids": ["feed-mcp:c1"]}),
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        gateway_deps=GatewayDeps(feed_fn=AsyncMock(return_value=[event])),
    )
    await tick.tick()
    assert "feed-mcp:c1" in tick.last_ctx.discarded_item_ids


@pytest.mark.asyncio
async def test_mark_not_interesting_multiple_items():
    llm = FakeLLM([
        ("mark_not_interesting", {"item_ids": ["feed-mcp:1", "feed-mcp:2"]}),
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
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
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        gateway_deps=GatewayDeps(context_fn=context_fn),
    )
    await tick.tick()
    assert context_fn.call_count == 1


# ── recall_memory 在 loop 内 ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recall_memory_in_loop():
    from unittest.mock import MagicMock
    memory = MagicMock()
    memory.retrieve_interest_block = AsyncMock(
        return_value=SimpleNamespace(hits=[{"text": "用户喜欢 RPG"}])
    )
    llm = FakeLLM([
        ("recall_memory", {"query": "RPG games"}),
        ("message_push", {"message": "RPG 推荐", "evidence": []}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(memory=memory),
    )
    await tick.tick()
    assert tick.last_ctx.terminal_action == "reply"
    memory.retrieve_interest_block.assert_awaited_once()


# ── user_busy skip 路径 ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_user_busy_skip():
    llm = FakeLLM([
        ("get_recent_chat", {}),
        ("finish_turn", {"decision": "skip", "reason": "user_busy"}),
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
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        gateway_deps=GatewayDeps(alert_fn=AsyncMock(return_value=[])),
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
    llm = FakeLLM([("finish_turn", {"decision": "skip", "reason": "no_content"})])
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
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),  # 不应执行
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
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(recent_chat_fn=AsyncMock(return_value=[])),
        gateway_deps=GatewayDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[]),
        ),
    )
    await tick.tick()
    assert tick.last_ctx.steps_taken == 4


# ── tool_choice="auto" ────────────────────────────────────────────────────
# 兼容性：当前 proactive 使用的上游在 thinking 模式下不支持 required/object。
# 因此主 loop 改回 auto，避免整轮 proactive 因 400 直接退出。

@pytest.mark.asyncio
async def test_main_loop_uses_auto_tool_choice():
    """主 loop 每一步都应以 tool_choice='auto' 调用 llm_fn。"""
    llm = FakeLLM([
        ("get_recent_chat", {}),
        ("finish_turn", {"decision": "skip", "reason": "no_content"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(recent_chat_fn=AsyncMock(return_value=[])),
    )
    await tick.tick()

    assert all(tc == "auto" for tc in llm.tool_choices), (
        f"expected all tool_choices to be 'auto', got {llm.tool_choices}"
    )


@pytest.mark.asyncio
async def test_alert_present_llm_called_with_auto_tool_choice():
    """有 Alert 时主 loop 也应使用 tool_choice='auto'。"""
    alert = {
        "ack_server": "health",
        "event_id": "recovery_001",
        "title": "恢复指标下降",
        "content": "连续三天 HRV 持续下降，建议早睡。",
    }
    llm = FakeLLM([
        ("get_recent_chat", {}),
        ("message_push", {"message": "最近恢复指标有点下滑，今天早点睡？", "evidence": ["health:recovery_001"]}),
        ("finish_turn", {"decision": "reply"}),
    ])

    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(recent_chat_fn=AsyncMock(return_value=[])),
        gateway_deps=GatewayDeps(
            alert_fn=AsyncMock(return_value=[alert]),
            feed_fn=AsyncMock(return_value=[]),
            context_fn=AsyncMock(return_value=[]),
        ),
    )
    await tick.tick()

    # 消息被正常发出
    assert tick.last_ctx.terminal_action == "reply"
    assert tick.last_ctx.steps_taken > 0

    assert all(tc == "auto" for tc in llm.tool_choices), (
        f"expected all tool_choices to be 'auto', got {llm.tool_choices}"
    )


@pytest.mark.asyncio
async def test_main_loop_stops_when_auto_tool_call_is_empty():
    calls: list[list[dict]] = []

    async def llm_fn(messages, schemas, tool_choice="auto"):
        calls.append(list(messages))
        return None

    tick = make_agent_tick(llm_fn=llm_fn)
    await tick.tick()

    assert tick.last_ctx.terminal_action is None
    assert tick.last_ctx.steps_taken == 0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_finish_turn_error_stops_under_auto_tool_choice():
    llm = FakeLLM([
        ("get_recent_chat", {"n": 10}),
        ("finish_turn", {"decision": "reply"}),
    ])
    tick = make_agent_tick(
        llm_fn=llm,
        tool_deps=ToolDeps(recent_chat_fn=AsyncMock(return_value=[])),
    )

    await tick.tick()

    assert tick.last_ctx.terminal_action is None
    assert tick.last_ctx.steps_taken == 2
