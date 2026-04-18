"""
TDD — Phase 3: proactive_v2/tools.py

测试覆盖：
  - TOOL_SCHEMAS 结构与必填字段
  - _web_fetch: 截断、错误透传
  - _recall_memory: list[dict] → {result, hits}
  - _finish_reply / _finish_skip: 写 ctx 终止状态
  - _mark_not_interesting: 写 ctx.discarded_item_ids
  - _get_alert_events / _get_content_events: 缓存保护
  - _get_context_data: 最多调用一次
  - execute(): 分发 + steps_taken 递增
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from proactive_v2.context import AgentTickContext
from proactive_v2.tools import (
    TOOL_SCHEMAS,
    ToolDeps,
    execute,
    _web_fetch,
    _web_search,
    _recall_memory,
    _finish_reply,
    _finish_skip,
    _finish_turn,
    _mark_not_interesting,
    _get_alert_events,
    _get_content_events,
    _get_context_data,
    _get_recent_chat,
)


# ── TOOL_SCHEMAS 结构 ─────────────────────────────────────────────────────

def test_tool_schemas_is_list():
    assert isinstance(TOOL_SCHEMAS, list)


def _fn(name: str) -> dict:
    """从 TOOL_SCHEMAS 取指定工具的 function 块。"""
    return next(s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == name)


def test_all_tools_present():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    required = {
        "get_alert_events",
        "get_content_events",
        "get_context_data",
        "web_fetch",
        "web_search",
        "recall_memory",
        "get_recent_chat",
        "mark_interesting",
        "mark_not_interesting",
        "finish_reply",
        "finish_skip",
    }
    assert required <= names  # 允许超集（未来可加工具），但必须包含以上全部


def test_each_schema_has_openai_format():
    """每条 schema 必须是 OpenAI function tool 格式。"""
    for s in TOOL_SCHEMAS:
        assert s.get("type") == "function", f"missing type=function: {s}"
        fn = s.get("function", {})
        assert "name" in fn, f"function missing name: {s}"
        assert "description" in fn, f"function missing description: {s}"
        assert "parameters" in fn, f"function missing parameters: {s}"


def test_finish_reply_schema_evidence_is_array():
    props = _fn("finish_reply")["parameters"]["properties"]
    assert "evidence" in props
    assert props["evidence"]["type"] == "array"
    assert props["evidence"]["items"]["type"] == "string"


def test_finish_reply_schema_content_required():
    assert "content" in _fn("finish_reply")["parameters"]["required"]


def test_finish_skip_schema_reason_is_supported():
    assert "reason" in _fn("finish_skip")["parameters"]["properties"]


def test_finish_reply_requires_non_empty_content():
    ctx = AgentTickContext()
    with pytest.raises(ValueError, match="requires non-empty content"):
        _finish_reply(ctx, {"content": "   ", "evidence": []})


def test_mark_not_interesting_schema_item_ids_is_array():
    props = _fn("mark_not_interesting")["parameters"]["properties"]
    assert "item_ids" in props
    assert props["item_ids"]["type"] == "array"


def test_mark_interesting_schema_item_ids_is_array():
    props = _fn("mark_interesting")["parameters"]["properties"]
    assert "item_ids" in props
    assert props["item_ids"]["type"] == "array"


def test_web_fetch_schema_url_required():
    assert "url" in _fn("web_fetch")["parameters"]["required"]


def test_web_search_schema_query_required():
    assert "query" in _fn("web_search")["parameters"]["required"]


def test_recall_memory_schema_query_required():
    assert "query" in _fn("recall_memory")["parameters"]["required"]


# ── _web_fetch ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_web_fetch_truncates_to_max_chars():
    fake_tool = AsyncMock()
    fake_tool.execute.return_value = json.dumps({
        "url": "https://example.com",
        "text": "x" * 20_000,
        "truncated": False,
    })
    result = json.loads(await _web_fetch(
        ctx=AgentTickContext(),
        args={"url": "https://example.com"},
        web_fetch_tool=fake_tool,
        max_chars=8_000,
    ))
    assert len(result["text"]) == 8_000
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_web_fetch_short_text_not_truncated():
    fake_tool = AsyncMock()
    fake_tool.execute.return_value = json.dumps({
        "url": "https://example.com",
        "text": "hello",
        "truncated": False,
    })
    result = json.loads(await _web_fetch(
        ctx=AgentTickContext(),
        args={"url": "https://example.com"},
        web_fetch_tool=fake_tool,
        max_chars=8_000,
    ))
    assert result["text"] == "hello"
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_web_fetch_exact_max_chars_not_truncated():
    fake_tool = AsyncMock()
    fake_tool.execute.return_value = json.dumps({
        "url": "https://example.com",
        "text": "y" * 8_000,
        "truncated": False,
    })
    result = json.loads(await _web_fetch(
        ctx=AgentTickContext(),
        args={"url": "https://example.com"},
        web_fetch_tool=fake_tool,
        max_chars=8_000,
    ))
    assert len(result["text"]) == 8_000
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_web_fetch_error_passthrough():
    fake_tool = AsyncMock()
    error_payload = json.dumps({"error": "timeout", "status": 504})
    fake_tool.execute.return_value = error_payload
    result = json.loads(await _web_fetch(
        ctx=AgentTickContext(),
        args={"url": "https://example.com"},
        web_fetch_tool=fake_tool,
        max_chars=8_000,
    ))
    assert result["error"] == "timeout"
    assert "text" not in result


@pytest.mark.asyncio
async def test_web_fetch_preserves_upstream_truncated_true():
    """上游已截断时，即使本次不截断也保持 truncated=True"""
    fake_tool = AsyncMock()
    fake_tool.execute.return_value = json.dumps({
        "url": "https://example.com",
        "text": "short",
        "truncated": True,   # 上游已截断
    })
    result = json.loads(await _web_fetch(
        ctx=AgentTickContext(),
        args={"url": "https://example.com"},
        web_fetch_tool=fake_tool,
        max_chars=8_000,
    ))
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_web_fetch_calls_execute_with_text_format():
    fake_tool = AsyncMock()
    fake_tool.execute.return_value = json.dumps({"url": "x", "text": "ok", "truncated": False})
    await _web_fetch(
        ctx=AgentTickContext(),
        args={"url": "https://example.com"},
        web_fetch_tool=fake_tool,
        max_chars=8_000,
    )
    fake_tool.execute.assert_called_once_with(url="https://example.com", format="text")


@pytest.mark.asyncio
async def test_web_search_passthrough():
    fake_tool = AsyncMock()
    fake_tool.execute.return_value = json.dumps({"query": "furia cs2", "result": "..."})
    result = json.loads(await _web_search(
        ctx=AgentTickContext(),
        args={"query": "furia cs2", "num_results": 3},
        web_search_tool=fake_tool,
    ))
    assert result["query"] == "furia cs2"
    fake_tool.execute.assert_called_once_with(query="furia cs2", num_results=3)


@pytest.mark.asyncio
async def test_web_search_without_tool_returns_error():
    result = json.loads(await _web_search(
        ctx=AgentTickContext(),
        args={"query": "hf speed-bench"},
        web_search_tool=None,
    ))
    assert "error" in result


# ── _recall_memory ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recall_memory_empty_hits():
    fake_memory = MagicMock()
    fake_memory.retrieve_interest_block = AsyncMock(return_value=SimpleNamespace(hits=[]))
    result = json.loads(await _recall_memory(
        ctx=AgentTickContext(),
        args={"query": "game news"},
        memory=fake_memory,
    ))
    assert result == {"result": "", "hits": 0}


@pytest.mark.asyncio
async def test_recall_memory_joins_texts():
    fake_memory = MagicMock()
    fake_memory.retrieve_interest_block = AsyncMock(
        return_value=SimpleNamespace(
            hits=[
                {"text": "用户喜欢 RPG"},
                {"text": "不喜欢 PvP"},
            ]
        )
    )
    result = json.loads(await _recall_memory(
        ctx=AgentTickContext(),
        args={"query": "game"},
        memory=fake_memory,
    ))
    assert result["hits"] == 2
    assert "用户喜欢 RPG" in result["result"]
    assert "不喜欢 PvP" in result["result"]


@pytest.mark.asyncio
async def test_recall_memory_skips_empty_text():
    fake_memory = MagicMock()
    fake_memory.retrieve_interest_block = AsyncMock(
        return_value=SimpleNamespace(
            hits=[
                {"text": ""},
                {"text": "有效记忆"},
            ]
        )
    )
    result = json.loads(await _recall_memory(
        ctx=AgentTickContext(),
        args={"query": "test"},
        memory=fake_memory,
    ))
    assert "有效记忆" in result["result"]
    assert result["hits"] == 2   # hits 按命中数统计，不过滤空 text


@pytest.mark.asyncio
async def test_recall_memory_passes_query_to_facade_interest_request():
    fake_memory = MagicMock()
    fake_memory.retrieve_interest_block = AsyncMock(return_value=SimpleNamespace(hits=[]))
    await _recall_memory(ctx=AgentTickContext(), args={"query": "q"}, memory=fake_memory)
    request = fake_memory.retrieve_interest_block.await_args.args[0]
    assert request.query == "q"
    assert request.top_k == 2


@pytest.mark.asyncio
async def test_recall_memory_prefers_facade_interest_block():
    fake_memory = MagicMock()
    fake_memory.retrieve_interest_block = AsyncMock(
        return_value=SimpleNamespace(
            hits=[
                {"id": "p1", "text": "用户偏好中文回复"},
                {"id": "u1", "text": "用户常用 Telegram"},
            ]
        )
    )

    result = json.loads(await _recall_memory(
        ctx=AgentTickContext(),
        args={"query": "q"},
        memory=fake_memory,
    ))

    assert result["hits"] == 2
    assert "用户偏好中文回复" in result["result"]
    fake_memory.retrieve_interest_block.assert_awaited_once()


@pytest.mark.asyncio
async def test_recall_memory_separator_between_hits():
    fake_memory = MagicMock()
    fake_memory.retrieve_interest_block = AsyncMock(
        return_value=SimpleNamespace(
            hits=[
                {"text": "A"},
                {"text": "B"},
            ]
        )
    )
    result = json.loads(await _recall_memory(
        ctx=AgentTickContext(), args={"query": "q"}, memory=fake_memory
    ))
    assert "---" in result["result"]


# ── _finish_reply / _finish_skip 终止语义 ─────────────────────────────────

def test_finish_reply_sets_terminal_reply():
    ctx = AgentTickContext()
    ctx.fetched_contents = [{"ack_server": "feed-mcp", "event_id": "1"}]
    result = json.loads(_finish_reply(ctx, {"content": "hello", "evidence": ["feed-mcp:1"]}))
    assert ctx.terminal_action == "reply"
    assert result["ok"] is True


def test_finish_reply_writes_content_and_evidence():
    ctx = AgentTickContext()
    ctx.fetched_contents = [{"ack_server": "feed-mcp", "event_id": "1"}]
    ctx.fetched_alerts = [{"ack_server": "alert-mcp", "event_id": "2"}]
    _finish_reply(ctx, {"content": "hello world", "evidence": ["feed-mcp:1", "alert-mcp:2"]})
    assert ctx.final_message == "hello world"
    assert ctx.cited_item_ids == ["feed-mcp:1", "alert-mcp:2"]


def test_finish_reply_evidence_added_to_interesting():
    ctx = AgentTickContext()
    ctx.fetched_contents = [
        {"ack_server": "feed-mcp", "event_id": "1"},
        {"ack_server": "feed-mcp", "event_id": "99"},
    ]
    ctx.discarded_item_ids = {"feed-mcp:99"}
    _finish_reply(ctx, {"content": "msg", "evidence": ["feed-mcp:1", "feed-mcp:99"]})
    assert "feed-mcp:1" in ctx.interesting_item_ids
    assert "feed-mcp:99" in ctx.interesting_item_ids
    assert "feed-mcp:99" not in ctx.discarded_item_ids


def test_finish_reply_invalid_evidence_raises():
    ctx = AgentTickContext()
    with pytest.raises(ValueError, match="invalid evidence ids"):
        _finish_reply(ctx, {"content": "hello", "evidence": ["fitbit:v2_x"]})


def test_finish_reply_rejects_prefixed_evidence_when_no_valid_ids():
    ctx = AgentTickContext()
    with pytest.raises(ValueError, match="invalid evidence ids"):
        _finish_reply(ctx, {"content": "hello", "evidence": ["feed-mcp:made-up"]})


def test_finish_skip_sets_reason_and_note():
    ctx = AgentTickContext()
    result = json.loads(_finish_skip(ctx, {"reason": "other", "note": "debug info"}))
    assert ctx.terminal_action == "skip"
    assert ctx.skip_reason == "other"
    assert ctx.skip_note == "debug info"
    assert result["ok"] is True


@pytest.mark.parametrize("reason", ["no_content", "user_busy", "already_sent_similar", "other"])
def test_finish_skip_valid_reasons(reason):
    ctx = AgentTickContext()
    _finish_skip(ctx, {"reason": reason})
    assert ctx.skip_reason == reason


def test_finish_skip_invalid_reason_raises():
    ctx = AgentTickContext()
    with pytest.raises(ValueError):
        _finish_skip(ctx, {"reason": "invalid_reason"})


def test_finish_skip_rejects_content_and_evidence():
    ctx = AgentTickContext()
    with pytest.raises(ValueError, match="does not accept content"):
        _finish_skip(ctx, {"reason": "no_content", "content": "x"})
    ctx.fetched_contents = [{"ack_server": "feed-mcp", "event_id": "1"}]
    with pytest.raises(ValueError, match="does not accept evidence"):
        _finish_skip(ctx, {"reason": "no_content", "evidence": ["feed-mcp:1"]})


# ── _mark_not_interesting ─────────────────────────────────────────────────

def test_mark_not_interesting_adds_to_discarded():
    ctx = AgentTickContext()
    result = json.loads(_mark_not_interesting(ctx, {"item_ids": ["feed-mcp:1", "feed-mcp:2"]}))
    assert "feed-mcp:1" in ctx.discarded_item_ids
    assert "feed-mcp:2" in ctx.discarded_item_ids
    assert result["ok"] is True


def test_mark_not_interesting_single_item():
    ctx = AgentTickContext()
    _mark_not_interesting(ctx, {"item_ids": ["alert-mcp:99"]})
    assert "alert-mcp:99" in ctx.discarded_item_ids


def test_mark_not_interesting_empty_list():
    ctx = AgentTickContext()
    result = json.loads(_mark_not_interesting(ctx, {"item_ids": []}))
    assert ctx.discarded_item_ids == set()
    assert result["ok"] is True


def test_mark_not_interesting_accumulates():
    ctx = AgentTickContext()
    _mark_not_interesting(ctx, {"item_ids": ["feed-mcp:1"]})
    _mark_not_interesting(ctx, {"item_ids": ["feed-mcp:2"]})
    assert "feed-mcp:1" in ctx.discarded_item_ids
    assert "feed-mcp:2" in ctx.discarded_item_ids


# ── _get_alert_events (缓存) ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_alert_events_caches_on_second_call():
    events = [
        {"id": "a1", "ack_server": "alert-mcp", "title": "CPU", "body": "", "severity": "high", "triggered_at": "2026-01-01T00:00:00Z"}
    ]
    ctx = AgentTickContext()
    ctx.mark_alerts_prefetched(events)
    await _get_alert_events(ctx, {})
    await _get_alert_events(ctx, {})
    assert ctx.fetched_alerts == events


@pytest.mark.asyncio
async def test_get_alert_events_stores_in_ctx():
    event = {"id": "a1", "ack_server": "alert-mcp", "title": "T", "body": "", "severity": "low", "triggered_at": "2026-01-01T00:00:00Z"}
    ctx = AgentTickContext()
    ctx.mark_alerts_prefetched([event])
    await _get_alert_events(ctx, {})
    assert ctx.fetched_alerts == [event]
    assert ctx.alerts_fetched is True


@pytest.mark.asyncio
async def test_get_alert_events_returns_json_list():
    ctx = AgentTickContext()
    raw = await _get_alert_events(ctx, {})
    parsed = json.loads(raw)
    assert isinstance(parsed, list)


# ── _get_content_events (缓存) ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_content_events_caches_on_second_call():
    events = [
        {"id": "c1", "ack_server": "feed-mcp", "url": "https://x.com", "title": "T", "source_name": "S", "published_at": "2026-01-01T00:00:00Z"}
    ]
    ctx = AgentTickContext()
    ctx.mark_contents_prefetched(events, {})
    await _get_content_events(ctx, {})
    await _get_content_events(ctx, {})
    assert ctx.fetched_contents == events


@pytest.mark.asyncio
async def test_get_content_events_stores_in_ctx():
    event = {"id": "c1", "ack_server": "feed-mcp", "url": "https://x.com", "title": "T", "source_name": "S", "published_at": "2026-01-01T00:00:00Z"}
    ctx = AgentTickContext()
    ctx.mark_contents_prefetched([event], {})
    await _get_content_events(ctx, {})
    assert ctx.fetched_contents == [event]
    assert ctx.contents_fetched is True


@pytest.mark.asyncio
async def test_get_content_events_passes_limit():
    ctx = AgentTickContext()
    await _get_content_events(ctx, {})
    assert ctx.contents_fetched is True


@pytest.mark.asyncio
async def test_get_content_events_returns_json_list():
    ctx = AgentTickContext()
    raw = await _get_content_events(ctx, {})
    assert isinstance(json.loads(raw), list)


# ── _get_context_data (最多调用一次) ─────────────────────────────────────

@pytest.mark.asyncio
async def test_get_context_data_max_one_call():
    rows = [{"title": "Steam", "body": "playing"}]
    ctx = AgentTickContext()
    ctx.mark_context_prefetched(rows)
    await _get_context_data(ctx, {})
    await _get_context_data(ctx, {})
    assert ctx.fetched_context == rows


@pytest.mark.asyncio
async def test_get_context_data_stores_in_ctx():
    item = {"title": "Steam", "body": "playing"}
    ctx = AgentTickContext()
    ctx.mark_context_prefetched([item])
    await _get_context_data(ctx, {})
    assert ctx.fetched_context == [item]
    assert ctx.context_fetched is True


@pytest.mark.asyncio
async def test_get_context_data_returns_json():
    ctx = AgentTickContext()
    raw = await _get_context_data(ctx, {})
    assert isinstance(json.loads(raw), list)


# ── _get_recent_chat ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_recent_chat_calls_fn_with_n():
    fake_chat_fn = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
    ctx = AgentTickContext()
    await _get_recent_chat(ctx, {"n": 10}, recent_chat_fn=fake_chat_fn)
    fake_chat_fn.assert_called_once_with(n=10)


@pytest.mark.asyncio
async def test_get_recent_chat_default_n_20():
    fake_chat_fn = AsyncMock(return_value=[])
    ctx = AgentTickContext()
    await _get_recent_chat(ctx, {}, recent_chat_fn=fake_chat_fn)
    fake_chat_fn.assert_called_once_with(n=20)


@pytest.mark.asyncio
async def test_get_recent_chat_returns_json():
    msgs = [{"role": "user", "content": "hi"}]
    fake_chat_fn = AsyncMock(return_value=msgs)
    ctx = AgentTickContext()
    raw = await _get_recent_chat(ctx, {}, recent_chat_fn=fake_chat_fn)
    assert json.loads(raw) == msgs


# ── execute() 分发 ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_increments_steps_taken():
    ctx = AgentTickContext()
    deps = ToolDeps()
    await execute("get_alert_events", {}, ctx, deps)
    assert ctx.steps_taken == 1


@pytest.mark.asyncio
async def test_execute_increments_each_call():
    ctx = AgentTickContext()
    deps = ToolDeps()
    await execute("get_alert_events", {}, ctx, deps)
    await execute("get_content_events", {}, ctx, deps)
    assert ctx.steps_taken == 2


@pytest.mark.asyncio
async def test_execute_dispatches_finish_reply():
    ctx = AgentTickContext()
    ctx.fetched_contents = [{"ack_server": "feed-mcp", "event_id": "1"}]
    deps = ToolDeps()
    await execute("finish_reply", {"content": "hi", "evidence": ["feed-mcp:1"]}, ctx, deps)
    assert ctx.terminal_action == "reply"


@pytest.mark.asyncio
async def test_execute_dispatches_finish_skip():
    ctx = AgentTickContext()
    deps = ToolDeps()
    await execute("finish_skip", {"reason": "no_content"}, ctx, deps)
    assert ctx.terminal_action == "skip"


@pytest.mark.asyncio
async def test_execute_dispatches_mark_not_interesting():
    ctx = AgentTickContext()
    deps = ToolDeps()
    await execute("mark_not_interesting", {"item_ids": ["feed-mcp:1"]}, ctx, deps)
    assert "feed-mcp:1" in ctx.discarded_item_ids


@pytest.mark.asyncio
async def test_execute_unknown_tool_raises():
    ctx = AgentTickContext()
    deps = ToolDeps()
    with pytest.raises(ValueError, match="unknown tool"):
        await execute("nonexistent_tool", {}, ctx, deps)


@pytest.mark.asyncio
async def test_execute_web_fetch_uses_max_chars_from_deps():
    fake_tool = AsyncMock()
    fake_tool.execute.return_value = json.dumps({"url": "x", "text": "z" * 5_000, "truncated": False})
    ctx = AgentTickContext()
    deps = ToolDeps(web_fetch_tool=fake_tool, max_chars=2_000)
    raw = await execute("web_fetch", {"url": "https://x.com"}, ctx, deps)
    result = json.loads(raw)
    assert len(result["text"]) == 2_000


@pytest.mark.asyncio
async def test_execute_web_search_uses_tool_from_deps():
    fake_tool = AsyncMock()
    fake_tool.execute.return_value = json.dumps({"query": "aurora furia", "result": "..."})
    ctx = AgentTickContext()
    deps = ToolDeps(web_search_tool=fake_tool)
    raw = await execute("web_search", {"query": "aurora furia", "type": "fast"}, ctx, deps)
    result = json.loads(raw)
    assert result["query"] == "aurora furia"
    fake_tool.execute.assert_called_once_with(query="aurora furia", type="fast")


@pytest.mark.asyncio
async def test_execute_recall_memory_uses_memory_from_deps():
    fake_memory = MagicMock()
    fake_memory.retrieve_interest_block = AsyncMock(
        return_value=SimpleNamespace(hits=[{"text": "pref"}])
    )
    ctx = AgentTickContext()
    deps = ToolDeps(memory=fake_memory)
    raw = await execute("recall_memory", {"query": "test"}, ctx, deps)
    result = json.loads(raw)
    assert result["hits"] == 1
