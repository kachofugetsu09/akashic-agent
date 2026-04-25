from typing import Any, cast
import json
from types import SimpleNamespace

from agent.looping.consolidation import (
    _build_consolidation_source_ref,
    _format_conversation_for_consolidation,
    _format_conversation_for_recent_context,
    _format_recent_context_messages,
    _parse_consolidation_payload,
    _select_consolidation_window,
)


def test_parse_consolidation_payload_supports_fenced_json():
    result = _parse_consolidation_payload(
        '```json\n{"history_entry":"[2026-03-09 12:00] 用户确认信息","pending_items":[{"tag":"preference","content":"回复保持简洁。"}]}\n```'
    )

    assert result is not None
    assert result["history_entry"].startswith("[2026-03-09 12:00]")
    assert result["pending_items"][0]["tag"] == "preference"


def test_parse_consolidation_payload_returns_none_for_non_object():
    assert _parse_consolidation_payload('["not","object"]') is None


def test_select_consolidation_window_uses_half_window_tail_keep():
    session = SimpleNamespace(
        key="cli:1",
        last_consolidated=2,
        messages=[{"content": str(i)} for i in range(10)],
    )

    window = _select_consolidation_window(
        session,
        keep_count=3,
        consolidation_min_new_messages=5,
        archive_all=False,
    )

    assert window is not None
    assert window.keep_count == 3
    assert window.consolidate_up_to == 7
    assert [m["content"] for m in window.old_messages] == ["2", "3", "4", "5", "6"]


def test_build_consolidation_source_ref_returns_message_id_list_json():
    window = SimpleNamespace(
        old_messages=[
            {"id": "telegram:123:4", "content": "a"},
            {"id": "telegram:123:5", "content": "b"},
            {"content": "missing id"},
        ]
    )

    ref = _build_consolidation_source_ref(cast(Any, window))
    assert json.loads(ref) == ["telegram:123:4", "telegram:123:5"]


def test_format_conversation_for_consolidation_skips_tool_messages():
    text = _format_conversation_for_consolidation(
        [
            {"role": "user", "content": "你好", "timestamp": "2026-03-09T10:00:00"},
            {"role": "tool", "content": "ignored", "timestamp": "2026-03-09T10:01:00"},
            {
                "role": "assistant",
                "content": "收到",
                "timestamp": "2026-03-09T10:02:00",
            },
        ]
    )

    assert "TOOL" not in text
    assert "[2026-03-09T10:00] USER: 你好" in text
    assert "[2026-03-09T10:02] ASSISTANT: 收到" in text


def test_format_conversation_for_consolidation_skips_proactive_assistant_messages():
    text = _format_conversation_for_consolidation(
        [
            {"role": "user", "content": "你好", "timestamp": "2026-03-09T10:00:00"},
            {
                "role": "assistant",
                "content": "这是一条主动 RSS 推送",
                "timestamp": "2026-03-09T10:01:00",
                "proactive": True,
            },
            {
                "role": "assistant",
                "content": "这是正常回复",
                "timestamp": "2026-03-09T10:02:00",
            },
        ]
    )

    assert "主动 RSS 推送" not in text
    assert "[2026-03-09T10:00] USER: 你好" in text
    assert "[2026-03-09T10:02] ASSISTANT: 这是正常回复" in text


def test_consolidation_formatters_skip_context_frame_messages():
    frame = '<system-reminder data-system-context-frame="true">\n内部上下文'
    messages = [
        {"id": "1", "role": "user", "content": frame, "timestamp": "2026-03-09T10:00:00"},
        {"id": "2", "role": "user", "content": "真实用户内容", "timestamp": "2026-03-09T10:01:00"},
    ]

    window = SimpleNamespace(old_messages=messages)

    assert json.loads(_build_consolidation_source_ref(cast(Any, window))) == ["2"]
    assert "内部上下文" not in _format_conversation_for_consolidation(messages)
    assert "内部上下文" not in _format_conversation_for_recent_context(messages)
    assert "内部上下文" not in _format_recent_context_messages(messages)
