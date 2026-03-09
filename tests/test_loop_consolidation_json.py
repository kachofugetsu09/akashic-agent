from types import SimpleNamespace

from agent.loop_consolidation import (
    _build_consolidation_source_ref,
    _format_conversation_for_consolidation,
    _parse_consolidation_payload,
    _select_consolidation_window,
)


def test_parse_consolidation_payload_supports_fenced_json():
    result = _parse_consolidation_payload(
        '```json\n{"history_entry":"[2026-03-09 12:00] 用户确认信息","pending_items":[{"tag":"preference","content":"不用 emoji。"}]}\n```'
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
        memory_window=6,
        archive_all=False,
    )

    assert window is not None
    assert window.keep_count == 3
    assert window.consolidate_up_to == 7
    assert [m["content"] for m in window.old_messages] == ["2", "3", "4", "5", "6"]


def test_build_consolidation_source_ref_matches_existing_shape():
    session = SimpleNamespace(key="telegram:123", last_consolidated=4)

    assert (
        _build_consolidation_source_ref(
            session,
            consolidate_up_to=9,
            archive_all=False,
        )
        == "telegram:123@4-9"
    )
    assert (
        _build_consolidation_source_ref(
            session,
            consolidate_up_to=9,
            archive_all=True,
        )
        == "telegram:123@archive_all"
    )


def test_format_conversation_for_consolidation_skips_tool_messages():
    text = _format_conversation_for_consolidation(
        [
            {"role": "user", "content": "你好", "timestamp": "2026-03-09T10:00:00"},
            {"role": "tool", "content": "ignored", "timestamp": "2026-03-09T10:01:00"},
            {"role": "assistant", "content": "收到", "timestamp": "2026-03-09T10:02:00"},
        ]
    )

    assert "TOOL" not in text
    assert "[2026-03-09T10:00] USER: 你好" in text
    assert "[2026-03-09T10:02] ASSISTANT: 收到" in text
