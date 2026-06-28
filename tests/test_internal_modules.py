from __future__ import annotations
from typing import Any, cast

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.memory.markdown import (
    _MarkdownConsolidationWorker as ConsolidationWorker,
    _build_consolidation_source_ref,
    _format_conversation_for_consolidation,
    _format_pending_items,
    _parse_consolidation_payload,
    _select_consolidation_window,
)
from memory2.post_response_worker import PostResponseMemoryWorker


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


class _ConsolidationHarness:
    def __init__(self, payload: str) -> None:
        self._memory_port = SimpleNamespace(
            read_long_term=MagicMock(return_value="MEM"),
            read_history=MagicMock(return_value=""),
            read_recent_context=MagicMock(return_value=""),
            append_history_once=MagicMock(return_value=True),
            append_pending_once=MagicMock(return_value=True),
            save_from_consolidation=AsyncMock(),
        )
        self.last_draft = None
        self.provider = SimpleNamespace(chat=AsyncMock(return_value=_Resp(payload)))
        self._consolidation = ConsolidationWorker(
            profile_maint=cast(Any, self._memory_port),
            provider=cast(Any, self.provider),
            model="lm",
            keep_count=2,
        )

    async def _consolidate_memory(
        self,
        session,
        archive_all: bool = False,
    ) -> None:
        self.last_draft = await self._consolidation.prepare_consolidation(
            session,
            archive_all=archive_all,
        )

@pytest.mark.asyncio
async def test_consolidation_helpers(
    monkeypatch: pytest.MonkeyPatch,
):
    assert _format_pending_items("x") == ""
    assert _format_pending_items(
        [
            {"tag": "preference", "content": "喜欢 A"},
            {"tag": "preference", "content": "喜欢 A"},
            {"tag": "bad", "content": "忽略"},
            "x",
        ]
    ) == "- [preference] 喜欢 A"
    assert _parse_consolidation_payload('{"x":1}') == {"x": 1}

    session = SimpleNamespace(
        key="telegram:1",
        last_consolidated=0,
        messages=[
            {"role": "user", "content": "u1", "timestamp": "2025-01-01T10:00:00"},
            {"role": "assistant", "content": "a1", "timestamp": "2025-01-01T10:01:00"},
            {"role": "tool", "content": "skip", "timestamp": "2025-01-01T10:02:00"},
            {
                "role": "assistant",
                "content": "skip proactive",
                "timestamp": "2025-01-01T10:03:00",
                "proactive": True,
            },
        ],
    )
    assert _select_consolidation_window(
        session,
        keep_count=5,
        consolidation_min_new_messages=10,
        archive_all=False,
    ) is None
    window = _select_consolidation_window(
        session,
        keep_count=2,
        consolidation_min_new_messages=10,
        archive_all=True,
    )
    assert window and window.consolidate_up_to == 4
    enough_messages_session = SimpleNamespace(
        key="telegram:2",
        last_consolidated=0,
        messages=[{"role": "user", "content": "u", "timestamp": "2025-01-01T10:00:00"}] * 16,
    )
    assert _select_consolidation_window(
        enough_messages_session,
        keep_count=5,
        consolidation_min_new_messages=10,
        archive_all=False,
    ) is not None
    assert _select_consolidation_window(
        enough_messages_session,
        keep_count=5,
        consolidation_min_new_messages=12,
        archive_all=False,
    ) is None
    window_with_ids = SimpleNamespace(
        old_messages=[
            {"id": "telegram:1:0"},
            {"id": "telegram:1:1"},
            {"content": "missing id"},
        ]
    )
    assert json.loads(
        _build_consolidation_source_ref(cast(Any, window_with_ids))
    ) == ["telegram:1:0", "telegram:1:1"]
    assert _format_conversation_for_consolidation(session.messages).count("USER") == 1

    harness = _ConsolidationHarness(
        json.dumps(
            {
                "history_entries": ["[2025-01-01 10:00] 主题A", "[2025-01-01 10:02] 主题B"],
                "pending_items": [{"tag": "preference", "content": "喜欢 A"}],
            }
        )
    )
    scheduled = []
    real_create_task = asyncio.create_task

    def _capture_task(coro):
        task = real_create_task(coro)
        scheduled.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", _capture_task)
    session._channel = "telegram"
    session._chat_id = "1"
    await harness._consolidate_memory(session, archive_all=True)
    if scheduled:
        await asyncio.gather(*scheduled)
    assert harness.last_draft is not None
    assert harness.last_draft.history_entry_payloads == [
        ("[2025-01-01 10:00] 主题A", 0),
        ("[2025-01-01 10:02] 主题B", 0),
    ]
    assert harness.last_draft.pending_items == "- [preference] 喜欢 A"
    assert session.last_consolidated == 0

    scheduled.clear()
    awaited = _ConsolidationHarness(
        json.dumps({"history_entries": ["[2025-01-01 10:00] 主题A"]})
    )
    awaited_session = SimpleNamespace(
        key="telegram:2",
        last_consolidated=0,
        messages=session.messages,
        _channel="telegram",
        _chat_id="2",
    )
    await awaited._consolidate_memory(
        awaited_session,
        archive_all=True,
    )
    assert awaited.last_draft is not None
    assert awaited.last_draft.history_entry_payloads == [
        ("[2025-01-01 10:00] 主题A", 0)
    ]
    assert scheduled == []

    empty = _ConsolidationHarness("")
    short_session = SimpleNamespace(key="s", messages=[{"role": "user", "content": "u"}], last_consolidated=0)
    await empty._consolidate_memory(short_session)
    assert empty.last_draft is None


@pytest.mark.asyncio
async def test_post_response_worker_invalidation_paths():
    memorizer = SimpleNamespace(
        save_item=AsyncMock(return_value="new:1"),
        supersede_batch=MagicMock(),
    )
    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            side_effect=[
                [{"id": "x1", "score": 0.9, "summary": "旧规则"}],
                [{"id": "x1", "score": 0.9, "summary": "旧规则"}],
            ]
        )
    )
    provider = SimpleNamespace(chat=AsyncMock(return_value=_Resp('["topic"]')))
    worker = PostResponseMemoryWorker(cast(Any, memorizer), cast(Any, retriever), cast(Any, provider), "lm")

    assert worker._consume_budget(10, 3) == (True, 7)
    assert worker._collect_explicit_memorized(
        [{"calls": [{"name": "memorize", "arguments": {"summary": "规则A"}, "result": "已记住（new:AbCDef12_34567890）：规则A"}]}]
    ) == (["规则A"], {"AbCDef12_34567890"})

    topics, remain = await worker._extract_invalidation_topics("你之前这个流程错了", 700)
    assert topics == ["topic"]

    provider.chat = AsyncMock(return_value=_Resp('["x1"]'))
    ids, remain = await worker._check_invalidate("topic", [{"id": "x1", "summary": "旧规则"}], remain)
    assert ids == ["x1"]


@pytest.mark.asyncio
async def test_post_response_worker_budget_exhausted_skips_invalidation():
    memorizer = SimpleNamespace(save_item=AsyncMock(return_value="new:2"), supersede_batch=MagicMock())
    retriever = SimpleNamespace(retrieve=AsyncMock(side_effect=RuntimeError("boom")))
    provider = SimpleNamespace(chat=AsyncMock(return_value=_Resp("bad json")))
    worker = PostResponseMemoryWorker(cast(Any, memorizer), cast(Any, retriever), cast(Any, provider), "lm")

    topics, remain = await worker._extract_invalidation_topics("也许这个流程不对", 0)
    assert topics == []
    assert remain == 0


@pytest.mark.asyncio
async def test_consolidation_long_term_prompt_contains_conversation():
    """consolidation 的合并长期记忆提取调用（第二次 LLM 调用）应包含窗口对话内容。"""
    captured_prompts: list[str] = []
    event_payload = json.dumps({
        "history_entries": [
            "[2026-03-17 15:07] 用户询问助手是否记得其开始佩戴 Fitbit 手环的具体时间。"
        ],
        "pending_items": [],
    })

    async def _capture_chat(*, messages, **kwargs):
        captured_prompts.append(str(messages[-1]["content"]))
        return _Resp(event_payload)

    harness = _ConsolidationHarness(event_payload)
    harness.provider.chat = _capture_chat
    harness._memory_port.save_item = AsyncMock(return_value="new:profile-1")
    session = SimpleNamespace(
        key="telegram:fitbit",
        last_consolidated=0,
        messages=[
            {
                "role": "assistant",
                "content": "嗯，刚看到个挺有意思的消息。",
                "timestamp": "2026-03-17T15:05:00",
                "proactive": True,
            },
            {
                "role": "assistant",
                "content": "嗯，刚看到个挺硬核的更新。",
                "timestamp": "2026-03-17T15:06:00",
                "proactive": True,
            },
            {
                "role": "user",
                "content": "你还记得我什么时候开始戴fitbit手环的吗",
                "timestamp": "2026-03-17T15:07:00",
            },
        ],
        _channel="telegram",
        _chat_id="fitbit",
    )

    await harness._consolidate_memory(
        session,
        archive_all=True,
    )

    assert len(captured_prompts) == 1
    assert "fitbit" in captured_prompts[0].lower()
    harness._memory_port.save_item.assert_not_awaited()
