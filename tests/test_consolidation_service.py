from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from core.memory.markdown import _MarkdownConsolidationWorker


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


def _prepare(worker: _MarkdownConsolidationWorker, session: object):
    return asyncio.run(worker.prepare_consolidation(session, archive_all=True))


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        key="cli:1",
        messages=[
            {
                "id": "cli:1:0",
                "role": "user",
                "content": "我买了 Zigbee 网关",
                "timestamp": "2026-03-15T10:00:00",
            },
            {
                "id": "cli:1:1",
                "role": "assistant",
                "content": "记住了",
                "timestamp": "2026-03-15T10:01:00",
            },
        ],
        last_consolidated=0,
    )


def test_consolidation_extracts_history_and_pending_without_side_projection():
    profile = SimpleNamespace(
        read_long_term=MagicMock(return_value="MEM"),
    )
    provider = SimpleNamespace(
        chat=AsyncMock(
            return_value=_Response(
                '{"history_entries":[{"summary":"[2026-03-15 10:00] 用户聊了 Zigbee 方案","emotional_weight":6}],"pending_items":[]}'
            )
        )
    )
    worker = _MarkdownConsolidationWorker(
        profile_maint=profile,
        provider=provider,
        model="m",
        keep_count=20,
    )

    draft = _prepare(worker, _session())

    assert draft is not None
    assert draft.history_entry_payloads == [
        ("[2026-03-15 10:00] 用户聊了 Zigbee 方案", 6),
    ]
    assert draft.pending_items == ""
    assert provider.chat.await_count == 1
    assert profile.read_long_term.call_count == 1
    assert all(
        call.kwargs.get("disable_thinking") is True
        for call in provider.chat.await_args_list
    )


def test_consolidation_empty_provider_response_is_failure():
    profile = SimpleNamespace(read_long_term=MagicMock(return_value=""))
    provider = SimpleNamespace(chat=AsyncMock(return_value=_Response("")))
    worker = _MarkdownConsolidationWorker(
        profile_maint=profile,
        provider=provider,
        model="m",
        keep_count=20,
    )

    result = _prepare(worker, _session())

    assert result is not None
    assert result.step == "event_extract"
    assert result.error == "empty_response"
