import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from agent.looping.consolidation import (
    ConsolidationService,
    _select_recent_history_entries,
)


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


def test_consolidation_service_archive_all_and_profile_extract():
    memory = SimpleNamespace(
        read_long_term=MagicMock(return_value="MEM"),
        read_history=MagicMock(
            return_value=(
                "[2026-03-15 09:00] 用户确认 Zigbee 需求\n\n"
                "[2026-03-15 09:30] 用户对本地控制方案感兴趣\n\n"
                "[2026-03-15 09:45] 用户准备下单 Zigbee 网关"
            )
        ),
        append_history_once=MagicMock(return_value=True),
        append_pending_once=MagicMock(return_value=True),
        save_from_consolidation=AsyncMock(),
        save_item=AsyncMock(return_value="new:profile-1"),
    )
    provider = SimpleNamespace(
        chat=AsyncMock(
            return_value=_Resp(
                '{"history_entries":["[2026-03-15 10:00] 用户聊了 Zigbee 方案"],"pending_items":[]}'
            )
        )
    )
    profile_extractor = SimpleNamespace(
        extract=AsyncMock(
            return_value=[
                SimpleNamespace(
                    summary="用户买了 Zigbee 网关",
                    category="device",
                    happened_at="2026-03-15T10:00:00",
                )
            ]
        )
    )

    service = ConsolidationService(
        memory_port=memory,
        provider=provider,
        model="m",
        memory_window=40,
        profile_extractor=profile_extractor,
    )
    session = SimpleNamespace(
        key="cli:1",
        messages=[
            {"role": "user", "content": "我买了 Zigbee 网关", "timestamp": "2026-03-15T10:00:00"},
            {"role": "assistant", "content": "记住了", "timestamp": "2026-03-15T10:01:00"},
        ],
        last_consolidated=0,
        _channel="cli",
        _chat_id="1",
    )

    asyncio.run(service.consolidate(session, archive_all=True, await_vector_store=True))

    memory.append_history_once.assert_called_once()
    memory.save_from_consolidation.assert_awaited_once()
    profile_extractor.extract.assert_awaited_once()
    memory.save_item.assert_awaited_once()
    prompt = provider.chat.await_args.kwargs["messages"][1]["content"]
    assert "## 最近三次 consolidation event" in prompt
    assert "用户准备下单 Zigbee 网关" in prompt
    assert "不能作为人物身份、说话人归属、关系判断或具体事实归属的直接证据" in prompt
    assert session.last_consolidated == 0


def test_consolidation_service_uses_profile_maint_for_file_side_io():
    memory_port = SimpleNamespace(
        save_from_consolidation=AsyncMock(),
        save_item=AsyncMock(return_value="new:profile-1"),
    )
    profile_maint = SimpleNamespace(
        read_long_term=MagicMock(return_value="MEM"),
        read_history=MagicMock(return_value=""),
        append_history_once=MagicMock(return_value=True),
        append_pending_once=MagicMock(return_value=True),
    )
    provider = SimpleNamespace(
        chat=AsyncMock(
            return_value=_Resp(
                '{"history_entries":["[2026-03-15 10:00] 用户聊了 Zigbee 方案"],"pending_items":[]}'
            )
        )
    )
    service = ConsolidationService(
        memory_port=memory_port,
        profile_maint=profile_maint,
        provider=provider,
        model="m",
        memory_window=40,
        profile_extractor=None,
    )
    session = SimpleNamespace(
        key="cli:1",
        messages=[
            {"role": "user", "content": "我买了 Zigbee 网关", "timestamp": "2026-03-15T10:00:00"},
            {"role": "assistant", "content": "记住了", "timestamp": "2026-03-15T10:01:00"},
        ],
        last_consolidated=0,
        _channel="cli",
        _chat_id="1",
    )

    asyncio.run(service.consolidate(session, archive_all=True, await_vector_store=True))

    profile_maint.read_long_term.assert_called_once()
    profile_maint.append_history_once.assert_called_once()
    memory_port.save_from_consolidation.assert_awaited_once()


def test_select_recent_history_entries_returns_last_three_chunks():
    history = (
        "[2026-03-15 09:00] A\n\n"
        "[2026-03-15 09:10] B\n\n"
        "[2026-03-15 09:20] C\n\n"
        "[2026-03-15 09:30] D"
    )
    assert _select_recent_history_entries(history, limit=3) == [
        "[2026-03-15 09:10] B",
        "[2026-03-15 09:20] C",
        "[2026-03-15 09:30] D",
    ]
