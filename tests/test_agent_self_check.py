import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import AsyncMock

import pytest
from agent.core.passive_support import collect_skill_mentions
from agent.core.passive_turn import DefaultReasoner
from prompts.agent import build_current_message_time_envelope
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, MemoryServices
from tests.memory_fakes import FakeMemoryEngine


def _make_loop(tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=MagicMock(),
            tools=MagicMock(),
            session_manager=MagicMock(),
            workspace=tmp_path,
            memory_services=MemoryServices(engine=FakeMemoryEngine(tmp_path)),
        ),
        AgentLoopConfig(),
    )


def test_collect_skill_mentions_returns_unique_existing_names(tmp_path):
    skills = [
        {"name": "feed-manage"},
        {"name": "refactor"},
    ]

    got = collect_skill_mentions(
        "请用 $feed-manage 然后 $refactor 再来一次 $feed-manage",
        skills,
    )

    assert got == ["feed-manage", "refactor"]


def test_collect_skill_mentions_ignores_unknown_skill(tmp_path):
    skills = [{"name": "known"}]

    got = collect_skill_mentions("$known $unknown", skills)

    assert got == ["known"]


def test_format_request_time_anchor_contains_iso_and_label():
    text = DefaultReasoner.format_request_time_anchor(None)
    assert text.startswith("request_time=")
    assert "(" in text and ")" in text


def test_build_current_message_time_envelope_contains_today_and_tomorrow():
    text = build_current_message_time_envelope(
        message_timestamp=datetime.fromisoformat("2026-04-08T17:57:00+08:00")
    )
    assert "当前消息时间: 2026-04-08 17:57" in text
    assert "今天=2026-04-08" in text
    assert "明天=2026-04-09" in text


@pytest.mark.asyncio
async def test_trigger_memory_consolidation_uses_real_entrypoint(tmp_path: Path):
    loop = _make_loop(tmp_path)
    session = SimpleNamespace(
        key="cli:test",
        messages=[{"role": "user", "content": "u"}] * 50,
        last_consolidated=0,
    )
    loop.session_manager.get_or_create = MagicMock(return_value=session)
    loop.session_manager.save_async = AsyncMock()
    loop._consolidate_memory = AsyncMock()

    triggered = await loop.trigger_memory_consolidation("cli:test")

    assert triggered is True
    loop._consolidate_memory.assert_awaited_once_with(
        session,
        archive_all=False,
        force=False,
    )
    loop.session_manager.save_async.assert_awaited_once_with(session)


@pytest.mark.asyncio
async def test_trigger_memory_consolidation_force_runs_below_threshold(tmp_path: Path):
    loop = _make_loop(tmp_path)
    session = SimpleNamespace(
        key="cli:test",
        messages=[{"role": "user", "content": "u"}] * 4,
        last_consolidated=0,
    )
    loop.session_manager.get_or_create = MagicMock(return_value=session)
    loop.session_manager.save_async = AsyncMock()
    loop._consolidate_memory = AsyncMock()

    triggered = await loop.trigger_memory_consolidation("cli:test", force=True)

    assert triggered is True
    loop._consolidate_memory.assert_awaited_once_with(
        session,
        archive_all=False,
        force=True,
    )
    loop.session_manager.save_async.assert_awaited_once_with(session)


@pytest.mark.asyncio
async def test_trigger_memory_consolidation_returns_false_when_not_needed(tmp_path: Path):
    loop = _make_loop(tmp_path)
    session = SimpleNamespace(
        key="cli:test",
        messages=[{"role": "user", "content": "u"}],
        last_consolidated=0,
    )
    loop.session_manager.get_or_create = MagicMock(return_value=session)
    loop.session_manager.save_async = AsyncMock()
    loop._consolidate_memory = AsyncMock()

    triggered = await loop.trigger_memory_consolidation("cli:test")

    assert triggered is False
    loop._consolidate_memory.assert_not_awaited()
    loop.session_manager.save_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_trigger_memory_consolidation_waits_for_inflight_task(tmp_path: Path):
    loop = _make_loop(tmp_path)
    session = SimpleNamespace(
        key="cli:test",
        messages=[{"role": "user", "content": "u"}] * 50,
        last_consolidated=0,
    )
    loop.session_manager.get_or_create = MagicMock(return_value=session)
    loop.session_manager.save_async = AsyncMock()
    loop._consolidate_memory = AsyncMock()
    loop._scheduler.mark_manual_start("cli:test")

    async def finish_existing_consolidation() -> None:
        await asyncio.sleep(0.01)
        session.last_consolidated = 30
        loop._scheduler.mark_manual_end("cli:test")

    waiter = asyncio.create_task(finish_existing_consolidation())
    try:
        triggered = await loop.trigger_memory_consolidation("cli:test")
    finally:
        await waiter

    assert triggered is True
    loop._consolidate_memory.assert_not_awaited()
    loop.session_manager.save_async.assert_not_awaited()
