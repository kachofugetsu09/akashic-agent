import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import AsyncMock

import pytest
from agent.looping.core import AgentLoop, AgentLoopConfig, AgentLoopDeps
from agent.memory import MemoryStore
from core.memory.port import DefaultMemoryPort


def _make_loop(tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=MagicMock(),
            tools=MagicMock(),
            session_manager=MagicMock(),
            workspace=tmp_path,
            memory_port=DefaultMemoryPort(MemoryStore(tmp_path)),
        ),
        AgentLoopConfig(),
    )


def test_collect_skill_mentions_returns_unique_existing_names(tmp_path):
    loop = _make_loop(tmp_path)
    loop.context.skills.list_skills = MagicMock(
        return_value=[
            {"name": "feed-manage"},
            {"name": "refactor"},
        ]
    )

    got = loop._collect_skill_mentions(
        "请用 $feed-manage 然后 $refactor 再来一次 $feed-manage"
    )

    assert got == ["feed-manage", "refactor"]


def test_collect_skill_mentions_ignores_unknown_skill(tmp_path):
    loop = _make_loop(tmp_path)
    loop.context.skills.list_skills = MagicMock(return_value=[{"name": "known"}])

    got = loop._collect_skill_mentions("$known $unknown")

    assert got == ["known"]


def test_format_request_time_anchor_contains_iso_and_label():
    text = AgentLoop._format_request_time_anchor(None)
    assert text.startswith("request_time=")
    assert "(" in text and ")" in text


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
        await_vector_store=True,
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
    loop._consolidating.add("cli:test")

    async def finish_existing_consolidation() -> None:
        await asyncio.sleep(0.01)
        session.last_consolidated = 30
        loop._consolidating.discard("cli:test")

    waiter = asyncio.create_task(finish_existing_consolidation())
    try:
        triggered = await loop.trigger_memory_consolidation("cli:test")
    finally:
        await waiter

    assert triggered is True
    loop._consolidate_memory.assert_not_awaited()
    loop.session_manager.save_async.assert_not_awaited()
