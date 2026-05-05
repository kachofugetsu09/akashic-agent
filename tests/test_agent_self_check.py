import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from unittest.mock import AsyncMock

import pytest
from agent.core.passive_support import collect_skill_mentions
from agent.core.passive_turn import DefaultReasoner
from prompts.agent import build_current_message_time_envelope
import agent.looping.core as loop_core
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps
from core.memory.markdown import ConsolidateResult
from core.memory.runtime import MemoryRuntime
from tests.memory_fakes import FakeMemoryEngine


def _make_loop(tmp_path: Path) -> AgentLoop:
    memory = FakeMemoryEngine(tmp_path)
    runtime = cast(
        MemoryRuntime,
        SimpleNamespace(
            engine=memory,
            markdown=SimpleNamespace(store=memory, maintenance=memory),
        ),
    )
    return AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=MagicMock(),
            tools=MagicMock(),
            session_manager=MagicMock(),
            workspace=tmp_path,
            memory_runtime=runtime,
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

    triggered = await loop.trigger_memory_consolidation("cli:test")

    assert triggered is True
    maintenance = cast(FakeMemoryEngine, loop._markdown_memory.maintenance)
    assert maintenance.consolidate_calls[0].session is session
    assert maintenance.consolidate_calls[0].archive_all is False
    assert maintenance.consolidate_calls[0].force is False
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

    triggered = await loop.trigger_memory_consolidation("cli:test", force=True)

    assert triggered is True
    maintenance = cast(FakeMemoryEngine, loop._markdown_memory.maintenance)
    assert maintenance.consolidate_calls[0].session is session
    assert maintenance.consolidate_calls[0].archive_all is False
    assert maintenance.consolidate_calls[0].force is True
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
    maintenance = cast(FakeMemoryEngine, loop._markdown_memory.maintenance)
    maintenance.consolidate = AsyncMock(
        return_value=ConsolidateResult(trace={"mode": "skipped"})
    )

    triggered = await loop.trigger_memory_consolidation("cli:test")

    assert triggered is False
    maintenance.consolidate.assert_awaited_once()
    loop.session_manager.save_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_trigger_memory_consolidation_times_out_when_busy(
    tmp_path: Path,
    monkeypatch,
):
    loop = _make_loop(tmp_path)
    session = SimpleNamespace(
        key="cli:test",
        messages=[{"role": "user", "content": "u"}] * 50,
        last_consolidated=0,
    )
    loop.session_manager.get_or_create = MagicMock(return_value=session)
    loop.session_manager.save_async = AsyncMock()

    async def _slow_consolidate(_request):
        await asyncio.sleep(1)
        return ConsolidateResult()

    maintenance = cast(FakeMemoryEngine, loop._markdown_memory.maintenance)
    maintenance.consolidate = AsyncMock(side_effect=_slow_consolidate)
    monkeypatch.setattr(loop_core, "_MANUAL_CONSOLIDATION_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(TimeoutError, match="memory consolidation busy"):
        await loop.trigger_memory_consolidation("cli:test")

    loop.session_manager.save_async.assert_not_awaited()
