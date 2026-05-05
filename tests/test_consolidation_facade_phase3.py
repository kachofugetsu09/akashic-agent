from __future__ import annotations
from typing import Any, cast

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.looping.consolidation import ConsolidationRuntime
from agent.looping.ports import TurnScheduler
from tests.memory_fakes import FakeMemoryEngine


@pytest.mark.asyncio
async def test_consolidation_runtime_delegates_to_memory_engine():
    memory = FakeMemoryEngine()
    runtime = ConsolidationRuntime(
        session_manager=MagicMock(),
        scheduler=MagicMock(),
        memory=memory,
        keep_count=20,
        wait_timeout_s=1.0,
    )
    session = object()

    await runtime.consolidate_memory(
        session,
        archive_all=True,
    )

    assert memory.consolidate_calls
    assert memory.consolidate_calls[0].session is session
    assert memory.consolidate_calls[0].archive_all is True


@pytest.mark.asyncio
async def test_turn_scheduler_runner_can_go_through_memory_engine():
    memory = FakeMemoryEngine()
    save_async = AsyncMock()
    session = SimpleNamespace(messages=[{"role": "user", "content": "u"}] * 30, last_consolidated=0)

    async def runner(session_obj):
        from core.memory.engine import ConsolidateRequest

        await memory.consolidate(ConsolidateRequest(session=session_obj))
        await save_async(session_obj)

    scheduler = TurnScheduler(
        consolidation_runner=runner,
        keep_count=20,
    )

    await scheduler._run_consolidation_bg(cast(Any, session), "telegram:1")

    assert memory.consolidate_calls[0].session is session
    save_async.assert_awaited_once_with(session)
