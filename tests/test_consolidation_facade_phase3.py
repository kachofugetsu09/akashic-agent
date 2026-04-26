from __future__ import annotations
from typing import Any, cast

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.looping.consolidation import ConsolidationRuntime
from agent.looping.ports import TurnScheduler
from core.memory.default_runtime_facade import DefaultMemoryRuntimeFacade


@pytest.mark.asyncio
async def test_consolidation_runtime_prefers_facade_run_consolidation():
    facade = MagicMock()
    facade.run_consolidation = AsyncMock()
    runtime = ConsolidationRuntime(
        session_manager=MagicMock(),
        scheduler=MagicMock(),
        consolidation=MagicMock(),
        facade=facade,
        keep_count=20,
        wait_timeout_s=1.0,
    )
    session = object()

    await runtime.consolidate_memory(
        session,
        archive_all=True,
    )

    facade.run_consolidation.assert_awaited_once_with(
        session,
        archive_all=True,
    )


def test_default_runtime_facade_binds_consolidation_runner():
    facade = DefaultMemoryRuntimeFacade(
        port=MagicMock(),
        engine=None,
        profile_maint=MagicMock(),
    )
    runner = AsyncMock()

    facade.bind_consolidation_runner(runner)

    assert facade._consolidation_runner is runner


@pytest.mark.asyncio
async def test_turn_scheduler_runner_can_go_through_facade():
    facade = MagicMock()
    facade.run_consolidation = AsyncMock()
    save_async = AsyncMock()
    session = SimpleNamespace(messages=[{"role": "user", "content": "u"}] * 30, last_consolidated=0)

    async def runner(session_obj):
        await facade.run_consolidation(session_obj)
        await save_async(session_obj)

    scheduler = TurnScheduler(
        consolidation_runner=runner,
        keep_count=20,
    )

    await scheduler._run_consolidation_bg(cast(Any, session), "telegram:1")

    facade.run_consolidation.assert_awaited_once_with(session)
    save_async.assert_awaited_once_with(session)
