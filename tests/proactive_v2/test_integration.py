from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from proactive_v2.config import ProactiveConfig
from proactive_v2.loop import ProactiveLoop


def make_loop() -> ProactiveLoop:
    loop = object.__new__(ProactiveLoop)
    loop._cfg = ProactiveConfig()
    loop._sense = SimpleNamespace(target_session_key=lambda: "telegram:1")
    loop._proactive_kernel = SimpleNamespace(run_tick=AsyncMock(return_value=None))
    return loop


@pytest.mark.asyncio
async def test_tick_calls_kernel() -> None:
    loop = make_loop()

    result = await loop._tick()

    loop._proactive_kernel.run_tick.assert_awaited_once_with("telegram:1")
    assert result is None


@pytest.mark.asyncio
async def test_tick_return_is_propagated() -> None:
    loop = make_loop()
    loop._proactive_kernel.run_tick = AsyncMock(return_value=42.0)

    assert await loop._tick() == 42.0


@pytest.mark.asyncio
async def test_kernel_route_stable_across_multiple_ticks() -> None:
    loop = make_loop()

    await loop._tick()
    await loop._tick()
    await loop._tick()

    assert loop._proactive_kernel.run_tick.await_count == 3
