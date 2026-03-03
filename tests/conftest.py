"""Shared fixtures and test bootstrap helpers."""
import asyncio
import sys
import types
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Provide a lightweight openai stub in test env so imports do not fail
# when optional runtime dependency is absent.
if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _DummyChatCompletions:
        async def create(self, *args, **kwargs):
            raise RuntimeError("openai stub: AsyncOpenAI.chat.completions.create not mocked")

    class _DummyChat:
        def __init__(self):
            self.completions = _DummyChatCompletions()

    class AsyncOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = _DummyChat()

    openai_stub.AsyncOpenAI = AsyncOpenAI
    sys.modules["openai"] = openai_stub

from agent.scheduler import LatencyTracker, SchedulerService, ScheduledJob


def make_job(
    trigger="at",
    tier="instant",
    fire_at=None,
    channel="telegram",
    chat_id="123",
    message="hello",
    prompt=None,
    name=None,
    interval_seconds=None,
    cron_expr=None,
    timezone_="UTC",
) -> ScheduledJob:
    if fire_at is None:
        fire_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    return ScheduledJob(
        trigger=trigger,
        tier=tier,
        fire_at=fire_at,
        channel=channel,
        chat_id=chat_id,
        message=message,
        prompt=prompt,
        name=name,
        interval_seconds=interval_seconds,
        cron_expr=cron_expr,
        timezone=timezone_,
    )


@pytest.fixture
def mock_push():
    m = AsyncMock()
    m.execute = AsyncMock(return_value="文本已发送")
    return m


@pytest.fixture
def mock_loop():
    m = AsyncMock()
    m.process_direct = AsyncMock(return_value="AI response")
    return m


@pytest.fixture
def fixed_now():
    return datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def store_path(tmp_path) -> Path:
    return tmp_path / "schedules.json"


@pytest.fixture
def tracker():
    return LatencyTracker(default=25.0, window=20)


@pytest.fixture
def service(store_path, mock_push, mock_loop, fixed_now, tracker):
    return SchedulerService(
        store_path=store_path,
        push_tool=mock_push,
        agent_loop=mock_loop,
        tracker=tracker,
        _now_fn=lambda: fixed_now,
    )


async def drain_tasks():
    """Let all pending asyncio tasks finish."""
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
