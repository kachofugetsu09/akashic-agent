"""Tests for ScheduleTool, ListSchedulesTool, CancelScheduleTool."""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from agent.scheduler import LatencyTracker, SchedulerService
from agent.tools.schedule import CancelScheduleTool, ListSchedulesTool, ScheduleTool
from tests.conftest import make_job

_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_NOW_FN = lambda: _NOW  # noqa: E731


def make_svc(tmp_path, mock_push, mock_loop):
    return SchedulerService(
        store_path=tmp_path / "jobs.json",
        push_tool=mock_push,
        agent_loop=mock_loop,
        tracker=LatencyTracker(default=25.0),
        _now_fn=_NOW_FN,
    )


# ── ScheduleTool: validation ──────────────────────────────────────


async def test_invalid_tier_returns_error(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc)
    result = await tool.execute(
        tier="precise", trigger="after", when="5m", channel="tg", chat_id="1"
    )
    assert "错误" in result
    assert "tier" in result


async def test_invalid_trigger_returns_error(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc)
    result = await tool.execute(
        tier="instant",
        trigger="sometime",
        when="5m",
        channel="tg",
        chat_id="1",
        message="hi",
    )
    assert "错误" in result
    assert "trigger" in result


async def test_instant_without_message_returns_error(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc)
    result = await tool.execute(
        tier="instant", trigger="after", when="5m", channel="tg", chat_id="1"
    )
    assert "错误" in result
    assert "message" in result


async def test_soft_without_prompt_returns_error(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc)
    result = await tool.execute(
        tier="soft", trigger="after", when="5m", channel="tg", chat_id="1"
    )
    assert "错误" in result
    assert "prompt" in result


async def test_invalid_when_returns_error(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc)
    result = await tool.execute(
        tier="instant",
        trigger="after",
        when="blah",
        channel="tg",
        chat_id="1",
        message="hi",
    )
    assert "错误" in result


# ── ScheduleTool: successful registration ────────────────────────


async def test_instant_after_registers_job(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc, default_tz="UTC")
    result = await tool.execute(
        tier="instant",
        trigger="after",
        when="5m",
        channel="telegram",
        chat_id="123",
        message="喝水了",
        request_time=_NOW.isoformat(),
    )
    assert "错误" not in result
    assert len(svc._jobs) == 1
    job = list(svc._jobs.values())[0]
    assert job.tier == "instant"
    assert job.message == "喝水了"


async def test_after_request_time_used_for_fire_at(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc, default_tz="UTC")
    await tool.execute(
        tier="instant",
        trigger="after",
        when="30s",
        channel="tg",
        chat_id="1",
        message="hi",
        request_time=_NOW.isoformat(),
    )
    job = list(svc._jobs.values())[0]
    expected_fire_at = _NOW + timedelta(seconds=30)
    assert abs((job.fire_at - expected_fire_at).total_seconds()) < 1


async def test_soft_at_registers_job(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc, default_tz="UTC")
    result = await tool.execute(
        tier="soft",
        trigger="at",
        when="2025-06-01T14:00:00",
        channel="telegram",
        chat_id="456",
        prompt="查询北京天气",
    )
    assert "错误" not in result
    job = list(svc._jobs.values())[0]
    assert job.tier == "soft"
    assert job.prompt == "查询北京天气"
    assert job.fire_at.hour == 14


async def test_every_interval_stores_interval_seconds(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc, default_tz="UTC")
    await tool.execute(
        tier="instant",
        trigger="every",
        when="1h",
        channel="tg",
        chat_id="1",
        message="提醒",
    )
    job = list(svc._jobs.values())[0]
    assert job.interval_seconds == 3600
    assert job.cron_expr is None


async def test_every_cron_stores_cron_expr(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc, default_tz="UTC")
    await tool.execute(
        tier="soft",
        trigger="every",
        when="0 9 * * *",
        channel="tg",
        chat_id="1",
        prompt="天气",
    )
    job = list(svc._jobs.values())[0]
    assert job.cron_expr == "0 9 * * *"
    assert job.interval_seconds is None


async def test_named_job(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ScheduleTool(svc, default_tz="UTC")
    await tool.execute(
        tier="instant",
        trigger="after",
        when="5m",
        channel="tg",
        chat_id="1",
        message="hi",
        name="my-reminder",
        request_time=_NOW.isoformat(),
    )
    job = list(svc._jobs.values())[0]
    assert job.name == "my-reminder"


# ── ListSchedulesTool ────────────────────────────────────────────


async def test_list_empty(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = ListSchedulesTool(svc)
    result = await tool.execute()
    assert "没有" in result


async def test_list_shows_jobs(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    job = make_job(name="喝水提醒", tier="instant")
    svc._jobs[job.id] = job

    tool = ListSchedulesTool(svc)
    result = await tool.execute()
    assert "喝水提醒" in result


# ── CancelScheduleTool ───────────────────────────────────────────


async def test_cancel_by_id(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    job = make_job()
    svc._jobs[job.id] = job

    tool = CancelScheduleTool(svc)
    result = await tool.execute(id=job.id)
    assert "已取消" in result
    assert job.id not in svc._jobs


async def test_cancel_by_name(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    job = make_job(name="daily-report")
    svc._jobs[job.id] = job

    tool = CancelScheduleTool(svc)
    result = await tool.execute(name="daily-report")
    assert "已取消" in result
    assert job.id not in svc._jobs


async def test_cancel_nonexistent_id(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = CancelScheduleTool(svc)
    result = await tool.execute(id="no-such-id")
    assert "未找到" in result


async def test_cancel_no_args_returns_error(tmp_path, mock_push, mock_loop):
    svc = make_svc(tmp_path, mock_push, mock_loop)
    tool = CancelScheduleTool(svc)
    result = await tool.execute()
    assert "错误" in result
