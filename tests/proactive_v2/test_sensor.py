from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

import pytest

from agent.prompting import SYSTEM_CONTEXT_FRAME_MARKER
from proactive_v2.config import ProactiveConfig
from proactive_v2.sensor import Sensor
from session.manager import SessionManager


@pytest.fixture
def sensor_env(tmp_path: Path) -> Iterator[tuple[Sensor, SessionManager]]:
    sessions = SessionManager(tmp_path)
    sensor = Sensor(
        cfg=ProactiveConfig(
            default_channel="telegram",
            default_chat_id="42",
            recent_chat_messages=10,
        ),
        sessions=sessions,
        presence=None,
    )
    try:
        yield sensor, sessions
    finally:
        sessions._store.close()


def test_collect_recent_filters_context_and_limits_content(
    sensor_env: tuple[Sensor, SessionManager],
) -> None:
    sensor, sessions = sensor_env
    session = sessions.get_or_create("telegram:42")
    session.messages = [
        {"role": "system", "content": "系统消息"},
        {"role": "user", "content": f"{SYSTEM_CONTEXT_FRAME_MARKER}\n上下文"},
        {"role": "user", "content": "问" * 250, "timestamp": "2026-07-13"},
        {"role": "assistant", "content": "回答", "timestamp": "2026-07-13"},
    ]

    assert sensor.collect_recent() == [
        {"role": "user", "content": "问" * 200, "timestamp": "2026-07-13"},
        {"role": "assistant", "content": "回答", "timestamp": "2026-07-13"},
    ]


def test_collect_recent_proactive_preserves_order_and_metadata(
    sensor_env: tuple[Sensor, SessionManager],
) -> None:
    sensor, sessions = sensor_env
    session = sessions.get_or_create("telegram:42")
    session.messages = [
        {
            "role": "assistant",
            "content": "第一条",
            "proactive": True,
            "timestamp": "2026-07-13T00:00:00+08:00",
            "state_summary_tag": "working",
            "source_refs": [{"id": "source-1"}],
        },
        {"role": "user", "content": "普通消息"},
        {
            "role": "assistant",
            "content": "第二条",
            "proactive": True,
            "timestamp": "invalid",
        },
    ]

    messages = sensor.collect_recent_proactive(2)

    assert [message.content for message in messages] == ["第一条", "第二条"]
    assert messages[0].timestamp == datetime.fromisoformat(
        "2026-07-13T00:00:00+08:00"
    )
    assert messages[0].state_summary_tag == "working"
    assert messages[0].source_refs == [{"id": "source-1"}]
    assert messages[1].timestamp is None


@pytest.mark.parametrize(
    "collect",
    [Sensor.collect_recent, Sensor.collect_recent_proactive],
)
def test_session_failure_is_not_reported_as_empty_history(
    sensor_env: tuple[Sensor, SessionManager],
    collect: Callable[[Sensor], object],
) -> None:
    sensor, sessions = sensor_env
    sessions._store.close()

    with pytest.raises(sqlite3.ProgrammingError):
        collect(sensor)
