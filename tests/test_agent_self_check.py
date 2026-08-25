from datetime import datetime, timedelta
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from agent.core.passive_support import collect_skill_mentions
from agent.core.passive_turn import DefaultReasoner
from prompts.agent import build_current_message_time_envelope


def test_collect_skill_mentions_returns_unique_existing_names(tmp_path):
    skills = [
        "feed-manage",
        "refactor",
    ]

    got = collect_skill_mentions(
        "请用 $feed-manage 然后 $refactor 再来一次 $feed-manage",
        skills,
    )

    assert got == ["feed-manage", "refactor"]


def test_collect_skill_mentions_ignores_unknown_skill(tmp_path):
    skills = ["known"]

    got = collect_skill_mentions("$known $unknown", skills)

    assert got == ["known"]


def test_format_request_time_anchor_contains_iso_and_label():
    text = DefaultReasoner.format_request_time_anchor(None)
    assert text.startswith("request_time=")
    assert "(" in text and ")" in text


def test_build_current_message_time_envelope_contains_today_and_tomorrow():
    message_timestamp = datetime.fromisoformat("2026-04-08T17:57:00+08:00")
    local_timestamp = message_timestamp.astimezone()

    text = build_current_message_time_envelope(message_timestamp=message_timestamp)

    assert f"当前消息时间: {local_timestamp:%Y-%m-%d %H:%M}" in text
    assert f"今天={local_timestamp:%Y-%m-%d}" in text
    assert f"明天={local_timestamp + timedelta(days=1):%Y-%m-%d}" in text
