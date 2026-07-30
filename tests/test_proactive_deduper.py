from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from plugins.default_proactive.deduper import (
    MessageDeduper,
    _format_recent_proactive_entries,
    _recent_meta,
)
from proactive_v2.sensor import RecentProactiveMessage


def _build_deduper(provider: Any) -> MessageDeduper:
    return MessageDeduper(
        provider=cast(Any, provider),
        model="test-model",
        max_tokens=128,
    )


def test_recent_meta_uses_the_producer_contract() -> None:
    timestamp = datetime(2026, 7, 13, tzinfo=timezone.utc)
    message = RecentProactiveMessage(
        content="hello",
        timestamp=timestamp,
        state_summary_tag="working",
    )

    assert _recent_meta(message) == [
        "time=2026-07-13T00:00:00+00:00",
        "state_tag=working",
    ]
    assert "time=2026-07-13T00:00:00+00:00" in _format_recent_proactive_entries(
        [message]
    )


def test_recent_meta_keeps_optional_datetime_and_default_tag() -> None:
    message = RecentProactiveMessage(content="hello")

    assert _recent_meta(message) == []


@pytest.mark.asyncio
async def test_provider_failure_is_fail_loud() -> None:
    provider = AsyncMock()
    provider.chat.side_effect = OSError("network down")

    with pytest.raises(OSError, match="network down"):
        await _build_deduper(provider).is_duplicate(
            "new",
            [RecentProactiveMessage(content="old")],
        )


@pytest.mark.asyncio
async def test_invalid_model_json_is_fail_loud() -> None:
    provider = AsyncMock()
    provider.chat.return_value = SimpleNamespace(content="not json")

    with pytest.raises(json.JSONDecodeError):
        await _build_deduper(provider).is_duplicate(
            "new",
            [RecentProactiveMessage(content="old")],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "error"),
    [
        ('{"is_duplicate": "false", "reason": "wrong type"}', "boolean"),
        ('{"is_duplicate": false, "reason": 1}', "string"),
        ('{"reason": "missing field"}', "is_duplicate"),
        ('{"is_duplicate": false}', "reason"),
    ],
)
async def test_model_schema_is_strict(content: str, error: str) -> None:
    provider = AsyncMock()
    provider.chat.return_value = SimpleNamespace(content=content)

    with pytest.raises(ValueError, match=error):
        await _build_deduper(provider).is_duplicate(
            "new",
            [RecentProactiveMessage(content="old")],
        )


@pytest.mark.asyncio
async def test_programming_error_propagates() -> None:
    provider = AsyncMock()
    provider.chat.side_effect = RuntimeError("programming error")

    with pytest.raises(RuntimeError, match="programming error"):
        await _build_deduper(provider).is_duplicate(
            "new",
            [RecentProactiveMessage(content="old")],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(("configured", "expected"), [(0, 128), (64, 64)])
async def test_deduper_keeps_its_local_output_limit(
    configured: int,
    expected: int,
) -> None:
    provider = AsyncMock()
    provider.chat.return_value = SimpleNamespace(
        content='{"is_duplicate": false, "reason": "new"}'
    )
    deduper = MessageDeduper(
        provider=cast(Any, provider),
        model="test-model",
        max_tokens=configured,
    )

    await deduper.is_duplicate(
        "new",
        [RecentProactiveMessage(content="old")],
    )

    assert provider.chat.await_args.kwargs["max_tokens"] == expected
