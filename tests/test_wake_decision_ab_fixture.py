from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from docker.debug.wake_decision_ab_fixture import (
    InvalidThenValidProvider,
    fixture_digest,
    load_fixture,
    run_runtime_fixture,
)


def test_frozen_fixture_has_stable_identity() -> None:
    fixture = load_fixture()
    candidates = fixture["candidates"]

    assert fixture["fixture_id"] == "wake-decision-ab-v1"
    assert isinstance(candidates, list)
    assert len(candidates) == 100
    assert fixture_digest(fixture) == (
        "dd269172d553d51cc3a3af8e2e54067bc76c81fb90a1a4ea305d96002f84c8f8"
    )


def test_frozen_baseline_matches_fixture_identity() -> None:
    fixture = load_fixture()
    path = Path("tests/fixtures/wake_decision_ab_v1.baseline.json")
    baseline = json.loads(path.read_text(encoding="utf-8"))

    assert baseline["source_commit"] == (
        "8296b3da76d191c99284b34a3eb4e33fecda5272"
    )
    assert baseline["fixture_digest"] == fixture_digest(fixture)
    assert baseline["valid_decision_rate"] == 0.0
    assert baseline["provider_decision_requests"] == 1
    assert baseline["content_counts"] == {"deferred": 1, "pending": 99}


def test_provider_freezes_invalid_then_valid_ab_sequence() -> None:
    provider = InvalidThenValidProvider()
    kwargs = {
        "tools": [{"type": "function"}],
        "messages": [
            {"role": "user", "content": "candidate_1234567890abcdef"}
        ],
    }

    first = asyncio.run(provider.chat(**kwargs))
    second = asyncio.run(provider.chat(**kwargs))

    assert first.tool_calls == []
    assert "INVALID_DECISION_MUST_NOT_LEAK_7F3A" in str(first.content)
    assert len(second.tool_calls) == 1
    assert second.tool_calls[0].name == "share_content"
    assert second.tool_calls[0].arguments["items"] == [
        "candidate_1234567890abcdef"
    ]


@pytest.mark.asyncio
async def test_runtime_fixture_improves_frozen_baseline(tmp_path: Path) -> None:
    baseline_path = Path("tests/fixtures/wake_decision_ab_v1.baseline.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    candidate = await run_runtime_fixture(tmp_path)

    assert candidate["fixture_digest"] == baseline["fixture_digest"]
    assert candidate["valid_decision_rate"] > baseline["valid_decision_rate"]
    assert candidate["provider_decision_requests"] == 2
    assert candidate["control_turn_count"] == 2
    assert candidate["delivery_count"] == 1
    assert candidate["session_projection_count"] == 1
    assert candidate["invalid_marker_user_leak"] is False
