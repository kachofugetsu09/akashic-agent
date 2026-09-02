from datetime import UTC, datetime, timedelta, timezone

import pytest

from agent.control.ids import new_item_id, new_thread_id, new_turn_id
from agent.control.models import (
    ThreadRecord,
    ThreadSource,
    TurnError,
    TurnItem,
    TurnItemKind,
    TurnRecord,
    TurnResult,
    TurnStatus,
    TurnUsage,
)

NOW = datetime(2026, 7, 14, 8, 0, tzinfo=UTC)


def test_turn_status_values_and_terminal_membership() -> None:
    assert [status.value for status in TurnStatus] == [
        "queued",
        "in_progress",
        "completed",
        "interrupted",
        "failed",
        "cancelled",
    ]
    assert TurnStatus.COMPLETED.is_terminal
    assert not TurnStatus.IN_PROGRESS.is_terminal


def test_ids_are_namespaced_and_unique() -> None:
    generators = (
        (new_thread_id, "programmatic:"),
        (new_turn_id, "turn:"),
        (new_item_id, "item:"),
    )
    for generate, prefix in generators:
        values = {generate() for _ in range(100)}
        assert len(values) == 100
        assert all(value.startswith(prefix) for value in values)


def test_thread_record_normalizes_utc_and_serializes_rfc3339() -> None:
    offset = timezone(timedelta(hours=8))
    record = ThreadRecord(
        id="telegram:1",
        source=ThreadSource.CHANNEL,
        created_at=datetime(2026, 7, 14, 16, tzinfo=offset),
        updated_at=datetime(2026, 7, 14, 16, 1, tzinfo=offset),
    )

    assert record.created_at.tzinfo is UTC
    assert record.to_dict()["createdAt"] == "2026-07-14T08:00:00Z"


def test_models_reject_naive_datetime() -> None:
    with pytest.raises(ValueError, match="必须包含时区"):
        ThreadRecord(
            id="thread",
            source=ThreadSource.INTERNAL,
            created_at=datetime(2026, 7, 14),
            updated_at=NOW,
        )


def test_turn_item_round_trip_keeps_discriminator() -> None:
    item = TurnItem(
        id="item:1",
        kind=TurnItemKind.TOOL_CALL,
        data={"name": "shell", "status": "completed"},
    )

    assert TurnItem.from_dict(item.to_dict()) == item
    assert item.to_dict()["type"] == "toolCall"


def test_turn_usage_rejects_fake_negative_counts() -> None:
    with pytest.raises(ValueError, match="不得为负数"):
        TurnUsage(input_tokens=-1)


def test_turn_result_has_stable_wire_shape_and_duration() -> None:
    record = TurnRecord(
        id="turn:1",
        thread_id="programmatic:1",
        status=TurnStatus.COMPLETED,
        input="你好",
        metadata={"source": "test"},
        items=[
            TurnItem(
                id="item:1",
                kind=TurnItemKind.ASSISTANT_MESSAGE,
                data={"content": "你好"},
            )
        ],
        usage=TurnUsage(input_tokens=10, output_tokens=2, coverage="exact"),
        error=None,
        created_at=NOW,
        started_at=NOW + timedelta(seconds=1),
        completed_at=NOW + timedelta(seconds=2, milliseconds=250),
        final_response="你好",
    )

    payload = TurnResult.from_record(record).to_dict()

    assert payload["threadId"] == "programmatic:1"
    assert payload["status"] == "completed"
    assert payload["durationMs"] == 1250
    assert payload["usage"] == {
        "inputTokens": 10,
        "cachedInputTokens": None,
        "outputTokens": 2,
        "reasoningOutputTokens": None,
        "requestCount": 0,
        "coveredRequestCount": 0,
        "coverage": "exact",
    }
    assert payload["error"] is None


def test_turn_error_requires_boolean_retryable_on_decode() -> None:
    with pytest.raises(ValueError, match="必须是布尔值"):
        TurnError.from_dict({"type": "provider", "message": "failed", "retryable": 1})
