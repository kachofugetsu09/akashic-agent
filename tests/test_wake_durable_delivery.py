from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from plugins.eventmail.store import EventMailStore


@pytest.mark.parametrize("selected_position", [0, 1])
def test_uncertain_batch_locks_only_cited_member_from_next_selection(
    tmp_path: Path, selected_position: int
) -> None:
    """Wake's Message source keeps uncited Content available after an uncertain send."""

    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = EventMailStore(tmp_path / "content.sqlite3")
    content.submit(
        "feed",
        "poll:first",
        (
            {
                "item_id": "uncited",
                "revision": "1",
                "payload": {"title": "uncited"},
                "not_before": now,
                "requires_ack": False,
            },
            {
                "item_id": "cited",
                "revision": "1",
                "payload": {"title": "cited"},
                "not_before": now,
                "requires_ack": False,
            },
        ),
    )

    snapshot = content.snapshot(now)
    accepted = {"session_id": "wake:default", "turn_id": "turn:uncertain"}
    selected = content.select_batch(
        tuple(item["ref"] for item in snapshot["items"]),
        snapshot["snapshot_seq"],
        accepted,
        now,
    )
    token = selected["selection_token"]
    assert isinstance(token, str)
    cited = snapshot["items"][selected_position]["ref"]
    assert content.transition(token, "ready_for_delivery", selected_refs=(cited,))["changed"] is True

    available = content.snapshot(now)
    available_ids = {str(item["ref"]["item_id"]) for item in available["items"]}
    expected_uncited = {"cited", "new"} if selected_position == 0 else {"uncited", "new"}
    assert available_ids == expected_uncited - {"new"}

    content.submit(
        "feed",
        "poll:second",
        (
            {
                "item_id": "new",
                "revision": "1",
                "payload": {"title": "new"},
                "not_before": now,
                "requires_ack": False,
            },
        ),
    )
    available = content.snapshot(now)
    available_ids = {str(item["ref"]["item_id"]) for item in available["items"]}
    assert available_ids == expected_uncited
    next_selected = content.select_batch(
        tuple(item["ref"] for item in available["items"]),
        available["snapshot_seq"],
        {"session_id": "wake:default", "turn_id": "turn:next"},
        now,
    )
    assert next_selected["selected"] is True
