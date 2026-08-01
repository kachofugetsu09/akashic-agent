import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import tomllib
import pytest

from docker.debug.full_replay_runner import (
    FeedItem,
    _patch_config,
    _target_channel,
    admit_replay_items,
    copy_context_snapshot,
    load_feed_union,
    load_score_map,
    load_wake_reservoir,
    occupied_hour_steps,
    refresh_replay_eligibility,
    wait_until_stable,
    write_replay_events,
)
from docker.debug.replay_controller import ReplayLayout, append_event, initialize


def _feed_db(path: Path, rows: list[tuple[object, ...]]) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE items(
            event_id TEXT PRIMARY KEY, source_id TEXT, source_name TEXT,
            source_type TEXT, title TEXT, content TEXT, url TEXT, author TEXT,
            published_at TEXT, first_seen_at TEXT, last_seen_at TEXT,
            content_hash TEXT
        )
        """
    )
    connection.executemany("INSERT INTO items VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    connection.commit()
    connection.close()


def test_replay_admission_uses_published_time_not_first_seen() -> None:
    seen = datetime(2026, 7, 12, tzinfo=UTC)
    items = [
        FeedItem(
            event_id=event_id,
            source_id="source",
            source_name="Feed",
            source_type="rss",
            title=event_id,
            content="",
            url="",
            author="",
            published_at=published_at,
            first_seen_at=seen,
            last_seen_at=seen,
            content_hash=event_id,
        )
        for event_id, published_at in (
            ("fresh", seen),
            ("stale", datetime(2026, 6, 12, tzinfo=UTC)),
            ("missing", None),
            ("missing-strong", None),
        )
    ]
    score_map = {
        item.event_id: (
            0.5,
            {"interest": 0.9 if item.event_id == "missing-strong" else 0.45},
            {},
        )
        for item in items
    }

    admitted = admit_replay_items(items, score_map)

    assert [item.event_id for item in admitted] == ["fresh", "missing-strong"]


def test_feed_union_uses_latest_snapshot_but_first_seen_for_availability(tmp_path) -> None:
    old = tmp_path / "old.db"
    current = tmp_path / "current.db"
    common = (
        "event-1", "source", "Feed", "rss", "old", "body", "https://x", "",
        "2026-07-01T00:00:00Z", "2026-07-11T03:12:00Z",
    )
    _feed_db(old, [(*common, "2026-07-11T03:13:00Z", "old-hash")])
    _feed_db(
        current,
        [
            (
                "event-1", "source", "Feed", "rss", "new", "body", "https://x", "",
                "2026-07-01T00:00:00Z", "2026-07-11T03:12:00Z",
                "2026-07-12T03:13:00Z", "new-hash",
            ),
            (
                "event-2", "source", "Feed", "rss", "later", "body", "https://y", "",
                "2026-07-02T00:00:00Z", "2026-07-11T04:01:00Z",
                "2026-07-11T04:02:00Z", "later-hash",
            ),
        ],
    )

    items = load_feed_union([old, current], datetime(2026, 7, 12, tzinfo=UTC))

    assert [item.event_id for item in items] == ["event-1", "event-2"]
    assert items[0].title == "new"
    assert items[0].first_seen_at == datetime(2026, 7, 11, 3, 12, tzinfo=UTC)


def test_wake_reservoir_supplies_real_events_and_scores(tmp_path) -> None:
    path = tmp_path / "wake.db"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE reservoir_events(
            item_id TEXT PRIMARY KEY, kind TEXT, source_id TEXT,
            original_source_id TEXT, ack_source_id TEXT, source_event_id TEXT,
            published_at TEXT, first_seen_at TEXT, preprocess_score REAL,
            payload_json TEXT, embedding_json TEXT, status TEXT, consumed_at TEXT
        )
        """
    )
    connection.execute(
        "INSERT INTO reservoir_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "feed:event-1", "content", "feed", "source", "feed", "event-1",
            "2026-07-12T00:00:00+00:00", "2026-07-12T01:00:00+00:00",
            0.4,
            json.dumps(
                {
                    "title": "真实池内容",
                    "source_name": "Source",
                    "preprocess_features": {"interest": 0.7},
                },
                ensure_ascii=False,
            ),
            None, "unread", None,
        ),
    )
    connection.commit()
    connection.close()

    items, scores = load_wake_reservoir(
        [path], datetime(2026, 7, 13, tzinfo=UTC)
    )

    assert [item.title for item in items] == ["真实池内容"]
    assert scores["event-1"][0] == 0.4
    assert scores["event-1"][1]["interest"] == 0.7


def test_occupied_hour_steps_groups_by_first_seen_hour(tmp_path) -> None:
    feed = tmp_path / "feed.db"
    _feed_db(
        feed,
        [
            (
                event_id, "source", "Feed", "rss", event_id, "body", "", "",
                "2026-07-01T00:00:00Z", first_seen, first_seen, event_id,
            )
            for event_id, first_seen in (
                ("one", "2026-07-11T03:12:00Z"),
                ("two", "2026-07-11T03:55:00Z"),
                ("three", "2026-07-11T05:01:00Z"),
            )
        ],
    )
    end_at = datetime(2026, 7, 11, 6, tzinfo=UTC)
    items = load_feed_union([feed], end_at)

    assert occupied_hour_steps(items, end_at) == [
        (datetime(2026, 7, 11, 4, tzinfo=UTC), 2),
        (datetime(2026, 7, 11, 5, tzinfo=UTC), 0),
        (datetime(2026, 7, 11, 6, tzinfo=UTC), 1),
    ]


def test_occupied_hour_steps_includes_partial_final_hour(tmp_path) -> None:
    item = FeedItem(
        event_id="last", source_id="source", source_name="Feed",
        source_type="rss", title="last", content="body", url="", author="",
        published_at=datetime(2026, 7, 13, 13, 5, tzinfo=UTC),
        first_seen_at=datetime(2026, 7, 13, 13, 10, tzinfo=UTC),
        last_seen_at=datetime(2026, 7, 13, 13, 10, tzinfo=UTC),
        content_hash="last",
    )

    assert occupied_hour_steps(
        [item], datetime(2026, 7, 13, 13, 30, tzinfo=UTC)
    ) == [(datetime(2026, 7, 13, 13, 30, tzinfo=UTC), 1)]


def test_feed_union_keeps_item_without_published_at(tmp_path) -> None:
    feed = tmp_path / "feed.db"
    _feed_db(
        feed,
        [
            (
                "no-published", "source", "Feed", "rss", "title", "body", "", "",
                None, "2026-07-11T03:12:00Z", "2026-07-11T03:13:00Z", "hash",
            )
        ],
    )

    items = load_feed_union([feed], datetime(2026, 7, 12, tzinfo=UTC))

    assert len(items) == 1
    assert items[0].published_at is None


def test_feed_union_deduplicates_x_backfill_and_recovers_status_time(tmp_path) -> None:
    old = tmp_path / "old.db"
    fixed = tmp_path / "fixed.db"
    url = "https://nitter.net/user/status/2074185390060110138#m"
    common = (
        "source", "X", "rss", "title", "body", url, "",
    )
    _feed_db(
        old,
        [
            (
                "old-id", *common, None, "2026-07-07T05:10:00Z",
                "2026-07-07T05:20:00Z", "old-hash",
            )
        ],
    )
    _feed_db(
        fixed,
        [
            (
                "fixed-id", *common, "2026-07-06T17:35:08.148Z",
                "2026-07-12T05:10:00Z", "2026-07-12T05:30:00Z", "new-hash",
            )
        ],
    )

    items = load_feed_union([old, fixed], datetime(2026, 7, 13, tzinfo=UTC))

    assert len(items) == 1
    assert items[0].event_id == "fixed-id"
    assert items[0].first_seen_at == datetime(
        2026, 7, 7, 5, 10, tzinfo=UTC
    )
    assert items[0].published_at == datetime(
        2026, 7, 6, 17, 35, 8, 148000, tzinfo=UTC
    )


def test_patch_config_preserves_history_target_and_disables_external_channel(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
        [channels.telegram]
        enabled = true
        [proactive]
        enabled = false
        [proactive.target]
        channel = "telegram"
        chat_id = "7674283004"
        """,
        encoding="utf-8",
    )

    _patch_config(path)
    config = tomllib.loads(path.read_text(encoding="utf-8"))

    assert config["proactive"]["target"] == {
        "channel": "telegram", "chat_id": "7674283004"
    }
    assert config["channels"]["telegram"]["enabled"] is False
    assert "integrations" not in config
    assert _target_channel(path) == "telegram"


def test_score_map_and_missing_scores_are_explicit(tmp_path) -> None:
    score_path = tmp_path / "scores.json"
    score_path.write_text(
        json.dumps(
            {
                "one": {
                    "score": 0.8,
                    "features": {"personal": 0.7},
                    "published_at": "2026-07-11T23:00:00+00:00",
                    "wake_eligible": False,
                    "freshness_reason": "outside_retention_window",
                }
            }
        ),
        encoding="utf-8",
    )
    score_map = load_score_map(score_path)
    replay_root = tmp_path / "replay"
    layout = ReplayLayout(
        tmp_path, replay_root, replay_root / "clock.json",
        replay_root / "events.jsonl", replay_root / "outbox.jsonl",
    )
    now = datetime(2026, 7, 12, tzinfo=UTC)
    _ = initialize(layout, now)
    items = [
        FeedItem(
            event_id=event_id, source_id="source", source_name="Feed",
            source_type="rss", title=event_id, content="body", url="", author="",
            published_at=now, first_seen_at=now, last_seen_at=now,
            content_hash=event_id,
        )
        for event_id in ("one", "two")
    ]

    missing = write_replay_events(layout, items, score_map)
    events = [json.loads(line) for line in layout.events_path.read_text().splitlines()]

    assert missing == 1
    assert events[0]["preprocess_score"] == 0.8
    assert events[0]["payload"]["features"] == {"personal": 0.7}
    assert events[0]["published_at"] == "2026-07-11T23:00:00+00:00"
    assert events[0]["wake_eligible"] is False
    assert events[1]["preprocess_score"] == 0.0


def test_replay_event_marks_missing_or_stale_publication_ineligible(tmp_path) -> None:
    replay_root = tmp_path / "replay"
    layout = ReplayLayout(
        tmp_path, replay_root, replay_root / "clock.json",
        replay_root / "events.jsonl", replay_root / "outbox.jsonl",
    )
    seen = datetime(2026, 7, 12, tzinfo=UTC)
    _ = initialize(layout, seen)
    items = [
        FeedItem(
            event_id="missing", source_id="source", source_name="Feed",
            source_type="rss", title="missing", content="", url="", author="",
            published_at=None, first_seen_at=seen, last_seen_at=seen,
            content_hash="missing",
        ),
        FeedItem(
            event_id="stale", source_id="source", source_name="Feed",
            source_type="rss", title="stale", content="", url="", author="",
            published_at=seen.replace(day=1), first_seen_at=seen, last_seen_at=seen,
            content_hash="stale",
        ),
    ]

    write_replay_events(layout, items)
    events = [json.loads(line) for line in layout.events_path.read_text().splitlines()]

    assert [event["wake_eligible"] for event in events] == [False, False]
    assert events[0]["published_at"] is None


def test_resume_refreshes_existing_replay_event_eligibility(tmp_path) -> None:
    replay_root = tmp_path / "replay"
    layout = ReplayLayout(
        tmp_path, replay_root, replay_root / "clock.json",
        replay_root / "events.jsonl", replay_root / "outbox.jsonl",
    )
    seen = datetime(2026, 7, 12, tzinfo=UTC)
    _ = initialize(layout, seen)
    item = FeedItem(
        event_id="old", source_id="source", source_name="Feed",
        source_type="rss", title="old", content="", url="", author="",
        published_at=seen.replace(day=1), first_seen_at=seen, last_seen_at=seen,
        content_hash="old",
    )
    _ = append_event(
        layout,
        {
            "event_id": "old", "source_id": "source", "kind": "content",
            "published_at": seen, "available_at": seen,
        },
    )

    refresh_replay_eligibility(layout, [item])

    event = json.loads(layout.events_path.read_text().strip())
    assert event["wake_eligible"] is False


def test_context_snapshot_is_copied_without_template_runtime_state(tmp_path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    sessions = sources / "sessions.db"
    akasha = sources / "akasha.db"
    memory = sources / "MEMORY.md"
    proactive = sources / "PROACTIVE_CONTEXT.md"
    for path, content in (
        (sessions, "sessions"), (akasha, "akasha"),
        (memory, "memory"), (proactive, "proactive"),
    ):
        path.write_text(content, encoding="utf-8")
    workspace = tmp_path / "workspace"

    copy_context_snapshot(
        workspace,
        sessions_db=sessions,
        akasha_db=akasha,
        memory_md=memory,
        proactive_context=proactive,
    )

    assert (workspace / "sessions.db").read_text() == "sessions"
    assert (workspace / "memory" / "akasha.db").read_text() == "akasha"
    assert (workspace / "memory" / "MEMORY.md").read_text() == "memory"
    assert (workspace / "PROACTIVE_CONTEXT.md").read_text() == "proactive"
    assert not (workspace / "wake_proactive.db").exists()


def test_wait_aborts_after_three_incomplete_wakes_at_same_time(tmp_path) -> None:
    replay_root = tmp_path / "workspace" / "replay"
    layout = ReplayLayout(
        tmp_path, replay_root, replay_root / "clock.json",
        replay_root / "events.jsonl", replay_root / "outbox.jsonl",
    )
    target = datetime(2026, 7, 12, tzinfo=UTC)
    _ = initialize(layout, target)
    connection = sqlite3.connect(tmp_path / "workspace" / "wake_proactive.db")
    connection.execute(
        """
        CREATE TABLE wake_runs(
            wake_id TEXT PRIMARY KEY, now_utc TEXT NOT NULL,
            terminal_action TEXT, final_message TEXT NOT NULL
        )
        """
    )
    connection.executemany(
        "INSERT INTO wake_runs VALUES(?,?,NULL,'')",
        [(f"wake-{index}", target.isoformat()) for index in range(3)],
    )
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="3 个未终止 wake"):
        wait_until_stable(
            layout,
            target=target,
            expected_events=0,
            timeout=1,
            quiet_seconds=0,
        )


def test_wait_does_not_advance_while_wake_is_incomplete(tmp_path) -> None:
    replay_root = tmp_path / "workspace" / "replay"
    layout = ReplayLayout(
        tmp_path, replay_root, replay_root / "clock.json",
        replay_root / "events.jsonl", replay_root / "outbox.jsonl",
    )
    target = datetime(2026, 7, 12, tzinfo=UTC)
    _ = initialize(layout, target)
    connection = sqlite3.connect(tmp_path / "workspace" / "wake_proactive.db")
    connection.execute(
        """
        CREATE TABLE wake_runs(
            wake_id TEXT PRIMARY KEY, now_utc TEXT NOT NULL,
            terminal_action TEXT, final_message TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO wake_runs VALUES('wake-1', ?, NULL, '')",
        (target.isoformat(),),
    )
    connection.commit()
    connection.close()

    with pytest.raises(TimeoutError):
        wait_until_stable(
            layout,
            target=target,
            expected_events=0,
            timeout=0.05,
            quiet_seconds=0,
        )


def test_wait_accepts_noop_target_after_minimum_wait(tmp_path) -> None:
    replay_root = tmp_path / "workspace" / "replay"
    layout = ReplayLayout(
        tmp_path, replay_root, replay_root / "clock.json",
        replay_root / "events.jsonl", replay_root / "outbox.jsonl",
    )
    target = datetime(2026, 7, 12, tzinfo=UTC)
    _ = initialize(layout, target)

    snapshot = wait_until_stable(
        layout,
        target=target,
        expected_events=0,
        timeout=1,
        quiet_seconds=0,
        minimum_wait_seconds=0.01,
    )

    assert target.isoformat() not in snapshot["processed_times"]


def test_wait_accepts_stale_incomplete_attempt_after_later_success(tmp_path) -> None:
    replay_root = tmp_path / "workspace" / "replay"
    layout = ReplayLayout(
        tmp_path, replay_root, replay_root / "clock.json",
        replay_root / "events.jsonl", replay_root / "outbox.jsonl",
    )
    target = datetime(2026, 7, 12, tzinfo=UTC)
    _ = initialize(layout, target)
    connection = sqlite3.connect(tmp_path / "workspace" / "wake_proactive.db")
    connection.execute(
        """
        CREATE TABLE wake_runs(
            wake_id TEXT PRIMARY KEY, now_utc TEXT NOT NULL,
            terminal_action TEXT, final_message TEXT NOT NULL
        )
        """
    )
    connection.executemany(
        "INSERT INTO wake_runs VALUES(?,?,?,?)",
        [
            ("failed", target.isoformat(), None, ""),
            ("success", target.isoformat(), "skip", ""),
        ],
    )
    connection.commit()
    connection.close()

    snapshot = wait_until_stable(
        layout,
        target=target,
        expected_events=0,
        timeout=1,
        quiet_seconds=0,
    )

    assert snapshot["incomplete_wake_times"] == {target.isoformat(): 1}
    assert snapshot["active_incomplete_wake_times"] == {}
