"""Real SQLite fixtures for the proactive-island active-state handoff."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

import agent.migrations.proactive_island.cli as handoff_cli
from agent.migrations.proactive_island.cli import apply as apply_cli
from agent.migrations.proactive_island.cli import backup_sources
from agent.migrations.proactive_island.cli import plan as plan_cli
from agent.migrations.proactive_island.cli import retire as retire_cli
from agent.migrations.proactive_island.handoff import (
    AdapterPlan,
    HandoffAdapter,
    HandoffStatus,
    TargetReceipt,
    apply_handoff,
    preflight_handoff,
    receipt_digest,
)
from agent.migrations.proactive_island.history import LegacyProactiveHistory
from agent.migrations.proactive_island.inventory import (
    LegacyFact,
    LegacyFactKind,
    inventory_digest,
    inventory_workspace,
)
from plugins.eventmail.store import EventMailStore
from plugins.wake.legacy_rules import read_archived_rules
from agent.migrations.proactive_island.wake_rules import WakeRulesArchiveAdapter
from scripts.proactive_island_handoff import main as handoff_main
from tests.fixtures.legacy_wake_state import (
    create_legacy_wake_database,
    populate_continuity_table,
)


def _workspace_state(root: Path) -> tuple[tuple[str, str, str], ...]:
    """Capture every path and file digest for a zero-write boundary oracle."""

    state = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_file():
            state.append(
                (relative, "file", hashlib.sha256(path.read_bytes()).hexdigest())
            )
        else:
            state.append((relative, "directory", ""))
    return tuple(state)


def _wake_db(
    workspace: Path,
    rows: list[tuple[object, ...]],
    acknowledgements: list[tuple[object, ...]] | None = None,
) -> Path:
    path = workspace / "wake_proactive.db"
    workspace.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE reservoir_events(
            item_id TEXT PRIMARY KEY, kind TEXT NOT NULL, source_id TEXT NOT NULL,
            original_source_id TEXT NOT NULL, ack_source_id TEXT,
            source_event_id TEXT NOT NULL, published_at TEXT NOT NULL,
            first_seen_at TEXT NOT NULL, preprocess_score REAL NOT NULL,
            payload_json TEXT NOT NULL, embedding_json TEXT, status TEXT NOT NULL,
            consumed_at TEXT
        );
        CREATE TABLE pending_acknowledgements(
            source_id TEXT NOT NULL, source_event_id TEXT NOT NULL,
            item_id TEXT NOT NULL DEFAULT '', action TEXT NOT NULL DEFAULT 'consume',
            queued_at TEXT NOT NULL,
            PRIMARY KEY(source_id, source_event_id, item_id)
        );
        CREATE TABLE wake_runs(
            wake_id TEXT PRIMARY KEY, session_key TEXT NOT NULL, now_utc TEXT NOT NULL,
            scratchpad_json TEXT NOT NULL, investigations_json TEXT NOT NULL,
            final_message TEXT NOT NULL, cited_ids_json TEXT NOT NULL,
            display_event_map_json TEXT NOT NULL, source_refs_json TEXT NOT NULL,
            investigation_completed INTEGER NOT NULL DEFAULT 0,
            terminal_action TEXT
        );
        """)
    connection.executemany(
        "INSERT INTO reservoir_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    connection.executemany(
        "INSERT INTO pending_acknowledgements VALUES(?,?,?,?,?)", acknowledgements or []
    )
    connection.commit()
    connection.close()
    return path


def _wake_row(
    index: int,
    *,
    source: str = "feed@github:subscriptions",
    status: str = "unread",
    kind: str = "content",
    event_id: str | None = None,
    payload: Mapping[str, object] | None = None,
) -> tuple[object, ...]:
    event = event_id or f"event-{index}"
    body = dict(payload or {"title": f"item {index}", "content": f"body {index}"})
    body.update({"event_id": event, "ack_server": source, "kind": kind})
    return (
        f"{source}:{event}",
        kind,
        source,
        "feed",
        source,
        event,
        f"2026-08-23T00:{index:02d}:00+00:00",
        f"2026-08-23T01:{index:02d}:00+00:00",
        0.5,
        json.dumps(body, sort_keys=True),
        None,
        status,
        None,
    )


def _provider_db(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE items(event_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO items VALUES(?, ?)",
        [(f"event-{index}", f"revision-{index}") for index in range(rows)],
    )
    connection.commit()
    connection.close()


def _proactive_db(workspace: Path) -> Path:
    """Create a reduced fixture with the exact formal proactive table set."""

    path = workspace / "proactive.db"
    workspace.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE deliveries(
            session_key TEXT, delivery_key TEXT, sent_at TEXT,
            PRIMARY KEY(session_key, delivery_key)
        );
        CREATE TABLE session_state(
            session_key TEXT, key TEXT, value TEXT,
            PRIMARY KEY(session_key, key)
        );
        CREATE TABLE context_only_timestamps(
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_key TEXT, ts TEXT
        );
        CREATE TABLE tick_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT, tick_id TEXT, session_key TEXT,
            started_at TEXT, finished_at TEXT
        );
        CREATE TABLE tick_step_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT, tick_id TEXT, step_index INTEGER,
            phase TEXT, tool_name TEXT
        );
        CREATE TABLE rejection_cooldown(item_id TEXT PRIMARY KEY, until_utc TEXT);
        CREATE TABLE seen_items(item_id TEXT PRIMARY KEY, seen_at TEXT);
        CREATE TABLE semantic_items(item_id TEXT PRIMARY KEY, embedding BLOB);
        CREATE TABLE kv_state(key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO deliveries VALUES('wake:default', 'delivery:1', '2026-08-23');
        INSERT INTO deliveries VALUES('wake:default', 'delivery:2', '2026-08-23');
        INSERT INTO session_state VALUES('wake:default', 'last_tick', 'one');
        INSERT INTO context_only_timestamps(session_key, ts)
            VALUES('wake:default', '2026-08-23');
        INSERT INTO tick_log(tick_id, session_key, started_at, finished_at)
            VALUES('tick:1', 'wake:default', '2026-08-23', '2026-08-23');
        INSERT INTO tick_step_log(tick_id, step_index, phase, tool_name)
            VALUES('tick:1', 0, 'content', 'poll');
        INSERT INTO rejection_cooldown VALUES('old:1', '2026-08-24');
        INSERT INTO seen_items VALUES('old:2', '2026-08-23');
        INSERT INTO semantic_items VALUES('old:3', X'0001');
        INSERT INTO kv_state VALUES('cursor', 'three');
        """)
    connection.commit()
    connection.close()
    return path


def _populate_wake_continuity(workspace: Path, table: str) -> Path:
    """Populate the frozen legacy schema with one continuity fact."""

    path = workspace / "wake_proactive.db"
    create_legacy_wake_database(path)
    populate_continuity_table(path, table)
    return path


class _SourceAdapter(HandoffAdapter):
    """Simulate only a source-owned provider join around the real Content store."""

    def __init__(self, provider: Path, content: EventMailStore) -> None:
        self.provider = provider
        self.content = content
        self.source_id = "feed-subscriptions"
        self.apply_calls = 0

    def accepts(self, fact: LegacyFact) -> bool:
        return (
            fact.kind is LegacyFactKind.WAKE_SOURCE_ITEM
            and fact.source_identity == "feed@github:subscriptions"
        )

    def plan(self, fact: LegacyFact) -> AdapterPlan:
        row = self._row(fact)
        event_id = cast(str, row["source_event_id"])
        connection = sqlite3.connect(
            self.provider.resolve().as_uri() + "?mode=ro", uri=True
        )
        result = connection.execute(
            "SELECT content_hash FROM items WHERE event_id=?", (event_id,)
        ).fetchone()
        connection.close()
        if result is None:
            raise RuntimeError("provider revision missing")
        return AdapterPlan(f"content:{self.source_id}:{event_id}:{str(result[0])}")

    def apply(self, fact: LegacyFact, plan: AdapterPlan) -> TargetReceipt:
        self.apply_calls += 1
        _, _, event_id, revision = plan.target_identity.split(":", 3)
        row = self._row(fact)
        payload = json.loads(cast(str, row["payload_json"]))
        batch_id = f"legacy-wake:{event_id}:{revision}"
        receipt = self.content.submit(
            self.source_id,
            batch_id,
            (
                {
                    "item_id": event_id,
                    "revision": revision,
                    "payload": payload,
                    "not_before": row["published_at"],
                    "requires_ack": True,
                },
            ),
        )
        normalized = {"target_identity": plan.target_identity, "receipt": receipt}
        return TargetReceipt(
            receipt_id=cast(str, receipt["receipt_id"]),
            receipt_digest=receipt_digest(normalized),
            target_identity=plan.target_identity,
        )

    def verify(self, fact: LegacyFact, receipt: TargetReceipt) -> bool:
        plan = self.plan(fact)
        if receipt.target_identity != plan.target_identity:
            return False
        _, _, event_id, revision = plan.target_identity.split(":", 3)
        batch_id = f"legacy-wake:{event_id}:{revision}"
        submission = self.content.read_submission(self.source_id, batch_id)
        item = self.content.read_revision(self.source_id, event_id, revision)
        if submission is None or item is None:
            return False
        normalized = {"target_identity": plan.target_identity, "receipt": submission}
        return receipt.receipt_id == submission[
            "receipt_id"
        ] and receipt.receipt_digest == receipt_digest(normalized)

    @staticmethod
    def _row(fact: LegacyFact) -> dict[str, object]:
        value = json.loads(fact.opaque)
        assert isinstance(value, dict)
        return cast(dict[str, object], value)


def test_empty_plan_and_apply_write_nothing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    before = tuple(workspace.rglob("*"))

    assert plan_cli(workspace).status is HandoffStatus.READY
    report = apply_cli(workspace.resolve(), tmp_path / "unused-backup")

    assert report.status is HandoffStatus.READY
    assert before == ()
    assert tuple(workspace.rglob("*")) == before
    assert not (tmp_path / "unused-backup").exists()


def test_active_source_plan_does_not_mount_or_initialize_content(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _wake_db(workspace, [_wake_row(1)])
    provider = tmp_path / "feed.sqlite3"
    _provider_db(provider, 2)
    target = workspace / "plugin-data" / "eventmail-builtin" / "eventmail.sqlite3"
    adapter = _SourceAdapter(provider, EventMailStore(target))
    before = _workspace_state(workspace)

    report = preflight_handoff(workspace, inventory_workspace(workspace), (adapter,))

    assert report.status is HandoffStatus.PLAN
    assert _workspace_state(workspace) == before
    assert not target.exists()


def test_null_ack_source_uses_exact_reservoir_source_owner(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    row = list(_wake_row(1, source="feed@github:subscriptions"))
    row[4] = None
    _wake_db(workspace, [tuple(row)])

    inventory = inventory_workspace(workspace)

    assert inventory.blocks == ()
    assert inventory.facts[0].source_identity == "feed@github:subscriptions"


def test_conflicting_wake_source_identity_blocks_with_exact_row_digest(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    first = _wake_row(1, event_id="same-event")
    second = list(_wake_row(2, event_id="same-event"))
    second[0] = "zz-another-item-id"
    _wake_db(workspace, [first, tuple(second)])

    inventory = inventory_workspace(workspace)

    conflict = next(
        block
        for block in inventory.blocks
        if block.reason == "source_identity_conflict"
    )
    assert conflict.locator == "wake:reservoir_events:zz-another-item-id"
    assert len(conflict.source_digest) == 64


def test_unknown_proactive_table_blocks_without_copying_rows(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = workspace / "proactive.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE future_state(id TEXT, payload TEXT)")
    connection.execute("INSERT INTO future_state VALUES('one', 'opaque')")
    connection.commit()
    connection.close()

    inventory = inventory_workspace(workspace)

    assert len(inventory.blocks) == 1
    assert inventory.blocks[0].locator == "proactive:future_state"
    assert inventory.blocks[0].reason == "unknown_proactive_table"
    assert len(inventory.blocks[0].source_digest) == 64


def test_duplicate_target_owners_block_without_calling_plan(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _wake_db(workspace, [_wake_row(1)])
    provider = tmp_path / "feed.sqlite3"
    _provider_db(provider, 2)
    adapter = _SourceAdapter(provider, EventMailStore(tmp_path / "unused.sqlite3"))

    report = preflight_handoff(
        workspace, inventory_workspace(workspace), (adapter, adapter)
    )

    assert report.status is HandoffStatus.BLOCK
    assert report.items[0].reason == "owner_adapter_conflict"
    assert not (tmp_path / "unused.sqlite3").exists()


def test_formal_shape_inventory_keeps_generic_job_and_terminal_drift_historical(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _wake_db(workspace, [_wake_row(index) for index in range(15)])
    rules = b"x" * 7474
    (workspace / "PROACTIVE_CONTEXT.md").write_bytes(rules)
    drift = workspace / "drift" / "drift.db"
    drift.parent.mkdir()
    connection = sqlite3.connect(drift)
    connection.executescript("""
        CREATE TABLE skill_continuum(skill_name TEXT, last_status TEXT);
        CREATE TABLE runs(
            id INTEGER PRIMARY KEY, event_id TEXT, run_at TEXT, skill_name TEXT,
            status TEXT, briefing TEXT, message_result TEXT
        );
        INSERT INTO skill_continuum VALUES('one', 'completed');
        """)
    connection.executemany(
        "INSERT INTO runs VALUES(?,?,?,?,?,?,?)",
        [
            (index, f"drift-{index}", "2026-08-23", "one", "paused", "done", "silent")
            for index in range(13)
        ],
    )
    connection.commit()
    connection.close()
    jobs = workspace / "runtime" / "plugin-jobs" / "outcomes.sqlite"
    jobs.parent.mkdir(parents=True)
    connection = sqlite3.connect(jobs)
    connection.execute(
        "CREATE TABLE job_outcomes(plugin_id TEXT, job_name TEXT, invocation_id TEXT, "
        "state TEXT, created_at TEXT, event_payload_json TEXT)"
    )
    connection.executemany(
        "INSERT INTO job_outcomes VALUES(?,?,?,?,?,?)",
        [
            ("github-watch", "poll", "running", "running", "2026-08-23", None),
            (
                "emotion",
                "merge_proactive_pending",
                "done",
                "succeeded",
                "2026-08-22",
                "{}",
            ),
        ],
    )
    connection.commit()
    connection.close()

    inventory = inventory_workspace(workspace)

    assert len(inventory.facts) == 16
    assert inventory.blocks == ()
    assert {
        fact.source_identity
        for fact in inventory.facts
        if fact.kind is LegacyFactKind.WAKE_SOURCE_ITEM
    } == {"feed@github:subscriptions"}
    rules_fact = next(
        fact for fact in inventory.facts if fact.kind is LegacyFactKind.WAKE_RULES
    )
    assert rules_fact.source_digest == hashlib.sha256(rules).hexdigest()
    history = LegacyProactiveHistory(workspace)
    assert len(history.drift_runs()) == 13
    assert {row["invocation_id"] for row in history.job_outcomes()} == {
        "running",
        "done",
    }


def test_proactive_continuity_blocks_once_per_table_and_history_stays_readable(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _proactive_db(workspace)

    inventory = inventory_workspace(workspace)

    assert {block.locator for block in inventory.blocks} == {
        "proactive:deliveries",
        "proactive:session_state",
        "proactive:context_only_timestamps",
        "proactive:rejection_cooldown",
        "proactive:seen_items",
        "proactive:kv_state",
    }
    deliveries = next(
        block for block in inventory.blocks if block.locator == "proactive:deliveries"
    )
    assert deliveries.reason == "proactive_continuity_owner_unavailable"
    assert deliveries.source_digest.startswith("rows=2;sha256=")
    history = LegacyProactiveHistory(workspace).proactive_tables()
    assert set(history) == {
        "deliveries",
        "session_state",
        "context_only_timestamps",
        "tick_log",
        "tick_step_log",
        "rejection_cooldown",
        "seen_items",
        "semantic_items",
        "kv_state",
    }
    assert history["semantic_items"][0]["embedding"] == {"sqlite_blob_hex": "0001"}


def test_existing_proactive_quota_blocks_by_exact_bytes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    content = b'{"version":1,"used":1,"window":"2026-07-12"}\n'
    (workspace / "proactive_quota.json").write_bytes(content)

    inventory = inventory_workspace(workspace)

    assert inventory.blocks[0].locator == "proactive:quota"
    assert inventory.blocks[0].reason == "proactive_quota_owner_unavailable"
    assert inventory.blocks[0].source_digest == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize(
    "table",
    [
        "reservoir_quarantine",
        "reservoir_tombstones",
        "hazard_state",
        "context_state",
        "context_reevaluate_state",
        "drift_state",
    ],
)
def test_real_wake_continuity_table_blocks_without_target_or_lineage(
    tmp_path: Path, table: str
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    _populate_wake_continuity(workspace, table)
    before = _workspace_state(workspace)

    planned = plan_cli(workspace)
    applied = apply_cli(workspace, (tmp_path / f"backup-{table}").resolve())

    assert planned.status is HandoffStatus.BLOCK
    assert applied.status is HandoffStatus.BLOCK
    block = next(item for item in planned.items if item.locator == f"wake:{table}")
    assert block.reason == "wake_continuity_owner_unavailable"
    assert block.source_digest.startswith("rows=1;sha256=")
    assert _workspace_state(workspace) == before
    assert not (workspace / "runtime").exists()
    assert not (workspace / "plugin-data").exists()
    assert not (tmp_path / f"backup-{table}").exists()


def test_real_wake_schema_unknown_table_blocks(tmp_path: Path) -> None:
    workspace = (tmp_path / "workspace").resolve()
    path = workspace / "wake_proactive.db"
    create_legacy_wake_database(path)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE future_wake_state(id TEXT, payload TEXT)")
    connection.execute("INSERT INTO future_wake_state VALUES('one', 'opaque')")
    connection.commit()
    connection.close()
    before = _workspace_state(workspace)

    backup = (tmp_path / "unknown-table-backup").resolve()
    planned = plan_cli(workspace)

    assert planned.status is HandoffStatus.BLOCK
    assert _workspace_state(workspace) == before
    assert not (workspace / "runtime").exists()
    assert not (
        workspace / "runtime" / "proactive-island-handoff" / "lineage.sqlite3"
    ).exists()
    assert not (workspace / "plugin-data").exists()
    assert not backup.exists()
    item = next(
        item for item in planned.items if item.locator == "wake:future_wake_state"
    )
    assert item.reason == "unknown_wake_table"
    assert len(item.source_digest) == 64

    applied = apply_cli(workspace, backup)

    assert applied.status is HandoffStatus.BLOCK
    assert any(item.reason == "unknown_wake_table" for item in applied.items)
    assert _workspace_state(workspace) == before
    assert not (workspace / "runtime").exists()
    assert not (
        workspace / "runtime" / "proactive-island-handoff" / "lineage.sqlite3"
    ).exists()
    assert not (workspace / "plugin-data").exists()
    assert not backup.exists()


def test_real_wake_history_tables_decode_without_blocking(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    path = workspace / "wake_proactive.db"
    now = datetime(2026, 8, 23, tzinfo=UTC)
    create_legacy_wake_database(path)
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO wake_observations(wake_id, session_key, kind, now_utc, "
        "trigger_json, candidates_json, llm_input_json) VALUES(?,?,?,?,?,?,?)",
        (
            "wake:one",
            "wake:default",
            "fixture",
            now.isoformat(),
            json.dumps({"timer": "one"}),
            json.dumps([{"item_id": "one"}]),
            json.dumps([{"role": "user"}]),
        ),
    )
    connection.execute(
        "INSERT INTO hazard_monitor VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "wake:default",
            0.1,
            0.2,
            0.1,
            0.5,
            0.3,
            0.2,
            "one",
            1,
            0,
            now.isoformat(),
        ),
    )
    connection.execute(
        "INSERT INTO wake_runs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "wake:one",
            "wake:default",
            now.isoformat(),
            "{}",
            "{}",
            "",
            "[]",
            "{}",
            "[]",
            0,
            "skip",
        ),
    )
    connection.commit()
    connection.close()

    inventory = inventory_workspace(workspace)
    history = LegacyProactiveHistory(workspace)

    assert inventory.blocks == ()
    assert inventory.facts == ()
    assert history.wake_runs()[0]["wake_id"] == "wake:one"
    assert history.wake_observations()[0]["trigger"] == {"timer": "one"}
    assert history.wake_observations()[0]["candidates"] == [{"item_id": "one"}]
    assert history.wake_hazard_monitor()[0]["driver_item_id"] == "one"


def test_target_first_crash_replays_without_duplicate_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _wake_db(workspace, [_wake_row(1)])
    provider = tmp_path / "feed.sqlite3"
    _provider_db(provider, 2)
    content = EventMailStore(tmp_path / "content.sqlite3")
    content.initialize()
    adapter = _SourceAdapter(provider, content)
    inventory = inventory_workspace(workspace)

    def crash(_fact: LegacyFact, _receipt: TargetReceipt) -> None:
        raise RuntimeError("crash after target")

    with pytest.raises(RuntimeError, match="crash after target"):
        apply_handoff(workspace, inventory, (adapter,), after_target=crash)
    assert content.state_counts() == {"pending": 1}
    assert (
        workspace / "runtime" / "proactive-island-handoff" / "lineage.sqlite3"
    ).is_file()
    connection = sqlite3.connect(
        workspace / "runtime" / "proactive-island-handoff" / "lineage.sqlite3"
    )
    assert connection.execute("SELECT count(*) FROM lineage").fetchone()[0] == 0
    connection.close()

    report = apply_handoff(workspace, inventory, (adapter,))

    assert report.status is HandoffStatus.APPLIED
    assert adapter.apply_calls == 2
    assert content.state_counts() == {"pending": 1}
    connection = sqlite3.connect(
        workspace / "runtime" / "proactive-island-handoff" / "lineage.sqlite3"
    )
    assert connection.execute("SELECT count(*) FROM lineage").fetchone()[0] == 1
    assert connection.execute(
        "SELECT completed_at IS NOT NULL, count(*) FROM attempts "
        "GROUP BY completed_at IS NOT NULL ORDER BY completed_at IS NOT NULL"
    ).fetchall() == [(0, 1), (1, 1)]
    connection.close()

    before_verify = _workspace_state(workspace)
    assert (
        preflight_handoff(workspace, inventory, (adapter,)).status
        is HandoffStatus.APPLIED
    )
    assert _workspace_state(workspace) == before_verify


def test_preflight_blocks_when_provider_replans_another_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _wake_db(workspace, [_wake_row(1)])
    provider = tmp_path / "feed.sqlite3"
    _provider_db(provider, 2)
    content = EventMailStore(tmp_path / "content.sqlite3")
    content.initialize()
    adapter = _SourceAdapter(provider, content)
    inventory = inventory_workspace(workspace)
    assert (
        apply_handoff(workspace, inventory, (adapter,)).status is HandoffStatus.APPLIED
    )
    connection = sqlite3.connect(provider)
    connection.execute(
        "UPDATE items SET content_hash='revision-changed' WHERE event_id='event-1'"
    )
    connection.commit()
    connection.close()

    report = preflight_handoff(workspace, inventory, (adapter,))

    assert report.status is HandoffStatus.BLOCK
    assert report.items[0].reason == "lineage_target_identity_drift"


def test_apply_keeps_the_preflight_target_when_provider_revision_drifts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _wake_db(workspace, [_wake_row(1)])
    provider = tmp_path / "feed.sqlite3"
    _provider_db(provider, 2)
    content = EventMailStore(tmp_path / "content.sqlite3")
    content.initialize()
    adapter = _SourceAdapter(provider, content)
    inventory = inventory_workspace(workspace)
    planned = preflight_handoff(workspace, inventory, (adapter,))
    connection = sqlite3.connect(provider)
    connection.execute(
        "UPDATE items SET content_hash='revision-changed' WHERE event_id='event-1'"
    )
    connection.commit()
    connection.close()

    report = apply_handoff(
        workspace,
        inventory,
        (adapter,),
        planned=planned,
    )

    assert report.status is HandoffStatus.BLOCK
    assert report.items[0].reason == "target_plan_drift_before_apply"
    assert content.state_counts() == {}
    assert not (
        workspace / "runtime" / "proactive-island-handoff" / "lineage.sqlite3"
    ).exists()


def test_wake_rules_archive_keeps_exact_bytes_and_verified_lineage(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    content = "主动规则\n".encode()
    (workspace / "PROACTIVE_CONTEXT.md").write_bytes(content)
    inventory = inventory_workspace(workspace)
    adapter = WakeRulesArchiveAdapter(workspace)

    assert (
        preflight_handoff(workspace, inventory, (adapter,)).status is HandoffStatus.PLAN
    )
    report = apply_handoff(workspace, inventory, (adapter,))

    assert report.status is HandoffStatus.APPLIED
    archive = (
        workspace
        / "plugin-data"
        / "wake-builtin"
        / "legacy-rules"
        / "PROACTIVE_CONTEXT.md"
    )
    assert archive.read_bytes() == content
    assert plan_cli(workspace).status is HandoffStatus.APPLIED


def test_archived_rules_are_read_from_handoff_archive(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "PROACTIVE_CONTEXT.md").write_text(
        "\n# exact legacy rules\n\n", encoding="utf-8"
    )
    inventory = inventory_workspace(workspace)
    assert (
        apply_handoff(
            workspace, inventory, (WakeRulesArchiveAdapter(workspace),)
        ).status
        is HandoffStatus.APPLIED
    )
    assert read_archived_rules(
        workspace / "plugin-data" / "wake-builtin"
    ) == "# exact legacy rules"


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        ("drift", "proposal_payload_unrecoverable"),
        ("documents", "paired_target_handoff_unavailable"),
        ("pending", "pending_document_owner_unavailable"),
    ],
)
def test_unrecoverable_active_categories_block(
    tmp_path: Path, setup: str, reason: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if setup == "drift":
        path = workspace / "drift" / "drift.db"
        path.parent.mkdir()
        connection = sqlite3.connect(path)
        connection.executescript(
            "CREATE TABLE skill_continuum(skill_name TEXT,last_status TEXT);"
            "CREATE TABLE runs(id INTEGER,event_id TEXT,message_result TEXT);"
            "INSERT INTO skill_continuum VALUES('paused-skill','paused');"
        )
        connection.commit()
        connection.close()
    elif setup == "documents":
        path = workspace / "runtime" / "proactive-documents" / "intents" / "one"
        path.mkdir(parents=True)
        (path / "intent.json").write_text("{}", encoding="utf-8")
    elif setup == "pending":
        (workspace / "proactive_pending.md").write_text("pending\n", encoding="utf-8")
    report = preflight_handoff(workspace, inventory_workspace(workspace), ())

    assert report.status is HandoffStatus.BLOCK
    assert any(item.reason == reason for item in report.items)


def test_historical_projection_never_creates_missing_legacy_state(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    before = tuple(workspace.rglob("*"))

    snapshot = LegacyProactiveHistory(workspace).snapshot()

    assert snapshot == {
        "proactive_tables": {},
        "wake_runs": (),
        "wake_observations": (),
        "wake_hazard_monitor": (),
        "drift_runs": (),
        "job_outcomes": (),
        "document_manifests": (),
    }
    assert tuple(workspace.rglob("*")) == before


def test_history_cli_projects_legacy_rows_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    wake_path = _wake_db(workspace, [])
    connection = sqlite3.connect(wake_path)
    connection.execute(
        "INSERT INTO wake_runs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "wake:one",
            "wake:default",
            "2026-08-23T00:00:00+00:00",
            "{}",
            "[]",
            "hello",
            "[]",
            "{}",
            "[]",
            1,
            "sent",
        ),
    )
    connection.commit()
    connection.close()
    before = _workspace_state(workspace)
    monkeypatch.setattr(
        sys,
        "argv",
        ["proactive_island_handoff.py", "--workspace", str(workspace), "--history"],
    )

    assert handoff_main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["wake_runs"][0]["wake_id"] == "wake:one"
    assert _workspace_state(workspace) == before


def test_cli_apply_requires_backup_and_preserves_rules_source(tmp_path: Path) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    source = workspace / "PROACTIVE_CONTEXT.md"
    source.write_bytes(b"rules fixture\n")
    backup = tmp_path / "backup"

    report = apply_cli(workspace, backup)

    assert report.status is HandoffStatus.APPLIED
    assert source.read_bytes() == b"rules fixture\n"
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["files"][0]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )


def test_backup_captures_proactive_database_and_quota(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    proactive = _proactive_db(workspace)
    quota = workspace / "proactive_quota.json"
    quota.write_text('{"version":1,"used":1}', encoding="utf-8")
    backup = tmp_path / "backup"

    backup_sources(workspace, backup)

    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    assert any(entry["source"] == str(proactive) for entry in manifest["sqlite"])
    assert any(entry["source"] == str(quota) for entry in manifest["files"])


def test_retire_exact_approved_blocks_keeps_sources_and_requires_backup(
    tmp_path: Path,
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    _proactive_db(workspace)
    inventory = inventory_workspace(workspace)
    backup = (tmp_path / "backup").resolve()

    report = retire_cli(workspace, backup, inventory_digest(inventory))

    assert report.status is HandoffStatus.READY
    assert (workspace / "proactive.db").is_file()
    receipt = json.loads(
        (
            workspace / "runtime" / "proactive-island-handoff" / "retirement.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["decision"] == "operator_approved_pre_cutover_supersession"
    assert len(receipt["blocks"]) == 6
    assert plan_cli(workspace).status is HandoffStatus.READY
    assert (
        retire_cli(workspace, backup, inventory_digest(inventory)).status
        is HandoffStatus.READY
    )


def test_retirement_does_not_hide_changed_legacy_state(tmp_path: Path) -> None:
    workspace = (tmp_path / "workspace").resolve()
    proactive = _proactive_db(workspace)
    inventory = inventory_workspace(workspace)
    _ = retire_cli(
        workspace,
        (tmp_path / "backup").resolve(),
        inventory_digest(inventory),
    )
    connection = sqlite3.connect(proactive)
    connection.execute("INSERT INTO seen_items VALUES('new-item', '2026-08-24')")
    connection.commit()
    connection.close()

    report = plan_cli(workspace)

    assert report.status is HandoffStatus.BLOCK
    assert any(item.locator == "proactive:seen_items" for item in report.items)


def test_retire_inventory_digest_mismatch_writes_nothing(tmp_path: Path) -> None:
    workspace = (tmp_path / "workspace").resolve()
    _proactive_db(workspace)
    before = _workspace_state(workspace)
    backup = (tmp_path / "backup").resolve()

    report = retire_cli(workspace, backup, "0" * 64)

    assert report.status is HandoffStatus.BLOCK
    assert report.items[0].reason == "source_inventory_digest_mismatch"
    assert _workspace_state(workspace) == before
    assert not backup.exists()


def test_retire_rejects_unknown_block_before_backup(tmp_path: Path) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    connection = sqlite3.connect(workspace / "proactive.db")
    connection.execute("CREATE TABLE future_state(id TEXT)")
    connection.execute("INSERT INTO future_state VALUES('one')")
    connection.commit()
    connection.close()
    inventory = inventory_workspace(workspace)
    backup = (tmp_path / "backup").resolve()

    with pytest.raises(RuntimeError, match="unknown_proactive_table"):
        retire_cli(workspace, backup, inventory_digest(inventory))

    assert not backup.exists()


def test_retirement_fails_loud_when_recovery_artifact_changes(tmp_path: Path) -> None:
    workspace = (tmp_path / "workspace").resolve()
    _proactive_db(workspace)
    inventory = inventory_workspace(workspace)
    backup = (tmp_path / "backup").resolve()
    _ = retire_cli(workspace, backup, inventory_digest(inventory))
    (backup / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backup manifest changed"):
        plan_cli(workspace)


def test_apply_blocks_source_drift_after_backup_before_any_target_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    rules = workspace / "PROACTIVE_CONTEXT.md"
    rules.write_bytes(b"version one")
    original = handoff_cli.backup_sources

    def backup_then_change(source: Path, target: Path) -> None:
        original(source, target)
        rules.write_bytes(b"version two")

    monkeypatch.setattr(handoff_cli, "backup_sources", backup_then_change)

    report = apply_cli(workspace, (tmp_path / "backup").resolve())

    assert report.status is HandoffStatus.BLOCK
    assert report.items[0].reason == "source_inventory_drift_after_backup"
    assert not (workspace / "plugin-data").exists()
    assert not (workspace / "runtime").exists()


@pytest.mark.parametrize("relationship", ["relative", "equal", "child", "parent"])
def test_apply_requires_disjoint_absolute_backup_root(
    tmp_path: Path, relationship: str
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    (workspace / "PROACTIVE_CONTEXT.md").write_bytes(b"rules")
    backup = {
        "relative": Path("backup"),
        "equal": workspace,
        "child": workspace / "backup",
        "parent": tmp_path.resolve(),
    }[relationship]

    with pytest.raises(ValueError):
        apply_cli(workspace, backup)


def _legacy_database_for_reader(workspace: Path, kind: str) -> Path:
    if kind == "proactive":
        return _proactive_db(workspace)
    if kind == "wake":
        return _wake_db(workspace, [])
    if kind == "drift":
        path = workspace / "drift" / "drift.db"
        path.parent.mkdir(parents=True)
        connection = sqlite3.connect(path)
        connection.executescript(
            "CREATE TABLE skill_continuum(skill_name TEXT,last_status TEXT);"
            "CREATE TABLE runs(id INTEGER,event_id TEXT,message_result TEXT);"
        )
        connection.commit()
        connection.close()
        return path
    path = workspace / "runtime" / "plugin-jobs" / "outcomes.sqlite"
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE job_outcomes(invocation_id TEXT,created_at TEXT,"
        "event_payload_json TEXT)"
    )
    connection.commit()
    connection.close()
    return path


@pytest.mark.parametrize("kind", ["proactive", "wake", "drift", "jobs"])
def test_legacy_readers_reject_uncheckpointed_wal(tmp_path: Path, kind: str) -> None:
    workspace = tmp_path / "workspace"
    path = _legacy_database_for_reader(workspace, kind)
    path.with_name(path.name + "-wal").write_bytes(b"uncheckpointed frames")

    with pytest.raises(RuntimeError, match="uncheckpointed WAL"):
        if kind == "jobs":
            LegacyProactiveHistory(workspace).job_outcomes()
        else:
            inventory_workspace(workspace)


@pytest.mark.parametrize("kind", ["proactive", "wake", "drift", "jobs"])
def test_legacy_readers_allow_empty_wal_and_existing_shm(
    tmp_path: Path, kind: str
) -> None:
    workspace = tmp_path / "workspace"
    path = _legacy_database_for_reader(workspace, kind)
    path.with_name(path.name + "-wal").write_bytes(b"")
    path.with_name(path.name + "-shm").write_bytes(b"retained shared memory")

    if kind == "jobs":
        assert LegacyProactiveHistory(workspace).job_outcomes() == ()
    else:
        _ = inventory_workspace(workspace)
