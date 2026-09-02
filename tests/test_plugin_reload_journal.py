import sqlite3
from pathlib import Path

import pytest

from agent.plugins.reload_journal import ReloadJournal


def test_reload_journal_records_durable_transaction_phases(tmp_path: Path) -> None:
    journal = ReloadJournal(tmp_path / "workspace")
    tx_id = journal.begin(
        plugin_id="weather",
        base_snapshot_id="snapshot-v1",
        generation_id="weather:source-v2:2",
        source_revision="source-v2",
        config_revision="config-v2",
        base_artifact_pointer=".artifacts/v1",
        candidate_artifact_pointer=".artifacts/v2",
    )
    journal.mark_runtime_owner(tx_id, "boot-v1")

    journal.advance(
        tx_id,
        "prepared",
        candidate_snapshot_id="snapshot-v2",
    )
    journal.advance(tx_id, "validating")
    journal.advance(tx_id, "commit_started")
    journal.advance(tx_id, "committed")
    journal.advance(tx_id, "draining")

    record = journal.get(tx_id)
    assert record.phase == "draining"
    assert record.plugin_id == "weather"
    assert record.base_snapshot_id == "snapshot-v1"
    assert record.candidate_snapshot_id == "snapshot-v2"
    assert record.source_revision == "source-v2"
    assert record.runtime_owner_boot_id == "boot-v1"
    assert record.base_artifact_pointer == ".artifacts/v1"
    assert record.candidate_artifact_pointer == ".artifacts/v2"
    assert [event.phase for event in journal.events(tx_id)] == [
        "preparing",
        "preparing",
        "prepared",
        "validating",
        "commit_started",
        "committed",
        "draining",
    ]

    reopened = ReloadJournal(tmp_path / "workspace")
    assert reopened.get(tx_id) == record

    with pytest.raises(RuntimeError, match="不可覆盖"):
        reopened.mark_runtime_owner(tx_id, "boot-v2")


def test_reload_journal_rejects_invalid_phase_transition(tmp_path: Path) -> None:
    journal = ReloadJournal(tmp_path / "workspace")
    tx_id = journal.begin(
        plugin_id="weather",
        base_snapshot_id="snapshot-v1",
        generation_id="weather:source-v2:2",
        source_revision="source-v2",
        config_revision="config-v2",
    )

    with pytest.raises(RuntimeError, match="ReloadTransaction 状态跳转无效"):
        journal.advance(tx_id, "committed")


def test_reload_journal_recovers_discarding_candidate(tmp_path: Path) -> None:
    journal = ReloadJournal(tmp_path / "workspace")
    tx_id = journal.begin(
        plugin_id="weather",
        base_snapshot_id="snapshot-v1",
        generation_id="weather:source-v2:2",
        source_revision="source-v2",
        config_revision="config-v2",
    )
    journal.advance(tx_id, "prepared", candidate_snapshot_id="snapshot-v2")
    journal.advance(tx_id, "validating")
    journal.advance(tx_id, "commit_started")
    journal.advance(tx_id, "latest_ready")
    journal.advance(tx_id, "discarding")

    action = journal.pending_recovery()[0]

    assert action.phase == "discarding"
    assert action.action == "discard_candidate"
    journal.finish_recovery(action)
    assert journal.get(tx_id).phase == "aborted"


def test_reload_journal_builds_crash_recovery_plan(tmp_path: Path) -> None:
    journal = ReloadJournal(tmp_path / "workspace")
    prepared = journal.begin(
        plugin_id="weather",
        base_snapshot_id="snapshot-v1",
        generation_id="weather:source-v2:2",
        source_revision="source-v2",
        config_revision="config-v2",
    )
    journal.advance(prepared, "prepared")
    committed = journal.begin(
        plugin_id="calendar",
        base_snapshot_id="snapshot-v1",
        generation_id="calendar:source-v3:3",
        source_revision="source-v3",
        config_revision="config-v3",
    )
    journal.advance(
        committed,
        "prepared",
        candidate_snapshot_id="snapshot-v3",
    )
    journal.advance(committed, "validating")
    journal.advance(committed, "commit_started")
    latest = journal.begin(
        plugin_id="feed",
        base_snapshot_id="snapshot-v1",
        generation_id="feed:source-v4:4",
        source_revision="source-v4",
        config_revision="config-v4",
    )
    journal.advance(latest, "prepared", candidate_snapshot_id="snapshot-v4")
    journal.advance(latest, "validating")
    journal.advance(latest, "commit_started")
    journal.advance(latest, "latest_ready")

    actions = journal.pending_recovery()

    assert (actions[0].tx_id, actions[0].phase, actions[0].action) == (
        committed,
        "commit_started",
        "restore_committed",
    )
    assert {
        (action.tx_id, action.phase, action.action) for action in actions[1:]
    } == {
        (latest, "latest_ready", "discard_candidate"),
        (prepared, "prepared", "discard_candidate"),
    }
    for action in actions:
        journal.finish_recovery(action)
    assert journal.get(committed).phase == "recovered"
    assert journal.get(latest).phase == "aborted"
    assert journal.get(prepared).phase == "aborted"
    assert journal.pending_recovery() == ()


def test_reload_journal_retains_cleanup_failure_evidence_across_restart(
    tmp_path: Path,
) -> None:
    journal = ReloadJournal(tmp_path / "workspace")
    tx_id = journal.begin(
        plugin_id="calendar",
        base_snapshot_id="stable-snapshot-v1",
        base_generation_id="calendar:stable:1",
        generation_id="calendar:candidate:2",
        source_revision="source-v2",
        config_revision="config-v2",
    )
    journal.advance(tx_id, "prepared", candidate_snapshot_id="candidate-snapshot-v2")
    journal.advance(tx_id, "validating")
    journal.advance(tx_id, "commit_started")
    journal.advance(tx_id, "latest_ready")
    journal.advance(tx_id, "promoting")
    journal.advance(
        tx_id,
        "cleanup_failed",
        resource="calendar_api@candidate:2",
        formal_effects=("old_endpoint_stopped", "new_endpoint_started"),
        error="terminate failed",
        recovery_target="base",
    )

    reopened = ReloadJournal(tmp_path / "workspace")
    record = reopened.get(tx_id)
    assert record.phase == "cleanup_failed"
    assert record.old_snapshot_id == "stable-snapshot-v1"
    assert record.new_snapshot_id == "candidate-snapshot-v2"
    assert record.old_generation_id == "calendar:stable:1"
    assert record.attempt_generation_id == "calendar:candidate:2"
    assert record.formal_effects == (
        "old_endpoint_stopped",
        "new_endpoint_started",
    )
    assert record.resource == "calendar_api@candidate:2"
    assert record.error == "terminate failed"
    assert record.attempt_count == 1
    failure_event = reopened.events(tx_id)[-1]
    assert failure_event.details["old_snapshot_id"] == "stable-snapshot-v1"
    assert failure_event.details["new_snapshot_id"] == "candidate-snapshot-v2"
    assert failure_event.details["attempt_generation_id"] == "calendar:candidate:2"
    assert failure_event.details["resource"] == "calendar_api@candidate:2"
    assert failure_event.details["error"] == "terminate failed"
    assert failure_event.details["attempt_count"] == 1

    action = reopened.pending_recovery()[0]
    assert action.action == "retry_generation_cleanup"
    assert action.error == "terminate failed"
    assert action.attempt_count == 1
    with pytest.raises(RuntimeError, match="状态跳转无效"):
        reopened.advance(tx_id, "complete")

    reopened.advance(
        tx_id,
        "cleanup_failed",
        resource="calendar_api@candidate:2",
        error="retry terminate failed",
    )
    assert reopened.get(tx_id).attempt_count == 2
    with pytest.raises(RuntimeError, match="recovery action 已失效"):
        reopened.finish_recovery(action)
    retry_action = reopened.pending_recovery()[0]
    with pytest.raises(RuntimeError, match="缺少 Host retry receipt"):
        reopened.finish_recovery(retry_action)
    with pytest.raises(RuntimeError, match="状态跳转无效"):
        reopened.advance(tx_id, "aborted")
    reopened.finish_recovery(
        retry_action,
        retry_receipt="managed-process-host:calendar:candidate:2:cleanup-complete",
    )
    recovered = reopened.get(tx_id)
    assert recovered.phase == "aborted"
    assert recovered.error == "retry terminate failed"
    assert reopened.pending_recovery() == ()


def test_reload_journal_degraded_runtime_recovery_is_explicit(tmp_path: Path) -> None:
    journal = ReloadJournal(tmp_path / "workspace")
    tx_id = journal.begin(
        plugin_id="calendar",
        base_snapshot_id="stable-snapshot-v1",
        base_generation_id="calendar:stable:1",
        generation_id="calendar:candidate:2",
        source_revision="source-v2",
        config_revision="config-v2",
    )
    journal.advance(tx_id, "prepared")
    journal.advance(tx_id, "validating")
    journal.advance(tx_id, "commit_started")
    journal.advance(tx_id, "latest_ready")
    journal.advance(tx_id, "promoting")
    journal.advance(
        tx_id,
        "degraded",
        resource="calendar_api@stable:1",
        formal_effects=("candidate_started", "old_restore_uncertain"),
        error="old endpoint restore failed",
        recovery_target="base",
    )

    action = journal.pending_recovery()[0]
    assert action.action == "retry_runtime_recovery"
    assert action.phase == "degraded"
    with pytest.raises(RuntimeError, match="状态跳转无效"):
        journal.advance(tx_id, "complete")
    with pytest.raises(RuntimeError, match="缺少 Host retry receipt"):
        journal.finish_recovery(action)
    with pytest.raises(RuntimeError, match="状态跳转无效"):
        journal.advance(tx_id, "recovered")
    journal.finish_recovery(
        action,
        retry_receipt="managed-process-host:calendar:stable:1:runtime-recovered",
    )
    record = journal.get(tx_id)
    assert record.phase == "recovered"
    assert record.error == "old endpoint restore failed"
    assert record.formal_effects == ("candidate_started", "old_restore_uncertain")
    assert journal.pending_recovery() == ()


def test_runtime_failure_evidence_upgrades_and_never_loses_owners(
    tmp_path: Path,
) -> None:
    journal = ReloadJournal(tmp_path / "workspace")
    tx_id = journal.begin(
        plugin_id="calendar",
        base_snapshot_id="stable-v1",
        base_generation_id="calendar:stable:1",
        generation_id="calendar:candidate:2",
        source_revision="source-v2",
        config_revision="config-v2",
    )
    journal.advance(
        tx_id,
        "cleanup_failed",
        resource="mcp:calendar",
        error="mcp cleanup failed",
        recovery_target="base",
    )
    journal.advance(
        tx_id,
        "degraded",
        resource="process:calendar_api",
        error="process watchdog failed",
        recovery_target="base",
    )

    action = journal.pending_recovery()[0]
    assert action.phase == "degraded"
    assert action.action == "retry_runtime_recovery"
    assert action.failure_resource == "mcp:calendar,process:calendar_api"
    assert action.attempt_count == 2
    with pytest.raises(RuntimeError, match="recovery target 不可覆盖"):
        journal.advance(
            tx_id,
            "degraded",
            resource="process:calendar_api",
            error="target drift",
            recovery_target="candidate",
        )


def test_cleanup_retry_preserves_a_committed_candidate_target(tmp_path: Path) -> None:
    journal = ReloadJournal(tmp_path / "workspace")
    tx_id = journal.begin(
        plugin_id="calendar@lab",
        base_snapshot_id="stable-v1",
        base_generation_id="calendar:stable:1",
        generation_id="calendar:candidate:2",
        source_revision="source-v2",
        config_revision="config-v2",
        base_artifact_pointer=".artifacts/v1",
        candidate_artifact_pointer=".artifacts/v2",
    )
    journal.advance(tx_id, "prepared")
    journal.advance(tx_id, "validating")
    journal.advance(tx_id, "commit_started")
    journal.advance(tx_id, "committed")
    journal.advance(tx_id, "draining")
    journal.advance(
        tx_id,
        "cleanup_failed",
        resource="composition-runtime:calendar:stable:1",
        error="old process group retained",
        recovery_target="candidate",
    )

    action = journal.pending_recovery()[0]
    journal.finish_recovery(
        action,
        retry_receipt="composition-runtime:calendar:stable:1:cleanup-complete",
    )

    record = journal.get(tx_id)
    assert record.phase == "recovered"
    assert record.error == "old process group retained"


def test_reload_journal_requires_complete_failure_evidence(tmp_path: Path) -> None:
    journal = ReloadJournal(tmp_path / "workspace")
    tx_id = journal.begin(
        plugin_id="calendar",
        base_snapshot_id="stable-snapshot-v1",
        base_generation_id="calendar:stable:1",
        generation_id="calendar:candidate:2",
        source_revision="source-v2",
        config_revision="config-v2",
    )

    with pytest.raises(ValueError, match="resource identity"):
        journal.advance(
            tx_id,
            "cleanup_failed",
            error="terminate failed",
            recovery_target="base",
        )
    with pytest.raises(ValueError, match="error evidence"):
        journal.advance(
            tx_id,
            "cleanup_failed",
            resource="calendar_api@candidate:2",
            recovery_target="base",
        )
    with pytest.raises(ValueError, match="recovery target"):
        journal.advance(
            tx_id,
            "cleanup_failed",
            resource="calendar_api@candidate:2",
            error="terminate failed",
        )

    assert journal.get(tx_id).phase == "preparing"


def test_reload_journal_migrates_legacy_schema_without_losing_transactions(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    database = workspace / "runtime" / "plugin-reloads.sqlite3"
    database.parent.mkdir(parents=True)
    conn = sqlite3.connect(database)
    try:
        conn.executescript(
            """
            CREATE TABLE reload_transactions (
                tx_id TEXT PRIMARY KEY,
                plugin_id TEXT NOT NULL,
                base_snapshot_id TEXT,
                candidate_snapshot_id TEXT,
                generation_id TEXT NOT NULL,
                source_revision TEXT NOT NULL,
                config_revision TEXT NOT NULL,
                phase TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error TEXT NOT NULL
            );
            CREATE TABLE reload_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_id TEXT NOT NULL REFERENCES reload_transactions(tx_id),
                phase TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO reload_transactions VALUES (
                'legacy-tx', 'calendar@lab', 'stable-v1', NULL,
                'calendar:candidate:2', 'source-v2', 'config-v2',
                'prepared', '2026-08-16T00:00:00+00:00',
                '2026-08-16T00:00:00+00:00', ''
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    journal = ReloadJournal(workspace)

    record = journal.get("legacy-tx")
    assert record.phase == "prepared"
    assert record.generation_id == "calendar:candidate:2"
    assert record.formal_effects == ()
    assert record.attempt_count == 0
    assert record.runtime_owner_boot_id is None
    assert record.base_artifact_pointer is None
    assert record.candidate_artifact_pointer is None
    assert record.recovery_target is None
    action = journal.pending_recovery()[0]
    assert action.action == "discard_candidate"
