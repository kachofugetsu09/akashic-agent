"""真实 SQLite tests for the C21 JobOutcomeLedger state contract."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from agent.plugins.job_outcome_ledger import (
    JobOutcomeIdentity,
    JobOutcomeIdentityError,
    JobOutcomeLedger,
    JobOutcomePhase,
    JobOutcomeState,
    JobOutcomeTransitionError,
    ProgrammaticTurnState,
)


def _identity(
    *,
    invocation_id: str = "invocation-1",
    event_id: str | None = "event-1",
    interval_bucket: str | None = None,
    snapshot_id: str = "snapshot-1",
    plugin_generation_id: str = "plugin-generation-1",
    model_generation_id: str = "model-generation-1",
) -> JobOutcomeIdentity:
    return JobOutcomeIdentity(
        plugin_id="plugin.example",
        job_name="merge_pending",
        invocation_id=invocation_id,
        event_id=event_id,
        interval_bucket=interval_bucket,
        snapshot_id=snapshot_id,
        plugin_generation_id=plugin_generation_id,
        model_generation_id=model_generation_id,
        artifact_identity="artifact-sha256:abc",
        source_revision="source-revision-1",
        handler_export="jobs.merge_pending",
        lifecycle_revision="lifecycle-3",
        api_revision="plugin-api-v3",
    )


def test_admission_persists_exact_identity_and_deduplicates_event(tmp_path: Path) -> None:
    path = tmp_path / "runtime" / "plugin-jobs" / "outcomes.sqlite"
    ledger = JobOutcomeLedger(path)

    first = ledger.admit(_identity())
    duplicate = ledger.admit(
        _identity(
            invocation_id="invocation-from-repeated-delivery",
            snapshot_id="new-snapshot-must-not-rerun",
            plugin_generation_id="new-plugin-generation",
            model_generation_id="new-model-generation",
        )
    )

    assert first.state is JobOutcomeState.QUEUED
    assert first.phase is JobOutcomePhase.HANDLER
    assert duplicate == first
    assert ledger.get("invocation-from-repeated-delivery") is None
    assert len(ledger.list_all()) == 1

    with closing(sqlite3.connect(path)) as connection:
        row = connection.execute(
            """
            SELECT semantic_job_id, invocation_id, event_id, snapshot_id,
                   plugin_generation_id, model_generation_id, artifact_identity,
                   source_revision, handler_export, lifecycle_revision, api_revision
            FROM job_outcomes
            """
        ).fetchone()
    assert row == (
        "plugin.example:merge_pending",
        "invocation-1",
        "event-1",
        "snapshot-1",
        "plugin-generation-1",
        "model-generation-1",
        "artifact-sha256:abc",
        "source-revision-1",
        "jobs.merge_pending",
        "lifecycle-3",
        "plugin-api-v3",
    )
    ledger.integrity_check()


def test_programmatic_turn_receipt_survives_process_reopen(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.sqlite"
    ledger = JobOutcomeLedger(path)
    admitted = ledger.admit(_identity())
    _ = ledger.transition(admitted.invocation_id, JobOutcomeState.RUNNING)
    submitting = ledger.begin_programmatic_turn(admitted.invocation_id)
    assert submitting.programmatic_turn_state is ProgrammaticTurnState.SUBMITTING
    ledger.close()

    reopened = JobOutcomeLedger(path)
    retained = reopened.require(admitted.invocation_id)
    assert retained.programmatic_turn_state is ProgrammaticTurnState.SUBMITTING
    assert retained.programmatic_turn_id is None
    committed = reopened.commit_programmatic_turn(admitted.invocation_id, "turn-1")
    assert committed.programmatic_turn_state is ProgrammaticTurnState.ADMITTED
    assert committed.programmatic_turn_id == "turn-1"


def test_invocation_id_cannot_be_reused_for_another_event(tmp_path: Path) -> None:
    ledger = JobOutcomeLedger(tmp_path / "outcomes.sqlite")
    ledger.admit(_identity())

    with pytest.raises(JobOutcomeIdentityError):
        ledger.admit(
            _identity(
                event_id="another-event",
                invocation_id="invocation-1",
            )
        )
    assert len(ledger.list_all()) == 1


def test_legal_transitions_increment_retry_attempt_and_terminal_once(
    tmp_path: Path,
) -> None:
    ledger = JobOutcomeLedger(tmp_path / "outcomes.sqlite")
    ledger.admit(_identity())

    running = ledger.transition("invocation-1", JobOutcomeState.RUNNING)
    pending = ledger.transition(
        "invocation-1",
        JobOutcomeState.RETRY_PENDING,
        phase=JobOutcomePhase.HANDLER,
        error="provider unavailable",
    )
    retried = ledger.transition("invocation-1", JobOutcomeState.RUNNING)
    failed = ledger.transition(
        "invocation-1",
        JobOutcomeState.FAILED,
        error="retry exhausted",
    )

    assert running.attempt == 1
    assert pending.phase is JobOutcomePhase.HANDLER
    assert retried.attempt == 2
    assert retried.error is None
    assert failed.state is JobOutcomeState.FAILED
    assert failed.attempt == 2
    with pytest.raises(JobOutcomeTransitionError):
        ledger.transition("invocation-1", JobOutcomeState.RUNNING)


def test_provider_retry_phase_is_reachable(tmp_path: Path) -> None:
    ledger = JobOutcomeLedger(tmp_path / "outcomes.sqlite")
    ledger.admit(_identity())
    ledger.transition("invocation-1", JobOutcomeState.RUNNING)

    pending = ledger.transition(
        "invocation-1",
        JobOutcomeState.RETRY_PENDING,
        phase=JobOutcomePhase.PROVIDER,
        error="provider request failed before domain effect",
    )

    assert pending.state is JobOutcomeState.RETRY_PENDING
    assert pending.phase is JobOutcomePhase.PROVIDER
    assert pending.error == "provider request failed before domain effect"


def test_running_binds_actual_model_generation_once(tmp_path: Path) -> None:
    ledger = JobOutcomeLedger(tmp_path / "outcomes.sqlite")
    pending_identity = replace(
        _identity(),
        model_generation_id="execution-pending",
    )
    ledger.admit(pending_identity)

    running = ledger.transition(
        "invocation-1",
        JobOutcomeState.RUNNING,
        model_generation_id="model-generation-2",
    )
    duplicate = ledger.admit(pending_identity)

    assert running.model_generation_id == "model-generation-2"
    assert duplicate == running

    ledger.transition(
        "invocation-1",
        JobOutcomeState.RETRY_PENDING,
        error="provider unavailable",
    )
    with pytest.raises(JobOutcomeIdentityError, match="model generation"):
        ledger.transition(
            "invocation-1",
            JobOutcomeState.RUNNING,
            model_generation_id="model-generation-3",
        )


def test_outcome_field_invariants_fail_loud(tmp_path: Path) -> None:
    ledger = JobOutcomeLedger(tmp_path / "outcomes.sqlite")
    ledger.admit(_identity())

    with pytest.raises(JobOutcomeTransitionError, match="running.*error"):
        ledger.transition("invocation-1", JobOutcomeState.RUNNING, error="stale")
    with pytest.raises(JobOutcomeTransitionError, match="running.*terminal"):
        ledger.transition(
            "invocation-1",
            JobOutcomeState.RUNNING,
            terminal_result_digest="stale-digest",
        )
    ledger.transition("invocation-1", JobOutcomeState.RUNNING)

    with pytest.raises(JobOutcomeTransitionError, match="retry_pending.*error"):
        ledger.transition("invocation-1", JobOutcomeState.RETRY_PENDING)
    with pytest.raises(JobOutcomeTransitionError, match="failed.*error"):
        ledger.transition("invocation-1", JobOutcomeState.FAILED)
    with pytest.raises(JobOutcomeTransitionError, match="succeeded.*terminal"):
        ledger.transition("invocation-1", JobOutcomeState.SUCCEEDED)
    with pytest.raises(JobOutcomeTransitionError, match="failed.*terminal"):
        ledger.transition(
            "invocation-1",
            JobOutcomeState.FAILED,
            error="failed",
            terminal_result_digest="unexpected",
        )

    assert ledger.require("invocation-1").state is JobOutcomeState.RUNNING


def test_documents_phase_is_forward_recovery_only(tmp_path: Path) -> None:
    ledger = JobOutcomeLedger(tmp_path / "outcomes.sqlite")
    ledger.admit(_identity())
    ledger.transition("invocation-1", JobOutcomeState.RUNNING)
    ledger.transition(
        "invocation-1",
        JobOutcomeState.RETRY_PENDING,
        phase=JobOutcomePhase.DOCUMENTS,
        error="document commit interrupted",
    )

    with pytest.raises(JobOutcomeTransitionError):
        ledger.transition("invocation-1", JobOutcomeState.CANCELLED)
    with pytest.raises(JobOutcomeTransitionError):
        ledger.transition("invocation-1", JobOutcomeState.RUNNING)
    with pytest.raises(JobOutcomeTransitionError):
        ledger.transition("invocation-1", JobOutcomeState.FAILED)

    succeeded = ledger.transition(
        "invocation-1",
        JobOutcomeState.SUCCEEDED,
        terminal_result_digest="result-sha256:done",
    )
    assert succeeded.phase is JobOutcomePhase.DOCUMENTS
    assert succeeded.result_digest == "result-sha256:done"
    assert succeeded.terminal


def test_restart_reads_pending_records_with_exact_identity(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.sqlite"
    first_ledger = JobOutcomeLedger(path)
    first_ledger.admit(_identity(interval_bucket="2026-08-17T03:00Z", event_id=None))
    first_ledger.transition("invocation-1", JobOutcomeState.RUNNING)
    first_ledger.close()

    restarted = JobOutcomeLedger(path)
    pending = restarted.list_pending()

    assert len(pending) == 1
    record = pending[0]
    assert record.invocation_id == "invocation-1"
    assert record.semantic_job_id == "plugin.example:merge_pending"
    assert record.interval_bucket == "2026-08-17T03:00Z"
    assert record.snapshot_id == "snapshot-1"
    assert record.plugin_generation_id == "plugin-generation-1"
    assert record.model_generation_id == "model-generation-1"
    assert record.state is JobOutcomeState.RUNNING


def test_restart_reads_pending_event_payload_without_changing_binding(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.sqlite"
    payload = {
        "event_id": "drift:durable-1",
        "session_key": "session-1",
        "skill_name": "explore-curiosity",
        "status": "completed",
        "briefing": "done",
        "message_result": "silent",
        "timestamp": "2026-08-17T03:00:00+00:00",
    }
    identity = _identity(event_id="drift:durable-1")
    first_ledger = JobOutcomeLedger(path)
    first_ledger.admit(identity=identity, event_payload=payload)
    first_ledger.transition("invocation-1", JobOutcomeState.RUNNING)
    first_ledger.close()

    restarted = JobOutcomeLedger(path)
    pending = restarted.list_pending()

    assert len(pending) == 1
    record = pending[0]
    assert record.event_id == "drift:durable-1"
    assert dict(record.event_payload or {}) == payload
    assert dict(record.identity().event_payload or {}) == payload
    assert record.snapshot_id == "snapshot-1"
    assert record.plugin_generation_id == "plugin-generation-1"


def test_schema_migrates_old_outcome_table_without_payload_column(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.sqlite"
    ledger = JobOutcomeLedger(path)
    ledger.close()
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("ALTER TABLE job_outcomes DROP COLUMN event_payload_json")
        connection.execute("ALTER TABLE job_outcomes DROP COLUMN programmatic_turn_state")
        connection.execute("ALTER TABLE job_outcomes DROP COLUMN programmatic_turn_id")
        connection.execute("PRAGMA user_version = 1")
        connection.commit()

    migrated = JobOutcomeLedger(path)
    with closing(sqlite3.connect(path)) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(job_outcomes)")
        }
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    assert "event_payload_json" in columns
    assert "programmatic_turn_state" in columns
    assert "programmatic_turn_id" in columns
    assert version == 3
    assert migrated.list_all() == ()


def test_transaction_rolls_back_real_sqlite_failure(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.sqlite"
    ledger = JobOutcomeLedger(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_outcome_insert
            BEFORE INSERT ON job_outcomes
            BEGIN
                SELECT RAISE(ABORT, 'durable insert rejected');
            END
            """
        )
        connection.commit()

    with pytest.raises(sqlite3.IntegrityError, match="durable insert rejected"):
        ledger.admit(_identity())

    assert ledger.list_all() == ()


def test_identity_requires_one_trigger_and_matching_semantic_key() -> None:
    with pytest.raises(ValueError, match="恰好提供"):
        _identity(event_id=None, interval_bucket=None)
    with pytest.raises(ValueError, match="恰好提供"):
        _identity(interval_bucket="bucket")
    with pytest.raises(JobOutcomeIdentityError):
        replace(_identity(), semantic_job_id="wrong-owner:job")
