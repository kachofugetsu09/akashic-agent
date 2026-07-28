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
    )

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
    assert [event.phase for event in journal.events(tx_id)] == [
        "preparing",
        "prepared",
        "validating",
        "commit_started",
        "committed",
        "draining",
    ]

    reopened = ReloadJournal(tmp_path / "workspace")
    assert reopened.get(tx_id) == record


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

    actions = journal.pending_recovery()

    assert [(action.tx_id, action.action) for action in actions] == [
        (committed, "restore_committed"),
        (prepared, "discard_candidate"),
    ]
    journal.finish_recovery(actions[0])
    journal.finish_recovery(actions[1])
    assert journal.get(committed).phase == "recovered"
    assert journal.get(prepared).phase == "aborted"
    assert journal.pending_recovery() == ()
