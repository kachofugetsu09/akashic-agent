from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from docker.debug import plugin_v3_e4_gate as gate


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _report_payload(label: str, head: str, tree: str, lock_sha: str) -> dict[str, object]:
    base: dict[str, object] = {
        "status": "passed",
        "core": {"head": head, "tree": tree},
    }
    if label == "E1":
        base.update({
            "phase": "e1",
            "lock": {"sha256": lock_sha},
            "scenarios": [{"id": "runtime_boot_akasha", "status": "passed"}],
            "runtime": {"akasha": {}},
        })
    elif label == "E2":
        base.update({
            "lock_sha256": lock_sha,
            "scenario_profile": gate.E2_PROFILE,
            "core_process_crash": {"status": "passed"},
        })
    elif label == "E3":
        base.update({
            "lock_sha256": lock_sha,
            "scenario_profile": gate.E3_PROFILE,
            "runtime": {"channel": {}, "message_push": {}, "channel_cleanup": {}},
        })
    else:
        base.update({
            "lock_sha256": lock_sha,
            "scenario_profile": gate.PASSIVE_PROFILE,
            "runtime": {"status": "passed"},
            "cleanup": {"residuals": [], "sandbox_removed": True},
        })
    return base


def test_validate_final_reports_requires_current_identity(tmp_path: Path) -> None:
    identity = {"head": "h", "tree": "t", "lock_sha256": "l"}
    paths = {
        label: _write(
            tmp_path / f"{label}.json", _report_payload(label, "h", "t", "l")
        )
        for label in ("E1", "E2", "E3", "Passive")
    }

    result = gate.validate_final_reports(
        e1_report=paths["E1"],
        e2_report=paths["E2"],
        e3_report=paths["E3"],
        passive_webui_report=paths["Passive"],
        current_identity=identity,
        lock_sha256="l",
    )

    assert result["status"] == "passed"
    assert len(result["reports"]) == 4  # type: ignore[arg-type]


def test_validate_final_reports_rejects_stale_core(tmp_path: Path) -> None:
    identity = {"head": "current", "tree": "t", "lock_sha256": "l"}
    paths = {
        label: _write(
            tmp_path / f"{label}.json", _report_payload(label, "old", "t", "l")
        )
        for label in ("E1", "E2", "E3", "Passive")
    }

    with pytest.raises(gate.GateBlocked, match="当前 HEAD/tree"):
        gate.validate_final_reports(
            e1_report=paths["E1"],
            e2_report=paths["E2"],
            e3_report=paths["E3"],
            passive_webui_report=paths["Passive"],
            current_identity=identity,
            lock_sha256="l",
        )


def test_validate_final_reports_rejects_incomplete_e1(tmp_path: Path) -> None:
    identity = {"head": "h", "tree": "t", "lock_sha256": "l"}
    payload = _report_payload("E1", "h", "t", "l")
    payload["scenarios"] = []
    e1 = _write(tmp_path / "E1.json", payload)
    paths = {
        label: _write(
            tmp_path / f"{label}.json", _report_payload(label, "h", "t", "l")
        )
        for label in ("E2", "E3", "Passive")
    }

    with pytest.raises(gate.GateBlocked, match="runtime_boot_akasha"):
        gate.validate_final_reports(
            e1_report=e1,
            e2_report=paths["E2"],
            e3_report=paths["E3"],
            passive_webui_report=paths["Passive"],
            current_identity=identity,
            lock_sha256="l",
        )


def test_validate_final_reports_rejects_stale_e3_cleanup_field(tmp_path: Path) -> None:
    identity = {"head": "h", "tree": "t", "lock_sha256": "l"}
    payload = _report_payload("E3", "h", "t", "l")
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    runtime["cleanup"] = runtime.pop("channel_cleanup")
    e3 = _write(tmp_path / "E3.json", payload)
    paths = {
        label: _write(
            tmp_path / f"{label}.json", _report_payload(label, "h", "t", "l")
        )
        for label in ("E1", "E2", "Passive")
    }

    with pytest.raises(gate.GateBlocked, match="channel_cleanup"):
        gate.validate_final_reports(
            e1_report=paths["E1"],
            e2_report=paths["E2"],
            e3_report=e3,
            passive_webui_report=paths["Passive"],
            current_identity=identity,
            lock_sha256="l",
        )


def test_fleet_coverage_is_explicitly_blocked_for_uncovered_ids(tmp_path: Path) -> None:
    lock = _write(tmp_path / "lock.json", {"plugins": [{"id": "citation"}, {"id": "future"}]})
    result = gate._fleet_coverage(lock)
    assert result["status"] == "blocked"
    assert result["missing_ids"] == ["future"]


def test_tree_summary_and_artifact_pointer_inventory(tmp_path: Path) -> None:
    root = tmp_path / "plugin-home"
    artifact = root / "webui" / ".artifacts" / "abc"
    artifact.mkdir(parents=True)
    (artifact / "plugin.py").write_text("v3", encoding="utf-8")
    (root / "webui" / "stable.json").write_text('{"stable":"abc"}', encoding="utf-8")

    summary = gate._tree_summary(root)
    inventory = gate._artifact_inventory(root)
    assert summary["exists"] is True
    assert inventory["artifact_digest"] != gate._digest_records([])
    assert len(inventory["pointer_files"]) == 1  # type: ignore[arg-type]
    before = summary["digest"]
    (artifact / "plugin.py").write_text("changed", encoding="utf-8")
    assert gate._tree_summary(root)["digest"] != before


def test_tmp_root_is_optional_and_caller_owned(tmp_path: Path) -> None:
    assert gate._resolve_tmp_root(None) is None
    assert gate._resolve_tmp_root(tmp_path) == tmp_path.resolve()

    invalid = tmp_path / "not-a-directory"
    invalid.write_text("fixture", encoding="utf-8")
    with pytest.raises(gate.GateFailure, match="tmp root"):
        gate._resolve_tmp_root(invalid)


def test_plugin_data_inventory_ignores_only_sqlite_runtime_sidecars(
    tmp_path: Path,
) -> None:
    plugin_data = tmp_path / "plugin-data"
    state = plugin_data / "github-watch-github" / "events.sqlite3"
    state.parent.mkdir(parents=True)
    state.write_bytes(b"SQLite format 3\x00authoritative")
    before = gate._artifact_inventory(plugin_data, exclude_sqlite_sidecars=True)

    (state.parent / "events.sqlite3-wal").write_bytes(b"runtime-wal")
    (state.parent / "events.sqlite3-shm").write_bytes(b"runtime-shm")
    after_sidecars = gate._artifact_inventory(
        plugin_data, exclude_sqlite_sidecars=True
    )
    assert after_sidecars == before

    unrelated = state.parent / "audit-wal"
    unrelated.write_bytes(b"plugin-owned")
    after_unrelated = gate._artifact_inventory(
        plugin_data, exclude_sqlite_sidecars=True
    )
    assert after_unrelated["tree_digest"] != before["tree_digest"]
    unrelated.unlink()

    state.write_bytes(b"SQLite format 3\x00changed")
    after_database = gate._artifact_inventory(
        plugin_data, exclude_sqlite_sidecars=True
    )
    assert after_database["tree_digest"] != before["tree_digest"]


def test_workspace_summary_uses_rehearsal_runtime_exclusions(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "stable.txt").write_text("stable", encoding="utf-8")
    (workspace / ".instance.lock").write_text("one", encoding="utf-8")
    (workspace / "runtime").mkdir()
    (workspace / "runtime" / "ephemeral").write_text("one", encoding="utf-8")
    before = gate._tree_summary(workspace, exclude_workspace_runtime=True)
    (workspace / ".instance.lock").write_text("two", encoding="utf-8")
    (workspace / "runtime" / "ephemeral").write_text("two", encoding="utf-8")
    after = gate._tree_summary(workspace, exclude_workspace_runtime=True)
    assert before["digest"] == after["digest"]


def _create_sessions(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE messages (id TEXT PRIMARY KEY, session_key TEXT, seq INTEGER, "
        "role TEXT, content TEXT, tool_chain TEXT, extra TEXT, ts TEXT)"
    )
    connection.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("m1", "s", 0, "user", "hello", "", "{}", "now"),
    )
    connection.commit()
    connection.close()


def test_sqlite_integrity_and_existing_rows_are_append_only(tmp_path: Path) -> None:
    database = tmp_path / "sessions.db"
    _create_sessions(database)
    before = gate._sqlite_snapshot(database)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("m2", "s", 1, "assistant", "world", "", "{}", "now"),
    )
    connection.commit()
    connection.close()
    after = gate._sqlite_snapshot(database)
    evidence = gate._append_only_evidence(before, after, label="test")
    assert evidence["status"] == "passed"
    assert evidence["new_message_count"] == 1


def test_sqlite_append_only_rejects_rewrite(tmp_path: Path) -> None:
    database = tmp_path / "sessions.db"
    _create_sessions(database)
    before = gate._sqlite_snapshot(database)
    connection = sqlite3.connect(database)
    connection.execute("UPDATE messages SET content = 'rewritten' WHERE id = 'm1'")
    connection.commit()
    connection.close()

    with pytest.raises(gate.GateFailure, match="append-only"):
        gate._append_only_evidence(before, gate._sqlite_snapshot(database), label="test")
