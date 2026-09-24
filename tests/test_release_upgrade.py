"""Release upgrade stops old owners before touching migrated state."""

from __future__ import annotations

import json
import argparse
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import pytest

from scripts.akashic_release import activate
from scripts.akashic_release import cli as release_cli
from scripts.akashic_release.manifest import write_json
from scripts.akashic_release.model import ReleasePaths
from agent.migrations.release_backup import backup_release_state


def _release_case(tmp_path: Path, monkeypatch) -> tuple[ReleasePaths, Path, Path, list[str]]:
    paths = ReleasePaths(tmp_path / "release")
    paths.create_layout()
    (paths.state / "workspace/runtime").mkdir(parents=True)
    write_json(paths.state / "workspace/runtime/plugin-stable.json",
               {"version": 1, "root_ref": "a" * 64})
    write_json(paths.activation / "active.json",
               {"targetCommit": "a" * 40, "status": "active"})
    manifest = paths.release("b" * 40)
    write_json(manifest, {"sourceCommit": "b" * 40, "imageId": "sha256:" + "b" * 64})
    environment = tmp_path / "runtime.env"
    environment.write_text("OLD=value\n")
    candidate = {
        "AKASHIC_CONFIG": str(paths.state / "config.toml"),
        "AKASHIC_WORKSPACE": str(paths.state / "workspace"),
        "AKASHIC_PLUGIN_HOME": str(paths.state / "plugin-home"),
        "AKASHIC_RUNTIME_COMMIT": "b" * 40,
        "AKASHIC_RUNTIME_TREE": "c" * 40,
        "AKASHIC_CONTAINER_NAME": "fixture-core",
    }
    events: list[str] = []
    monkeypatch.setattr(activate, "_verify_state_ready", lambda paths: None)
    monkeypatch.setattr(activate, "_prepare_workload_dirs", lambda paths: None)
    monkeypatch.setattr(activate, "release_environment", lambda **kwargs: candidate)
    monkeypatch.setattr(activate, "stop_runtime", lambda **kwargs: events.append("stop"))
    monkeypatch.setattr(activate, "start_bridge", lambda **kwargs: events.append("bridge"))
    monkeypatch.setattr(activate, "start_core", lambda **kwargs: events.append("core"))
    return paths, manifest, environment, events


def test_release_upgrade_command_failure_keeps_old_runtime_stopped(tmp_path, monkeypatch):
    paths, manifest, environment, events = _release_case(tmp_path, monkeypatch)

    def run(command, **kwargs):
        events.append("docker")
        assert command[:2] == ["docker", "run"]
        assert "upgrade-bundled" in command
        assert command[command.index("--expected-root-ref") + 1] == "a" * 64
        assert command[-2:] == ["--previous-source-commit", "a" * 40]
        raise subprocess.CalledProcessError(1, command, output='{"status":"failed"}')

    with pytest.raises(RuntimeError, match="旧 runtime 保持停止"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True)
    assert events == ["stop", "docker"]
    assert environment.read_text() == "OLD=value\n"
    failures = list(paths.activation.glob("failed-*.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["status"] == "maintenance_required"


def test_release_upgrade_readiness_failure_does_not_restart_old_image(tmp_path, monkeypatch):
    paths, manifest, environment, events = _release_case(tmp_path, monkeypatch)
    monkeypatch.setattr(activate, "verify_release", lambda environment: None)

    def run(command, **kwargs):
        if command[:2] == ["docker", "run"]:
            events.append("migrate")
            backup = Path(command[command.index("--backup-dir") + 1])
            backup.mkdir()
            (backup / "manifest.json").write_text("{}")
            return subprocess.CompletedProcess(command, 0, json.dumps({
                "status": "selected_not_started", "new_root_ref": "b" * 64,
                "old_root_ref": "a" * 64, "backup_dir": str(backup),
            }), "")
        if command[:2] == ["docker", "exec"]:
            events.append("readiness")
            return subprocess.CompletedProcess(command, 0, json.dumps({
                "selection_ref": "a" * 64, "plugins": [],
            }), "")
        raise AssertionError(command)

    with pytest.raises(RuntimeError, match="旧 runtime 保持停止"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True)
    assert events == ["stop", "migrate", "bridge", "core", "readiness", "stop"]
    assert json.loads((paths.activation / "active.json").read_text())["targetCommit"] == "a" * 40
    failures = list(paths.activation.glob("failed-*.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["status"] == "maintenance_required"


def test_external_upgrade_crash_marker_blocks_retry_before_stop(tmp_path, monkeypatch):
    paths, manifest, environment, events = _release_case(tmp_path, monkeypatch)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    plan = inputs / "plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "expected_root_ref": "a" * 64,
                                "targets": [{"plugin_id": "outside@external",
                                             "bundle_relative_path": "outside.bundle",
                                             "bundle_sha256": "b" * 64,
                                             "target_commit": "c" * 40}]}))
    monkeypatch.setattr(activate, "verify_release", lambda environment: None)

    def stopped_upgrade(**kwargs):
        events.append("preflight" if kwargs.get("preflight_only") else "upgrade")
        if kwargs.get("preflight_only"):
            return {"status": "preflight_ok"}
        raise RuntimeError("after CAS injected interruption")

    monkeypatch.setattr(activate, "_stopped_upgrade", stopped_upgrade)
    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(RuntimeError, match="旧 runtime 保持停止"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True,
                                  external_plan=plan, external_inputs=inputs)
    assert events == ["stop", "preflight", "upgrade"]
    attempts = list(paths.activation.glob("attempt-external-*.json"))
    assert len(attempts) == 1
    assert json.loads(attempts[0].read_text())["status"] == "pending"
    assert json.loads((paths.activation / "active.json").read_text())["targetCommit"] == "a" * 40
    before = list(events)
    with pytest.raises(RuntimeError, match="未结算 release failure"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True,
                                  external_plan=plan, external_inputs=inputs)
    assert events == before


def test_release_backup_reads_committed_wal_and_keeps_sidecars_forensics(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    database = state / "facts.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("CREATE TABLE facts (value TEXT NOT NULL)")
        connection.execute("INSERT INTO facts VALUES ('committed-in-wal')")
        connection.commit()
        assert (state / "facts.sqlite3-wal").exists()
        manifest = backup_release_state(state, tmp_path / "recovery")
    with closing(sqlite3.connect(tmp_path / "recovery/state/facts.sqlite3")) as saved:
        assert saved.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert saved.execute("SELECT value FROM facts").fetchone() == ("committed-in-wal",)
    restored = tmp_path / "new-state"
    shutil.copytree(tmp_path / "recovery/state", restored, symlinks=True)
    with closing(sqlite3.connect(restored / "facts.sqlite3")) as saved:
        assert saved.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert saved.execute("SELECT value FROM facts").fetchone() == ("committed-in-wal",)
    sidecars = manifest["forensic_sidecars"]
    assert isinstance(sidecars, list) and sidecars
    assert all(isinstance(row, dict) and row["restore"] is False for row in sidecars)


def test_release_backup_restores_opaque_suffixes_without_runtime_controls(tmp_path):
    state = tmp_path / "state"
    data = state / "workspace/plugin-data/opaque-builtin"
    data.mkdir(parents=True)
    payloads = {"-wal": b"exact opaque wal", "-shm": b"exact opaque shm",
                "absent-wal": b"no base", "plain-shm": b"plain sidecar",
                "plain": b"not sqlite", "owner.lock": b"plugin fact",
                ".instance.lock": b"plugin fact"}
    for name, value in payloads.items():
        (data / name).write_bytes(value)
    workspace = state / "workspace"
    for name in (".instance.lock", ".supervisor.lock", ".supervisor.pid",
                 ".runtime-ready.json", "akashic.sock"):
        (workspace / name).write_text("stale")
    (state / "plugin-home").mkdir()
    (state / "plugin-home/.publication.lock").write_text("stale")
    (workspace / "runtime").mkdir()
    for name in ("chat.sock", "web-chat.sock", "dashboard.sock"):
        (workspace / "runtime" / name).write_text("stale")
    (state / "config.toml").write_text("setting = 'kept'\n")
    (data / "current").symlink_to("plain")
    backup = tmp_path / "recovery"
    manifest = backup_release_state(state, backup)
    target = tmp_path / "new-state"
    assert not target.exists()
    shutil.copytree(backup / "state", target, symlinks=True)
    assert {name: (target / "workspace/plugin-data/opaque-builtin" / name).read_bytes()
            for name in payloads} == payloads
    assert (target / "config.toml").read_text() == "setting = 'kept'\n"
    assert (target / "workspace/plugin-data/opaque-builtin/current").is_symlink()
    assert (target / "workspace/plugin-data/opaque-builtin/current").read_bytes() == b"not sqlite"
    assert all(not (target / "workspace" / name).exists() for name in
               (".instance.lock", ".supervisor.lock", ".supervisor.pid",
                ".runtime-ready.json", "akashic.sock"))
    assert not (target / "plugin-home/.publication.lock").exists()
    assert all(not (target / "workspace/runtime" / name).exists() for name in
               ("chat.sock", "web-chat.sock", "dashboard.sock"))
    assert {name: (data / name).read_bytes() for name in payloads} == payloads
    files = manifest["files"]
    assert isinstance(files, list)
    for name in ("-wal", "-shm"):
        path = f"workspace/plugin-data/opaque-builtin/{name}"
        assert any(row["path"] == path and row["kind"] == "file" for row in files)
    sidecars = manifest["forensic_sidecars"]
    assert isinstance(sidecars, list)
    assert {"workspace/plugin-data/opaque-builtin/absent-wal",
            "workspace/plugin-data/opaque-builtin/-wal",
            "workspace/plugin-data/opaque-builtin/-shm"}.isdisjoint(
                {row["path"] for row in sidecars})


def test_release_readiness_requires_every_selected_fiber() -> None:
    status = {"selection_ref": "a" * 64, "selection_components": ["b" * 64],
              "plugins": [{"plugin_id": "target@release", "selected_ref": "b" * 64,
                           "archive_ref": "b" * 64, "state": "active", "fiber_state": "active"}]}

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, json.dumps(status), "")

    candidate = {"AKASHIC_CONTAINER_NAME": "fixture-core",
                 "AKASHIC_CONFIG": "/state/config.toml", "AKASHIC_WORKSPACE": "/state/workspace"}
    assert activate._verify_selected_runtime(candidate=candidate, root_ref="a" * 64,
                                             run=run)["active_selected"] == 1
    status["plugins"] = []
    with pytest.raises(RuntimeError, match="完整 selection"):
        activate._verify_selected_runtime(candidate=candidate, root_ref="a" * 64,
                                          run=run)


def test_rollback_rejects_unsettled_upgrade_failure(tmp_path):
    paths = ReleasePaths(tmp_path / "release")
    paths.create_layout()
    write_json(paths.activation / "active.json", {"targetCommit": "a" * 40})
    write_json(paths.activation / "previous.json", {"targetCommit": "z" * 40})
    write_json(paths.activation / "failed-b.json", {"status": "maintenance_required"})
    args = argparse.Namespace(root=paths.root, runtime_env=tmp_path / "runtime.env",
                              mise=tmp_path / "mise", yes=True)
    with pytest.raises(RuntimeError, match="待结算的停机恢复记录"):
        release_cli.rollback(args)
