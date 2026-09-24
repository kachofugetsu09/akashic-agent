"""Release upgrade stops old owners before touching migrated state."""

from __future__ import annotations

import json
import hashlib
import argparse
import shutil
import sqlite3
import subprocess
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.akashic_release import activate
from scripts.akashic_release import cli as release_cli
from scripts.akashic_release.manifest import read_json, write_json
from scripts.akashic_release.model import ReleasePaths
from agent.migrations.release_backup import backup_release_state
from agent.plugins.reload_journal import ReloadJournal
from agent.plugins.selection import PluginSelection


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


def _external_activation_case(tmp_path: Path, monkeypatch):
    """Use real selection and backup owners around a controlled service boundary."""

    paths = ReleasePaths(tmp_path / "release")
    paths.create_layout()
    workspace = paths.state / "workspace"
    home = paths.state / "plugin-home"
    workspace.mkdir()
    home.mkdir()
    (paths.state / "config.toml").write_text("[runtime]\n")
    data = workspace / "plugin-data/fixture"
    data.mkdir(parents=True)
    (data / "data.txt").write_text("v1")
    selection = PluginSelection(workspace)
    selection.initialize()
    old_root = selection.commit((), expected_ref=None)
    _ = ReloadJournal(workspace)
    write_json(paths.activation / "active.json", {"status": "active", "targetCommit": "a" * 40})
    manifest = paths.release("b" * 40)
    image = "sha256:" + "b" * 64
    write_json(manifest, {"sourceCommit": "b" * 40, "imageId": image})
    environment = tmp_path / "runtime.env"
    environment.write_text("OLD=value\n")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    events: list[str] = []
    ticks = iter(datetime(2026, 9, 24, 12, minute, tzinfo=timezone.utc) for minute in range(10))
    monkeypatch.setattr(activate, "datetime", type("Clock", (), {"now": staticmethod(lambda tz: next(ticks))}))
    monkeypatch.setattr(activate, "release_environment", lambda **kwargs: {
        "AKASHIC_CONFIG": str(paths.state / "config.toml"),
        "AKASHIC_WORKSPACE": str(workspace),
        "AKASHIC_PLUGIN_HOME": str(home),
        "AKASHIC_RUNTIME_COMMIT": "b" * 40,
        "AKASHIC_RUNTIME_TREE": "c" * 40,
        "AKASHIC_CONTAINER_NAME": "fixture-core",
    })
    monkeypatch.setattr(activate, "verify_release", lambda path: events.append("doctor"))
    monkeypatch.setattr(activate, "_prepare_workload_dirs", lambda paths: events.append("workload_dirs"))
    monkeypatch.setattr(activate, "stop_runtime", lambda **kwargs: events.append("stop"))
    monkeypatch.setattr(activate, "start_bridge", lambda **kwargs: events.append("bridge"))
    monkeypatch.setattr(activate, "start_core", lambda **kwargs: events.append("core"))

    def stopped_upgrade(**kwargs):
        if kwargs.get("preflight_only"):
            events.append("preflight")
            return {"status": "preflight_ok"}
        events.append("upgrade")
        before = selection.read()
        backup = kwargs["backup_dir"]
        backup_release_state(paths.state, backup)
        (data / "data.txt").write_text("v2")
        new_root = selection.commit((), expected_ref=before)
        return {"status": "selected_not_started", "old_root_ref": before,
                "new_root_ref": new_root, "ordered_components": [],
                "backup_dir": str(backup),
                "external_plan_sha256": kwargs["expected_plan_sha256"]}

    monkeypatch.setattr(activate, "_stopped_upgrade", stopped_upgrade)

    def run(command, **kwargs):
        if command[:2] == ["docker", "exec"]:
            events.append("live")
            return subprocess.CompletedProcess(command, 0, json.dumps({
                "selection_ref": selection.read(), "selection_components": [], "plugins": [],
            }), "")
        if command[:2] == ["systemctl", "is-active"]:
            return subprocess.CompletedProcess(command, 3, "inactive\n", "")
        raise AssertionError(command)

    def plan_for(root: str, name: str) -> Path:
        plan = inputs / f"{name}.json"
        plan.write_text(json.dumps({"schema_version": 1, "expected_root_ref": root,
                                    "targets": [{"plugin_id": "outside@external",
                                                 "bundle_relative_path": "outside.bundle",
                                                 "bundle_sha256": "d" * 64,
                                                 "target_commit": "c" * 40}]}))
        return plan

    return paths, manifest, environment, inputs, selection, data, events, run, plan_for


@pytest.mark.parametrize("failing_write", ["workload_dirs", "environment"])
def test_external_prestart_write_failure_can_settle_full_restore(tmp_path, monkeypatch, failing_write):
    paths, manifest, environment, inputs, selection, data, events, run, plan_for = (
        _external_activation_case(tmp_path, monkeypatch))
    old_root = selection.read()
    assert old_root is not None
    plan = plan_for(old_root, "first")
    original_atomic = activate.atomic_write
    if failing_write == "workload_dirs":
        monkeypatch.setattr(activate, "_prepare_workload_dirs",
                            lambda paths: (_ for _ in ()).throw(OSError("workload dir failed")))
    else:
        def fail_environment(path, content):
            if path == environment:
                raise OSError("runtime.env replace failed")
            return original_atomic(path, content)
        monkeypatch.setattr(activate, "atomic_write", fail_environment)

    with pytest.raises(RuntimeError, match="启动前"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True, external_plan=plan,
                                  external_inputs=inputs)
    assert "upgrade" in events and "bridge" not in events and "core" not in events
    attempt_path = next(paths.activation.glob("attempt-external-*.json"))
    failed_path = next(paths.activation.glob("failed-*.json"))
    attempt = read_json(attempt_path)
    failed = read_json(failed_path)
    assert failed["status"] == "maintenance_required"
    assert failed["phase"] == "before_target_start" and failed["targetStarted"] is False
    assert failed["attemptPath"] == str(attempt_path)
    assert failed["upgrade"]["new_root_ref"] == selection.read()
    assert failed["backupDir"] == attempt["backupDir"]
    assert failed["environmentBackup"] == attempt["environmentBackup"]
    assert failed["environmentBackupSha256"] == attempt["environmentBackupSha256"]
    assert "failed" in failed["detail"]
    before = list(events)
    with pytest.raises(RuntimeError, match="未结算 release failure"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True, external_plan=plan,
                                  external_inputs=inputs)
    assert events == [*before, "stop"]
    with pytest.raises(RuntimeError, match="未完整恢复"):
        activate.settle_restored_failure(paths=paths, failed_path=failed_path,
                                         environment_file=environment, run=run)
    post_failure = tmp_path / "post-failure-state"
    shutil.copytree(paths.state, post_failure, symlinks=True)
    shutil.rmtree(paths.state)
    shutil.copytree(Path(attempt["backupDir"]) / "state", paths.state, symlinks=True)
    environment.write_bytes(Path(attempt["environmentBackup"]).read_bytes())
    assert selection.read() == old_root and (data / "data.txt").read_text() == "v1"
    settled = activate.settle_restored_failure(paths=paths, failed_path=failed_path,
                                               environment_file=environment, run=run)
    assert settled["status"] == "verified_full_restore"
    monkeypatch.setattr(activate, "_prepare_workload_dirs", lambda paths: events.append("workload_dirs"))
    monkeypatch.setattr(activate, "atomic_write", original_atomic)
    second = plan_for(old_root, "second")
    assert activate.activate_release(paths=paths, manifest_path=manifest,
                                     environment_file=environment, mise=tmp_path / "mise",
                                     run=run, upgrade=True, external_plan=second,
                                     external_inputs=inputs) == "activated"


def test_active_receipt_survives_terminal_attempt_write_failure(tmp_path, monkeypatch):
    paths, manifest, environment, inputs, selection, data, events, run, plan_for = (
        _external_activation_case(tmp_path, monkeypatch))
    old_root = selection.read()
    assert old_root is not None
    first = plan_for(old_root, "first")
    original_write = activate.write_json
    failed_once = False

    def fail_terminal(path, document):
        nonlocal failed_once
        if (path.name.startswith("attempt-external-")
            and document.get("status") == "active" and not failed_once):
            failed_once = True
            raise OSError("terminal attempt fsync failed")
        return original_write(path, document)

    monkeypatch.setattr(activate, "write_json", fail_terminal)
    with pytest.raises(OSError, match="terminal attempt"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True, external_plan=first,
                                  external_inputs=inputs)
    assert read_json(paths.activation / "active.json")["status"] == "active"
    first_attempt = next(paths.activation.glob("attempt-external-*.json"))
    assert read_json(first_attempt)["status"] == "pending"
    after_success = list(events)
    assert activate.activate_release(paths=paths, manifest_path=manifest,
                                     environment_file=environment, mise=tmp_path / "mise",
                                     run=run, upgrade=True, external_plan=first,
                                     external_inputs=inputs) == "already_active"
    assert events == [*after_success, "doctor", "live"]
    current_root = selection.read()
    assert current_root is not None
    second = plan_for(current_root, "second")
    assert activate.activate_release(paths=paths, manifest_path=manifest,
                                     environment_file=environment, mise=tmp_path / "mise",
                                     run=run, upgrade=True, external_plan=second,
                                     external_inputs=inputs) == "activated"
    assert read_json(first_attempt)["status"] == "active"
    assert read_json(first_attempt)["backupDir"]
    assert read_json(first_attempt)["imageId"] == read_json(manifest)["imageId"]
    current_root = selection.read()
    assert current_root is not None
    third = plan_for(current_root, "third")
    assert activate.activate_release(paths=paths, manifest_path=manifest,
                                     environment_file=environment, mise=tmp_path / "mise",
                                     run=run, upgrade=True, external_plan=third,
                                     external_inputs=inputs) == "activated"
    assert len(list(paths.activation.glob("attempt-external-*.json"))) == 3


def test_active_replay_refuses_unrelated_pending_attempt(tmp_path, monkeypatch):
    paths, manifest, environment, inputs, selection, data, events, run, plan_for = (
        _external_activation_case(tmp_path, monkeypatch))
    root = selection.read()
    assert root is not None
    first = plan_for(root, "first")
    assert activate.activate_release(paths=paths, manifest_path=manifest,
                                     environment_file=environment, mise=tmp_path / "mise",
                                     run=run, upgrade=True, external_plan=first,
                                     external_inputs=inputs) == "activated"
    other = paths.activation / "attempt-external-unrelated.json"
    write_json(other, {"status": "pending", "targetCommit": "f" * 40})
    before = list(events)
    with pytest.raises(RuntimeError, match="其他不完整 external attempt"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True, external_plan=first,
                                  external_inputs=inputs)
    assert events == [*before, "doctor", "live", "stop"]
    assert read_json(other)["status"] == "pending"


@pytest.mark.parametrize("mismatch", [
    "attempt_image", "attempt_plan", "attempt_old_root", "active_new_root",
    "active_order", "active_attempt_path", "missing_image_both", "doctor", "live",
])
def test_pending_success_needs_exact_bound_live_proof(tmp_path, monkeypatch, mismatch):
    paths, manifest, environment, inputs, selection, data, events, run, plan_for = (
        _external_activation_case(tmp_path, monkeypatch))
    old_root = selection.read()
    assert old_root is not None
    first = plan_for(old_root, "first")
    original_write = activate.write_json

    def fail_terminal(path, document):
        if path.name.startswith("attempt-external-") and document.get("status") == "active":
            raise OSError("terminal attempt fsync failed")
        return original_write(path, document)

    monkeypatch.setattr(activate, "write_json", fail_terminal)
    with pytest.raises(OSError, match="terminal attempt"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True, external_plan=first,
                                  external_inputs=inputs)
    monkeypatch.setattr(activate, "write_json", original_write)
    attempt_path = next(paths.activation.glob("attempt-external-*.json"))
    active_path = paths.activation / "active.json"
    attempt = read_json(attempt_path)
    active = read_json(active_path)
    if mismatch == "attempt_image":
        attempt["imageId"] = "sha256:" + "f" * 64
    elif mismatch == "attempt_plan":
        attempt["externalPlanSha256"] = "f" * 64
    elif mismatch == "attempt_old_root":
        attempt["oldRootRef"] = "f" * 64
    elif mismatch == "active_new_root":
        active["upgrade"]["new_root_ref"] = "f" * 64
    elif mismatch == "active_order":
        active["upgrade"]["ordered_components"] = ["f" * 64]
    elif mismatch == "active_attempt_path":
        active["attemptPath"] = str(paths.activation / "unrelated.json")
    elif mismatch == "missing_image_both":
        attempt.pop("imageId")
        active.pop("imageId")
    elif mismatch == "doctor":
        monkeypatch.setattr(activate, "verify_release",
                            lambda path: (_ for _ in ()).throw(RuntimeError("doctor failed")))
    elif mismatch == "live":
        monkeypatch.setattr(activate, "_verify_selected_runtime",
                            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("live failed")))
    original_write(attempt_path, attempt)
    original_write(active_path, active)
    current_root = selection.read()
    assert current_root is not None
    second = plan_for(current_root, "second")
    before = list(events)
    with pytest.raises(RuntimeError):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True, external_plan=second,
                                  external_inputs=inputs)
    assert events[-1] == "stop"
    assert "preflight" not in events[len(before):]
    assert "upgrade" not in events[len(before):]
    assert "bridge" not in events[len(before):] and "core" not in events[len(before):]
    assert read_json(attempt_path)["status"] == "pending"


def test_prestart_failure_retains_both_errors_if_receipt_write_fails(tmp_path, monkeypatch):
    paths, manifest, environment, inputs, selection, data, events, run, plan_for = (
        _external_activation_case(tmp_path, monkeypatch))
    root = selection.read()
    assert root is not None
    plan = plan_for(root, "first")
    monkeypatch.setattr(activate, "_prepare_workload_dirs",
                        lambda paths: (_ for _ in ()).throw(OSError("original workload failure")))
    original_write = activate.write_json

    def fail_receipt(path, document):
        if path.name.startswith("failed-"):
            raise OSError("receipt fsync failure")
        return original_write(path, document)

    monkeypatch.setattr(activate, "write_json", fail_receipt)
    with pytest.raises(RuntimeError, match="original workload failure.*receipt fsync failure"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True, external_plan=plan,
                                  external_inputs=inputs)
    assert "bridge" not in events and "core" not in events
    assert read_json(next(paths.activation.glob("attempt-external-*.json")))["status"] == "pending"


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
    assert events == [*before, "stop"]


def test_external_active_receipt_failure_stops_and_blocks_reentry(tmp_path, monkeypatch):
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
    monkeypatch.setattr(activate, "_stopped_upgrade", lambda **kwargs:
                        {"status": "preflight_ok"} if kwargs.get("preflight_only") else
                        {"status": "selected_not_started", "old_root_ref": "a" * 64,
                         "new_root_ref": "b" * 64, "ordered_components": ["d" * 64]})
    monkeypatch.setattr(activate, "_verify_selected_runtime", lambda **kwargs:
                        {"selection_ref": "b" * 64, "active_selected": 1})
    original_write = activate.write_json

    def fail_active(path, document):
        if path == paths.activation / "active.json":
            raise OSError("injected active receipt fsync failure")
        return original_write(path, document)

    monkeypatch.setattr(activate, "write_json", fail_active)

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(RuntimeError, match="目标 runtime 已停"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True, external_plan=plan,
                                  external_inputs=inputs)
    assert events == ["stop", "bridge", "core", "stop"]
    failed = next(paths.activation.glob("failed-*.json"))
    assert read_json(failed)["phase"] == "active_receipt"
    before = list(events)
    with pytest.raises(RuntimeError, match="未结算 release failure"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True, external_plan=plan,
                                  external_inputs=inputs)
    assert events == [*before, "stop"]


def test_same_core_new_external_plan_runs_stopped_preflight(tmp_path, monkeypatch):
    paths, manifest, environment, events = _release_case(tmp_path, monkeypatch)
    write_json(paths.activation / "active.json", {"status": "active", "targetCommit": "b" * 40})
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    plan = inputs / "plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "expected_root_ref": "a" * 64,
                                "targets": [{"plugin_id": "outside@external",
                                             "bundle_relative_path": "outside.bundle",
                                             "bundle_sha256": "b" * 64,
                                             "target_commit": "c" * 40}]}))
    monkeypatch.setattr(activate, "verify_release", lambda environment: None)

    def conflict(**kwargs):
        events.append("preflight")
        raise ValueError("explicit target disabled")

    monkeypatch.setattr(activate, "_stopped_upgrade", conflict)

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(RuntimeError, match="旧 runtime 已恢复"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True, external_plan=plan,
                                  external_inputs=inputs)
    assert events == ["stop", "preflight", "bridge", "core"]


def test_exact_active_external_replay_is_read_only(tmp_path, monkeypatch):
    paths = ReleasePaths(tmp_path / "release")
    paths.create_layout()
    workspace = paths.state / "workspace"
    workspace.mkdir()
    selection = PluginSelection(workspace)
    selection.initialize()
    selected_root = selection.commit((), expected_ref=None)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    plan = inputs / "plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "expected_root_ref": "a" * 64,
                                "targets": [{"plugin_id": "outside@external",
                                             "bundle_relative_path": "outside.bundle",
                                             "bundle_sha256": "b" * 64,
                                             "target_commit": "c" * 40}]}))
    image = "sha256:" + "b" * 64
    manifest = paths.release("b" * 40)
    write_json(manifest, {"sourceCommit": "b" * 40, "imageId": image})
    attempt_path = paths.activation / "attempt-external-replay.json"
    plan_digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    write_json(attempt_path, {"status": "pending", "targetCommit": "b" * 40,
                              "imageId": image, "externalPlanSha256": plan_digest,
                              "oldRootRef": "a" * 64, "backupDir": str(paths.backups / "upgrade-replay"),
                              "environmentBackup": str(paths.backups / "runtime.env.replay"),
                              "environmentBackupSha256": "e" * 64})
    write_json(paths.activation / "active.json", {
        "status": "active", "targetCommit": "b" * 40, "imageId": image,
        "upgrade": {"old_root_ref": "a" * 64, "new_root_ref": selected_root,
                    "ordered_components": [],
                    "backup_dir": str(paths.backups / "upgrade-replay"),
                    "external_plan_sha256": plan_digest},
        "attemptPath": str(attempt_path),
        "environmentBackup": str(paths.backups / "runtime.env.replay"),
        "environmentBackupSha256": "e" * 64,
        "runtimeCheck": {"selection_ref": selected_root, "active_selected": 0},
    })
    environment = tmp_path / "runtime.env"
    environment.write_text("AKASHIC_CONTAINER_NAME=fixture-core\n")
    monkeypatch.setattr(activate, "verify_release", lambda environment: None)
    monkeypatch.setattr(activate, "_verify_selected_runtime", lambda **kwargs:
                        {"selection_ref": selected_root, "active_selected": 0})
    monkeypatch.setattr(activate, "stop_runtime", lambda **kwargs:
                        pytest.fail("verified active replay must not stop"))

    def run(command, **kwargs):
        pytest.fail("verified active replay must not call Docker")

    assert activate.activate_release(paths=paths, manifest_path=manifest,
                                     environment_file=environment, mise=tmp_path / "mise",
                                     run=run, upgrade=True, external_plan=plan,
                                     external_inputs=inputs) == "already_active"


def test_environment_backup_is_exact_and_durable(tmp_path):
    source = tmp_path / "runtime.env"
    source.write_bytes(b"A=old\nB=\xc3\xa9\n")
    backup = tmp_path / "backups/runtime.env"
    digest = activate._save_environment_backup(source, backup)
    assert backup.read_bytes() == source.read_bytes()
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        activate._save_environment_backup(source, backup)


def test_external_preflight_conflict_restores_old_runtime_without_attempt(tmp_path, monkeypatch):
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

    def preflight_conflict(**kwargs):
        raise ValueError("disabled target")

    monkeypatch.setattr(activate, "_stopped_upgrade", preflight_conflict)

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(RuntimeError, match="旧 runtime 已恢复"):
        activate.activate_release(paths=paths, manifest_path=manifest,
                                  environment_file=environment, mise=tmp_path / "mise",
                                  run=run, upgrade=True, external_plan=plan,
                                  external_inputs=inputs)
    assert events == ["stop", "bridge", "core"]
    assert not list(paths.activation.glob("attempt-external-*.json"))
    failed = next(paths.activation.glob("failed-*.json"))
    assert json.loads(failed.read_text())["status"] == "preflight_conflict"


def test_stopped_external_failure_settles_only_after_complete_restore(tmp_path):
    paths = ReleasePaths(tmp_path / "release")
    paths.create_layout()
    workspace = paths.state / "workspace"
    home = paths.state / "plugin-home"
    workspace.mkdir()
    home.mkdir()
    (paths.state / "config.toml").write_text("[runtime]\n")
    data = workspace / "plugin-data/fixture"
    data.mkdir(parents=True)
    opaque = data / "-wal"
    opaque.write_bytes(b"original opaque bytes")
    selection = PluginSelection(workspace)
    selection.initialize()
    old_root = selection.commit((), expected_ref=None)
    _ = ReloadJournal(workspace)
    active = paths.activation / "active.json"
    write_json(active, {"status": "active", "targetCommit": "a" * 40})
    environment = tmp_path / "runtime.env"
    environment.write_bytes(b"OLD=exact\n")
    env_backup = paths.backups / "runtime.env.before-test"
    env_digest = activate._save_environment_backup(environment, env_backup)
    recovery = paths.backups / "upgrade-test"
    backup_release_state(paths.state, recovery)
    attempt_path = paths.activation / "attempt-external-test.json"
    write_json(attempt_path, {"status": "pending", "targetCommit": "b" * 40,
                              "backupDir": str(recovery), "oldRootRef": old_root})
    failed_path = paths.activation / "failed-test.json"
    failure = {"status": "maintenance_required", "phase": "stopped_upgrade",
               "targetCommit": "b" * 40, "previousCommit": "a" * 40,
               "backupDir": str(recovery), "environmentBackup": str(env_backup),
               "environmentBackupSha256": env_digest, "attemptPath": str(attempt_path)}
    write_json(failed_path, failure)

    def inactive(command, **kwargs):
        return subprocess.CompletedProcess(command, 3, "inactive\n", "")

    opaque.write_bytes(b"partial restore")
    with pytest.raises(RuntimeError, match="字节未完整恢复"):
        activate.settle_restored_failure(paths=paths, failed_path=failed_path,
                                         environment_file=environment, run=inactive)
    assert not activate.failure_settled(paths, failed_path)
    shutil.rmtree(paths.state)
    shutil.copytree(recovery / "state", paths.state, symlinks=True)
    settled = activate.settle_restored_failure(paths=paths, failed_path=failed_path,
                                               environment_file=environment, run=inactive)
    assert settled["status"] == "verified_full_restore"
    assert settled["oldRootRef"] == old_root
    assert activate.failure_settled(paths, failed_path)
    assert activate._attempt_settled(paths, attempt_path)
    assert read_json(failed_path) == failure
    second = paths.activation / "failed-post-start.json"
    write_json(second, {**failure, "phase": "target_start_or_readiness"})
    with pytest.raises(ValueError, match="尚未启动目标 runtime"):
        activate.settle_restored_failure(paths=paths, failed_path=second,
                                         environment_file=environment, run=inactive)


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
