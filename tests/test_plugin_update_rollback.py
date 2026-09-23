"""Crash and pointer recovery around the live plugin selection."""
from __future__ import annotations

import asyncio
from contextlib import closing
from pathlib import Path
import signal
import shutil
import sqlite3
import subprocess
import sys
from uuid import uuid4

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from agent.plugin_composition import ServiceKey
from agent.plugins.artifacts import ArtifactPointer, ArtifactPointers, read_pointers, write_pointers
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from agent.plugins.manifest import load_plugin_manifest, set_plugin_enabled, write_plugin_manifest
from agent.plugins.reload_journal import ReloadJournal
from agent.plugins.selection import PluginSelection
from bus.event_bus import EventBus
from tests.test_plugin_install import _commit, _write_v3_plugin

CHILD = '''
import asyncio, os, signal, sys
from pathlib import Path
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
workspace, home, source = map(Path, sys.argv[1:4])
cut = sys.argv[4]
def kill():
    os.kill(os.getpid(), signal.SIGKILL)
async def run():
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    await host.load_all()
    original = host._selection.commit
    def switched(*args, **kwargs):
        if cut == "before":
            kill()
        result = original(*args, **kwargs)
        kill()
        return result
    host._selection.commit = switched
    accepted = await host.install(source=str(source), marketplace="lab", ref_name="", sparse_paths=[], update_id="crash-update")
    assert accepted.state == "accepted"
    operation = host._operation
    assert operation is not None
    await operation.task
    raise AssertionError("crash cut was not reached")
asyncio.run(run())
'''

def prepare(tmp_path):
    source = tmp_path / "source"
    module = '''from agent.plugin_composition import ServiceKey
api_version = 3
name = "probe"
version = "1.0.0"
inject = ()
async def apply(ctx):
    await ctx.provide(ServiceKey("version.probe"), lambda: "old")
'''
    _write_v3_plugin(source, name="probe", module_source=module)
    _commit(source)
    home, workspace = tmp_path / "home", tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    old = install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    (old.data_path / "history.txt").write_text("existing durable data")
    (source / "plugin.py").write_text(module.replace('"old"', '"new"'))
    _commit(source)
    return source, home, workspace, old


def arm_historical_update(source, home, workspace, previous, previous_enabled):
    """Build a real artifact, then record the legacy pending rollback input."""
    installed = install_git_plugin(
        workspace=workspace, source=str(source), marketplace="lab", plugins_home=home,
    )
    base = installed.installed_path.parents[1]
    pointers = read_pointers(base)
    assert pointers is not None and pointers.stable == pointers.latest
    update_id = "historical-" + uuid4().hex
    ReloadJournal(workspace).arm_update(
        update_id=update_id, plugin_id=f"{installed.plugin_name}@lab",
        plugin_base=base, previous=previous, candidate=pointers.stable,
        previous_enabled=previous_enabled,
    )
    return installed, update_id


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", ["before", "after"])
async def test_killed_update_boots_exact_selected_archive(tmp_path: Path, cut: str) -> None:
    """A process crash cannot turn an uncertain CAS into an implicit rollback."""
    source, home, workspace, old = prepare(tmp_path)
    initial = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    await initial.load_all()
    await initial.terminate_all()
    selected_old = PluginSelection(workspace).read()
    assert selected_old is not None
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", CHILD, str(workspace), str(home), str(source), cut],
        cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == -signal.SIGKILL, result.stdout + result.stderr
    selected_after = PluginSelection(workspace).read()
    assert (selected_after == selected_old) is (cut == "before")
    (source / "plugin.py").unlink()
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    try:
        await host.load_all()
        assert PluginSelection(workspace).read() == selected_after
        root = host.live_root
        assert root is not None
        assert root.context.require(ServiceKey("version.probe"))() == (
            "old" if cut == "before" else "new"
        )
        assert (old.data_path / "history.txt").read_text() == "existing durable data"
        assert old.installed_path.exists()
    finally:
        await host.terminate_all()


def test_rollback_keeps_prior_disabled_state_and_unrelated_manifest_entries(tmp_path):
    source, home, workspace, old = prepare(tmp_path)
    set_plugin_enabled("probe@lab", enabled=False, plugins_home=home)
    entries = load_plugin_manifest(home)
    entries["other@lab"] = False
    write_plugin_manifest(entries, plugins_home=home)
    previous = read_pointers(old.installed_path.parents[1])
    assert previous is not None
    _, update_id = arm_historical_update(source, home, workspace, previous, False)
    assert load_plugin_manifest(home)["probe@lab"] is True
    journal = ReloadJournal(workspace)
    journal.rollback_updates(home)
    journal.rollback_updates(home)
    assert journal.update(update_id).phase == "rolled_back"
    assert load_plugin_manifest(home) == {"probe@lab": False, "other@lab": False}
    assert (old.data_path / "history.txt").read_text() == "existing durable data"


def test_unknown_pointer_is_not_overwritten_by_startup_rollback(tmp_path):
    source, home, workspace, old = prepare(tmp_path)
    previous = read_pointers(old.installed_path.parents[1])
    assert previous is not None
    _, update_id = arm_historical_update(source, home, workspace, previous, True)
    base = old.installed_path.parents[1]
    other = base / ".artifacts/third"
    shutil.copytree(old.installed_path, other)
    write_pointers(base, stable=ArtifactPointer(".artifacts/third"), latest=ArtifactPointer(".artifacts/third"))
    before = (base / ".pointers.json").read_bytes()
    journal = ReloadJournal(workspace)
    with pytest.raises(RuntimeError, match="其他操作改变"):
        journal.rollback_updates(home)
    assert (base / ".pointers.json").read_bytes() == before
    assert journal.update(update_id).phase == "armed"


@pytest.mark.parametrize("empty_pointer", [False, True])
def test_first_install_rollback_restores_absent_pointer_and_manifest_entry(tmp_path, empty_pointer):
    source = tmp_path / "source"
    _write_v3_plugin(source, name="new")
    _commit(source)
    home, workspace = tmp_path / "home", tmp_path / "workspace"
    base = home / "cache/lab/new"
    if empty_pointer:
        base.mkdir(parents=True)
        write_pointers(base, stable=ArtifactPointer(None), latest=ArtifactPointer(None))
    previous = ArtifactPointers(ArtifactPointer(None), ArtifactPointer(None)) if empty_pointer else None
    result, _ = arm_historical_update(source, home, workspace, previous, None)
    journal = ReloadJournal(workspace)
    journal.rollback_updates(home)
    pointers = read_pointers(result.installed_path.parents[1])
    assert (pointers is None) is (not empty_pointer)
    if empty_pointer:
        assert pointers.stable.path is None and pointers.latest.path is None
    assert "new@lab" not in load_plugin_manifest(home)
    assert result.installed_path.exists()


@pytest.mark.parametrize("invalid", ["null", "[]", "false"])
def test_first_install_rollback_rejects_existing_nonobject_pointer(tmp_path, invalid):
    source = tmp_path / "source"
    _write_v3_plugin(source, name="new")
    _commit(source)
    home, workspace = tmp_path / "home", tmp_path / "workspace"
    result, update_id = arm_historical_update(source, home, workspace, None, None)
    path = result.installed_path.parents[1] / ".pointers.json"
    path.write_text(invalid)
    before_manifest = load_plugin_manifest(home)
    journal = ReloadJournal(workspace)
    with pytest.raises(ValueError, match="必须是对象"):
        journal.rollback_updates(home)
    assert path.read_text() == invalid
    assert load_plugin_manifest(home) == before_manifest
    assert journal.update(update_id).phase == "armed"


def test_reload_link_and_commit_are_atomic_with_update_guard(tmp_path):
    source, home, workspace, old = prepare(tmp_path)
    previous = read_pointers(old.installed_path.parents[1])
    assert previous is not None
    _, update_id = arm_historical_update(source, home, workspace, previous, True)
    journal = ReloadJournal(workspace)
    update = journal.update(update_id)
    with pytest.raises(RuntimeError, match="换候选"):
        journal.begin(plugin_id="probe@lab", base_snapshot_id=None, generation_id="bad",
            source_revision="source", config_revision="config", candidate_artifact_pointer="wrong")
    assert journal.update(update_id).reload_tx_id is None
    tx = journal.begin(plugin_id="probe@lab", base_snapshot_id=None, generation_id="candidate",
        source_revision="source", config_revision="config", candidate_artifact_pointer=update.candidate.path)
    for phase in ("prepared", "validating", "commit_started", "latest_ready", "promoting"):
        journal.advance(tx, phase)
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.execute("""CREATE TRIGGER reject_commit BEFORE UPDATE OF phase ON plugin_updates
            WHEN NEW.phase='committed' BEGIN SELECT RAISE(ABORT, 'injected cut'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="injected cut"):
        journal.advance(tx, "committed")
    assert journal.get(tx).phase == "promoting"
    assert journal.update(update_id).phase == "armed"
