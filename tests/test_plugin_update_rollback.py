"""用真实被杀进程验证更新中断后只回退，不重建或续跑候选。"""
import asyncio
from contextlib import closing
from pathlib import Path
import signal
import shutil
import sqlite3
import subprocess
import sys

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugins.artifacts import ArtifactPointer, read_pointers, write_pointers
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from agent.plugins.manifest import load_plugin_manifest, set_plugin_enabled, write_plugin_manifest
from agent.plugins.reload_journal import ReloadJournal
from bus.event_bus import EventBus
from tests.test_plugin_install import _commit, _write_v3_plugin

CHILD = '''
import asyncio, os, signal, sys
from pathlib import Path
import agent.plugins.install as installer
import agent.plugins.manager as runtime
from agent.plugins.reload_journal import ReloadJournal
from agent.plugin_composition import ServiceKey
from bus.event_bus import EventBus
workspace, home, source = map(Path, sys.argv[1:4])
cut = sys.argv[4]
def kill():
    os.kill(os.getpid(), signal.SIGKILL)
async def run():
    host = runtime.PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    await host.load_all()
    if cut in {"latest", "manifest"}:
        name = "write_pointers" if cut == "latest" else "upsert_plugin_manifest"
        original = getattr(installer, name)
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            kill()
            return result
        setattr(installer, name, changed)
    result, status = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[], update_id="crash-update")
    assert host.latest_snapshot.composition_root.context.require(ServiceKey("version.probe"))() == "new"
    if cut == "promoting":
        original = runtime._switch_ready_pointer
        def switched(*args, **kwargs):
            result = original(*args, **kwargs)
            kill()
            return result
        runtime._switch_ready_pointer = switched
    elif cut == "committed":
        original = ReloadJournal.advance
        def advanced(self, tx_id, phase, **kwargs):
            original(self, tx_id, phase, **kwargs)
            if phase == "committed":
                kill()
        ReloadJournal.advance = advanced
    await host.switch_ready("probe@lab")
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
async def apply(ctx, config):
    await ctx.provide(ServiceKey("version.probe"), lambda: "old")
'''
    _write_v3_plugin(source, name="probe", module_source=module)
    _commit(source)
    home, workspace = tmp_path / "home", tmp_path / "workspace"
    old = install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    (old.data_path / "history.txt").write_text("existing durable data")
    (source / "plugin.py").write_text(module.replace('"old"', '"new"'))
    _commit(source)
    return source, home, workspace, old


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["commit", "pointer_failure", "shutdown", "cancel_after_commit", "retry_after_cancel"])
async def test_publication_returns_to_caller_before_waiting_for_its_generation(tmp_path, monkeypatch, finish):
    """真实发布等待调用者归还旧租约，失败及关闭仍由原切换 owner 结算。"""
    from agent.plugins.snapshot import get_current_runtime_lease
    from agent.plugin_composition.context import RuntimeScope
    import agent.plugins.manager as runtime

    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    lease = None
    shutdown = None
    try:
        await host.load_all()
        stable = host.current_snapshot
        result, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        waiting = asyncio.Event()
        original_wait = host._snapshot_store.wait_for_no_leases
        async def wait(snapshot):
            assert get_current_runtime_lease() is None
            if snapshot is stable:
                waiting.set()
            await original_wait(snapshot)
        monkeypatch.setattr(host._snapshot_store, "wait_for_no_leases", wait)
        if finish == "pointer_failure":
            def fail_pointer(*args):
                raise OSError("injected publication pointer failure")
            monkeypatch.setattr(runtime, "_switch_ready_pointer", fail_pointer)
        elif finish == "cancel_after_commit":
            original_track = host._track_reload_drain
            def cancel_after_commit(*args):
                original_track(*args)
                assert host._reload_journal.update(result.update_id).phase == "committed"
                asyncio.current_task().cancel()
            monkeypatch.setattr(host, "_track_reload_drain", cancel_after_commit)
        lease = await host._snapshot_store.acquire()
        async with RuntimeScope(lease):
            host.start_update_publication(result.update_id)
            await asyncio.wait_for(waiting.wait(), 10)
            publication = host._update_publication[1]
            assert host.update_is_publishing(result.update_id)
            assert host._reload_journal.update(result.update_id).phase == "armed"
            assert host.current_snapshot is stable
            if finish == "retry_after_cancel":
                publication.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await publication
                assert host._reload_journal.update(result.update_id).error == "publication cancelled"
                host.start_update_publication(result.update_id)
                publication = host._update_publication[1]
            if finish == "shutdown":
                shutdown = asyncio.create_task(host.terminate_all())
                with pytest.raises(asyncio.CancelledError):
                    await publication
        if shutdown is not None:
            await asyncio.wait_for(shutdown, 10)
        else:
            await asyncio.wait_for(publication, 10)
        update = host._reload_journal.update(result.update_id)
        assert not host.update_is_publishing(result.update_id)
        if finish in {"commit", "cancel_after_commit", "retry_after_cancel"}:
            assert update.phase == "committed"
            assert update.error == ""
            assert host.current_snapshot.composition_root.context.require(ServiceKey("version.probe"))() == "new"
        else:
            assert update.phase != "committed"
            assert update.error == ("publication cancelled" if finish == "shutdown" else "injected publication pointer failure")
            assert read_pointers(old.installed_path.parents[1]).stable == update.previous.stable
            if finish == "pointer_failure":
                assert update.phase == "rolled_back"
                assert host._reload_journal.get(update.reload_tx_id).phase == "aborted"
                assert read_pointers(old.installed_path.parents[1]).latest == update.previous.latest
                assert host.current_snapshot.composition_root.context.require(ServiceKey("version.probe"))() == "old"
                fresh, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
                assert fresh.update_id != result.update_id
                assert host._reload_journal.update(fresh.update_id).phase == "armed"
    finally:
        if lease is not None:
            await lease.release()
        if shutdown is not None:
            await asyncio.gather(shutdown, return_exceptions=True)
        await host.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", ["latest", "manifest", "promoting", "committed"])
async def test_killed_update_returns_to_old_pointer_until_commit(tmp_path, cut):
    source, home, workspace, old = prepare(tmp_path)
    result = await asyncio.to_thread(subprocess.run,
        [sys.executable, "-c", CHILD, str(workspace), str(home), str(source), cut],
        cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == -signal.SIGKILL, result.stdout + result.stderr
    journal = ReloadJournal(workspace)
    before = journal.update("crash-update")
    assert before.phase == ("committed" if cut == "committed" else "armed")
    # 删除原 Git source，启动不能靠重新拉取或重建候选解决中断。
    (source / "plugin.py").unlink()
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    try:
        await host.load_all()
        value = host.current_snapshot.composition_root.context.require(ServiceKey("version.probe"))()
        assert value == ("new" if cut == "committed" else "old")
        update = journal.update("crash-update")
        assert update.phase == ("committed" if cut == "committed" else "rolled_back")
        pointers = read_pointers(old.installed_path.parents[1])
        assert pointers.stable == pointers.latest
        expected = before.candidate if cut == "committed" else before.previous.stable
        assert pointers.stable == expected
        assert host.ready_candidate is None
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
    staged = install_git_plugin(workspace=workspace, source=str(source), marketplace="lab",
                                plugins_home=home, stage_candidate=True)
    assert load_plugin_manifest(home)["probe@lab"] is True
    journal = ReloadJournal(workspace)
    journal.rollback_updates(home)
    journal.rollback_updates(home)
    assert journal.update(staged.update_id).phase == "rolled_back"
    assert load_plugin_manifest(home) == {"probe@lab": False, "other@lab": False}
    assert (old.data_path / "history.txt").read_text() == "existing durable data"


def test_unknown_pointer_is_not_overwritten_by_startup_rollback(tmp_path):
    source, home, workspace, old = prepare(tmp_path)
    staged = install_git_plugin(workspace=workspace, source=str(source), marketplace="lab",
                                plugins_home=home, stage_candidate=True)
    base = old.installed_path.parents[1]
    other = base / ".artifacts/third"
    shutil.copytree(old.installed_path, other)
    write_pointers(base, stable=ArtifactPointer(".artifacts/third"), latest=ArtifactPointer(".artifacts/third"))
    before = (base / ".pointers.json").read_bytes()
    journal = ReloadJournal(workspace)
    with pytest.raises(RuntimeError, match="其他操作改变"):
        journal.rollback_updates(home)
    assert (base / ".pointers.json").read_bytes() == before
    assert journal.update(staged.update_id).phase == "armed"


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
    result = install_git_plugin(workspace=workspace, source=str(source), marketplace="lab",
                                plugins_home=home, stage_candidate=True)
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
    result = install_git_plugin(workspace=workspace, source=str(source), marketplace="lab",
                                plugins_home=home, stage_candidate=True)
    path = result.installed_path.parents[1] / ".pointers.json"
    path.write_text(invalid)
    before_manifest = load_plugin_manifest(home)
    journal = ReloadJournal(workspace)
    with pytest.raises(ValueError, match="必须是对象"):
        journal.rollback_updates(home)
    assert path.read_text() == invalid
    assert load_plugin_manifest(home) == before_manifest
    assert journal.update(result.update_id).phase == "armed"


def test_reload_link_and_commit_are_atomic_with_update_guard(tmp_path):
    source, home, workspace, old = prepare(tmp_path)
    staged = install_git_plugin(workspace=workspace, source=str(source), marketplace="lab",
                                plugins_home=home, stage_candidate=True)
    journal = ReloadJournal(workspace)
    update = journal.update(staged.update_id)
    with pytest.raises(RuntimeError, match="换候选"):
        journal.begin(plugin_id="probe@lab", base_snapshot_id=None, generation_id="bad",
            source_revision="source", config_revision="config", candidate_artifact_pointer="wrong")
    assert journal.update(staged.update_id).reload_tx_id is None
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
    assert journal.update(staged.update_id).phase == "armed"


@pytest.mark.asyncio
async def test_same_artifact_enable_commits_only_after_runtime_activation(tmp_path):
    source, home, workspace, _ = prepare(tmp_path)
    installed = install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    set_plugin_enabled("probe@lab", enabled=False, plugins_home=home)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    await host.load_all()
    try:
        result, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        assert result.installed_path == installed.installed_path
        assert host.current_snapshot.composition_root.context.require(ServiceKey("version.probe"))() == "new"
        assert ReloadJournal(workspace).update(result.update_id).phase == "committed"
    finally:
        await host.terminate_all()


@pytest.mark.parametrize("interrupt", [False, True])
def test_update_migration_preserves_old_resource_rows_with_native_backup(tmp_path, monkeypatch, interrupt):
    from yoyo import get_backend, read_migrations
    from agent.migrations.context import bind_migration_context
    journal = ReloadJournal(tmp_path)
    tx = journal.begin(plugin_id="sample", base_snapshot_id=None, generation_id="generation", source_revision="source", config_revision="config")
    with closing(sqlite3.connect(journal.path)) as conn, conn:
        conn.execute("DROP TABLE plugin_updates")
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "20260906_02_session_attributes.py").write_text('from yoyo import step\nsteps = [step("SELECT 1")]\n')
    source = Path(__file__).parents[1] / "migrations/yoyo/20260906_03_plugin_update_rollback.py"
    (directory / source.name).write_bytes(source.read_bytes())
    backend = get_backend(f'sqlite:///{tmp_path / "ledger.db"}')
    migrations = read_migrations(str(directory))
    before = journal.get(tx)
    with backend, bind_migration_context(config_path=tmp_path / "config.toml", workspace=tmp_path):
        if interrupt:
            migrations[-1].load()
            module = migrations[-1].module
            with monkeypatch.context() as patch:
                patch.setattr(module, "SCHEMA", {**module.SCHEMA, "failure": "INVALID SQL"})
                with pytest.raises(sqlite3.OperationalError):
                    module.migrate_updates(None)
            with closing(sqlite3.connect(journal.path)) as conn:
                assert conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'plugin_update%'").fetchall() == []
            assert journal.get(tx) == before
        backend.apply_migrations(backend.to_apply(migrations))
        assert not backend.to_apply(migrations)
        migrations[-1].module.migrate_updates(None)
    assert journal.get(tx) == before
    backups = tuple((tmp_path / "backups/plugin-update-rollback").glob("*/plugin-reloads.sqlite3"))
    assert len(backups) == (2 if interrupt else 1)
    with closing(sqlite3.connect(backups[0])) as conn:
        assert conn.execute("SELECT tx_id FROM reload_transactions").fetchall() == [(tx,)]
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
