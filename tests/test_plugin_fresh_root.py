"""整体换代使用固定输入和独立物理实例，失败资源由原 Root 保留。"""
import asyncio
import sys

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from tests.test_plugin_install import _commit, _write_v3_plugin


MODULE = '''from agent.plugin_composition import ServiceKey, RUNTIME_STARTED
api_version = 3
name = "NAME"
version = "1.0.0"
STATE = {"mounts": 0, "starts": 0, "closes": 0}
async def apply(ctx):
    STATE["mounts"] += 1
    def started(_):
        STATE["starts"] += 1
    async def close():
        STATE["closes"] += 1
        if FAIL_CLOSE and STATE["closes"] == 1:
            raise OSError("connection still open")
    await ctx.effect(lambda: close)
    await ctx.provide(ServiceKey("NAME.state"), STATE)
    await ctx.on(RUNTIME_STARTED, started)
    if REJECT_FORMAL and (ctx.runtime.workspace / "reject-formal").exists():
        raise ValueError("formal rejected after acquisition")
'''


def module(name, *, reject=False, fail_close=False):
    return MODULE.replace("NAME", name).replace("REJECT_FORMAL", repr(reject)).replace("FAIL_CLOSE", repr(fail_close))


def installed_pair(tmp_path):
    workspace, home = tmp_path / "workspace", tmp_path / "home"
    for name in ("changed", "peer"):
        source = tmp_path / name
        _write_v3_plugin(source, name=name, module_source=module(name))
        _commit(source)
        install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    return PluginManager(
        [], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache",
    )


async def candidate(host, tmp_path, *, reject=False, fail_close=False):
    source = tmp_path / "changed"
    (source / "plugin.py").write_text(module("changed", reject=reject, fail_close=fail_close) + "\nrevision = 2\n")
    _commit(source)
    await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
    return host.latest_snapshot


def inputs(snapshot):
    return {key: item.archive_ref for key, item in snapshot.generations.items()}


def assert_actual_instances(snapshot):
    for plugin_id, generation in snapshot.generations.items():
        runtime = snapshot.composition_root.plugin_runtime(plugin_id)
        assert runtime.generation_id == generation.generation_id
        assert runtime.data_dir == generation.data_dir
        name = plugin_id.split("@", 1)[0]
        assert snapshot.composition_root.context.require(ServiceKey(name + ".state")) is generation.instance.module.STATE
        assert generation.runtime_snapshot is snapshot


@pytest.mark.asyncio
async def test_cancel_during_external_switch_joins_cleanup_without_committing_candidate(tmp_path):
    host = installed_pair(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    switches = []
    publication = None

    async def switch(old, new):
        switches.append((old, new))
        if len(switches) == 1:
            entered.set()
            await release.wait()

    try:
        await host.load_all()
        stable = host.current_snapshot
        checked = await candidate(host, tmp_path)
        host.bind_endpoint_switcher(switch)
        publication = asyncio.create_task(host.switch_ready("changed@lab"))
        await entered.wait()
        publication.cancel()
        asyncio.get_running_loop().call_soon(publication.cancel)
        asyncio.get_running_loop().call_soon(release.set)
        with pytest.raises(asyncio.CancelledError):
            await publication
        assert inputs(host.current_snapshot) == inputs(stable)
        assert host.current_snapshot is not stable
        assert host.current_snapshot is not checked
        assert host.current_snapshot.accepting_leases
        assert len(switches) == 3  # 新端点尝试、撤销、重建旧组合后重新接入。
        assert host._building_roots == {}
        assert_actual_instances(host.current_snapshot)
    finally:
        release.set()
        if publication is not None and not publication.done():
            publication.cancel()
            await asyncio.gather(publication, return_exceptions=True)
        await host.terminate_all()


@pytest.mark.asyncio
async def test_candidate_and_formal_mount_fresh_instances_for_every_plugin(tmp_path):
    host = installed_pair(tmp_path)
    try:
        await host.load_all()
        stable = host.current_snapshot
        ready = await candidate(host, tmp_path)
        candidate_dirs = {key: item.data_dir for key, item in ready.generations.items()}
        assert_actual_instances(stable)
        assert_actual_instances(ready)
        for key, item in ready.generations.items():
            old = stable.generations[key]
            assert item is not old
            assert item.instance.module is not old.instance.module
            assert item.scope is not old.scope
            assert item.data_dir != old.data_dir
        assert ready.generations["peer@lab"].archive_ref == stable.generations["peer@lab"].archive_ref

        await host.switch_ready("changed@lab")
        formal = host.current_snapshot
        assert inputs(formal) == inputs(ready)
        assert len({stable.snapshot_id, ready.snapshot_id, formal.snapshot_id}) == 3
        tx_id = ready.generations["changed@lab"].reload_tx_id
        assert host.reload_journal.get(tx_id).generation_id == ready.generations["changed@lab"].generation_id
        assert formal.generations["changed@lab"].generation_id in host.reload_journal.runtime_generation_ids(tx_id)
        assert_actual_instances(formal)
        for key, actual in formal.generations.items():
            old, checked = stable.generations[key], ready.generations[key]
            assert actual is not checked and actual is not old
            assert actual.instance.module is not checked.instance.module
            assert actual.instance.module is not old.instance.module
            assert actual.scope is not checked.scope and actual.scope is not old.scope
            assert len({actual.generation_id, old.generation_id, checked.generation_id}) == 3
            assert actual.data_dir == old.data_dir
            assert checked.data_dir == candidate_dirs[key]
            assert checked.scope.closed and old.scope.closed
            assert actual.instance.module.STATE == {"mounts": 1, "starts": 1, "closes": 0}
            assert host.generation(key) is actual
            assert sys.modules[host._stable_aliases[actual.module_path]] is actual.instance.module
        assert host._building_roots == {}
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_store_rejects_republishing_a_shared_generation(tmp_path):
    from agent.plugins.snapshot import RuntimeSnapshot

    host = installed_pair(tmp_path)
    try:
        await host.load_all()
        stable = host.current_snapshot
        shared = RuntimeSnapshot(snapshot_id="shared-generation", generations=stable.generations)
        with pytest.raises(RuntimeError, match="共享 PluginGeneration"):
            host.snapshot_store.begin_publish(shared)
        assert host.current_snapshot is stable
        assert host.snapshot_store.pending_transaction is None
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_import_failure_keeps_module_tree_when_scope_cleanup_fails(tmp_path, monkeypatch):
    source = tmp_path / "plugins/import_owner"
    _write_v3_plugin(source, name="import_owner", module_source="api_version = 3\nname = 'import_owner'\nversion = '1.0.0'\nfrom . import helper\nraise ValueError('partial import')\n")
    (source / "helper.py").write_text("RESOURCE = object()\n")
    host = PluginManager(
        [source.parent], event_bus=EventBus(), workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home/cache",
    )
    close_scope = host._close_root_scope
    attempts = 0

    async def fail_once(scope, module_path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("scope resource still open")
        await close_scope(scope, module_path)

    monkeypatch.setattr(host, "_close_root_scope", fail_once)
    with pytest.raises(BaseExceptionGroup):
        await host._load_one(host.discover()[0])
    [(root, generations)] = host._building_roots.items()
    assert generations == ()
    [(module_path, scope)] = host._scopes.items()
    assert module_path in sys.modules and module_path + ".helper" in sys.modules
    assert not scope.closed
    assert attempts == 1
    await host.terminate_all()
    assert attempts == 2 and scope.closed
    assert root not in host._building_roots
    assert module_path not in sys.modules and module_path + ".helper" not in sys.modules


@pytest.mark.asyncio
async def test_formal_failure_rebuilds_old_inputs_as_a_new_snapshot(tmp_path):
    host = installed_pair(tmp_path)
    try:
        await host.load_all()
        stable = host.current_snapshot
        old_root = stable.composition_root
        ready = await candidate(host, tmp_path, reject=True)
        (tmp_path / "workspace/reject-formal").touch()
        with pytest.raises(Exception, match="formal rejected"):
            await host.switch_ready("changed@lab")
        restored = host.current_snapshot
        assert restored is not stable and restored is not ready
        assert stable.composition_root is old_root
        assert inputs(restored) == inputs(stable)
        assert restored.accepting_leases
        assert_actual_instances(restored)
        for key, generation in restored.generations.items():
            assert generation is not stable.generations[key]
            assert generation.instance.module is not stable.generations[key].instance.module
            assert generation.scope is not stable.generations[key].scope
            assert stable.generations[key].scope.closed
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["mount", "post_compile"])
async def test_partial_formal_mount_retains_failed_owner_until_explicit_recovery(tmp_path, monkeypatch, stage):
    host = installed_pair(tmp_path)
    try:
        await host.load_all()
        stable = host.current_snapshot
        # 候选关闭必须成功；只让正式实例取得的连接首次关闭失败。
        source = tmp_path / "changed"
        text = module("changed", reject=stage == "mount", fail_close=True).replace(
            "if True and STATE[\"closes\"] == 1:",
            "if (ctx.runtime.workspace / 'reject-formal').exists() and STATE[\"closes\"] == 1:",
        )
        (source / "plugin.py").write_text(text)
        _commit(source)
        await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        (tmp_path / "workspace/reject-formal").touch()
        captured = []
        invariants = host._post_snapshot_invariants

        async def fail_after_compile(snapshot):
            captured.append(snapshot)
            raise ValueError("formal validation failed after compilation")

        if stage == "post_compile":
            monkeypatch.setattr(host, "_post_snapshot_invariants", fail_after_compile)
        with pytest.raises(BaseExceptionGroup):
            await host.switch_ready("changed@lab")
        assert host.current_snapshot is stable
        assert not stable.accepting_leases
        if stage == "mount":
            [(root, owners)] = host._building_roots.items()
            failed = next(item for item in owners if item.plugin_id == "changed@lab")
            assert failed.runtime_snapshot is None
        else:
            [snapshot] = captured
            assert snapshot.snapshot_id in host.snapshot_store.retained_snapshot_ids
            assert snapshot.state == "aborted"
            root = snapshot.composition_root
            failed = snapshot.generations["changed@lab"]
            assert failed.runtime_snapshot is snapshot
            monkeypatch.setattr(host, "_post_snapshot_invariants", invariants)
        physical_module = failed.instance.module
        assert physical_module.STATE["closes"] == 1
        assert sys.modules[failed.module_path] is physical_module
        assert not failed.scope.closed
        await host.retry_runtime_recovery("changed@lab")
        assert physical_module.STATE["closes"] == 2
        assert failed.module_path not in sys.modules
        assert failed.scope.closed
        assert root not in host._building_roots
        assert host.current_snapshot is not stable
        assert inputs(host.current_snapshot) == inputs(stable)
        assert_actual_instances(host.current_snapshot)
    finally:
        await host.terminate_all()
