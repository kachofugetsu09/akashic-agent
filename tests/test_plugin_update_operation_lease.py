"""真实更新能力只能在正式租约内准备候选，不能同步排空自己。"""
import asyncio
from contextlib import asynccontextmanager

import pytest

from agent.plugin_composition import CompositionRoot, RuntimeScope, ServiceKey
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
from agent.plugins.manager import PluginManager
from agent.plugins.install import install_git_plugin
from agent.plugins.snapshot import get_current_runtime_lease, lease_runtime_snapshot
from bus.event_bus import EventBus
from session.log import MessageLog
from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from tests.test_plugin_install import _commit, _write_v3_plugin


CALLER = '''
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
api_version = 3
name = "caller"
version = "1.0.0"
inject = (PLUGIN_UPDATES,)
async def apply(ctx):
    await ctx.provide(ServiceKey("test.context"), ctx)
'''

VALUE = '''
from agent.plugin_composition import ServiceKey
api_version = 3
name = "NAME"
version = "1.0.0"
async def apply(ctx):
    await ctx.provide(ServiceKey("test.NAME"), lambda: "old")
'''


@asynccontextmanager
async def installed_host(tmp_path, *, existing=True, changed=True):
    """新 workspace 的正式调用插件与待安装目标各自拥有真实源码。"""
    workspace, home = tmp_path / "workspace", tmp_path / "home"
    builtin, source = tmp_path / "builtin", tmp_path / "source"
    _write_v3_plugin(builtin / "caller", name="caller", module_source=CALLER)
    _write_v3_plugin(builtin / "peer", name="peer", module_source=VALUE.replace("NAME", "peer"))
    target = VALUE.replace("NAME", "target")
    _write_v3_plugin(source, name="target", module_source=target)
    _commit(source)
    initialize_plugin_workspace(workspace)
    if existing:
        install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    log = MessageLog(workspace / "sessions.db")
    host = PluginManager([builtin], event_bus=EventBus(), workspace=workspace,
                         message_log=log, installed_cache_root=home / "cache")
    try:
        await host.load_all()
        if changed:
            (source / "plugin.py").write_text(target.replace('lambda: "old"', 'lambda: "new"'))
            _commit(source)
        yield host, source, workspace, builtin
    finally:
        await host.terminate_all()
        log.close()


def update_api(snapshot):
    root = snapshot.composition_root.context
    return root.require(PLUGIN_UPDATES), root.require(ServiceKey("test.context"))


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
async def test_real_api_prepares_validates_and_discards_without_changing_stable(tmp_path, existing):
    async with installed_host(tmp_path, existing=existing) as (host, source, workspace, builtin):
        selection = workspace / "runtime" / "plugin-stable.json"
        before = selection.read_bytes()
        # 两次旧全局 reconcile 都会读到这些变更；本次安装不得执行它们。
        (builtin / "peer" / "plugin.py").write_text(
            VALUE.replace("NAME", "peer").replace('lambda: "old"', 'lambda: "changed"')
        )
        _write_v3_plugin(builtin / "extra", name="extra", module_source=VALUE.replace("NAME", "extra"))
        async with lease_runtime_snapshot(host.snapshot_store) as stable:
            api, ctx = update_api(stable)
            status = await api.install(ctx, "request", source=str(source), marketplace="lab")
            assert status.phase == "armed" and status.ready and not status.publishing
            candidate = host.latest_snapshot
            assert candidate is not stable and host.current_snapshot is stable
            assert selection.read_bytes() == before
            assert set(candidate.generations) == set(stable.generations) | {"target@lab"}
            for plugin_id, generation in stable.generations.items():
                if plugin_id != "target@lab":
                    assert candidate.generations[plugin_id].archive_ref == generation.archive_ref
            assert stable.composition_root.context.require(ServiceKey("test.peer"))() == "old"
            if existing:
                assert stable.composition_root.context.require(ServiceKey("test.target"))() == "old"
            else:
                assert "target@lab" not in stable.generations
            async with api.open_validation(ctx, "request") as scope:
                assert scope.require(ServiceKey("test.target"))() == "new"
                assert scope.require(ServiceKey("test.peer"))() == "old"
                child_api = scope.require(PLUGIN_UPDATES)
                child_ctx = scope.require(ServiceKey("test.context"))
                with pytest.raises(PermissionError, match="候选验证不能"):
                    await child_api.discard(child_ctx, "request")
                with pytest.raises(RuntimeError, match="不属于当前 runtime scope"):
                    await api.discard(ctx, "request")
            assert not host._validation_hosts
            await api.discard(ctx, "request")
            assert api.read(ctx, "request").phase == "rolled_back"
            assert host.ready_candidate is None and host.current_snapshot is stable
            assert selection.read_bytes() == before and stable.accepting_leases


@pytest.mark.asyncio
async def test_same_artifact_install_keeps_completed_receipt_without_candidate(tmp_path):
    async with installed_host(tmp_path, changed=False) as (host, source, workspace, _):
        selection = workspace / "runtime" / "plugin-stable.json"
        before = selection.read_bytes()
        async with lease_runtime_snapshot(host.snapshot_store) as stable:
            api, ctx = update_api(stable)
            status = await api.install(ctx, "same", source=str(source), marketplace="lab")
            assert status.phase == "committed" and not status.ready
            assert host.ready_candidate is None and host.current_snapshot is stable
            assert selection.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["load", "publish", "switch", "reconcile", "unload", "terminate"])
async def test_stable_self_wait_is_rejected_before_any_operation(tmp_path, action):
    async with installed_host(tmp_path) as (host, source, workspace, _):
        async with lease_runtime_snapshot(host.snapshot_store) as stable:
            api, ctx = update_api(stable)
            await api.install(ctx, "request", source=str(source), marketplace="lab")
            candidate, operation = host.ready_candidate, host._operation
            selection = (workspace / "runtime" / "plugin-stable.json").read_bytes()
            receipt = host.reload_journal.update("request")
            calls = {
                "load": host.load_all,
                "publish": lambda: host.publish_prepared("target@lab"),
                "switch": lambda: host.switch_ready("target@lab"),
                "reconcile": host.reconcile_changed,
                "unload": lambda: host.reconcile_disabled_and_drain("target@lab"),
                "terminate": host.terminate_all,
            }
            with pytest.raises(RuntimeError, match="持有本 Manager"):
                await calls[action]()
            assert host._operation is operation and not operation.revoked
            assert host.current_snapshot is stable and stable.accepting_leases
            assert host.ready_candidate is candidate and not host._stopping
            assert host.reload_journal.update("request") == receipt
            assert (workspace / "runtime" / "plugin-stable.json").read_bytes() == selection
            await api.discard(ctx, "request")


@pytest.mark.asyncio
async def test_candidate_lease_cannot_close_itself_and_unknown_callers_cannot_install(tmp_path):
    async with installed_host(tmp_path) as (host, source, _, _):
        async with lease_runtime_snapshot(host.snapshot_store) as stable:
            api, ctx = update_api(stable)
            operation = host._operation
            foreign = CompositionRoot("foreign")
            try:
                with pytest.raises(PermissionError, match="Context 不属于"):
                    await api.install(foreign.context, "foreign", source=str(source), marketplace="lab")
                with pytest.raises(RuntimeError, match="实际 runtime scope"):
                    await asyncio.create_task(api.install(ctx, "inherited", source=str(source), marketplace="lab"))
                assert host._operation is operation and host.ready_candidate is None
            finally:
                await foreign.dispose()
            await api.install(ctx, "request", source=str(source), marketplace="lab")
            candidate, operation = host.latest_snapshot, host._operation
            async with RuntimeScope(host.snapshot_store.lease(candidate.snapshot_id)):
                for call in (
                    lambda: host.discard_update("request"),
                    lambda: host.drop_candidate("target@lab"),
                    lambda: host.discard_prepared("target@lab"),
                    lambda: host.retry_validation_cleanup("unknown"),
                    host.terminate_all,
                ):
                    with pytest.raises(RuntimeError, match="持有本 Manager"):
                        await call()
                    assert host._operation is operation and not operation.revoked
                    assert host.latest_snapshot is candidate and candidate.accepting_leases
                    assert host.read_update("request").phase == "armed"
            await api.discard(ctx, "request")


@pytest.mark.asyncio
async def test_cancelled_api_caller_leaves_operation_lease_until_work_exits(tmp_path, monkeypatch):
    async with installed_host(tmp_path) as (host, source, _, _):
        entered, release = asyncio.Event(), asyncio.Event()
        leases, commits = [], []

        async def blocked_install(**kwargs):
            """模拟已开始且必须收尾的操作；撤销后仍不能提交。"""
            leases.append(get_current_runtime_lease())
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            host._check_operation_commit()
            commits.append("unauthorized")

        monkeypatch.setattr(host, "_install_candidate", blocked_install)

        async def caller():
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                api, ctx = update_api(snapshot)
                await api.install(ctx, "request", source=str(source), marketplace="lab")

        task = asyncio.create_task(caller())
        try:
            await entered.wait()
            operation = host._operation
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert operation.revoked and not operation.task.done()
            assert leases[0].active and leases[0].snapshot is host.current_snapshot
            assert host.current_snapshot.lease_count == 1
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.gather(host._operation.task, return_exceptions=True)
        assert not leases[0].active and host.current_snapshot.lease_count == 0
        assert not commits and host.ready_candidate is None
        with pytest.raises(KeyError):
            host.read_update("request")
