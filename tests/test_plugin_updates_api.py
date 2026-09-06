"""普通插件使用公开更新能力；候选不能把验证变成另一次安装。"""
import asyncio

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from session.log import MessageLog
from tests.test_plugin_install import _commit, _write_v3_plugin


MODULE = '''
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
api_version = 3
name = "probe"
version = "1.0.0"
inject = (PLUGIN_UPDATES,)
async def apply(ctx, config):
    await ctx.provide(ServiceKey("test.context"), ctx)
    await ctx.provide(ServiceKey("test.version"), lambda: "old")
'''


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["publish", "discard"])
async def test_ordinary_update_api_checks_scope_and_isolates_validation(tmp_path, finish):
    source, workspace, home = (tmp_path / name for name in ("source", "workspace", "home"))
    _write_v3_plugin(source, name="probe", module_source=MODULE)
    _commit(source)
    install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    log = MessageLog(workspace / "sessions.db")
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, message_log=log,
                         installed_cache_root=home / "cache")
    stream = None
    notification = None
    try:
        await host.load_all()
        (source / "plugin.py").write_text(MODULE.replace('lambda: "old"', 'lambda: "new"'))
        _commit(source)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            api = root.require(PLUGIN_UPDATES)
            ctx = root.require(ServiceKey("test.context"))
            assert api.read(ctx, "request") is None
            pointers = next(home.rglob(".pointers.json"))
            before = pointers.read_bytes()
            for invalid in (None, False, 0, "", " padded "):
                with pytest.raises(ValueError, match="更新 ID"):
                    await api.install(ctx, invalid, source=str(source), marketplace="lab")
                assert host.ready_candidate is None and pointers.read_bytes() == before
            status = await api.install(ctx, "request", source=str(source), marketplace="lab")
            assert status.phase == "armed" and not status.publishing
            with pytest.raises(RuntimeError, match="不能重跑安装"):
                await api.install(ctx, "request", source=str(source), marketplace="lab")
            async with api.open_validation(ctx, "request") as scope:
                assert scope.require(ServiceKey("test.version"))() == "new"
                child_api = scope.require(PLUGIN_UPDATES)
                child_ctx = scope.require(ServiceKey("test.context"))
                with pytest.raises(PermissionError, match="候选验证不能"):
                    child_api.publish(child_ctx, "request")
                with pytest.raises(RuntimeError, match="不属于当前 runtime scope"):
                    api.read(ctx, "request")
            if finish == "discard":
                await api.discard(ctx, "request")
                assert api.read(ctx, "request").phase == "rolled_back"
            else:
                stream = api.changes(ctx)
                await anext(stream)
                notification = asyncio.create_task(anext(stream))
                api.publish(ctx, "request")
                await asyncio.wait_for(notification, 10)
                assert api.read(ctx, "request").publishing
        if finish == "publish":
            await asyncio.wait_for(host._update_publication[1], 10)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            api = root.require(PLUGIN_UPDATES)
            ctx = root.require(ServiceKey("test.context"))
            status = api.read(ctx, "request")
            assert status.phase == ("committed" if finish == "publish" else "rolled_back")
            assert root.require(ServiceKey("test.version"))() == ("new" if finish == "publish" else "old")
            if finish == "publish":
                await stream.aclose()
                stream = api.changes(ctx)
                await anext(stream)
                notification = asyncio.create_task(anext(stream))
                unchanged = await api.install(ctx, "same-artifact", source=str(source), marketplace="lab")
                assert unchanged.phase == "committed" and not unchanged.ready
                await asyncio.wait_for(notification, 10)
        with pytest.raises(RuntimeError, match="实际 runtime scope"):
            api.read(ctx, "request")
    finally:
        if notification is not None and not notification.done():
            notification.cancel()
            await asyncio.gather(notification, return_exceptions=True)
        if stream is not None:
            await stream.aclose()
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_queued_publication_cannot_publish_a_replacement_candidate(tmp_path, monkeypatch):
    """发布真正取得候选锁时再核对原收据，不能只按同一插件 ID 切换。"""
    from tests.test_plugin_update_rollback import prepare

    source, home, workspace, _ = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    entered, release = asyncio.Event(), asyncio.Event()
    original_publish = host._publish_update
    async def publish(update_id, plugin_id):
        entered.set()
        await release.wait()
        await original_publish(update_id, plugin_id)
    monkeypatch.setattr(host, "_publish_update", publish)
    try:
        await host.load_all()
        first, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        host.start_update_publication(first.update_id)
        await asyncio.wait_for(entered.wait(), 10)
        await host.discard_update(first.update_id)
        second, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        release.set()
        await asyncio.wait_for(host._update_publication[1], 10)
        assert host.reload_journal.update(first.update_id).phase == "rolled_back"
        assert "不匹配" in host.reload_journal.update(first.update_id).error
        assert host.reload_journal.update(second.update_id).phase == "armed"
        assert host.current_snapshot.composition_root.context.require(ServiceKey("version.probe"))() == "old"
        assert host.ready_candidate.reload_tx_id == host.reload_journal.update(second.update_id).reload_tx_id
    finally:
        release.set()
        await host.terminate_all()
