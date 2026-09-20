"""通用丢弃入口必须结算实际安装更新，保留正式选择和业务数据。"""

import asyncio

import pytest

from agent.plugins.artifacts import read_pointers
from agent.plugins.manager import PluginManager
from agent.plugins.manifest import load_plugin_manifest
from agent.plugins.selection import PluginSelection
from agent.plugins import update_rollback
from bus.event_bus import EventBus
from tests.test_plugin_update_rollback import prepare


@pytest.mark.asyncio
async def test_drop_installed_candidate_settles_update_and_allows_next_install(tmp_path):
    """真实安装后按插件 ID 丢弃，不能留下孤立 armed 更新或 latest 指针。"""
    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager(
        [], event_bus=EventBus(), workspace=workspace,
        installed_cache_root=home / "cache",
    )
    try:
        await host.load_all()
        stable = host.current_snapshot
        selection = PluginSelection(workspace).read()
        pointers = read_pointers(old.installed_path.parents[1])
        manifest = load_plugin_manifest(home)
        result, _ = await host.install_candidate(
            source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
        )
        assert host.reload_journal.update(result.update_id).phase == "armed"
        assert host.ready_candidate is not None
        assert read_pointers(old.installed_path.parents[1]) != pointers

        discarded = await host.drop_candidate("probe@lab")

        assert discarded["publication_state"] == "discarded"
        assert host.reload_journal.update(result.update_id).phase == "rolled_back"
        assert host.ready_candidate is None
        assert host.current_snapshot is stable
        assert PluginSelection(workspace).read() == selection
        assert read_pointers(old.installed_path.parents[1]) == pointers
        assert load_plugin_manifest(home) == manifest
        assert (old.data_path / "history.txt").read_text() == "existing durable data"
        assert result.installed_path.exists()

        replacement, _ = await host.install_candidate(
            source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
        )
        assert replacement.update_id != result.update_id
        assert host.reload_journal.update(replacement.update_id).phase == "armed"
        replacement_ready = host.ready_candidate
        await host.discard_update(result.update_id)
        assert host.ready_candidate is replacement_ready
        assert host.reload_journal.update(replacement.update_id).phase == "armed"
        await host.discard_update(replacement.update_id)
        assert host.reload_journal.update(replacement.update_id).phase == "rolled_back"
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", ["closing_candidate", "before_formal_root"])
async def test_revert_during_publication_settles_closed_candidate(tmp_path, monkeypatch, cut):
    """发布与撤销交错时，两条实际关闭后的 abort 路径都能完成安装回退。"""
    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    entered, release, revoked = asyncio.Event(), asyncio.Event(), asyncio.Event()
    discard = None
    try:
        await host.load_all()
        selection = PluginSelection(workspace).read()
        pointers = read_pointers(old.installed_path.parents[1])
        result, _ = await host.install_candidate(
            source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
        )
        candidate = host.latest_snapshot
        if cut == "closing_candidate":
            original_close = host.snapshot_store.discard_latest

            async def close_then_wait(*args, **kwargs):
                closed = await original_close(*args, **kwargs)
                entered.set()
                await release.wait()
                return closed

            monkeypatch.setattr(host.snapshot_store, "discard_latest", close_then_wait)
        else:
            original_replace = host._replace_formal_root

            async def wait_before_formal(*args, **kwargs):
                entered.set()
                await release.wait()
                return await original_replace(*args, **kwargs)

            monkeypatch.setattr(host, "_replace_formal_root", wait_before_formal)
        original_error = host.reload_journal.record_update_error

        def record_revoke(update_id, error):
            original_error(update_id, error)
            if update_id == result.update_id and error == "test revoke":
                revoked.set()

        monkeypatch.setattr(host.reload_journal, "record_update_error", record_revoke)
        host.start_update_publication(result.update_id)
        await entered.wait()
        assert candidate.snapshot_id not in host.snapshot_store.retained_snapshot_ids
        discard = asyncio.create_task(host.discard_update(result.update_id, reason="test revoke"))
        await revoked.wait()
        if cut == "closing_candidate":
            release.set()
        await discard
        assert host.ready_candidate is None
        update = host.reload_journal.update(result.update_id)
        assert update.phase == "rolled_back"
        assert host.reload_journal.get(update.reload_tx_id).phase == "aborted"
        assert host.reload_journal.events(update.reload_tx_id)[-1].details["cleanup_receipt"] == "candidate-root-closed"
        assert PluginSelection(workspace).read() == selection
        assert read_pointers(old.installed_path.parents[1]) == pointers
        await host.discard_update(result.update_id)
        assert host.reload_journal.update(result.update_id).phase == "rolled_back"
    finally:
        release.set()
        if discard is not None and not discard.done():
            discard.cancel()
            with pytest.raises(asyncio.CancelledError):
                await discard
        await host.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", ["pointer_before", "pointer_after", "manifest_before", "manifest_after", "cancel_after_close"])
async def test_discard_retries_file_settlement_after_candidate_closed(tmp_path, monkeypatch, cut):
    """实际关闭后取消或文件写入失败，精确重试不再次关闭候选。"""
    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    try:
        await host.load_all()
        stable = host.current_snapshot
        selection = PluginSelection(workspace).read()
        pointers = read_pointers(old.installed_path.parents[1])
        manifest = load_plugin_manifest(home)
        result, _ = await host.install_candidate(
            source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
        )
        candidate = host.latest_snapshot
        with monkeypatch.context() as patch:
            if cut == "cancel_after_close":
                original_drop = host._drop_ready

                async def cancel_after_close(*args, **kwargs):
                    status = await original_drop(*args, **kwargs)
                    asyncio.current_task().cancel()
                    return status

                patch.setattr(host, "_drop_ready", cancel_after_close)
            else:
                name = "write_pointers" if cut.startswith("pointer") else "write_plugin_manifest"
                original_write = getattr(update_rollback, name)

                def fail_write(*args, **kwargs):
                    if cut.endswith("after"):
                        original_write(*args, **kwargs)
                    raise OSError("injected settlement failure")

                patch.setattr(update_rollback, name, fail_write)
            with pytest.raises(asyncio.CancelledError if cut == "cancel_after_close" else OSError):
                await host.discard_update(result.update_id)

        update = host.reload_journal.update(result.update_id)
        assert update.phase == "armed"
        assert host.ready_candidate is None
        assert candidate.snapshot_id not in host.snapshot_store.retained_snapshot_ids
        assert host.reload_journal.get(update.reload_tx_id).phase == "aborted"
        assert host.reload_journal.events(update.reload_tx_id)[-1].details["cleanup_receipt"] == "candidate-root-closed"

        async def forbid_second_close(*args, **kwargs):
            raise AssertionError("closed candidate must not be closed again")

        monkeypatch.setattr(host, "_drop_ready", forbid_second_close)
        await host.discard_update(result.update_id)
        assert host.reload_journal.update(result.update_id).phase == "rolled_back"
        assert host.current_snapshot is stable
        assert PluginSelection(workspace).read() == selection
        assert read_pointers(old.installed_path.parents[1]) == pointers
        assert load_plugin_manifest(home) == manifest
        assert (old.data_path / "history.txt").read_text() == "existing durable data"
        assert result.installed_path.exists()
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_discard_keeps_failed_cleanup_owner_until_real_close(tmp_path, monkeypatch):
    """关闭失败不能拿 aborted 回执跳过仍由 snapshot 持有的资源。"""
    source, home, workspace, _ = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    try:
        await host.load_all()
        result, _ = await host.install_candidate(
            source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
        )
        candidate = host.latest_snapshot
        original_drain = host.snapshot_store._on_drained

        async def fail_candidate_close(snapshot):
            if snapshot is candidate:
                raise OSError("injected resource close failure")
            await original_drain(snapshot)

        with monkeypatch.context() as patch:
            patch.setattr(host.snapshot_store, "_on_drained", fail_candidate_close)
            with pytest.raises(RuntimeError, match="drain"):
                await host.discard_update(result.update_id)
        update = host.reload_journal.update(result.update_id)
        assert update.phase == "armed"
        assert host.ready_candidate is not None
        assert candidate.snapshot_id in host.snapshot_store.retained_snapshot_ids
        assert host.reload_journal.get(update.reload_tx_id).phase == "discarding"
        assert "cleanup_receipt" not in host.reload_journal.events(update.reload_tx_id)[-1].details
        await host.discard_update(result.update_id)
        assert host.reload_journal.update(result.update_id).phase == "rolled_back"
        assert candidate.snapshot_id not in host.snapshot_store.retained_snapshot_ids
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("external_change", ["pointers", "enabled", "selection"])
async def test_closed_candidate_settlement_rejects_external_changes(tmp_path, monkeypatch, external_change):
    """有关闭回执也不能覆盖外部改写或已改变的正式选择。"""
    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    try:
        await host.load_all()
        result, _ = await host.install_candidate(
            source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
        )

        def fail_manifest(*args, **kwargs):
            raise OSError("injected manifest failure")

        with monkeypatch.context() as patch:
            patch.setattr(update_rollback, "write_plugin_manifest", fail_manifest)
            with pytest.raises(OSError):
                await host.discard_update(result.update_id)
        if external_change == "pointers":
            path = old.installed_path.parents[1] / ".pointers.json"
            path.write_text('{"stable": null, "latest": null}')
        elif external_change == "enabled":
            from agent.plugins.manifest import set_plugin_enabled
            set_plugin_enabled("probe@lab", enabled=False, plugins_home=home)
            path = home / "manifest.toml"
        else:
            selection = PluginSelection(workspace)
            selection.commit((), expected_ref=selection.read())
            path = workspace / "runtime" / "plugin-stable.json"
        changed = path.read_bytes()
        with pytest.raises(RuntimeError, match="其他操作改变|stable 选择已改变"):
            await host.discard_update(result.update_id)
        assert path.read_bytes() == changed
        assert host.reload_journal.update(result.update_id).phase == "armed"
    finally:
        await host.terminate_all()
