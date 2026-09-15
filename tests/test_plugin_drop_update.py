"""通用丢弃入口必须结算实际安装更新，保留正式选择和业务数据。"""

import pytest

from agent.plugins.artifacts import read_pointers
from agent.plugins.manager import PluginManager
from agent.plugins.manifest import load_plugin_manifest
from agent.plugins.selection import PluginSelection
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
        await host.discard_update(replacement.update_id)
        assert host.reload_journal.update(replacement.update_id).phase == "rolled_back"
    finally:
        await host.terminate_all()
