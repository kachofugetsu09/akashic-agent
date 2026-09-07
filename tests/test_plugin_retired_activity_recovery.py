from __future__ import annotations

from pathlib import Path

import pytest

from agent.plugins.manager import PluginManager
from agent.plugins.reload_journal import ReloadJournal
from bus.event_bus import EventBus


def _write_plugin(root: Path) -> None:
    plugin = root / "builtin" / "baseline"
    plugin.mkdir(parents=True)
    (plugin / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'baseline'\n"
        "version = '1.0.0'\n\n"
        "async def apply(ctx, config):\n"
        "    return None\n",
        encoding="utf-8",
    )


def _write_recovery_action(workspace: Path, *, resource: str) -> ReloadJournal:
    journal = ReloadJournal(workspace)
    tx_id = journal.begin(
        plugin_id="baseline",
        base_snapshot_id=None,
        generation_id="legacy-generation",
        source_revision="legacy-source",
        config_revision="legacy-config",
    )
    journal.mark_runtime_owner(tx_id, "old-boot")
    journal.advance(
        tx_id,
        "degraded",
        error="legacy owner retained",
        resource=resource,
        recovery_action="retry_runtime_recovery",
        recovery_target="candidate",
    )
    return journal


def _manager(root: Path, workspace: Path, *, with_plugin: bool) -> PluginManager:
    return PluginManager(
        plugin_dirs=[root / "builtin"] if with_plugin else [],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=root / "cache",
    )


@pytest.mark.asyncio
async def test_startup_keeps_retired_activity_recovery_pending(tmp_path: Path) -> None:
    """旧 Activity owner 记录不能在启动时被伪造为 recovered。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_plugin(tmp_path)
    journal = _write_recovery_action(
        workspace,
        resource="channel-publication, activity-publication ,channel-binding:old",
    )
    manager = _manager(tmp_path, workspace, with_plugin=True)
    try:
        with pytest.raises(RuntimeError, match="retired activity-publication owner"):
            await manager.load_all()
    finally:
        await manager.terminate_all()

    record = journal.latest(plugin_id="baseline")
    assert record is not None
    assert record.phase == "degraded"
    assert record.failure_resource == (
        "channel-publication, activity-publication ,channel-binding:old"
    )


@pytest.mark.asyncio
async def test_manual_retry_keeps_retired_activity_recovery_pending(tmp_path: Path) -> None:
    """手工 retry 不能调用已删除 owner，也不能完成 journal。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journal = _write_recovery_action(
        workspace,
        resource="channel-publication,activity-publication",
    )
    manager = _manager(tmp_path, workspace, with_plugin=False)
    try:
        with pytest.raises(RuntimeError, match="retired activity-publication owner"):
            await manager.retry_runtime_recovery("baseline")
    finally:
        await manager.terminate_all()

    record = journal.latest(plugin_id="baseline")
    assert record is not None
    assert record.phase == "degraded"


@pytest.mark.asyncio
async def test_startup_still_finishes_non_activity_runtime_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未涉及旧 Activity owner 的正常 runtime recovery 仍可完成。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_plugin(tmp_path)
    journal = _write_recovery_action(workspace, resource="channel-publication")
    monkeypatch.setenv("AKASHIC_SUPERVISED", "1")
    monkeypatch.setenv("AKASHIC_BOOT_ID", "new-boot")
    import agent.background.boot_guardian as guardian

    monkeypatch.setattr(guardian, "_cleanup_boot_processes", lambda **_: None)
    manager = _manager(tmp_path, workspace, with_plugin=True)
    try:
        await manager.load_all()
    finally:
        await manager.terminate_all()

    record = journal.latest(plugin_id="baseline")
    assert record is not None
    assert record.phase == "recovered"
