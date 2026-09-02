import asyncio
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from plugins.eventmail.store import EventMailStore
from plugins.drift.store import DriftStore


async def _eventually(predicate) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not settle")


@pytest.mark.asyncio
async def test_fixture_submits_through_two_narrow_services_only_on_formal_start(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    plugin_root = tmp_path / "plugins"
    content_dir = plugin_root / "content"
    drift_dir = plugin_root / "drift"
    fixture_dir = plugin_root / "wake_drift_gate"
    shutil.copytree(root / "plugins" / "eventmail", content_dir)
    shutil.copytree(root / "plugins" / "drift", drift_dir)
    shutil.copytree(root / "tests" / "fixtures" / "wake_drift_gate", fixture_dir)
    workspace = tmp_path / "workspace"
    manager = PluginManager(
        plugin_dirs=[content_dir, drift_dir, fixture_dir],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    content = EventMailStore(
        workspace / "plugin-data" / "eventmail-builtin" / "eventmail.sqlite3"
    )
    drift = DriftStore(
        workspace / "plugin-data" / "drift-builtin" / "drift.sqlite3"
    )
    assert content.state_counts() == {}
    assert drift.snapshot(datetime.now(UTC))["proposals"] == ()

    lifecycle = asyncio.create_task(manager.run_runtime_services())
    try:
        await _eventually(lambda: content.state_counts() == {"pending": 1})
        assert len(drift.snapshot(datetime.now(UTC))["proposals"]) == 1

        with (fixture_dir / "plugin.py").open("a", encoding="utf-8") as handle:
            handle.write("\n# candidate fixture revision\n")
        candidate = await manager.prepare_candidate("wake_drift_gate")
        assert candidate is not None
        assert content.state_counts() == {"pending": 1}
        assert len(drift.snapshot(datetime.now(UTC))["proposals"]) == 1
        await manager.discard_prepared("wake_drift_gate")
    finally:
        lifecycle.cancel()
        _ = await asyncio.gather(lifecycle, return_exceptions=True)
        await manager.terminate_all()


def test_fixture_declares_structural_services_without_importing_domain_plugins() -> None:
    source = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "wake_drift_gate"
        / "plugin.py"
    ).read_text(encoding="utf-8")

    assert "from plugins.eventmail" not in source
    assert "from plugins.drift" not in source
    assert 'ServiceKey[ContentSourceServices]("eventmail.content_source.v1")' in source
    assert 'ServiceKey[DriftProposalServices]("drift.proposals.v1")' in source
    assert "SCOPED_TURNS" not in source
    assert "TIMERS" not in source
