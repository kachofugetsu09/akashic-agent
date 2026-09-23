"""完整选择的真实 boot 与 selection 结算边界。"""

from pathlib import Path

import pytest

import agent.plugins.manager as manager_module
from agent.plugin_composition.config_input import save_config
from agent.plugins.artifacts import ArtifactPointer, write_pointers
from agent.plugins.manager import PluginManager
from agent.plugins.manifest import write_plugin_manifest
from agent.plugins.selection import PluginSelection, SelectionFormatError
from bus.event_bus import EventBus


def plugin(tmp_path: Path, name: str) -> Path:
    root = tmp_path / "plugins" / name
    root.mkdir(parents=True)
    (root / "plugin.py").write_text(
        f"name = {name!r}\nversion = '1.0'\napi_version = 3\n"
        "async def apply(ctx):\n    return None\n",
    )
    return root


def manager(tmp_path: Path) -> PluginManager:
    return PluginManager(
        [tmp_path / "plugins"], event_bus=EventBus(), workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


def initialize(tmp_path: Path) -> PluginSelection:
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    selection = PluginSelection(tmp_path / "workspace")
    selection.initialize()
    return selection


def forbidden(*args, **kwargs):
    raise AssertionError("archived boot must not read mutable installation inputs")


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [None, "{}", "broken"])
async def test_boot_requires_explicit_valid_selection(tmp_path, monkeypatch, raw):
    owner = manager(tmp_path)
    if raw is not None:
        owner._selection.path.write_text(raw)
    monkeypatch.setattr(owner, "discover", forbidden)
    with pytest.raises(SelectionFormatError):
        await owner.load_all()
    assert owner.live_root is None
    await owner.terminate_all()


@pytest.mark.asyncio
async def test_boot_restores_complete_archives_and_ignores_live_inputs(tmp_path, monkeypatch):
    roots = [plugin(tmp_path, name) for name in ("alpha", "beta")]
    install_base = tmp_path / "home" / "cache" / "lab" / "alpha"
    artifact = install_base / ".artifacts" / "v1"
    artifact.parent.mkdir(parents=True)
    roots[0].rename(artifact)
    roots[0] = artifact
    pointer = ArtifactPointer(".artifacts/v1")
    write_pointers(install_base, stable=pointer, latest=pointer)
    selection = initialize(tmp_path)
    first = manager(tmp_path)
    await first.load_all()
    ref = selection.read()
    assert ref is not None
    before = first.live_root
    expected = {key: (item.archive_ref, dict(item.config_projection)) for key, item in first._active_generations.items()}
    assert len(expected) == 2
    assert "alpha@lab" in expected
    for item in first._active_generations.values():
        save_config(item.data_dir, {"changed": True})
    await first.terminate_all()
    for root in roots:
        (root / "plugin.py").write_text("raise AssertionError('live source must not import')\n")
    write_pointers(install_base, stable=ArtifactPointer(None), latest=ArtifactPointer(None))
    write_plugin_manifest({"alpha@lab": False, "beta": False}, plugins_home=tmp_path / "home")
    second = manager(tmp_path)
    monkeypatch.setattr(second, "discover", forbidden)
    monkeypatch.setattr(manager_module, "load_plugin_manifest", forbidden)
    monkeypatch.setattr(manager_module, "load_config", forbidden)
    await second.load_all()
    try:
        restored = second.live_root
        assert {key: (item.archive_ref, dict(item.config_projection)) for key, item in second._active_generations.items()} == expected
        assert restored is not before
        assert all(item.plugin_dir == item.code_dir for item in second._active_generations.values())
        assert selection.read() == ref
        assert restored is second.live_root
    finally:
        await second.terminate_all()


@pytest.mark.asyncio
async def test_explicit_null_commits_empty_once_and_empty_boot_does_not_discover(tmp_path, monkeypatch):
    selection = initialize(tmp_path)
    assert selection.read() is None
    first = manager(tmp_path)
    await first.load_all()
    ref = selection.read()
    assert ref is not None
    assert selection.archive.read_descriptor(ref)["components"] == ()
    await first.terminate_all()
    plugin(tmp_path, "unselected")
    second = manager(tmp_path)
    monkeypatch.setattr(second, "discover", forbidden)
    await second.load_all()
    assert not second._active_generations
    assert selection.read() == ref
    await second.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["committed", "committed_then_changed", "uncommitted", "later_same_components", "unknown"])
async def test_boot_settles_exact_transition_without_resuming_or_rolling_back_install(tmp_path, monkeypatch, transition):
    plugin(tmp_path, "alpha")
    selection = initialize(tmp_path)
    first = manager(tmp_path)
    await first.load_all()
    base = selection.read()
    assert base is not None
    components = selection.archive.read_descriptor(base)["components"]
    assert isinstance(components, tuple)
    await first.terminate_all()
    journal = first._reload_journal
    install_base = tmp_path / "installation-not-opened"
    journal.arm_update(update_id="update", plugin_id="alpha", plugin_base=install_base,
                       previous=None, candidate=ArtifactPointer(".artifacts/new"), previous_enabled=None)
    tx_id = journal.begin(
        plugin_id="alpha", base_snapshot_id="old", generation_id="candidate",
        source_revision="same-revision-is-not-evidence", config_revision="config",
        candidate_artifact_pointer=".artifacts/new", details={"base_selection_ref": base},
    )
    if transition != "unknown":
        journal.annotate(tx_id, {"event": "selection_candidate", "components": []})
    for phase in ("prepared", "validating", "commit_started", "latest_ready", "promoting"):
        journal.advance(tx_id, phase)
    committed = transition in {"committed", "committed_then_changed"}
    if committed:
        next_ref = selection.commit((), expected_ref=base)
        if transition == "committed_then_changed":
            selection.commit(components, expected_ref=next_ref)
    elif transition == "later_same_components":
        later_base = selection.commit(components, expected_ref=base)
        selection.commit((), expected_ref=later_base)
    selected = selection.read()
    assert selected is not None
    second = manager(tmp_path)
    monkeypatch.setattr(second, "discover", forbidden)
    monkeypatch.setattr(second._reload_journal, "rollback_updates", forbidden)
    await second.load_all()
    try:
        assert selection.read() == selected
        assert second.live_root is not None
        assert tuple(item.archive_ref for item in second._active_generations.values()) == selection.archive.read_descriptor(selected)["components"]
        phase = second._reload_journal.get(tx_id).phase
        assert phase == ("recovered" if committed else "promoting" if transition == "unknown" else "aborted")
        update = second._reload_journal.update("update")
        assert update.phase == ("committed" if committed else "armed")
        if not committed:
            assert "explicit settlement" in update.error
        assert not install_base.exists()
    finally:
        await second.terminate_all()
