"""完整选择的真实 boot/publication 边界；本层只编写，未执行。"""

import asyncio
from pathlib import Path

import pytest

import agent.plugins.manager as manager_module
import agent.plugins.selection as selection_module
from agent.plugin_composition.config_input import save_config
from agent.plugins.artifacts import ArtifactPointer, write_pointers
from agent.plugins.manager import PluginManager
from agent.plugins.manifest import write_plugin_manifest
from agent.plugins.selection import PluginSelection, SelectionConflictError, SelectionFormatError, SelectionWriteError
from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore
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


async def replace_root(owner, expected_ref):
    """经实际操作接纳入口替换全组，不绕过取消与截止许可。"""
    return await owner._run_operation(
        lambda: owner._replace_formal_root((), expected_ref=expected_ref),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [None, "{}", "broken"])
async def test_boot_requires_explicit_valid_selection(tmp_path, monkeypatch, raw):
    owner = manager(tmp_path)
    if raw is not None:
        owner._selection.path.write_text(raw)
    monkeypatch.setattr(owner, "discover", forbidden)
    with pytest.raises(SelectionFormatError):
        await owner.load_all()
    assert owner.current_snapshot is None
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
    before = first.current_snapshot
    expected = {key: (item.archive_ref, dict(item.config_projection)) for key, item in before.generations.items()}
    assert len(expected) == 2
    assert "alpha@lab" in expected
    for item in before.generations.values():
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
        restored = second.current_snapshot
        assert {key: (item.archive_ref, dict(item.config_projection)) for key, item in restored.generations.items()} == expected
        assert restored.composition_root is not before.composition_root
        assert all(item.plugin_dir == item.code_dir for item in restored.generations.values())
        assert selection.read() == ref
        assert second.ready_candidate is None
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
    assert not second.current_snapshot.generations
    assert selection.read() == ref
    await second.terminate_all()


@pytest.mark.asyncio
async def test_initializer_failure_does_not_commit_and_rebuilds_exact_old_root(tmp_path, monkeypatch):
    plugin(tmp_path, "alpha")
    selection = initialize(tmp_path)
    owner = manager(tmp_path)
    await owner.load_all()
    ref, old = selection.read(), owner.current_snapshot
    start = owner._start_closed_runtime_snapshot
    calls = 0

    async def fail_new(lease):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("closed init failed")
        await start(lease)

    monkeypatch.setattr(owner, "_start_closed_runtime_snapshot", fail_new)
    with pytest.raises(RuntimeError, match="closed init failed"):
        await replace_root(owner, ref)
    assert calls == 2
    assert selection.read() == ref
    assert owner.current_snapshot is not old
    assert set(owner.current_snapshot.generations) == set(old.generations)
    assert owner.current_snapshot.accepting_leases
    await owner.terminate_all()


@pytest.mark.asyncio
async def test_post_commit_failure_keeps_new_closed_owner(tmp_path, monkeypatch):
    plugin(tmp_path, "alpha")
    selection = initialize(tmp_path)
    owner = manager(tmp_path)
    await owner.load_all()
    previous = selection.read()

    def fail_after_commit(snapshot, old):
        raise RuntimeError("publication callback failed")

    monkeypatch.setattr(owner, "_activate_snapshot", fail_after_commit)
    with pytest.raises(RuntimeError, match="publication callback failed"):
        await replace_root(owner, previous)
    target = selection.read()
    assert target != previous
    assert selection.archive.read_descriptor(target)["previous"] == previous
    assert owner._publication.selection_result == target
    assert owner.current_snapshot is owner._publication.candidate
    assert not owner.current_snapshot.generations
    assert not owner.current_snapshot.accepting_leases
    with pytest.raises(RuntimeError, match="maintenance"):
        await replace_root(owner, target)
    await owner.terminate_all()


@pytest.mark.asyncio
async def test_uncertain_pointer_write_never_restores_old_selection(tmp_path, monkeypatch):
    plugin(tmp_path, "alpha")
    selection = initialize(tmp_path)
    owner = manager(tmp_path)
    await owner.load_all()
    previous = selection.read()

    def fail_sync(path):
        raise OSError("directory sync failed")

    with monkeypatch.context() as patch:
        patch.setattr(selection_module, "sync_directory", fail_sync)
        with pytest.raises(SelectionWriteError) as caught:
            await replace_root(owner, previous)
    error = caught.value
    assert error.outcome == "uncertain"
    assert owner._publication.selection_result is error
    assert selection.read() == error.target_ref != previous
    assert owner.current_snapshot is owner._publication.candidate
    assert not owner.current_snapshot.accepting_leases
    with pytest.raises(RuntimeError, match="耐久性未确认"):
        await owner.retry_runtime_recovery("alpha")
    await owner.terminate_all()
    restarted = manager(tmp_path)
    await restarted.load_all()
    assert not restarted.current_snapshot.generations
    assert selection.read() == error.target_ref
    await restarted.terminate_all()


@pytest.mark.asyncio
async def test_stale_aba_is_rejected_before_closing_live_root(tmp_path):
    plugin(tmp_path, "alpha")
    selection = initialize(tmp_path)
    owner = manager(tmp_path)
    await owner.load_all()
    first, root = selection.read(), owner.current_snapshot
    components = selection.archive.read_descriptor(first)["components"]
    second = selection.commit((), expected_ref=first)
    third = selection.commit(components, expected_ref=second)
    with pytest.raises(SelectionConflictError):
        await replace_root(owner, first)
    assert owner.current_snapshot is root and root.accepting_leases
    assert selection.read() == third != first
    await owner.terminate_all()


@pytest.mark.asyncio
async def test_snapshot_callback_failure_cannot_undo_durable_result():
    store = RuntimeSnapshotStore()
    old = RuntimeSnapshotCompiler().compile({}, snapshot_revision="old")
    store.install(old)
    new = RuntimeSnapshotCompiler().compile({}, snapshot_revision="new")
    transaction = store.begin_publish(new)
    await store.commit_provisional(transaction)

    def commit():
        transaction.selection_result = "confirmed-selection-ref"

    def fail():
        raise RuntimeError("after durable commit")

    with pytest.raises(RuntimeError, match="after durable commit"):
        await store.finalize_provisional(transaction, before_open=commit, after_open=fail)
    assert store.current is new and not new.accepting_leases
    with pytest.raises(RuntimeError, match="不能回滚"):
        await store.rollback_published(transaction, keep_candidate_latest=False, reopen_previous=True)
    await store.close()


@pytest.mark.asyncio
async def test_runtime_started_finishes_with_closed_exact_scope_before_commit(tmp_path, monkeypatch):
    root = plugin(tmp_path, "alpha")
    (root / "plugin.py").write_text(
        "name = 'alpha'\nversion = '1.0'\napi_version = 3\n"
        "from agent.plugin_composition import RUNTIME_STARTED\n"
        "from agent.plugins.snapshot import get_current_runtime_snapshot\n"
        "started = False\n"
        "async def apply(ctx):\n"
        "    async def start(event):\n"
        "        global started\n"
        "        async with ctx.runtime_scope():\n"
        "            assert not get_current_runtime_snapshot().accepting_leases\n"
        "            started = True\n"
        "    await ctx.on(RUNTIME_STARTED, start)\n",
    )
    initialize(tmp_path)
    owner = manager(tmp_path)
    commit = owner._selection.commit
    calls = []

    def check_ready(components, *, expected_ref):
        snapshot = owner._publication.candidate
        assert snapshot.generations["alpha"].instance.module.started
        assert not snapshot.accepting_leases
        calls.append(components)
        return commit(components, expected_ref=expected_ref)

    monkeypatch.setattr(owner._selection, "commit", check_ready)
    await owner.load_all()
    assert len(calls) == 1 and owner.current_snapshot.accepting_leases
    await owner.terminate_all()


@pytest.mark.asyncio
async def test_cancelled_publisher_cannot_commit_after_closed_initialization(tmp_path, monkeypatch):
    selection = initialize(tmp_path)
    owner = manager(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    start = owner._start_closed_runtime_snapshot

    async def wait_then_start(lease):
        entered.set()
        await release.wait()
        await start(lease)

    monkeypatch.setattr(owner, "_start_closed_runtime_snapshot", wait_then_start)
    task = asyncio.create_task(owner.load_all())
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert selection.read() is None
    assert owner.current_snapshot is None
    await owner.terminate_all()


@pytest.mark.asyncio
async def test_sync_commit_without_returned_result_is_retained_as_uncertain(tmp_path, monkeypatch):
    selection = initialize(tmp_path)
    owner = manager(tmp_path)
    commit = owner._selection.commit

    def interrupted(components, *, expected_ref):
        commit(components, expected_ref=expected_ref)
        raise asyncio.CancelledError

    monkeypatch.setattr(owner._selection, "commit", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await owner.load_all()
    assert selection.read() is not None
    assert owner._publication.selection_result.outcome == "uncertain"
    assert owner.current_snapshot is owner._publication.candidate
    assert not owner.current_snapshot.accepting_leases
    await owner.terminate_all()


@pytest.mark.asyncio
async def test_unchanged_write_failure_rebuilds_old_without_another_commit(tmp_path, monkeypatch):
    plugin(tmp_path, "alpha")
    selection = initialize(tmp_path)
    owner = manager(tmp_path)
    await owner.load_all()
    ref, old = selection.read(), owner.current_snapshot
    attempts = []
    error = SelectionWriteError(operation="commit", target_ref=None, outcome="unchanged",
                                observed_ref=ref, observation_error=None)

    def fail_write(components, *, expected_ref):
        attempts.append((components, expected_ref))
        raise error

    monkeypatch.setattr(owner._selection, "commit", fail_write)
    with pytest.raises(SelectionWriteError) as caught:
        await replace_root(owner, ref)
    assert caught.value is error and attempts == [((), ref)]
    assert selection.read() == ref
    assert owner.current_snapshot is not old and owner.current_snapshot.accepting_leases
    assert set(owner.current_snapshot.generations) == set(old.generations)
    await owner.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["committed", "committed_then_changed", "uncommitted", "later_same_components", "unknown"])
async def test_boot_settles_exact_transition_without_resuming_or_rolling_back_install(tmp_path, monkeypatch, transition):
    plugin(tmp_path, "alpha")
    selection = initialize(tmp_path)
    first = manager(tmp_path)
    await first.load_all()
    base = selection.read()
    components = selection.archive.read_descriptor(base)["components"]
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
    second = manager(tmp_path)
    monkeypatch.setattr(second, "discover", forbidden)
    monkeypatch.setattr(second._reload_journal, "rollback_updates", forbidden)
    await second.load_all()
    try:
        assert selection.read() == selected
        assert second.ready_candidate is None
        assert tuple(item.archive_ref for item in second.current_snapshot.generations.values()) == selection.archive.read_descriptor(selected)["components"]
        phase = second._reload_journal.get(tx_id).phase
        assert phase == ("recovered" if committed else "promoting" if transition == "unknown" else "aborted")
        update = second._reload_journal.update("update")
        assert update.phase == ("committed" if committed else "armed")
        if not committed:
            assert "explicit settlement" in update.error
        assert not install_base.exists()
    finally:
        await second.terminate_all()
