from __future__ import annotations

# pyright: reportPrivateUsage=false

import asyncio
import importlib.util
import json
import shutil
import sqlite3
import sys
import tomllib
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest
import agent.plugins.service_hold as service_hold_module

from agent.plugin_composition import CompositionError, CompositionRoot, PluginRuntime
from agent.plugin_composition.model import ServiceKey
from agent.plugins.artifact_pins import _decode_artifact
from agent.plugins.generation import GateResult, PluginContributions, PluginGeneration
from agent.plugins.reload_journal import ReloadJournal
from agent.plugins.scope import PluginScope
from agent.plugins.service_hold import (
    HoldId,
    _HoldRun,
    _PluginRef,
    _RootRef,
    _hold_key,
)
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    get_current_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.plugins.source_hash import file_revision, source_revision


def _write_plugin(plugin_dir: Path, data_dir: Path, value: str) -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(
        "from agent.plugin_composition import ServiceKey\n"
        "VALUE = ServiceKey('test.value')\n"
        "async def apply(ctx, config):\n"
        "    await ctx.provide(VALUE, config['value'])\n",
        encoding="utf-8",
    )
    (data_dir / "config.local.toml").write_text(
        f'value = "{value}"\n',
        encoding="utf-8",
    )


def _load_module(path: Path) -> ModuleType:
    name = f"service_hold_fixture_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"fixture module 无法导入: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


async def _build_snapshot(
    plugin_dir: Path,
    data_dir: Path,
    *,
    generation_id: str,
    config_path: Path | None = None,
    source_type: str = "builtin",
    source_value: str | None = None,
    config_value: str | None = None,
) -> tuple[RuntimeSnapshot, CompositionRoot]:
    config_path = config_path or data_dir / "config.local.toml"
    module = _load_module(plugin_dir / "plugin.py")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    root = CompositionRoot("service-hold-root")
    runtime = PluginRuntime(
        plugin_id="fixture",
        generation_id=generation_id,
        plugin_dir=plugin_dir,
        data_dir=data_dir,
        workspace=data_dir.parent.parent,
        config=config,
    )

    async def apply(ctx) -> None:
        await module.apply(ctx, config)

    _ = await root.mount(apply, name="fixture", runtime=runtime)
    generation = PluginGeneration(
        plugin_id="fixture",
        generation_id=generation_id,
        module_path=module.__name__,
        source_revision=source_value or source_revision(plugin_dir),
        config_revision=config_value or file_revision(config_path),
        plugin_dir=plugin_dir,
        data_dir=data_dir,
        config=config,
        instance=module,
        scope=PluginScope("fixture", generation_id=generation_id),
        contributions=PluginContributions(manifest={}),
        gate_result=GateResult(
            gate_id="fixture",
            plugin_id="fixture",
            candidate_revision=source_revision(plugin_dir),
            status="passed",
            checks=(),
        ),
        config_path=config_path,
        source_type=source_type,  # type: ignore[arg-type]
    )
    snapshot = RuntimeSnapshotCompiler().compile(
        {"fixture": generation},
        composition_root=root,
    )
    return snapshot, root


class _FreshLoader:
    """Fresh-import the complete held Root from its real disk pins."""

    def __init__(self) -> None:
        self.calls = 0
        self.roots: list[CompositionRoot] = []

    async def load(self, ref: _RootRef) -> RuntimeSnapshot:
        self.calls += 1
        if len(ref.plugins) != 1:
            raise RuntimeError("fixture 只支持一只 plugin")
        item = ref.plugins[0]
        artifact = _decode_artifact(item.artifact)
        snapshot, root = await _build_snapshot(
            Path(artifact.path),
            Path(artifact.data_path),
            generation_id=item.generation,
            config_path=Path(artifact.config_path),
            source_type=artifact.source_type,
            source_value=artifact.source_revision,
            config_value=artifact.config_revision,
        )
        self.roots.append(root)
        return snapshot


@dataclass
class _Fixture:
    workspace: Path
    plugin_dir: Path
    data_dir: Path
    key: ServiceKey[str]
    root: CompositionRoot
    snapshot: RuntimeSnapshot
    store: RuntimeSnapshotStore
    journal: ReloadJournal
    run: _HoldRun


async def _fixture(tmp_path: Path) -> _Fixture:
    workspace = tmp_path / "workspace"
    plugin_dir = tmp_path / "plugins" / "fixture"
    data_dir = workspace / "plugin-data" / "fixture"
    _write_plugin(plugin_dir, data_dir, "one")
    snapshot, root = await _build_snapshot(
        plugin_dir,
        data_dir,
        generation_id="fixture-v1",
    )

    async def dispose(item: RuntimeSnapshot) -> None:
        if item.composition_root is not None:
            await item.composition_root.dispose()

    store = RuntimeSnapshotStore(dispose)
    store.install(snapshot)
    journal = ReloadJournal(workspace)
    return _Fixture(
        workspace=workspace,
        plugin_dir=plugin_dir,
        data_dir=data_dir,
        key=ServiceKey("test.value"),
        root=root,
        snapshot=snapshot,
        store=store,
        journal=journal,
        run=_HoldRun(store, journal),
    )


async def _reserve(fixture: _Fixture, holder: object) -> HoldId:
    lease = fixture.store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        return await holder.reserve()  # type: ignore[union-attr]
    finally:
        reset_runtime_snapshot(token)
        await lease.release()


@pytest.mark.asyncio
async def test_service_hold_flow_is_fixed_and_isolated(tmp_path: Path) -> None:
    assert service_hold_module.__all__ == ["HoldId", "ServiceHold"]
    fixture = await _fixture(tmp_path)
    first = fixture.run.bind(fixture.key, _hold_key("source:first"))
    second = fixture.run.bind(fixture.key, _hold_key("source:second"))

    with pytest.raises(CompositionError) as caught:
        await first.reserve()
    assert caught.value.code == "HOLD_SCOPE"

    hold_id = await _reserve(fixture, first)
    second_id = await _reserve(fixture, second)
    assert hold_id != second_id
    assert fixture.snapshot.hold_count == 2
    assert await first.pending() == (hold_id,)
    assert await second.pending() == (second_id,)
    with pytest.raises(PermissionError):
        await second.activate(hold_id)

    await first.activate(hold_id)

    async def read(value: str) -> str:
        assert get_current_runtime_snapshot() is fixture.snapshot
        return value

    assert await first.call(hold_id, read) == "one"
    with pytest.raises(PermissionError):
        await second.call(hold_id, read)
    wrong = fixture.run.bind(
        ServiceKey[str]("test.other"),
        _hold_key("source:first"),
    )
    with pytest.raises(PermissionError):
        await wrong.call(hold_id, read)
    with pytest.raises(PermissionError):
        await second.drop(hold_id)

    record = fixture.journal._hold_record(str(hold_id))
    assert record.state == "active"
    assert record.hold_key != "source:first"
    assert "one" not in record.root_json
    assert all(
        field not in record.root_json
        for field in (
            "source_name",
            "payload",
            "channel",
            "target",
            "delivery",
            "outcome",
        )
    )
    assert all(
        Path(_decode_artifact(value).path).is_relative_to(
            fixture.workspace / "runtime" / "artifact-pins" / "artifacts"
        )
        for value in fixture.journal._artifact_refs()
    )

    await first.drop(hold_id)
    await first.drop(hold_id)
    await second.drop(second_id)
    assert await first.pending() == ()
    assert fixture.snapshot.hold_count == 0
    assert fixture.journal._hold_record(str(hold_id)).state == "dropped"
    await fixture.store.close()


@pytest.mark.asyncio
async def test_service_hold_cancel_releases_call_but_keeps_hold(
    tmp_path: Path,
) -> None:
    fixture = await _fixture(tmp_path)
    holder = fixture.run.bind(fixture.key, _hold_key("source:cancel-call"))
    hold_id = await _reserve(fixture, holder)
    await holder.activate(hold_id)
    started = asyncio.Event()

    async def block(_value: str) -> None:
        started.set()
        await asyncio.Event().wait()

    call = asyncio.create_task(holder.call(hold_id, block))
    await started.wait()
    assert fixture.snapshot.lease_count == 1
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert fixture.snapshot.lease_count == 0
    assert fixture.snapshot.hold_count == 1
    assert await holder.pending() == (hold_id,)
    await holder.drop(hold_id)
    await fixture.store.close()


@pytest.mark.asyncio
async def test_service_hold_rejects_candidate_without_journal_row(
    tmp_path: Path,
) -> None:
    fixture = await _fixture(tmp_path)
    candidate, _candidate_root = await _build_snapshot(
        fixture.plugin_dir,
        fixture.data_dir,
        generation_id="fixture-v2",
    )
    transaction = fixture.store.begin_publish(candidate)
    await fixture.store.commit_latest(transaction)
    holder = fixture.run.bind(fixture.key, _hold_key("source:candidate"))
    lease = fixture.store.lease(selector="latest")
    token = bind_runtime_snapshot(lease)
    try:
        with pytest.raises(RuntimeError, match="live exact snapshot lease"):
            await holder.reserve()
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
    assert fixture.journal._pending_holds() == ()
    assert fixture.journal._artifact_refs() == ()
    await fixture.store.discard_latest(candidate)
    await fixture.store.close()


@pytest.mark.asyncio
async def test_service_hold_reboots_from_exact_pins(tmp_path: Path) -> None:
    fixture = await _fixture(tmp_path)
    key = _hold_key("source:reboot")
    first = fixture.run.bind(fixture.key, key)
    hold_id = await _reserve(fixture, first)
    await first.activate(hold_id)

    # The live source and config drift after reserve; reboot must ignore both.
    _write_plugin(fixture.plugin_dir, fixture.data_dir, "two")
    shutil.rmtree(fixture.plugin_dir)

    loader = _FreshLoader()

    async def dispose(item: RuntimeSnapshot) -> None:
        if item.composition_root is not None:
            await item.composition_root.dispose()

    reopened_store = RuntimeSnapshotStore(dispose)
    reopened = _HoldRun(reopened_store, ReloadJournal(fixture.workspace))
    await reopened.recover(loader)
    second = reopened.bind(ServiceKey("test.value"), key)
    assert await second.pending() == (hold_id,)

    async def read(value: str) -> str:
        snapshot = get_current_runtime_snapshot()
        assert snapshot is not None
        assert snapshot is not fixture.snapshot
        return value

    assert await second.call(hold_id, read) == "one"
    assert loader.calls == 1
    await second.drop(hold_id)
    await reopened_store.close()


@pytest.mark.asyncio
async def test_missing_pin_degrades_without_stable_fallback(tmp_path: Path) -> None:
    fixture = await _fixture(tmp_path)
    holder = fixture.run.bind(fixture.key, _hold_key("source:missing"))
    hold_id = await _reserve(fixture, holder)
    await holder.activate(hold_id)
    record = fixture.journal._hold_record(str(hold_id))
    ref = json.loads(record.root_json)
    pinned = Path(_decode_artifact(ref["plugins"][0]["artifact"]).path)
    shutil.rmtree(pinned)

    reopened = _HoldRun(
        RuntimeSnapshotStore(),
        ReloadJournal(fixture.workspace),
    )
    with pytest.raises(BaseExceptionGroup):
        await reopened.recover(_FreshLoader())
    failed = fixture.journal._hold_record(str(hold_id))
    assert failed.state == "active"
    assert failed.error


@pytest.mark.asyncio
async def test_hold_and_live_lease_both_block_drain(tmp_path: Path) -> None:
    fixture = await _fixture(tmp_path)
    holder = fixture.run.bind(fixture.key, _hold_key("source:drain"))
    hold_id = await _reserve(fixture, holder)
    await holder.activate(hold_id)
    generation = fixture.snapshot.generations["fixture"]
    live = fixture.store.lease()
    wait = asyncio.create_task(fixture.store.wait_for_generation_drained(generation))
    await asyncio.sleep(0)
    assert not wait.done()

    await holder.drop(hold_id)
    await asyncio.sleep(0)
    assert generation.hold_count == 0
    assert not wait.done()
    await live.release()
    await wait
    await fixture.store.close()


class _SourceDb:
    """Model the fake source ledger that Core must never inspect."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with closing(sqlite3.connect(path)) as conn:
            conn.execute(
                """
                CREATE TABLE work (
                    hold_id TEXT PRIMARY KEY,
                    target TEXT NOT NULL,
                    delivery_key TEXT NOT NULL,
                    outcome TEXT,
                    degraded INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()

    def add(self, hold_id: HoldId) -> None:
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                "INSERT INTO work VALUES (?, 'target', 'stable-key', NULL, 0)",
                (str(hold_id),),
            )
            conn.commit()

    def finish(self, hold_id: HoldId, outcome: str) -> None:
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                "UPDATE work SET outcome = ?, degraded = ? WHERE hold_id = ?",
                (outcome, int(outcome == "unknown"), str(hold_id)),
            )
            conn.commit()

    def delete(self, hold_id: HoldId) -> None:
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute("DELETE FROM work WHERE hold_id = ?", (str(hold_id),))
            conn.commit()

    def row(self, hold_id: HoldId) -> tuple[object, ...] | None:
        with closing(sqlite3.connect(self.path)) as conn:
            return conn.execute(
                "SELECT hold_id, target, delivery_key, outcome, degraded "
                "FROM work WHERE hold_id = ?",
                (str(hold_id),),
            ).fetchone()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "outcome"),
    (
        ("accept-failure", "abort"),
        ("run-failure", "abort"),
        ("cancel", "abort"),
        ("no-sink", "done"),
        ("stream-failure", "unknown"),
        ("projection-failure", "unknown"),
    ),
)
async def test_fake_source_owns_outcome(
    tmp_path: Path,
    case: str,
    outcome: str,
) -> None:
    fixture = await _fixture(tmp_path)
    source = _SourceDb(tmp_path / "source.sqlite3")
    holder = fixture.run.bind(fixture.key, _hold_key(f"source:{case}"))
    hold_id = await _reserve(fixture, holder)
    source.add(hold_id)
    await holder.activate(hold_id)

    async def run(value: str) -> str:
        assert value == "one"
        assert source.row(hold_id) is not None
        assert fixture.journal._hold_record(str(hold_id)).state == "active"
        if case in {"accept-failure", "run-failure"}:
            raise RuntimeError(case)
        if case == "cancel":
            raise asyncio.CancelledError
        return case

    if case in {"accept-failure", "run-failure"}:
        with pytest.raises(RuntimeError, match=case):
            await holder.call(hold_id, run)
    elif case == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await holder.call(hold_id, run)
    else:
        assert await holder.call(hold_id, run) == case
    source.finish(hold_id, outcome)

    if outcome == "unknown":
        assert source.row(hold_id) == (
            str(hold_id),
            "target",
            "stable-key",
            "unknown",
            1,
        )
        assert await holder.pending() == (hold_id,)
        assert fixture.snapshot.hold_count == 1
    else:
        await holder.drop(hold_id)
        source.delete(hold_id)
        assert source.row(hold_id) is None
        assert await holder.pending() == ()
        assert fixture.snapshot.hold_count == 0

    with closing(sqlite3.connect(fixture.journal.path)) as conn:
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(service_holds)")
        }
    assert columns.isdisjoint(
        {"source", "payload", "channel", "target", "delivery_key", "outcome"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "crash_after",
    (
        "before-reserve",
        "reserve",
        "source-row",
        "activate",
        "outcome",
        "drop",
        "delete",
    ),
)
async def test_fake_source_closes_each_crash_gap(
    tmp_path: Path,
    crash_after: str,
) -> None:
    """Reboot after each durable fence and converge in the required order."""

    fixture = await _fixture(tmp_path)
    source = _SourceDb(tmp_path / "source.sqlite3")
    key = _hold_key("source:crash-gaps")
    first = fixture.run.bind(fixture.key, key)
    hold_id: HoldId | None = None
    if crash_after != "before-reserve":
        hold_id = await _reserve(fixture, first)
    if hold_id is not None and crash_after in {
        "source-row",
        "activate",
        "outcome",
        "drop",
        "delete",
    }:
        source.add(hold_id)
    if hold_id is not None and crash_after in {
        "activate",
        "outcome",
        "drop",
        "delete",
    }:
        await first.activate(hold_id)
    if hold_id is not None and crash_after in {"outcome", "drop", "delete"}:
        source.finish(hold_id, "abort")
    if hold_id is not None and crash_after in {"drop", "delete"}:
        await first.drop(hold_id)
    if hold_id is not None and crash_after == "delete":
        source.delete(hold_id)

    reopened_store = RuntimeSnapshotStore()
    reopened_journal = ReloadJournal(fixture.workspace)
    reopened = _HoldRun(reopened_store, reopened_journal)
    await reopened.recover(_FreshLoader())
    second = reopened.bind(ServiceKey("test.value"), key)
    pending = await second.pending()

    if hold_id is None:
        assert pending == ()
        return
    row = source.row(hold_id)
    if row is None:
        # reserve -> source-row crash leaves an orphan Core hold.
        if pending:
            await second.drop(hold_id)
    elif row[3] is None:
        # source row exists, so boot may idempotently activate, then abort.
        if reopened_journal._hold_record(str(hold_id)).state == "reserved":
            await second.activate(hold_id)
        source.finish(hold_id, "abort")
        await second.drop(hold_id)
        source.delete(hold_id)
    else:
        # outcome was durable before drop; drop and delete are both idempotent.
        await second.drop(hold_id)
        source.delete(hold_id)

    assert await second.pending() == ()
    assert source.row(hold_id) is None
    assert reopened_journal._hold_record(str(hold_id)).state == "dropped"
