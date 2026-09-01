from __future__ import annotations

# pyright: reportPrivateUsage=false

import ast
import asyncio
import json
import pickle
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.plugin_composition import (
    BACKGROUND_JOBS,
    ROOT_SWITCH,
    CompositionError,
    CompositionRoot,
    PluginRuntime,
    PluginBackgroundJobs,
    RootSwitch,
    ServiceKey,
    SwitchInput,
    SwitchPart,
)
from agent.plugin_composition.root_switch import (
    _PartEntry,
    _PartRef,
    _PartSet,
)
from agent.plugin_composition.effect import Effect
from agent.plugins.reload_journal import ReloadJournal
from agent.plugins.artifact_pins import _ArtifactPins
from agent.plugins.artifacts import ArtifactPointer, read_pointer, write_pointers
from agent.plugins.root_switch import (
    _SwitchError,
    _SwitchRun,
    _decode_ref,
    _pin_parts,
    _ref_value,
)
from agent.plugins.generation import PluginContributions
from agent.plugins.manager import PluginManager
from agent.plugins.install import finalize_uninstall_plugin
from agent.plugins.generation_activity_host import ActivityCatalog, ActivityHost
from agent.plugins.service_hold import _HoldRun, _hold_key
from agent.plugins.snapshot import (
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.plugins.source_hash import file_hash, file_revision, source_revision
from bus.event_bus import EventBus


class _Owner:
    def __init__(
        self,
        label: str,
        events: list[str],
        *,
        active: bool,
        fail: frozenset[str] = frozenset(),
    ) -> None:
        self.label = label
        self.events = events
        self.active = active
        self.present = active
        self.fail = fail
        self.wait_on: str | None = None
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.recover_calls: list[bool] = []
        self.snapshots: list[tuple[str, str]] = []

    def part(self, name: str = "shared") -> SwitchPart:
        return SwitchPart(
            name=name,
            stop=lambda snapshot: self._run("stop", snapshot),
            leave=lambda snapshot: self._run("leave", snapshot),
            enter=lambda snapshot: self._run("enter", snapshot),
            start=lambda snapshot: self._run("start", snapshot),
            recover=self.recover,
        )

    async def _run(self, action: str, snapshot: str) -> None:
        self.snapshots.append((action, snapshot))
        self.events.append(f"{self.label}.{action}")
        if action == "stop":
            self.active = False
        elif action == "leave":
            self.present = False
        elif action == "enter":
            self.present = True
        elif action == "start":
            self.active = True
        if self.wait_on == action:
            self.waiting.set()
            await self.release.wait()
        if action in self.fail:
            raise RuntimeError(f"{self.label} {action} failed")

    async def recover(self, snapshot: str, active: bool) -> None:
        self.snapshots.append(("recover", snapshot))
        self.events.append(f"{self.label}.recover:{active}")
        self.recover_calls.append(active)
        if "recover" in self.fail:
            raise RuntimeError(f"{self.label} recover failed")
        self.present = active
        self.active = active


class _Loader:
    def __init__(self, entries: tuple[_PartEntry, ...]) -> None:
        self._entries = {entry.ref: entry for entry in entries}

    async def load(self, ref: _PartRef) -> _PartEntry:
        try:
            return self._entries[ref]
        except KeyError as error:
            expected = json.loads(ref.artifact)
            for known, entry in self._entries.items():
                artifact = json.loads(known.artifact)
                if (
                    known.name == ref.name
                    and known.owner == ref.owner
                    and known.generation == ref.generation
                    and artifact["source_revision"] == expected["source_revision"]
                    and artifact["config_revision"] == expected["config_revision"]
                ):
                    return _PartEntry(ref, entry.part)
            raise FileNotFoundError(ref.artifact) from error


def _entry(
    owner: _Owner,
    *,
    name: str = "shared",
    plugin: str,
    generation: str,
    artifact: str,
    fiber: str | None = None,
) -> _PartEntry:
    return _PartEntry(
        ref=_PartRef(name, plugin, generation, artifact, fiber or plugin),
        part=owner.part(name),
    )


def _parts(*entries: _PartEntry) -> _PartSet:
    return _PartSet({entry.ref.name: entry for entry in entries})


def _target_parts(target: Any, *entries: _PartEntry) -> _PartSet | None:
    """Bind test callbacks to the exact refs selected by a durable target."""

    callbacks = {entry.ref.name: entry.part for entry in entries}
    selected = {}
    for move in target.moves:
        ref = move.new if target.use_new else move.old
        if ref is not None:
            selected[ref.name] = _PartEntry(ref, callbacks[ref.name])
    return _PartSet(selected) if selected else None


async def _recover_target(target: Any, *entries: _PartEntry) -> None:
    """Run the two closed sides after journal preflight."""

    loader = _Loader(entries)
    inactive = ((move.old if target.use_new else move.new) for move in target.moves)
    for ref in inactive:
        if ref is not None:
            assert target.other_snapshot is not None
            entry = await loader.load(ref)
            await entry.part.recover(target.other_snapshot, False)
    active = ((move.new if target.use_new else move.old) for move in target.moves)
    for ref in active:
        if ref is not None:
            assert target.snapshot is not None
            entry = await loader.load(ref)
            await entry.part.recover(target.snapshot, True)


def _artifact(root: Path, name: str) -> str:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text("# switch fixture\n", encoding="utf-8")
    data_dir = root / f"{name}-data"
    data_dir.mkdir(exist_ok=True)
    config_path = data_dir / "config.local.toml"
    return json.dumps(
        {
            "config_hash": file_hash(config_path),
            "config_path": str(config_path.resolve(strict=False)),
            "config_revision": file_revision(config_path),
            "data_path": str(data_dir.resolve()),
            "entrypoint": "plugin.py",
            "path": str(plugin_dir.resolve()),
            "source_revision": source_revision(plugin_dir),
            "source_type": "installed",
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _builtin_artifact(
    root: Path,
    label: str,
    log: Path,
    *,
    fail_recover: bool = False,
) -> str:
    plugin_dir = root / label
    plugin_dir.mkdir(parents=True)
    recover_failure = (
        "    raise RuntimeError('recover failed')\n" if fail_recover else ""
    )
    (plugin_dir / "plugin.py").write_text(
        "from pathlib import Path\n"
        "from agent.plugin_composition import ROOT_SWITCH, SwitchPart\n"
        "api_version = 3\n"
        "name = 'plugin'\n"
        f"version = {label!r}\n"
        "inject = (ROOT_SWITCH,)\n"
        "async def _none(_snapshot): pass\n"
        "async def _recover(_snapshot, active):\n"
        f"{recover_failure}"
        f"    path = Path({str(log)!r})\n"
        f"    prior = path.read_text(encoding='utf-8') if path.exists() else ''\n"
        f"    path.write_text(prior + {label!r} + ':' + str(active) + '\\n', encoding='utf-8')\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(ROOT_SWITCH).add(ctx, SwitchPart(\n"
        "        name='shared', stop=_none, leave=_none, enter=_none,\n"
        "        start=_none, recover=_recover,\n"
        "    ))\n",
        encoding="utf-8",
    )
    data_dir = root / f"{label}-data"
    data_dir.mkdir()
    config_path = data_dir / "config.local.toml"
    return json.dumps(
        {
            "config_hash": file_hash(config_path),
            "config_path": str(config_path.resolve(strict=False)),
            "config_revision": file_revision(config_path),
            "data_path": str(data_dir.resolve()),
            "entrypoint": "plugin.py",
            "path": str(plugin_dir.resolve()),
            "source_revision": source_revision(plugin_dir),
            "source_type": "builtin",
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_builtin(
    plugin_dir: Path,
    data_dir: Path,
    *,
    label: str,
    log: Path,
    plugin_name: str = "plugin",
) -> str:
    """Write one same-path builtin revision used by restart tests."""

    plugin_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(
        "from pathlib import Path\n"
        "from agent.plugin_composition import ROOT_SWITCH, SwitchPart\n"
        "api_version = 3\n"
        f"name = {plugin_name!r}\n"
        f"version = {label!r}\n"
        "inject = (ROOT_SWITCH,)\n"
        "async def _none(_snapshot): pass\n"
        "async def _recover(_snapshot, active):\n"
        f"    path = Path({str(log)!r})\n"
        "    prior = path.read_text(encoding='utf-8') if path.exists() else ''\n"
        f"    path.write_text(prior + {label!r} + ':' + str(active) + '\\n', encoding='utf-8')\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(ROOT_SWITCH).add(ctx, SwitchPart(\n"
        "        name='shared', stop=_none, leave=_none, enter=_none,\n"
        "        start=_none, recover=_recover,\n"
        "    ))\n",
        encoding="utf-8",
    )
    config_path = data_dir / "config.local.toml"
    return json.dumps(
        {
            "config_hash": file_hash(config_path),
            "config_path": str(config_path.resolve(strict=False)),
            "config_revision": file_revision(config_path),
            "data_path": str(data_dir.resolve()),
            "entrypoint": "plugin.py",
            "path": str(plugin_dir.resolve()),
            "source_revision": source_revision(plugin_dir),
            "source_type": "builtin",
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_installed(
    plugin_base: Path,
    artifact_name: str,
    data_dir: Path,
    *,
    plugin_name: str,
    log: Path,
) -> str:
    artifact_dir = plugin_base / ".artifacts" / artifact_name
    artifact = cast(
        dict[str, str],
        json.loads(
            _write_builtin(
                artifact_dir,
                data_dir,
                label=artifact_name,
                log=log,
                plugin_name=plugin_name,
            )
        ),
    )
    (artifact_dir / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        f"name = {plugin_name!r}\n"
        f"version = {artifact_name!r}\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n",
        encoding="utf-8",
    )
    artifact["source_type"] = "installed"
    artifact["source_revision"] = source_revision(artifact_dir)
    return json.dumps(artifact, separators=(",", ":"), sort_keys=True)


def _journal(
    workspace: Path,
    *,
    old_snapshot: str | None = "snapshot-old",
    new_snapshot: str = "snapshot-new",
) -> tuple[ReloadJournal, str]:
    journal = ReloadJournal(workspace)
    tx_id = journal.begin(
        plugin_id="fixture",
        base_snapshot_id=old_snapshot,
        base_generation_id="fixture-old",
        generation_id="fixture-new",
        source_revision="source-new",
        config_revision="config-new",
    )
    journal.advance(
        tx_id,
        "prepared",
        candidate_snapshot_id=new_snapshot,
    )
    journal.advance(tx_id, "validating")
    return journal, tx_id


def _generation(
    plugin_id: str,
    generation_id: str,
    plugin_dir: Path,
) -> Any:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text("# switch fixture\n", encoding="utf-8")
    data_dir = plugin_dir.parent / f"{plugin_dir.name}-data"
    data_dir.mkdir(exist_ok=True)
    config_path = data_dir / "config.local.toml"
    return SimpleNamespace(
        plugin_id=plugin_id,
        generation_id=generation_id,
        source_revision=source_revision(plugin_dir),
        config_revision=file_revision(config_path),
        config_path=config_path,
        plugin_dir=plugin_dir,
        data_dir=data_dir,
        entrypoint="plugin.py",
        source_type="installed",
        static_manifest=None,
        skill_catalog=None,
        contributions=PluginContributions(manifest={}),
        config_projection={},
        lease_count=0,
        hold_count=0,
        reload_tx_id=None,
    )


def test_root_switch_public_surface_stays_small_and_source_neutral() -> None:
    import agent.plugin_composition.root_switch as module

    assert module.__all__ == [
        "ROOT_SWITCH",
        "RootSwitch",
        "SwitchInput",
        "SwitchPart",
    ]
    root = Path(__file__).parents[1]
    violations: list[str] = []
    for relative in (
        "agent/plugin_composition/root_switch.py",
        "agent/plugins/root_switch.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names = (node.module or "",)
            else:
                continue
            for name in names:
                if name == "agent.control" or name.startswith("agent.control."):
                    violations.append(f"{relative}:{node.lineno}:{name}")
    assert violations == []


@pytest.mark.asyncio
async def test_snapshot_seal_freezes_exact_part_without_calling_it(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    owner = _Owner("only", events, active=False)
    root = CompositionRoot("switch-root")
    switch = RootSwitch(root.instance_token)
    _ = await root.context.provide(ROOT_SWITCH, switch)
    plugin_dir = tmp_path / ".artifacts" / "plugin-v1"
    plugin_dir.mkdir(parents=True)
    runtime = PluginRuntime(
        plugin_id="plugin",
        generation_id="plugin-v1",
        plugin_dir=plugin_dir,
        data_dir=tmp_path / "data",
        workspace=tmp_path / "workspace",
        config={},
    )
    plugin_ctx = None

    async def plugin(ctx) -> None:
        nonlocal plugin_ctx
        plugin_ctx = ctx
        _ = await ctx.require(ROOT_SWITCH).add(ctx, owner.part())

    _ = await root.mount(plugin, name="plugin", runtime=runtime)
    generation = _generation("plugin", "plugin-v1", plugin_dir)
    snapshot = RuntimeSnapshotCompiler().compile(
        {"plugin": generation},
        composition_root=root,
    )

    assert events == []
    assert snapshot.switch_parts is not None
    entry = snapshot.switch_parts["shared"]
    artifact = cast(dict[str, str], json.loads(entry.ref.artifact))
    assert entry.ref.owner == "plugin"
    assert entry.ref.generation == "plugin-v1"
    assert artifact == {
        "config_hash": file_hash(generation.config_path),
        "config_path": str(generation.config_path.resolve()),
        "config_revision": generation.config_revision,
        "data_path": str(generation.data_dir.resolve()),
        "entrypoint": "plugin.py",
        "path": str(plugin_dir.resolve()),
        "source_revision": generation.source_revision,
        "source_type": "installed",
    }
    assert entry.part is not owner
    assert entry.part.name == "shared"
    assert plugin_ctx is not None
    with pytest.raises(CompositionError) as caught:
        await switch.add(plugin_ctx, owner.part("late"))
    assert caught.value.code == "ROOT_SWITCH_FROZEN"
    await root.dispose()


@pytest.mark.asyncio
async def test_switch_inputs_freeze_direct_multiset_and_real_closure(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("input-root")
    switch = RootSwitch(root.instance_token)
    _ = await root.context.provide(ROOT_SWITCH, switch)
    value_key = ServiceKey[str]("test.input.value")
    noise_key = ServiceKey[str]("test.input.noise")
    tokens: list[SwitchInput] = []
    contexts: list[Any] = []
    generations: dict[str, Any] = {}

    async def mount_plugin(
        plugin_id: str,
        generation_id: str,
        apply: Any,
    ) -> None:
        plugin_dir = tmp_path / plugin_id
        runtime = PluginRuntime(
            plugin_id=plugin_id,
            generation_id=generation_id,
            plugin_dir=plugin_dir,
            data_dir=tmp_path / f"{plugin_id}-data",
            workspace=tmp_path / "workspace",
            config={},
        )
        _ = await root.mount(apply, name=plugin_id, runtime=runtime)
        generations[plugin_id] = _generation(
            plugin_id,
            generation_id,
            plugin_dir,
        )

    async def provider(ctx) -> None:
        _ = await ctx.provide(value_key, "ready")

    async def noise(ctx) -> None:
        _ = await ctx.provide(noise_key, "unused")

    async def contributor(ctx) -> None:
        contexts.append(ctx)

        async def child(child_ctx) -> None:
            assert child_ctx.require(value_key) == "ready"

            def add_input() -> None:
                tokens.append(child_ctx.require(ROOT_SWITCH).input(child_ctx))

            _ = await child_ctx.effect(add_input, label="registry:item-one")
            _ = await child_ctx.effect(add_input, label="registry:item-two")

        _ = await ctx.mount(
            child,
            name="input-child",
            inject=(ROOT_SWITCH, value_key),
        )

        async def unrelated(child_ctx) -> None:
            assert child_ctx.require(noise_key) == "unused"

        _ = await ctx.mount(
            unrelated,
            name="unrelated-child",
            inject=(noise_key,),
        )

    owner = _Owner("part", [], active=False)

    async def part(ctx) -> None:
        _ = await ctx.require(ROOT_SWITCH).add(
            ctx,
            SwitchPart(
                name="shared",
                stop=owner.part().stop,
                leave=owner.part().leave,
                enter=owner.part().enter,
                start=owner.part().start,
                recover=owner.recover,
                inputs=tuple(tokens),
            ),
        )

    await mount_plugin("provider", "provider-v1", provider)
    await mount_plugin("noise", "noise-v1", noise)
    await mount_plugin("contributor", "contributor-v1", contributor)
    await mount_plugin("part", "part-v1", part)
    snapshot = RuntimeSnapshotCompiler().compile(
        generations,
        composition_root=root,
    )

    assert snapshot.switch_parts is not None
    ref = snapshot.switch_parts["shared"].ref
    assert ref.fiber == "part"
    assert tuple((item.owner, item.generation, item.fiber) for item in ref.inputs) == (
        ("contributor", "contributor-v1", "input-child"),
        ("contributor", "contributor-v1", "input-child"),
    )
    assert tuple(item.owner for item in ref.needs) == (
        "contributor",
        "provider",
    )
    run = _SwitchRun(ReloadJournal(tmp_path / "input-workspace"))
    boot = run.prepare_boot(
        "input-boot",
        new_snapshot=snapshot.snapshot_id,
        new_parts=snapshot.switch_parts,
    )
    assert boot is not None
    boot.commit()
    boot.finish()
    stored = run.choices()[0].ref
    assert stored is not None and stored.fiber == "part"
    assert len(stored.inputs) == 2
    assert stored.inputs[0] == stored.inputs[1]
    contributor_need = next(
        item for item in stored.needs if item.owner == "contributor"
    )
    assert all(item.artifact == contributor_need.artifact for item in stored.inputs)
    old_shape = _ref_value(stored)
    assert old_shape is not None
    del old_shape["inputs"]
    with pytest.raises(RuntimeError, match="结构无效"):
        _decode_ref(old_shape)
    assert tokens[0] is not tokens[1]
    with pytest.raises(TypeError, match="只能由"):
        SwitchInput()
    with pytest.raises(TypeError, match="不能序列化"):
        pickle.dumps(tokens[0])
    with pytest.raises(CompositionError) as outside:
        switch.input(contexts[0])
    assert outside.value.code == "INPUT_MISMATCH"
    await root.dispose()


@pytest.mark.asyncio
async def test_switch_input_rejects_an_unowned_effect(tmp_path: Path) -> None:
    root = CompositionRoot("input-owner-root")
    switch = RootSwitch(root.instance_token)
    _ = await root.context.provide(ROOT_SWITCH, switch)
    contexts: list[Any] = []
    runtime = PluginRuntime(
        plugin_id="owner",
        generation_id="owner-v1",
        plugin_dir=tmp_path / "owner",
        data_dir=tmp_path / "owner-data",
        workspace=tmp_path / "workspace",
        config={},
    )

    async def plugin(ctx) -> None:
        contexts.append(ctx)

    _ = await root.mount(plugin, name="owner", runtime=runtime)
    ctx = contexts[0]
    rogue = Effect(
        label="rogue",
        remove_from_owner=lambda _effect: None,
        plugin_id=runtime.plugin_id,
        generation_id=runtime.generation_id,
        fiber=ctx.fiber.name,
        root_token=ctx.root_instance_token,
        activation_token=ctx.fiber.activation_token,
    )

    with pytest.raises(CompositionError) as caught:
        await rogue.start(lambda: switch.input(ctx))
    assert caught.value.code == "INPUT_MISMATCH"
    await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("unused", "duplicate", "cleaned", "other-root"))
async def test_switch_input_must_be_live_and_consumed_once(
    tmp_path: Path,
    mode: str,
) -> None:
    source_root = CompositionRoot(f"input-source-{mode}")
    source_switch = RootSwitch(source_root.instance_token)
    _ = await source_root.context.provide(ROOT_SWITCH, source_switch)
    source_dir = tmp_path / "source"
    source_runtime = PluginRuntime(
        plugin_id="source",
        generation_id="source-v1",
        plugin_dir=source_dir,
        data_dir=tmp_path / "source-data",
        workspace=tmp_path / "workspace",
        config={},
    )
    tokens: list[SwitchInput] = []
    effects: list[Any] = []

    async def source(ctx) -> None:
        def setup() -> None:
            tokens.append(ctx.require(ROOT_SWITCH).input(ctx))

        effects.append(await ctx.effect(setup, label="registry:item"))

    _ = await source_root.mount(source, name="source", runtime=source_runtime)
    if mode == "cleaned":
        await effects[0].aclose()

    target_root = source_root
    target_switch = source_switch
    generations = {
        "source": _generation("source", "source-v1", source_dir),
    }
    if mode == "other-root":
        target_root = CompositionRoot("other-root")
        target_switch = RootSwitch(target_root.instance_token)
        _ = await target_root.context.provide(ROOT_SWITCH, target_switch)
        generations = {}
    part_dir = tmp_path / "part"
    part_runtime = PluginRuntime(
        plugin_id="part",
        generation_id="part-v1",
        plugin_dir=part_dir,
        data_dir=tmp_path / "part-data",
        workspace=tmp_path / "workspace",
        config={},
    )
    owner = _Owner("part", [], active=False)
    selected = (
        ()
        if mode == "unused"
        else ((tokens[0], tokens[0]) if mode == "duplicate" else (tokens[0],))
    )

    async def part(ctx) -> None:
        _ = await ctx.require(ROOT_SWITCH).add(
            ctx,
            SwitchPart(
                name="shared",
                stop=owner.part().stop,
                leave=owner.part().leave,
                enter=owner.part().enter,
                start=owner.part().start,
                recover=owner.recover,
                inputs=selected,
            ),
        )

    _ = await target_root.mount(part, name="part", runtime=part_runtime)
    generations["part"] = _generation("part", "part-v1", part_dir)
    expected = {
        "unused": "UNUSED_INPUT",
        "duplicate": "DUPLICATE_INPUT",
        "cleaned": "INVALID_INPUT",
        "other-root": "INVALID_INPUT",
    }[mode]
    with pytest.raises(CompositionError) as caught:
        RuntimeSnapshotCompiler().compile(
            generations,
            composition_root=target_root,
        )
    assert caught.value.code == expected
    if target_root is not source_root:
        await target_root.dispose()
    await source_root.dispose()


@pytest.mark.asyncio
async def test_candidate_seal_does_not_write_builtin_pin(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_dir = tmp_path / "plugins" / "plugin"
    root = CompositionRoot("candidate-root")
    switch = RootSwitch(root.instance_token)
    _ = await root.context.provide(ROOT_SWITCH, switch)
    runtime = PluginRuntime(
        plugin_id="plugin",
        generation_id="plugin-v1",
        plugin_dir=plugin_dir,
        data_dir=tmp_path / "data",
        workspace=workspace,
        config={},
    )
    owner = _Owner("candidate", [], active=False)

    async def plugin(ctx) -> None:
        _ = await ctx.require(ROOT_SWITCH).add(ctx, owner.part())

    _ = await root.mount(plugin, name="plugin", runtime=runtime)
    generation = _generation("plugin", "plugin-v1", plugin_dir)
    generation.source_type = "builtin"
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )

    snapshot = manager._snapshot_compiler.compile(
        {"plugin": generation},
        composition_root=root,
    )

    assert snapshot.switch_parts is not None
    assert not (workspace / "runtime" / "artifact-pins" / "artifacts").exists()
    await root.dispose()


def test_config_drift_before_prepare_leaves_no_plan_or_pin(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    old_data = workspace / "plugin-data" / "old-builtin"
    new_data = workspace / "plugin-data" / "new-builtin"
    old_artifact = _write_builtin(
        tmp_path / "plugins" / "old",
        old_data,
        label="old",
        log=tmp_path / "recover.log",
        plugin_name="old",
    )
    new_artifact = _write_builtin(
        tmp_path / "plugins" / "new",
        new_data,
        label="new",
        log=tmp_path / "recover.log",
        plugin_name="new",
    )
    (new_data / "config.local.toml").write_text(
        'value = "changed-after-seal"\n',
        encoding="utf-8",
    )
    events: list[str] = []
    run = _SwitchRun(ReloadJournal(workspace))

    with pytest.raises(RuntimeError, match="pin 前已漂移"):
        run.prepare(
            "config-drift",
            old_snapshot="old-snapshot",
            old_parts=_parts(
                _entry(
                    _Owner("old", events, active=True),
                    plugin="old",
                    generation="old-generation",
                    artifact=old_artifact,
                )
            ),
            new_snapshot="new-snapshot",
            new_parts=_parts(
                _entry(
                    _Owner("new", events, active=False),
                    plugin="new",
                    generation="new-generation",
                    artifact=new_artifact,
                )
            ),
        )

    assert run.targets() == ()
    assert run.choices() == ()
    assert events == []
    pin_root = workspace / "runtime" / "artifact-pins" / "artifacts"
    assert not pin_root.exists() or not tuple(pin_root.iterdir())


@pytest.mark.asyncio
async def test_duplicate_part_name_rejects_snapshot_seal(tmp_path: Path) -> None:
    root = CompositionRoot("duplicate-root")
    switch = RootSwitch(root.instance_token)
    _ = await root.context.provide(ROOT_SWITCH, switch)
    generations: dict[str, Any] = {}
    for index in (1, 2):
        plugin_id = f"plugin-{index}"
        generation_id = f"generation-{index}"
        plugin_dir = tmp_path / plugin_id
        plugin_dir.mkdir()
        runtime = PluginRuntime(
            plugin_id=plugin_id,
            generation_id=generation_id,
            plugin_dir=plugin_dir,
            data_dir=tmp_path / f"data-{index}",
            workspace=tmp_path / "workspace",
            config={},
        )
        owner = _Owner(plugin_id, [], active=False)

        async def plugin(ctx, part=owner.part()) -> None:
            _ = await ctx.require(ROOT_SWITCH).add(ctx, part)

        _ = await root.mount(plugin, name=plugin_id, runtime=runtime)
        generations[plugin_id] = _generation(
            plugin_id,
            generation_id,
            plugin_dir,
        )

    assert root.receipt().ready
    with pytest.raises(CompositionError) as caught:
        RuntimeSnapshotCompiler().compile(generations, composition_root=root)
    assert caught.value.code == "DUPLICATE_ROOT_SWITCH"
    await root.dispose()


@pytest.mark.asyncio
async def test_duplicate_owner_rejects_snapshot_seal(tmp_path: Path) -> None:
    root = CompositionRoot("owner-root")
    switch = RootSwitch(root.instance_token)
    _ = await root.context.provide(ROOT_SWITCH, switch)
    plugin_dir = tmp_path / "plugin"
    generation = _generation("plugin", "generation", plugin_dir)
    runtime = PluginRuntime(
        plugin_id="plugin",
        generation_id="generation",
        plugin_dir=plugin_dir,
        data_dir=generation.data_dir,
        workspace=tmp_path / "workspace",
        config={},
    )
    owners = (_Owner("first", [], active=False), _Owner("second", [], active=False))

    async def plugin(ctx) -> None:
        _ = await ctx.require(ROOT_SWITCH).add(ctx, owners[0].part("first"))
        _ = await ctx.require(ROOT_SWITCH).add(ctx, owners[1].part("second"))

    _ = await root.mount(plugin, name="plugin", runtime=runtime)
    with pytest.raises(CompositionError) as caught:
        RuntimeSnapshotCompiler().compile(
            {"plugin": generation},
            composition_root=root,
        )
    assert caught.value.code == "DUPLICATE_OWNER"
    await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("old_present", "new_present", "expected"),
    (
        (False, True, ["new.enter", "new.start"]),
        (True, False, ["old.stop", "old.leave"]),
        (
            True,
            True,
            ["old.stop", "old.leave", "new.enter", "new.start"],
        ),
    ),
)
async def test_switch_runs_fixed_install_remove_replace_order(
    tmp_path: Path,
    old_present: bool,
    new_present: bool,
    expected: list[str],
) -> None:
    events: list[str] = []
    old_owner = _Owner("old", events, active=True)
    new_owner = _Owner("new", events, active=False)
    old_entry = _entry(
        old_owner,
        plugin="plugin",
        generation="old-generation",
        artifact=_artifact(tmp_path, "artifact-old"),
    )
    new_entry = _entry(
        new_owner,
        plugin="plugin",
        generation="new-generation",
        artifact=_artifact(tmp_path, "artifact-new"),
    )
    journal, tx_id = _journal(tmp_path / "workspace")
    work = _SwitchRun(journal).prepare(
        tx_id,
        old_snapshot="snapshot-old",
        old_parts=_parts(old_entry) if old_present else None,
        new_snapshot="snapshot-new",
        new_parts=_parts(new_entry) if new_present else None,
    )
    assert work is not None

    await work.apply()
    work.commit()

    assert events == expected
    assert all(snapshot == "snapshot-old" for _, snapshot in old_owner.snapshots)
    assert all(snapshot == "snapshot-new" for _, snapshot in new_owner.snapshots)
    record = journal.switch_record(tx_id)
    assert record is not None and record.use_new and not record.cleared
    assert len(_SwitchRun(journal).pins()) == int(old_present) + int(new_present)
    journal.finish_switch(tx_id, use_new=True)
    assert journal.pending_switches() == ()


@pytest.mark.asyncio
async def test_same_exact_owner_needs_no_switch_plan(tmp_path: Path) -> None:
    events: list[str] = []
    old_owner = _Owner("old", events, active=True)
    new_owner = _Owner("new", events, active=False)
    old_entry = _entry(
        old_owner,
        plugin="plugin",
        generation="same-generation",
        artifact=_artifact(tmp_path, "same-artifact"),
    )
    new_entry = _PartEntry(old_entry.ref, new_owner.part())
    journal, tx_id = _journal(tmp_path / "workspace")

    work = _SwitchRun(journal).prepare(
        tx_id,
        old_snapshot="snapshot-old",
        old_parts=_parts(old_entry),
        new_snapshot="snapshot-new",
        new_parts=_parts(new_entry),
    )

    assert work is None
    assert journal.switch_record(tx_id) is None
    assert events == []


@pytest.mark.asyncio
async def test_start_failure_restores_old_side_in_reverse_order(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    old_owner = _Owner("old", events, active=True)
    new_owner = _Owner(
        "new",
        events,
        active=False,
        fail=frozenset({"start"}),
    )
    old_entry = _entry(
        old_owner,
        plugin="plugin",
        generation="old",
        artifact=_artifact(tmp_path, "artifact-old"),
    )
    new_entry = _entry(
        new_owner,
        plugin="plugin",
        generation="new",
        artifact=_artifact(tmp_path, "artifact-new"),
    )
    journal, tx_id = _journal(tmp_path / "workspace")
    work = _SwitchRun(journal).prepare(
        tx_id,
        old_snapshot="snapshot-old",
        old_parts=_parts(old_entry),
        new_snapshot="snapshot-new",
        new_parts=_parts(new_entry),
    )
    assert work is not None

    with pytest.raises(RuntimeError, match="new start failed"):
        await work.apply()

    assert events == [
        "old.stop",
        "old.leave",
        "new.enter",
        "new.start",
        "new.stop",
        "new.leave",
        "old.enter",
        "old.start",
    ]
    record = journal.switch_record(tx_id)
    assert record is not None and record.cleared and not record.use_new
    assert old_owner.active and old_owner.present
    assert not new_owner.active and not new_owner.present


@pytest.mark.asyncio
async def test_restore_failure_keeps_degraded_record_and_both_pins(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    old_owner = _Owner("old", events, active=True)
    new_owner = _Owner(
        "new",
        events,
        active=False,
        fail=frozenset({"start", "leave"}),
    )
    old_entry = _entry(
        old_owner,
        plugin="plugin",
        generation="old",
        artifact=_artifact(tmp_path, "artifact-old"),
    )
    new_entry = _entry(
        new_owner,
        plugin="plugin",
        generation="new",
        artifact=_artifact(tmp_path, "artifact-new"),
    )
    journal, tx_id = _journal(tmp_path / "workspace")
    work = _SwitchRun(journal).prepare(
        tx_id,
        old_snapshot="snapshot-old",
        old_parts=_parts(old_entry),
        new_snapshot="snapshot-new",
        new_parts=_parts(new_entry),
    )
    assert work is not None

    with pytest.raises(BaseExceptionGroup):
        await work.apply()

    record = journal.switch_record(tx_id)
    assert record is not None
    assert record.state == "degraded"
    assert not record.cleared
    assert record.failure_resource == "root-switch:shared:leave"
    assert old_owner.active and old_owner.present


@pytest.mark.asyncio
async def test_caller_cancel_waits_for_step_then_allows_old_restore(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    old_owner = _Owner("old", events, active=True)
    new_owner = _Owner("new", events, active=False)
    new_owner.wait_on = "enter"
    old_entry = _entry(
        old_owner,
        plugin="plugin",
        generation="old",
        artifact=_artifact(tmp_path, "artifact-old"),
    )
    new_entry = _entry(
        new_owner,
        plugin="plugin",
        generation="new",
        artifact=_artifact(tmp_path, "artifact-new"),
    )
    journal, tx_id = _journal(tmp_path / "workspace")
    work = _SwitchRun(journal).prepare(
        tx_id,
        old_snapshot="snapshot-old",
        old_parts=_parts(old_entry),
        new_snapshot="snapshot-new",
        new_parts=_parts(new_entry),
    )
    assert work is not None
    apply = asyncio.create_task(work.apply())
    await new_owner.waiting.wait()

    apply.cancel()
    await asyncio.sleep(0)
    assert not apply.done()
    new_owner.release.set()
    with pytest.raises(asyncio.CancelledError):
        await apply
    record = journal.switch_record(tx_id)
    assert record is not None and record.next_step == record.step_count

    await work.rollback(asyncio.CancelledError())

    assert old_owner.active and old_owner.present
    assert not new_owner.active and not new_owner.present
    assert journal.switch_record(tx_id).cleared  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_new_side_mark_is_inside_provisional_pointer_before_admission(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    old_owner = _Owner("old", events, active=True)
    new_owner = _Owner("new", events, active=False)
    old_entry = _entry(
        old_owner,
        plugin="plugin",
        generation="old",
        artifact=_artifact(tmp_path, "artifact-old"),
    )
    new_entry = _entry(
        new_owner,
        plugin="plugin",
        generation="new",
        artifact=_artifact(tmp_path, "artifact-new"),
    )
    compiler = RuntimeSnapshotCompiler()
    old_snapshot = compiler.compile({}, snapshot_revision="old")
    new_snapshot = compiler.compile({}, snapshot_revision="new")
    store = RuntimeSnapshotStore()
    store.install(old_snapshot)
    transaction = store.begin_publish(new_snapshot)
    await store.commit_provisional(transaction)
    journal, tx_id = _journal(
        tmp_path / "workspace",
        old_snapshot=old_snapshot.snapshot_id,
        new_snapshot=new_snapshot.snapshot_id,
    )
    work = _SwitchRun(journal).prepare(
        tx_id,
        old_snapshot=old_snapshot.snapshot_id,
        old_parts=_parts(old_entry),
        new_snapshot=new_snapshot.snapshot_id,
        new_parts=_parts(new_entry),
    )
    assert work is not None
    await work.apply()

    def mark_new_side() -> None:
        assert store.current is new_snapshot
        assert not new_snapshot.accepting_leases
        work.commit()

    await store.finalize_provisional(transaction, after_open=mark_new_side)

    assert store.current is new_snapshot
    assert new_snapshot.accepting_leases
    assert journal.switch_record(tx_id).use_new  # type: ignore[union-attr]
    journal.finish_switch(tx_id, use_new=True)
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("crash_step", (1, 2, 3, 4))
@pytest.mark.parametrize("step_recorded", (False, True))
async def test_each_forward_step_crash_recovers_old_side(
    tmp_path: Path,
    crash_step: int,
    step_recorded: bool,
) -> None:
    events: list[str] = []
    old_owner = _Owner("old", events, active=True)
    new_owner = _Owner("new", events, active=False)
    old_entry = _entry(
        old_owner,
        plugin="plugin",
        generation="old",
        artifact=_artifact(tmp_path, "artifact-old"),
    )
    new_entry = _entry(
        new_owner,
        plugin="plugin",
        generation="new",
        artifact=_artifact(tmp_path, "artifact-new"),
    )
    workspace = tmp_path / f"workspace-{crash_step}-{step_recorded}"
    journal, tx_id = _journal(workspace)
    run = _SwitchRun(journal)
    work = run.prepare(
        tx_id,
        old_snapshot="snapshot-old",
        old_parts=_parts(old_entry),
        new_snapshot="snapshot-new",
        new_parts=_parts(new_entry),
    )
    assert work is not None
    for index, step in enumerate(work._steps[:crash_step], start=1):
        await getattr(step.entry.part, step.action)(step.snapshot)
        if index < crash_step or step_recorded:
            journal.advance_switch(tx_id, index)

    record = journal.switch_record(tx_id)
    assert record is not None
    assert record.next_step == crash_step - (0 if step_recorded else 1)

    reopened = ReloadJournal(workspace)
    recovery = _SwitchRun(reopened)
    targets = await recovery.preflight(_Loader((old_entry, new_entry)))
    await _recover_target(targets[0], old_entry, new_entry)

    assert len(targets) == 1
    assert not targets[0].use_new
    assert old_owner.active and old_owner.present
    assert not new_owner.active and not new_owner.present
    recovery.finish_recovery(targets[0], _target_parts(targets[0], old_entry))
    assert reopened.pending_switches() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(("old_present", "new_present"), ((False, True), (True, False)))
@pytest.mark.parametrize("step_recorded", (False, True))
@pytest.mark.parametrize("crash_step", (1, 2))
async def test_install_remove_step_crash_recovers_old_side(
    tmp_path: Path,
    old_present: bool,
    new_present: bool,
    step_recorded: bool,
    crash_step: int,
) -> None:
    old_owner = _Owner("old", [], active=old_present)
    new_owner = _Owner("new", [], active=False)
    old_entry = _entry(
        old_owner,
        plugin="plugin",
        generation="old",
        artifact=_artifact(tmp_path, "artifact-old"),
    )
    new_entry = _entry(
        new_owner,
        plugin="plugin",
        generation="new",
        artifact=_artifact(tmp_path, "artifact-new"),
    )
    workspace = tmp_path / "workspace"
    journal, tx_id = _journal(workspace)
    work = _SwitchRun(journal).prepare(
        tx_id,
        old_snapshot="snapshot-old",
        old_parts=_parts(old_entry) if old_present else None,
        new_snapshot="snapshot-new",
        new_parts=_parts(new_entry) if new_present else None,
    )
    assert work is not None
    for index, step in enumerate(work._steps[:crash_step], start=1):
        await getattr(step.entry.part, step.action)(step.snapshot)
        if index < crash_step or step_recorded:
            journal.advance_switch(tx_id, index)

    recovery = _SwitchRun(ReloadJournal(workspace))
    entries = tuple(
        entry
        for entry, present in ((old_entry, old_present), (new_entry, new_present))
        if present
    )
    targets = await recovery.preflight(_Loader(entries))
    await _recover_target(targets[0], *entries)

    assert old_owner.active is old_present
    assert old_owner.present is old_present
    assert not new_owner.active and not new_owner.present
    recovery.finish_recovery(
        targets[0],
        _target_parts(targets[0], old_entry) if old_present else None,
    )


@pytest.mark.asyncio
async def test_many_parts_keep_stage_order(tmp_path: Path) -> None:
    events: list[str] = []
    entries: list[_PartEntry] = []
    for name in ("alpha", "beta"):
        entries.append(
            _entry(
                _Owner(f"old-{name}", events, active=True),
                name=name,
                plugin=f"old-{name}",
                generation=f"old-{name}",
                artifact=_artifact(tmp_path, f"old-{name}"),
            )
        )
        entries.append(
            _entry(
                _Owner(f"new-{name}", events, active=False),
                name=name,
                plugin=f"new-{name}",
                generation=f"new-{name}",
                artifact=_artifact(tmp_path, f"new-{name}"),
            )
        )
    old = _parts(entries[0], entries[2])
    new = _parts(entries[1], entries[3])
    journal, tx_id = _journal(tmp_path / "workspace")
    work = _SwitchRun(journal).prepare(
        tx_id,
        old_snapshot="snapshot-old",
        old_parts=old,
        new_snapshot="snapshot-new",
        new_parts=new,
    )
    assert work is not None

    await work.apply()

    assert events == [
        "old-alpha.stop",
        "old-beta.stop",
        "old-alpha.leave",
        "old-beta.leave",
        "new-alpha.enter",
        "new-beta.enter",
        "new-alpha.start",
        "new-beta.start",
    ]


@pytest.mark.asyncio
async def test_crash_after_new_side_mark_recovers_new_side(tmp_path: Path) -> None:
    events: list[str] = []
    old_owner = _Owner("old", events, active=True)
    new_owner = _Owner("new", events, active=False)
    old_entry = _entry(
        old_owner,
        plugin="plugin",
        generation="old",
        artifact=_artifact(tmp_path, "artifact-old"),
    )
    new_entry = _entry(
        new_owner,
        plugin="plugin",
        generation="new",
        artifact=_artifact(tmp_path, "artifact-new"),
    )
    workspace = tmp_path / "workspace"
    journal, tx_id = _journal(workspace)
    work = _SwitchRun(journal).prepare(
        tx_id,
        old_snapshot="snapshot-old",
        old_parts=_parts(old_entry),
        new_snapshot="snapshot-new",
        new_parts=_parts(new_entry),
    )
    assert work is not None
    await work.apply()
    work.commit()
    old_owner.active = True
    old_owner.present = True
    new_owner.active = False
    new_owner.present = False

    reopened = ReloadJournal(workspace)
    recovery = _SwitchRun(reopened)
    targets = await recovery.preflight(_Loader((old_entry, new_entry)))
    await _recover_target(targets[0], old_entry, new_entry)

    assert targets[0].use_new
    assert not old_owner.active and not old_owner.present
    assert new_owner.active and new_owner.present
    recovery.finish_recovery(targets[0], _target_parts(targets[0], new_entry))


@pytest.mark.asyncio
async def test_missing_artifact_stays_degraded(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    old_owner = _Owner("old", events, active=True)
    new_owner = _Owner("new", events, active=False)
    old_entry = _entry(
        old_owner,
        plugin="plugin",
        generation="old",
        artifact=_artifact(tmp_path, "artifact-old"),
    )
    new_entry = _entry(
        new_owner,
        plugin="plugin",
        generation="new",
        artifact=_artifact(tmp_path, "artifact-new"),
    )
    journal, tx_id = _journal(tmp_path / "missing")
    work = _SwitchRun(journal).prepare(
        tx_id,
        old_snapshot="snapshot-old",
        old_parts=_parts(old_entry),
        new_snapshot="snapshot-new",
        new_parts=_parts(new_entry),
    )
    assert work is not None
    await work.apply()
    work.commit()

    with pytest.raises(_SwitchError) as missing:
        await _SwitchRun(ReloadJournal(tmp_path / "missing")).preflight(
            _Loader((new_entry,))
        )
    assert missing.value.resources == ("root-switch:shared:preflight",)
    missing_record = journal.switch_record(tx_id)
    assert missing_record is not None and missing_record.state == "degraded"
    assert not missing_record.cleared


@pytest.mark.asyncio
async def test_manager_provides_root_switch_and_recovers_before_boot_open(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "plugins" / "switch_probe"
    plugin_dir.mkdir(parents=True)
    marker = tmp_path / "workspace" / "switch-state"
    (plugin_dir / "plugin.py").write_text(
        "from pathlib import Path\n"
        "from agent.plugin_composition import ROOT_SWITCH, RUNTIME_STARTED, SwitchPart\n"
        "api_version = 3\n"
        "name = 'switch_probe'\n"
        "version = '1.0.0'\n"
        "inject = (ROOT_SWITCH,)\n"
        "async def _write(value):\n"
        f"    path = Path({str(marker)!r})\n"
        "    prior = path.read_text(encoding='utf-8') if path.exists() else ''\n"
        "    path.write_text(prior + value + '\\n', encoding='utf-8')\n"
        "async def _recover(_snapshot, active):\n"
        "    await _write('active' if active else 'inactive')\n"
        "async def _started(_event):\n"
        "    await _write('runtime')\n"
        "async def apply(ctx, config):\n"
        "    await ctx.on(RUNTIME_STARTED, _started)\n"
        "    part = SwitchPart(\n"
        "        name='shared', stop=lambda _snapshot: _write('stopped'),\n"
        "        leave=lambda _snapshot: _write('left'), enter=lambda _snapshot: _write('entered'),\n"
        "        start=lambda _snapshot: _write('started'), recover=_recover,\n"
        "    )\n"
        "    await ctx.require(ROOT_SWITCH).add(ctx, part)\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )

    await manager.load_all()

    snapshot = manager.current_snapshot
    assert snapshot is not None and snapshot.accepting_leases
    assert snapshot.switch_parts is not None
    assert tuple(snapshot.switch_parts) == ("shared",)
    assert marker.read_text(encoding="utf-8").splitlines() == ["active"]
    await manager._start_runtime_snapshot(snapshot)
    assert marker.read_text(encoding="utf-8").splitlines() == [
        "active",
        "runtime",
    ]
    assert manager.reload_journal.pending_switches() == ()


@pytest.mark.asyncio
async def test_first_recover_has_a_durable_boot_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_dir = tmp_path / "plugins" / "switch_probe"
    plugin_dir.mkdir(parents=True)
    marker = tmp_path / "switch-state"
    (plugin_dir / "plugin.py").write_text(
        "from pathlib import Path\n"
        "from agent.plugin_composition import ROOT_SWITCH, SwitchPart\n"
        "api_version = 3\nname = 'switch_probe'\nversion = '1'\n"
        "inject = (ROOT_SWITCH,)\n"
        "async def _none(_snapshot): pass\n"
        "async def _recover(_snapshot, active):\n"
        f"    path = Path({str(marker)!r})\n"
        "    prior = path.read_text(encoding='utf-8') if path.exists() else ''\n"
        "    path.write_text(prior + ('active' if active else 'inactive') + '\\n', encoding='utf-8')\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(ROOT_SWITCH).add(ctx, SwitchPart(\n"
        "        name='shared', stop=_none, leave=_none, enter=_none,\n"
        "        start=_none, recover=_recover,\n"
        "    ))\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[plugin_dir.parent],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    recover_parts = manager._recover_parts

    async def crash_after_recover(snapshot, active: bool) -> None:
        await recover_parts(snapshot, active)
        if active:
            records = manager.reload_journal.pending_switches()
            assert len(records) == 1
            assert not records[0].use_new
            assert manager._switch_run.pins()
            raise RuntimeError("crash after recover")

    monkeypatch.setattr(manager, "_recover_parts", crash_after_recover)

    with pytest.raises(RuntimeError, match="crash after recover"):
        await manager.load_all()
    assert marker.read_text(encoding="utf-8").splitlines() == ["active"]
    failed = manager.reload_journal.pending_switches()
    assert len(failed) == 1 and failed[0].state == "degraded"

    shutil.rmtree(plugin_dir)
    reopened = PluginManager(
        plugin_dirs=[plugin_dir.parent],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await reopened.load_all()

    assert marker.read_text(encoding="utf-8").splitlines() == [
        "active",
        "inactive",
    ]
    assert reopened.reload_journal.pending_switches() == ()
    choices = reopened._switch_run.choices()
    assert len(choices) == 1 and choices[0].ref is None
    assert reopened.current_snapshot is not None
    assert reopened.current_snapshot.switch_parts is None


@pytest.mark.asyncio
async def test_plugin_cannot_add_two_switch_parts(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins" / "owner"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "from agent.plugin_composition import ROOT_SWITCH, SwitchPart\n"
        "api_version = 3\nname = 'owner'\nversion = '1'\n"
        "inject = (ROOT_SWITCH,)\n"
        "async def _none(_snapshot): pass\n"
        "async def _recover(_snapshot, _active): pass\n"
        "async def apply(ctx, config):\n"
        "    switch = ctx.require(ROOT_SWITCH)\n"
        "    for name in ('first', 'second'):\n"
        "        await switch.add(ctx, SwitchPart(\n"
        "            name=name, stop=_none, leave=_none, enter=_none,\n"
        "            start=_none, recover=_recover,\n"
        "        ))\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[plugin_dir.parent],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )

    await manager.load_all()

    assert manager.current_snapshot is None
    assert manager.active_plugins() == []
    assert manager.reload_journal.switch_choices() == ()


@pytest.mark.asyncio
async def test_manager_switch_waits_changed_generation_and_uses_one_pointer(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    old_owner = _Owner("old", events, active=True)
    new_owner = _Owner("new", events, active=False)
    value_key = ServiceKey[str]("test.value")

    async def build(label: str, owner: _Owner):
        root = CompositionRoot(f"root-{label}")
        switch = RootSwitch(root.instance_token)
        _ = await root.context.provide(ROOT_SWITCH, switch)
        _ = await root.context.provide(value_key, label)
        plugin_dir = tmp_path / f"artifact-{label}"
        generation = _generation("plugin", label, plugin_dir)
        runtime = PluginRuntime(
            plugin_id="plugin",
            generation_id=label,
            plugin_dir=plugin_dir,
            data_dir=generation.data_dir,
            workspace=tmp_path / "workspace",
            config={},
        )

        async def apply(ctx) -> None:
            _ = await ctx.require(ROOT_SWITCH).add(ctx, owner.part())

        _ = await root.mount(apply, name="plugin", runtime=runtime)
        snapshot = RuntimeSnapshotCompiler().compile(
            {"plugin": generation},
            composition_root=root,
            snapshot_revision=label,
        )
        return generation, snapshot

    old_generation, old_snapshot = await build("old", old_owner)
    _, new_snapshot = await build("new", new_owner)
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    manager.snapshot_store.install(old_snapshot)
    lease = await manager.snapshot_store.acquire()
    holds = _HoldRun(manager.snapshot_store, manager.reload_journal)
    holder = holds.bind(value_key, _hold_key("source:root-switch"))
    token = bind_runtime_snapshot(lease)
    try:
        hold_id = await holder.reserve()
    finally:
        reset_runtime_snapshot(token)
    await holder.activate(hold_id)
    transaction = manager.snapshot_store.begin_publish(new_snapshot)
    publish = asyncio.create_task(
        manager._commit_snapshot_with_publication_participants(
            transaction,
            old_commands=(),
            new_commands=(),
            promote_latest=False,
        )
    )
    await asyncio.sleep(0)
    assert not publish.done()
    assert old_generation.lease_count == 1
    assert old_generation.hold_count == 1

    await lease.release()
    await asyncio.sleep(0)
    assert not publish.done()
    await holder.drop(hold_id)
    await publish

    assert manager.current_snapshot is new_snapshot
    assert new_snapshot.accepting_leases
    assert events == [
        "old.stop",
        "old.leave",
        "new.enter",
        "new.start",
    ]
    assert manager.reload_journal.pending_switches() == ()


@pytest.mark.asyncio
async def test_open_cancel_keeps_provisional_closed(tmp_path: Path) -> None:
    """Cancellation while waiting for the state lock must change no state."""

    compiler = RuntimeSnapshotCompiler()
    old = compiler.compile(
        {"plugin": _generation("plugin", "old", tmp_path / "old")},
        snapshot_revision="old",
    )
    new = compiler.compile(
        {"plugin": _generation("plugin", "new", tmp_path / "new")},
        snapshot_revision="new",
    )
    store = RuntimeSnapshotStore()
    store.install(old)
    transaction = store.begin_publish(new)
    await store.commit_provisional(transaction)
    await store.select_provisional(transaction)
    await store._condition.acquire()
    opening = asyncio.create_task(store.open_provisional(transaction))
    try:
        await asyncio.sleep(0)
        opening.cancel()
        with pytest.raises(asyncio.CancelledError):
            await opening
    finally:
        store._condition.release()

    assert store.current is new
    assert not new.accepting_leases
    assert store._provisional is transaction
    await store.rollback_provisional(transaction, keep_candidate_latest=False)
    assert store.current is old
    assert old.accepting_leases


@pytest.mark.asyncio
async def test_installed_plugin_cannot_drop_its_switch_part(tmp_path: Path) -> None:
    owner = _Owner("old", [], active=True)
    generation = _generation("plugin", "old", tmp_path / "plugin-old")
    candidate = _generation("plugin", "new", tmp_path / "plugin-new")
    compiler = RuntimeSnapshotCompiler()
    old_snapshot = compiler.compile({}, snapshot_revision="old")
    old_snapshot.generations = {"plugin": generation}
    old_snapshot.switch_parts = _parts(
        _entry(
            owner,
            plugin="plugin",
            generation="old",
            artifact=_artifact(tmp_path, "old-artifact"),
        )
    )
    new_snapshot = compiler.compile({}, snapshot_revision="new")
    new_snapshot.generations = {"plugin": candidate}
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    manager.snapshot_store.install(old_snapshot)
    transaction = manager.snapshot_store.begin_publish(new_snapshot)

    with pytest.raises(RuntimeError, match="不能移除 RootSwitch part"):
        await manager._commit_snapshot_with_publication_participants(
            transaction,
            old_commands=(),
            new_commands=(),
            promote_latest=False,
        )


@pytest.mark.asyncio
async def test_part_transfer_requires_owner_remove_and_install(tmp_path: Path) -> None:
    old_a = _generation("a", "a-old", tmp_path / "a-old")
    old_b = _generation("b", "b-old", tmp_path / "b-old")
    new_a = _generation("a", "a-new", tmp_path / "a-new")
    new_b = _generation("b", "b-new", tmp_path / "b-new")
    compiler = RuntimeSnapshotCompiler()
    old_snapshot = compiler.compile({}, snapshot_revision="old")
    old_snapshot.generations = {"a": old_a, "b": old_b}
    old_snapshot.switch_parts = _parts(
        _entry(
            _Owner("a", [], active=True),
            plugin="a",
            generation="a-old",
            artifact=_artifact(tmp_path, "a-part"),
        )
    )
    new_snapshot = compiler.compile({}, snapshot_revision="new")
    new_snapshot.generations = {"a": new_a, "b": new_b}
    new_snapshot.switch_parts = _parts(
        _entry(
            _Owner("b", [], active=False),
            plugin="b",
            generation="b-new",
            artifact=_artifact(tmp_path, "b-part"),
        )
    )
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    manager.snapshot_store.install(old_snapshot)

    with pytest.raises(RuntimeError, match="同时移除旧 owner"):
        await manager._commit_snapshot_with_publication_participants(
            manager.snapshot_store.begin_publish(new_snapshot),
            old_commands=(),
            new_commands=(),
            promote_latest=False,
        )


@pytest.mark.asyncio
async def test_activity_change_refuses_root_switch(tmp_path: Path) -> None:
    """Keep two commit owners apart until Activity becomes a SwitchPart."""

    class _FailOpen:
        name = "probe"

        def __init__(self) -> None:
            self.fail = False

        def prepare_components(self, _tx, lease, catalog):
            assert lease.active
            assert isinstance(catalog, ActivityCatalog)
            return lease.snapshot.snapshot_id

        def discard_plan(self, _tx, _plan) -> None:
            return None

        async def stop_components(self, _tx, _binding) -> None:
            return None

        async def materialize_closed(self, _tx, plan):
            return plan

        def finalize_components(self, _tx, _binding) -> None:
            return None

        async def open_components(self, _tx, _binding) -> None:
            if self.fail:
                raise RuntimeError("activity open failed")

        def pause_components(self, _binding) -> None:
            return None

        async def restore_components(self, _tx, _binding) -> None:
            return None

        async def close_components(self, _tx, _binding) -> None:
            return None

    async def build(label: str, owner: _Owner):
        root = CompositionRoot(f"root-{label}")
        _ = await root.context.provide(
            BACKGROUND_JOBS,
            PluginBackgroundJobs(root.instance_token),
        )
        switch = RootSwitch(root.instance_token)
        _ = await root.context.provide(ROOT_SWITCH, switch)
        plugin_dir = tmp_path / f"artifact-{label}"
        generation = _generation("plugin", label, plugin_dir)
        runtime = PluginRuntime(
            plugin_id="plugin",
            generation_id=label,
            plugin_dir=plugin_dir,
            data_dir=generation.data_dir,
            workspace=tmp_path / "workspace",
            config={},
        )

        async def apply(ctx) -> None:
            _ = await ctx.require(ROOT_SWITCH).add(ctx, owner.part())

        _ = await root.mount(apply, name="plugin", runtime=runtime)
        return RuntimeSnapshotCompiler().compile(
            {"plugin": generation},
            composition_root=root,
            snapshot_revision=label,
        )

    old_owner = _Owner("old", [], active=True)
    new_owner = _Owner("new", [], active=False)
    old_snapshot = await build("old", old_owner)
    new_snapshot = await build("new", new_owner)
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    child = _FailOpen()
    host = ActivityHost((child,))
    manager.bind_activity_host(host)
    manager.snapshot_store.install(old_snapshot)
    first = await host.prepare_transaction(
        manager.snapshot_store.lease(old_snapshot.snapshot_id)
    )
    await host.pause_and_drain(first)
    _ = await host.materialize_closed(first)
    host.finalize(first)
    await host.open(first)
    child.fail = True

    transaction = manager.snapshot_store.begin_publish(new_snapshot)
    with pytest.raises(RuntimeError, match="不能与 RootSwitch 同批"):
        await manager._commit_snapshot_with_publication_participants(
            transaction,
            old_commands=(),
            new_commands=(),
            promote_latest=False,
        )

    assert manager.reload_journal.pending_switches() == ()
    assert manager.current_snapshot is old_snapshot
    assert old_snapshot.accepting_leases
    assert old_owner.active and old_owner.present
    assert not new_owner.active and not new_owner.present
    await host.close()


@pytest.mark.asyncio
async def test_manager_retry_requires_restart_for_root_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    recovery = SimpleNamespace(
        plugin_id="plugin",
        action="retry_runtime_recovery",
        failure_resource="activity-publication,root-switch:plugin-generation",
    )
    monkeypatch.setattr(
        manager.reload_journal,
        "pending_recovery",
        lambda: (recovery,),
    )
    with pytest.raises(RuntimeError, match="需要重启恢复"):
        await manager.retry_runtime_recovery("plugin")


@pytest.mark.asyncio
async def test_boot_loads_fresh_recovery_closures_from_builtin_pins(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    log = tmp_path / "recover.log"
    old_ref = _builtin_artifact(tmp_path / "sources", "old", log)
    new_ref = _builtin_artifact(tmp_path / "sources", "new", log)
    old_owner = _Owner("live-old", [], active=True)
    new_owner = _Owner("live-new", [], active=False)
    old_entry = _entry(
        old_owner,
        plugin="plugin",
        generation="old-generation",
        artifact=old_ref,
    )
    new_entry = _entry(
        new_owner,
        plugin="plugin",
        generation="new-generation",
        artifact=new_ref,
    )
    journal = ReloadJournal(workspace)
    work = _SwitchRun(journal).prepare(
        "crash-switch",
        old_snapshot="old-snapshot",
        old_parts=_parts(old_entry),
        new_snapshot="new-snapshot",
        new_parts=_parts(new_entry),
    )
    assert work is not None
    first = work._steps[0]
    await getattr(first.entry.part, first.action)(first.snapshot)
    journal.advance_switch("crash-switch", 1)
    shutil.rmtree(tmp_path / "sources")

    reopened = ReloadJournal(workspace)
    pinned = _SwitchRun(reopened).pins()
    assert len(pinned) == 2
    assert all(json.loads(item)["source_type"] == "builtin" for item in pinned)
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )

    await manager._recover_switches()

    assert not log.exists()
    await manager._recover_inactive_parts()
    target = manager._boot_switches[0]
    selected = tuple(
        ref
        for move in target.moves
        for ref in (move.new if target.use_new else move.old,)
        if ref is not None
    )
    entries, root = await manager._load_parts(selected)
    try:
        assert target.snapshot is not None
        for ref in selected:
            await entries[ref].part.recover(target.snapshot, True)
    finally:
        await root.dispose()
    assert log.read_text(encoding="utf-8").splitlines() == [
        "new:False",
        "old:True",
    ]
    record = reopened.switch_record("crash-switch")
    assert record is not None and not record.cleared and not record.use_new


@pytest.mark.asyncio
async def test_boot_callback_failure_keeps_switch_degraded(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    log = tmp_path / "recover.log"
    old_ref = _builtin_artifact(tmp_path / "sources", "old", log)
    new_ref = _builtin_artifact(
        tmp_path / "sources",
        "new",
        log,
        fail_recover=True,
    )
    old_entry = _entry(
        _Owner("old", [], active=True),
        plugin="plugin",
        generation="old-generation",
        artifact=old_ref,
    )
    new_entry = _entry(
        _Owner("new", [], active=False),
        plugin="plugin",
        generation="new-generation",
        artifact=new_ref,
    )
    journal = ReloadJournal(workspace)
    work = _SwitchRun(journal).prepare(
        "failed-recovery",
        old_snapshot="old-snapshot",
        old_parts=_parts(old_entry),
        new_snapshot="new-snapshot",
        new_parts=_parts(new_entry),
    )
    assert work is not None
    shutil.rmtree(tmp_path / "sources")
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )

    await manager._recover_switches()
    assert not log.exists()
    with pytest.raises(BaseExceptionGroup, match="inactive recovery"):
        await manager._recover_inactive_parts()
    record = journal.switch_record("failed-recovery")
    assert record is not None
    assert record.state == "degraded"
    assert record.failure_resource == "root-switch:inactive:recover"


@pytest.mark.asyncio
async def test_full_root_boot_failure_keeps_switch_degraded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    old_entry = _entry(
        _Owner("old", [], active=True),
        plugin="plugin",
        generation="old-generation",
        artifact=_artifact(tmp_path, "old"),
    )
    new_entry = _entry(
        _Owner("new", [], active=False),
        plugin="plugin",
        generation="new-generation",
        artifact=_artifact(tmp_path, "new"),
    )
    journal = ReloadJournal(workspace)
    work = _SwitchRun(journal).prepare(
        "full-root-failure",
        old_snapshot="old-snapshot",
        old_parts=_parts(old_entry),
        new_snapshot="new-snapshot",
        new_parts=_parts(new_entry),
    )
    assert work is not None
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )

    async def set_targets() -> None:
        manager._boot_switches = manager._switch_run.targets()

    async def fail_build(_mods: object) -> None:
        raise RuntimeError("full Root build failed")

    monkeypatch.setattr(manager, "_recover_switches", set_targets)
    monkeypatch.setattr(manager, "_load_stable_batch", fail_build)

    with pytest.raises(RuntimeError, match="full Root build failed"):
        await manager.load_all()
    record = journal.switch_record("full-root-failure")
    assert record is not None
    assert record.state == "degraded"
    assert record.failure_resource == "root-switch:selected:build"
    assert manager.current_snapshot is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "resource"),
    (
        ("close", "root-switch:preflight:close"),
        ("pointer", "root-switch:pointer:restore"),
    ),
)
async def test_boot_setup_failure_keeps_switch_degraded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    resource: str,
) -> None:
    workspace = tmp_path / "workspace"
    old_entry = _entry(
        _Owner("old", [], active=True),
        plugin="plugin",
        generation="old-generation",
        artifact=_artifact(tmp_path, "old"),
    )
    new_entry = _entry(
        _Owner("new", [], active=False),
        plugin="plugin",
        generation="new-generation",
        artifact=_artifact(tmp_path, "new"),
    )
    journal = ReloadJournal(workspace)
    work = _SwitchRun(journal).prepare(
        "boot-setup-failure",
        old_snapshot="old-snapshot",
        old_parts=_parts(old_entry),
        new_snapshot="new-snapshot",
        new_parts=_parts(new_entry),
    )
    assert work is not None
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    class RecoveryRoot:
        async def dispose(self) -> None:
            if mode == "close":
                raise RuntimeError("recovery Root close failed")

    async def load_part(ref: _PartRef):
        part = old_entry.part if ref.generation == "old-generation" else new_entry.part
        return _PartEntry(ref, part), cast(Any, RecoveryRoot())

    def fail_pointer(_targets: object) -> None:
        if mode == "pointer":
            raise RuntimeError("pointer restore failed")

    monkeypatch.setattr(manager, "_load_part", load_part)
    monkeypatch.setattr(manager, "_set_pointers", fail_pointer)

    if mode == "close":
        with pytest.raises(BaseExceptionGroup, match="recovery Root"):
            await manager._recover_switches()
    else:
        with pytest.raises(RuntimeError, match="pointer restore failed"):
            await manager._recover_switches()
    record = journal.switch_record("boot-setup-failure")
    assert record is not None
    assert record.state == "degraded"
    assert record.failure_resource == resource
    assert manager._boot_switches == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("use_new", "generation"),
    ((False, "old-generation"), (True, "new-generation")),
)
async def test_same_path_builtin_choice_survives_two_boots(
    tmp_path: Path,
    use_new: bool,
    generation: str,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_dir = tmp_path / "plugins" / "plugin"
    data_dir = workspace / "plugin-data" / "plugin-builtin"
    log = tmp_path / "recover.log"
    old_artifact = _write_builtin(
        plugin_dir,
        data_dir,
        label="old",
        log=log,
    )
    old_entry = _entry(
        _Owner("old-live", [], active=True),
        plugin="plugin",
        generation="old-generation",
        artifact=old_artifact,
    )
    old_parts = _pin_parts(
        _ArtifactPins(workspace, ReloadJournal(workspace)),
        _parts(old_entry),
    )
    new_artifact = _write_builtin(
        plugin_dir,
        data_dir,
        label="new",
        log=log,
    )
    new_entry = _entry(
        _Owner("new-live", [], active=False),
        plugin="plugin",
        generation="new-generation",
        artifact=new_artifact,
    )
    journal = ReloadJournal(workspace)
    work = _SwitchRun(journal).prepare(
        "same-path",
        old_snapshot="old-snapshot",
        old_parts=old_parts,
        new_snapshot="new-snapshot",
        new_parts=_parts(new_entry),
    )
    assert work is not None
    if use_new:
        await work.apply()
        work.commit()
    first = PluginManager(
        plugin_dirs=[plugin_dir.parent],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await first.load_all()
    assert first.current_snapshot is not None
    assert first.current_snapshot.generations["plugin"].generation_id == generation
    assert first.current_snapshot.generations["plugin"].config_projection == {}
    assert journal.pending_switches() == ()

    second = PluginManager(
        plugin_dirs=[plugin_dir.parent],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await second.load_all()
    selected = second.current_snapshot
    assert selected is not None
    assert selected.generations["plugin"].generation_id == generation
    assert selected.switch_parts is not None
    assert selected.switch_parts["shared"].ref.generation == generation


@pytest.mark.asyncio
async def test_manager_reuses_old_pin_after_builtin_source_changes(
    tmp_path: Path,
) -> None:
    """The live snapshot may keep its original ref after first boot save."""

    workspace = tmp_path / "workspace"
    plugin_dir = tmp_path / "plugins" / "plugin"
    data_dir = workspace / "plugin-data" / "plugin-builtin"
    _write_builtin(
        plugin_dir,
        data_dir,
        label="old",
        log=tmp_path / "recover.log",
    )
    manager = PluginManager(
        plugin_dirs=[plugin_dir.parent],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    old_snapshot = manager.current_snapshot
    assert old_snapshot is not None and old_snapshot.switch_parts is not None
    old_ref = old_snapshot.switch_parts["shared"].ref
    assert Path(json.loads(old_ref.artifact)["path"]) == plugin_dir.resolve()

    _write_builtin(
        plugin_dir,
        data_dir,
        label="new",
        log=tmp_path / "recover.log",
    )
    assert await manager.prepare_candidate("plugin") is not None
    await manager.publish_prepared("plugin")

    selected = manager.current_snapshot
    assert selected is not None and selected is not old_snapshot
    assert selected.generations["plugin"].instance.version == "new"
    assert manager.reload_journal.pending_switches() == ()


@pytest.mark.asyncio
async def test_owner_transfer_choice_survives_two_boots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugins = tmp_path / "plugins"
    old_artifact = _write_builtin(
        plugins / "old",
        workspace / "plugin-data" / "old-builtin",
        label="old",
        log=tmp_path / "recover.log",
        plugin_name="old",
    )
    new_artifact = _write_builtin(
        plugins / "new",
        workspace / "plugin-data" / "new-builtin",
        label="new",
        log=tmp_path / "recover.log",
        plugin_name="new",
    )
    run = _SwitchRun(ReloadJournal(workspace))
    work = run.prepare(
        "owner-transfer",
        old_snapshot="old-snapshot",
        old_parts=_parts(
            _entry(
                _Owner("old", [], active=True),
                plugin="old",
                generation="old-generation",
                artifact=old_artifact,
            )
        ),
        new_snapshot="new-snapshot",
        new_parts=_parts(
            _entry(
                _Owner("new", [], active=False),
                plugin="new",
                generation="new-generation",
                artifact=new_artifact,
            )
        ),
    )
    assert work is not None
    await work.apply()
    work.commit()

    for _index in range(2):
        manager = PluginManager(
            plugin_dirs=[plugins],
            event_bus=EventBus(),
            workspace=workspace,
            installed_cache_root=tmp_path / "home" / "cache",
        )
        await manager.load_all()
        snapshot = manager.current_snapshot
        assert snapshot is not None
        assert set(snapshot.generations) == {"new"}
        choice = manager._switch_run.choices()[0]
        assert choice.ref is not None and choice.ref.owner == "new"
    assert run.targets() == ()


@pytest.mark.asyncio
async def test_installed_owner_transfer_recovers_two_boots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache = tmp_path / "home" / "cache" / "market"
    old_base = cache / "old"
    new_base = cache / "new"
    old_artifact = _write_installed(
        old_base,
        "old-v1",
        workspace / "plugin-data" / "old-market",
        plugin_name="old",
        log=tmp_path / "recover.log",
    )
    new_artifact = _write_installed(
        new_base,
        "new-v1",
        workspace / "plugin-data" / "new-market",
        plugin_name="new",
        log=tmp_path / "recover.log",
    )
    _ = write_pointers(
        old_base,
        stable=ArtifactPointer(".artifacts/old-v1"),
        latest=ArtifactPointer(".artifacts/old-v1"),
    )
    _ = write_pointers(
        new_base,
        stable=ArtifactPointer(None),
        latest=ArtifactPointer(".artifacts/new-v1"),
    )
    run = _SwitchRun(ReloadJournal(workspace))
    work = run.prepare(
        "installed-transfer",
        old_snapshot="old-snapshot",
        old_parts=_parts(
            _entry(
                _Owner("old", [], active=True),
                plugin="old@market",
                generation="old-generation",
                artifact=old_artifact,
            )
        ),
        new_snapshot="new-snapshot",
        new_parts=_parts(
            _entry(
                _Owner("new", [], active=False),
                plugin="new@market",
                generation="new-generation",
                artifact=new_artifact,
            )
        ),
    )
    assert work is not None
    await work.apply()
    work.commit()

    for _index in range(2):
        manager = PluginManager(
            plugin_dirs=[],
            event_bus=EventBus(),
            workspace=workspace,
            installed_cache_root=tmp_path / "home" / "cache",
        )
        await manager.load_all()
        snapshot = manager.current_snapshot
        assert snapshot is not None
        assert set(snapshot.generations) == {"new@market"}
        assert read_pointer(old_base, "stable") == ArtifactPointer(None)
        assert read_pointer(new_base, "stable") == ArtifactPointer(
            ".artifacts/new-v1"
        )
    assert run.targets() == ()


@pytest.mark.asyncio
async def test_config_secret_stays_out_of_journal_and_old_revision_recovers(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_dir = tmp_path / "plugins" / "plugin"
    data_dir = workspace / "plugin-data" / "plugin-builtin"
    data_dir.mkdir(parents=True)
    secret = "raw-secret-value"
    config_path = data_dir / "config.local.toml"
    config_path.write_text(f'api_key = "{secret}"\n', encoding="utf-8")
    artifact = _write_builtin(
        plugin_dir,
        data_dir,
        label="new",
        log=tmp_path / "recover.log",
    )
    entry = _entry(
        _Owner("new", [], active=False),
        plugin="plugin",
        generation="new-generation",
        artifact=artifact,
    )
    journal = ReloadJournal(workspace)
    work = _SwitchRun(journal).prepare(
        "secret-switch",
        old_snapshot=None,
        old_parts=None,
        new_snapshot="new-snapshot",
        new_parts=_parts(entry),
    )
    assert work is not None
    await work.apply()
    work.commit()

    assert secret not in journal.pending_switches()[0].plan_json
    config_root = workspace / "runtime" / "artifact-pins" / "configs"
    pinned = tuple(config_root.glob("*/config.local.toml"))
    assert len(pinned) == 1
    assert pinned[0].read_text(encoding="utf-8") == f'api_key = "{secret}"\n'
    assert config_root.stat().st_mode & 0o777 == 0o700
    assert pinned[0].stat().st_mode & 0o777 == 0o600

    config_path.write_text('api_key = "changed"\n', encoding="utf-8")
    manager = PluginManager(
        plugin_dirs=[plugin_dir.parent],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()

    snapshot = manager.current_snapshot
    assert snapshot is not None
    assert snapshot.generations["plugin"].config_projection == {"api_key": secret}
    assert journal.pending_switches() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("use_new", "expected"),
    ((False, "one"), (True, "two")),
)
async def test_config_change_recovers_either_exact_side(
    tmp_path: Path,
    use_new: bool,
    expected: str,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_dir = tmp_path / "plugins" / "plugin"
    data_dir = workspace / "plugin-data" / "plugin-builtin"
    data_dir.mkdir(parents=True)
    config_path = data_dir / "config.local.toml"
    config_path.write_text('value = "one"\n', encoding="utf-8")
    old_artifact = _write_builtin(
        plugin_dir,
        data_dir,
        label="same-code",
        log=tmp_path / "recover.log",
    )
    old_entry = _entry(
        _Owner("old", [], active=True),
        plugin="plugin",
        generation="old-generation",
        artifact=old_artifact,
    )
    journal = ReloadJournal(workspace)
    run = _SwitchRun(journal)
    boot = run.prepare_boot(
        "config-boot",
        new_snapshot="old-snapshot",
        new_parts=_parts(old_entry),
    )
    assert boot is not None
    boot.commit()
    boot.finish()

    config_path.write_text('value = "two"\n', encoding="utf-8")
    new_artifact = _write_builtin(
        plugin_dir,
        data_dir,
        label="same-code",
        log=tmp_path / "recover.log",
    )
    new_entry = _entry(
        _Owner("new", [], active=False),
        plugin="plugin",
        generation="new-generation",
        artifact=new_artifact,
    )
    work = run.prepare(
        "config-change",
        old_snapshot="old-snapshot",
        old_parts=_parts(old_entry),
        new_snapshot="new-snapshot",
        new_parts=_parts(new_entry),
    )
    assert work is not None
    if use_new:
        await work.apply()
        work.commit()

    for value in ("three", "four"):
        config_path.write_text(f'value = "{value}"\n', encoding="utf-8")
        manager = PluginManager(
            plugin_dirs=[plugin_dir.parent],
            event_bus=EventBus(),
            workspace=workspace,
            installed_cache_root=tmp_path / "home" / "cache",
        )
        await manager.load_all()
        snapshot = manager.current_snapshot
        assert snapshot is not None
        assert snapshot.generations["plugin"].config_projection == {
            "value": expected
        }
    assert journal.pending_switches() == ()


@pytest.mark.asyncio
async def test_builtin_absence_choice_survives_two_boots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_dir = tmp_path / "plugins" / "plugin"
    data_dir = workspace / "plugin-data" / "plugin-builtin"
    artifact = _write_builtin(
        plugin_dir,
        data_dir,
        label="new",
        log=tmp_path / "recover.log",
    )
    entry = _entry(
        _Owner("new-live", [], active=False),
        plugin="plugin",
        generation="new-generation",
        artifact=artifact,
    )
    journal = ReloadJournal(workspace)
    work = _SwitchRun(journal).prepare(
        "install-crash",
        old_snapshot=None,
        old_parts=None,
        new_snapshot="new-snapshot",
        new_parts=_parts(entry),
    )
    assert work is not None

    for index in range(2):
        manager = PluginManager(
            plugin_dirs=[plugin_dir.parent],
            event_bus=EventBus(),
            workspace=workspace,
            installed_cache_root=tmp_path / "home" / "cache",
        )
        await manager.load_all()
        snapshot = manager.current_snapshot
        if snapshot is None:
            assert index == 1
        else:
            assert "plugin" not in snapshot.generations
            assert snapshot.switch_parts is None
        choices = _SwitchRun(ReloadJournal(workspace)).choices()
        assert len(choices) == 1 and choices[0].ref is None
    assert journal.pending_switches() == ()


def test_recovery_checks_full_part_and_snapshot_identity(tmp_path: Path) -> None:
    old = _entry(
        _Owner("old", [], active=True),
        plugin="plugin",
        generation="old-generation",
        artifact=_artifact(tmp_path, "old"),
    )
    new = _entry(
        _Owner("new", [], active=False),
        plugin="plugin",
        generation="new-generation",
        artifact=_artifact(tmp_path, "new"),
    )
    run = _SwitchRun(ReloadJournal(tmp_path / "workspace"))
    work = run.prepare(
        "exact-check",
        old_snapshot="old-snapshot",
        old_parts=_parts(old),
        new_snapshot="new-snapshot",
        new_parts=_parts(new),
    )
    assert work is not None
    target = run.targets()[0]
    with pytest.raises(RuntimeError, match="snapshot"):
        run.check_recovery(target, _parts(old), "wrong-snapshot")
    wrong = _PartEntry(
        _PartRef(
            old.ref.name,
            old.ref.owner,
            "wrong-generation",
            old.ref.artifact,
            old.ref.fiber,
            old.ref.inputs,
            old.ref.needs,
        ),
        old.part,
    )
    with pytest.raises(RuntimeError, match="stable part"):
        run.check_recovery(target, _parts(wrong), "old-snapshot")


@pytest.mark.asyncio
async def test_fresh_recovery_loads_ordinary_dependency_closure(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    plugins = tmp_path / "plugins"
    provider = plugins / "provider"
    owner = plugins / "owner"
    provider.mkdir(parents=True)
    owner.mkdir(parents=True)
    log = tmp_path / "dependency.log"
    (provider / "plugin.py").write_text(
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\nname = 'provider'\nversion = '1'\ninject = ()\n"
        "VALUE = ServiceKey('test.value')\n"
        "async def apply(ctx, config):\n"
        "    await ctx.provide(VALUE, 'ready')\n",
        encoding="utf-8",
    )
    (owner / "plugin.py").write_text(
        "from pathlib import Path\n"
        "from agent.plugin_composition import ROOT_SWITCH, ServiceKey, SwitchPart\n"
        "api_version = 3\nname = 'owner'\nversion = '1'\n"
        "VALUE = ServiceKey('test.value')\ninject = (ROOT_SWITCH,)\n"
        "async def _none(_snapshot): pass\n"
        "async def child(ctx):\n"
        "    value = ctx.require(VALUE)\n"
        "    async def recover(_snapshot, active):\n"
        f"        Path({str(log)!r}).write_text(value + ':' + str(active), encoding='utf-8')\n"
        "    await ctx.require(ROOT_SWITCH).add(ctx, SwitchPart(\n"
        "        name='shared', stop=_none, leave=_none, enter=_none,\n"
        "        start=_none, recover=recover,\n"
        "    ))\n"
        "async def apply(ctx, config):\n"
        "    await ctx.mount(child, name='switch-child', inject=(ROOT_SWITCH, VALUE))\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[plugins],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    snapshot = manager.current_snapshot
    assert snapshot is not None and snapshot.switch_parts is not None
    ref = manager._switch_run.choices()[0].ref
    assert ref is not None
    assert tuple(need.owner for need in ref.needs) == ("provider",)
    assert all(
        Path(json.loads(need.artifact)["path"]).is_relative_to(
            workspace / "runtime" / "artifact-pins" / "artifacts"
        )
        for need in ref.needs
    )

    entry, root = await manager._load_part(ref)
    try:
        await entry.part.recover("recovery-snapshot", True)
    finally:
        await root.dispose()
    assert log.read_text(encoding="utf-8") == "ready:True"


def test_uninstall_cannot_delete_a_pinned_installed_artifact(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_base = tmp_path / "home" / "cache" / "market" / "probe"
    artifact_dir = plugin_base / ".artifacts" / "probe-v1"
    artifact = _artifact(artifact_dir.parent, artifact_dir.name)
    owner = _Owner("owner", [], active=True)
    journal = ReloadJournal(workspace)
    work = _SwitchRun(journal).prepare(
        "delete-guard",
        old_snapshot="old-snapshot",
        old_parts=_parts(
            _entry(
                owner,
                plugin="probe@market",
                generation="old-generation",
                artifact=artifact,
            )
        ),
        new_snapshot="new-snapshot",
        new_parts=None,
    )
    assert work is not None

    with pytest.raises(RuntimeError, match="durable work pin"):
        finalize_uninstall_plugin(
            "probe@market",
            workspace=workspace,
            plugins_home=tmp_path / "home",
        )
    assert artifact_dir.is_dir()


def test_pointer_recovery_waits_for_the_pin_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    entry = _entry(
        _Owner("old", [], active=True),
        plugin="probe@market",
        generation="old-generation",
        artifact=_artifact(
            tmp_path / "home" / "cache" / "market" / "probe" / ".artifacts",
            "old",
        ),
    )
    run = _SwitchRun(ReloadJournal(workspace))
    work = run.prepare(
        "pointer-lock",
        old_snapshot="old-snapshot",
        old_parts=_parts(entry),
        new_snapshot="new-snapshot",
        new_parts=None,
    )
    assert work is not None
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    started = threading.Event()
    entered = threading.Event()
    errors: list[BaseException] = []

    def set_pointer(*_args) -> None:
        entered.set()

    monkeypatch.setattr(manager, "_set_pointer", set_pointer)

    def recover() -> None:
        started.set()
        try:
            manager._set_pointers(run.targets())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=recover)
    with _ArtifactPins(workspace, ReloadJournal(workspace)).lock():
        thread.start()
        assert started.wait(1)
        assert not entered.wait(0.05)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert entered.is_set()
    assert errors == []
