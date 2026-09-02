from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

import agent.plugins.manager as plugin_manager_module
from agent.plugin_composition import (
    BACKGROUND_JOBS,
    CHANNELS,
    AttachmentKind,
    ChannelCapability,
    ChannelDefinition,
    CompositionOverlay,
    CompositionRoot,
    CredentialRef,
    InboundIdentity,
    PluginBackgroundJobs,
    PluginChannels,
    PluginRuntime,
    ProviderClientFactory,
    ServiceView,
)
from agent.plugins.composable import ComposablePlugin
from agent.plugins.artifacts import ArtifactPointer, read_pointer, write_pointers
from agent.plugins.dashboard_host import DashboardBinding, PluginDashboardHost
from agent.plugins.generation import PluginGeneration
from agent.plugins.generation_activity_host import ActivityHost
from agent.plugins.manager import PluginManager
from agent.plugins.manifest import write_plugin_manifest
from agent.plugins.skill_links import PluginSkillLinker
from agent.plugins.snapshot import (
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.tools.message_push import MessagePushTool
from bootstrap.tools import _dispatch_v3_channel_push
from bus.event_bus import EventBus
from bus.queue import MessageBus
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from session.store import SessionStore


def _write_plugin(root: Path, name: str, source: str) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_dir


def _manager(tmp_path: Path) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


def _active_channel_generation(manager: PluginManager):
    snapshot = manager.current_snapshot
    return (
        None
        if snapshot is None
        else manager.channel_generation_host.get(snapshot.snapshot_id)
    )


@pytest.mark.asyncio
async def test_installed_plugin_without_static_manifest_fails_before_import(
    tmp_path: Path,
) -> None:
    """在任何插件代码或正式数据写入前拒绝无 manifest 的 installed artifact。"""

    # 1. 构造缺少静态 admission manifest 的旧 installed artifact。
    plugin_base = tmp_path / "home" / "cache" / "lab" / "missing_manifest"
    plugin_dir = plugin_base / ".artifacts" / "1.0.0-test"
    plugin_dir.mkdir(parents=True)
    import_marker = plugin_dir / "imported"
    (plugin_dir / "plugin.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(import_marker)!r}).write_text('imported')\n"
        "api_version = 3\n"
        "name = 'missing_manifest'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    (plugin_base / ".pointers.json").write_text(
        '{"stable":".artifacts/1.0.0-test",' '"latest":".artifacts/1.0.0-test"}\n',
        encoding="utf-8",
    )
    manager = _manager(tmp_path)

    # 2. Admission 必须在 import、generation 和正式 data root 之前失败。
    with pytest.raises(ValueError, match="缺少静态 manifest"):
        await manager.load_all()
    assert not import_marker.exists()
    assert manager.current_snapshot is None
    assert manager.generation("missing_manifest@lab") is None
    assert not (
        tmp_path / "workspace" / "plugin-data" / "missing_manifest-lab"
    ).exists()


@pytest.mark.asyncio
async def test_replace_snapshot_payload_rebinds_all_exact_root_activity_catalogs() -> (
    None
):
    """让全部 activity catalog 随载荷一起切换到正式 Root。"""

    async def compile_snapshot(label: str):
        # 1. 每棵 Root 独立拥有 background-job activity catalog。
        root = CompositionRoot(label)
        _ = await root.context.provide(
            BACKGROUND_JOBS,
            PluginBackgroundJobs(root.instance_token),
        )
        snapshot = RuntimeSnapshotCompiler().compile(
            {},
            composition_root=root,
        )
        return root, snapshot

    validation_root, target = await compile_snapshot("activity:validation")
    formal_root, source = await compile_snapshot("activity:formal")
    try:
        # 2. identity 值应等价，但对象保持可区分，确保六个字段都真实替换。
        identity_fields = ("background_job_catalog_identity",)
        for name in identity_fields:
            target_identity = getattr(target, name)
            source_identity = getattr(source, name)
            assert isinstance(target_identity, str)
            assert target_identity == source_identity
            distinct_source_identity = (source_identity + "#")[:-1]
            assert distinct_source_identity == source_identity
            assert distinct_source_identity is not target_identity
            setattr(source, name, distinct_source_identity)

        old_catalogs = (target.background_job_catalog,)
        assert all(catalog is not None for catalog in old_catalogs)
        target.state = "validating"

        plugin_manager_module._replace_snapshot_payload(  # pyright: ignore[reportPrivateUsage]
            target,
            source,
        )

        # 3. catalog、identity 与 Root 必须来自同一份 formal snapshot。
        catalog_fields = ("background_job_catalog",)
        for name, old_catalog in zip(catalog_fields, old_catalogs, strict=True):
            catalog = getattr(target, name)
            assert catalog is getattr(source, name)
            assert catalog is not old_catalog
            assert catalog.root_instance_token is formal_root.instance_token
        for name in identity_fields:
            assert getattr(target, name) is getattr(source, name)
        assert target.composition_root is formal_root
        RuntimeSnapshotStore._validate_composition(  # pyright: ignore[reportPrivateUsage]
            target
        )
    finally:
        await validation_root.dispose()
        await formal_root.dispose()


def _channel_plugin_source(
    version: str,
    *,
    fail_start: bool = False,
    fail_stop: bool = False,
    block_deliver: bool = False,
) -> str:
    delivery_body = (
        "            DELIVERY_ENTERED.set()\n"
        "            await asyncio.Event().wait()\n"
        if block_deliver
        else "            return ProviderDeliveryReceipt(request.delivery_id, DeliveryStatus.DELIVERED)\n"
    )
    return (
        "import asyncio\n"
        "from pydantic import AliasChoices, BaseModel, Field\n"
        "from agent.plugin_composition import (\n"
        "    CHANNELS, ChannelCapability, ChannelDefinition, ChannelReady, CredentialRef,\n"
        "    DeliveryStatus, InboundIdentity, ProviderDeliveryReceipt, StopReceipt,\n"
        ")\n"
        "api_version = 3\n"
        "name = 'channel_probe'\n"
        f"version = {version!r}\n"
        "DELIVERY_ENTERED = asyncio.Event()\n"
        "inject = (CHANNELS,)\n"
        "class Config(BaseModel):\n"
        "    app_id: str\n"
        "    app_secret: CredentialRef = Field(\n"
        "        validation_alias=AliasChoices('app_secret', 'appSecret'),\n"
        "    )\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(CHANNELS).register(ctx, ChannelDefinition(\n"
        "        name='feishu',\n"
        "        capabilities=frozenset({ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}),\n"
        "        factory_export='build_adapter',\n"
        "        inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,\n"
        "        credential_paths=('appSecret', 'app_secret'),\n"
        "    ))\n"
        "class Adapter:\n"
        "    def __init__(self, context):\n"
        "        self.context = context\n"
        "        self.ports = None\n"
        "        self.admission_open = False\n"
        "        self._in_flight = 0\n"
        "        self._drained = asyncio.Event()\n"
        "        self._drained.set()\n"
        f"        self.fail_start = {fail_start!r}\n"
        f"        self.fail_stop = {fail_stop!r}\n"
        "    def attach_runtime(self, ports):\n"
        "        if self.admission_open: raise RuntimeError('channel admission already open')\n"
        "        if ports.binding_token != self.context.binding_token: raise RuntimeError('channel binding mismatch')\n"
        "        if ports.ingress is None: raise RuntimeError('channel ingress missing')\n"
        "        self.ports = ports\n"
        "    def open_admission(self):\n"
        "        if self.ports is None: raise RuntimeError('channel runtime not attached')\n"
        "        self.admission_open = True\n"
        "    def close_admission(self):\n"
        "        self.admission_open = False\n"
        "    async def start(self):\n"
        "        if self.fail_start: raise RuntimeError('channel start failed')\n"
        "        if self.ports is None: raise RuntimeError('channel runtime not attached')\n"
        "        return ChannelReady(self.context.binding_token)\n"
        "    async def deliver(self, request):\n"
        "        self._in_flight += 1\n"
        "        self._drained.clear()\n"
        "        try:\n" + delivery_body + "        finally:\n"
        "            self._in_flight -= 1\n"
        "            if self._in_flight == 0: self._drained.set()\n"
        "    async def stop(self):\n"
        "        self.close_admission()\n"
        "        await self._drained.wait()\n"
        "        if self.fail_stop: raise RuntimeError('channel stop failed')\n"
        "        return StopReceipt(self.context.binding_token, True)\n"
        "def build_adapter(context): return Adapter(context)\n"
    )


def _channel_static_manifest(version: str) -> str:
    return (
        "schema_version = 1\n"
        "name = 'channel_probe'\n"
        f"version = {version!r}\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n\n"
        "[channel_credentials]\n"
        "feishu = ['app_secret', 'appSecret']\n"
    )


def _write_static_v3_manifest(
    root: Path,
    name: str,
    version: str,
) -> None:
    (root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        f"name = {name!r}\n"
        f"version = {version!r}\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n",
        encoding="utf-8",
    )


def _shared_writer_source(version: str) -> str:
    return f"""\
api_version = 3
name = 'shared_writer'
version = {version!r}
import asyncio
import sqlite3
from agent.plugin_composition import RUNTIME_STARTED, RUNTIME_STOPPING
writer_task = None

async def apply(ctx, config):
    database = ctx.data_root / 'writer.sqlite3'
    root_token = id(ctx._root_instance_token())
    ctx.data_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute(
        'CREATE TABLE IF NOT EXISTS owner ('
        'slot INTEGER PRIMARY KEY CHECK (slot = 1), version TEXT NOT NULL)'
    )
    connection.execute(
        'CREATE TABLE IF NOT EXISTS trace ('
        'seq INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL)'
    )
    connection.execute(
        'CREATE TABLE IF NOT EXISTS writes ('
        'seq INTEGER PRIMARY KEY AUTOINCREMENT, version TEXT NOT NULL, '
        'root_token INTEGER NOT NULL)'
    )
    connection.commit()
    connection.close()

    async def started(_event):
        global writer_task
        connection = sqlite3.connect(database)
        connection.execute('INSERT INTO owner VALUES (1, ?)', ({version!r},))
        connection.execute('INSERT INTO trace(event) VALUES (?)', ('start:{version}',))
        connection.commit()
        connection.close()

        async def write_forever():
            connection = sqlite3.connect(database)
            try:
                while True:
                    connection.execute(
                        'INSERT INTO writes(version, root_token) VALUES (?, ?)',
                        ({version!r}, root_token),
                    )
                    connection.commit()
                    await asyncio.sleep(0)
            finally:
                connection.close()

        writer_task = asyncio.create_task(write_forever())

    async def stopping(_event):
        global writer_task
        if writer_task is not None:
            writer_task.cancel()
            try:
                await writer_task
            except asyncio.CancelledError:
                pass
            writer_task = None
        connection = sqlite3.connect(database)
        connection.execute('DELETE FROM owner WHERE version = ?', ({version!r},))
        connection.execute('INSERT INTO trace(event) VALUES (?)', ('stop:{version}',))
        connection.commit()
        connection.close()

    await ctx.on(RUNTIME_STARTED, started)
    await ctx.on(RUNTIME_STOPPING, stopping)
"""


def _sqlite_scalar(database: Path, query: str) -> object:
    connection = sqlite3.connect(database)
    try:
        return connection.execute(query).fetchone()[0]
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_candidate_uses_isolated_data_copy(
    tmp_path: Path,
) -> None:
    _ = _write_plugin(
        tmp_path / "plugins",
        "isolated_reader",
        "api_version = 3\n"
        "name = 'isolated_reader'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config):\n"
        "    ctx.data_root.mkdir(parents=True, exist_ok=True)\n"
        "    (ctx.data_root / 'isolated.txt').write_text('isolated')\n",
    )
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "shared_reader",
        "api_version = 3\n"
        "name = 'shared_reader'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config):\n"
        "    import sqlite3\n"
        "    ctx.data_root.mkdir(parents=True, exist_ok=True)\n"
        "    connection = sqlite3.connect(ctx.data_root / 'state.sqlite3')\n"
        "    connection.execute('CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY)')\n"
        "    connection.execute('INSERT INTO items DEFAULT VALUES')\n"
        "    connection.commit()\n"
        "    connection.close()\n",
    )
    _write_static_v3_manifest(plugin_dir, "shared_reader", "1.0.0")
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.generation("shared_reader")
    isolated = manager.generation("isolated_reader")
    assert stable is not None and isolated is not None
    assert stable.source_type == "builtin"
    assert stable.static_manifest is not None
    database = stable.data_dir / "state.sqlite3"
    sparse = stable.data_dir / "large.sparse"
    with sparse.open("wb") as stream:
        stream.truncate(1024 * 1024)
    formal_inode = database.stat().st_ino
    formal_digest = hashlib.sha256(database.read_bytes()).hexdigest()
    proactive = tmp_path / "workspace" / "proactive.db"
    wake_proactive = tmp_path / "workspace" / "wake_proactive.db"
    proactive.write_bytes(b"legacy proactive island")
    wake_proactive.write_bytes(b"legacy wake island")
    proactive_inode = proactive.stat().st_ino
    wake_proactive_inode = wake_proactive.stat().st_ino

    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'shared_reader'\n"
        "version = '2.0.0'\n"
        "async def apply(ctx, config):\n"
        "    import json, sqlite3\n"
        "    (ctx.data_root / 'candidate-marker').write_text('candidate')\n"
        "    database = ctx.data_root / 'state.sqlite3'\n"
        "    connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)\n"
        "    rows = connection.execute('SELECT COUNT(*) FROM items').fetchone()[0]\n"
        "    rejected = False\n"
        "    try:\n"
        "        connection.execute('INSERT INTO items DEFAULT VALUES')\n"
        "    except sqlite3.OperationalError:\n"
        "        rejected = True\n"
        "    connection.close()\n"
        "    ctx.runtime.workspace.mkdir(parents=True, exist_ok=True)\n"
        "    (ctx.runtime.workspace / 'shared-read.json').write_text(\n"
        "        json.dumps({'rows': rows, 'write_rejected': rejected})\n"
        "    )\n",
        encoding="utf-8",
    )
    _write_static_v3_manifest(plugin_dir, "shared_reader", "2.0.0")

    candidate = await manager.prepare_candidate("shared_reader")

    assert candidate is not None and candidate.runtime_snapshot is not None
    assert candidate.validation_workspace is not None
    assert "state.sqlite3" in candidate.validation_data_inventory
    assert "large.sparse" in candidate.validation_data_inventory
    candidate_root = candidate.runtime_snapshot.composition_root
    assert candidate_root is not None
    candidate_fibers = {
        fiber.name: fiber for fiber in candidate_root.root_fiber.children
    }
    candidate_runtime = candidate_fibers["shared_reader"].runtime
    assert candidate_runtime is not None
    assert candidate_runtime.data_dir != stable.data_dir
    assert "isolated_reader" not in candidate_fibers
    assert (isolated.data_dir / "isolated.txt").read_text() == "isolated"
    validation_root = candidate.validation_workspace.parent
    observations = tuple(validation_root.rglob("shared-read.json"))
    assert len(observations) == 1
    assert json.loads(observations[0].read_text(encoding="utf-8")) == {
        "rows": 1,
        "write_rejected": True,
    }
    assert len(tuple(validation_root.rglob("state.sqlite3"))) == 1
    assert len(tuple(validation_root.rglob("large.sparse"))) == 1
    assert len(tuple(validation_root.rglob("candidate-marker"))) == 1
    assert not tuple(validation_root.rglob("proactive.db"))
    assert not tuple(validation_root.rglob("wake_proactive.db"))
    assert not (stable.data_dir / "candidate-marker").exists()
    assert database.stat().st_ino == formal_inode
    assert hashlib.sha256(database.read_bytes()).hexdigest() == formal_digest
    assert proactive.stat().st_ino == proactive_inode
    assert wake_proactive.stat().st_ino == wake_proactive_inode
    formal = sqlite3.connect(database)
    try:
        assert formal.execute("SELECT COUNT(*) FROM items").fetchone() == (1,)
    finally:
        formal.close()

    await manager.discard_prepared("shared_reader")
    assert not validation_root.exists()
    assert database.is_file() and sparse.is_file()
    assert proactive.is_file() and wake_proactive.is_file()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_unrelated_candidate_does_not_mount_or_copy_stateful_plugin(
    tmp_path: Path,
) -> None:
    stateful_dir = _write_plugin(
        tmp_path / "plugins",
        "stateful",
        "api_version = 3\n"
        "name = 'stateful'\n"
        "version = '1.0.0'\n"
        "workspace_roots = ('memory',)\n"
        "workspace_files = ('sessions.db',)\n"
        "async def apply(ctx, config):\n"
        "    ctx.data_root.mkdir(parents=True, exist_ok=True)\n"
        "    marker = ctx.data_root / 'mount-count'\n"
        "    count = int(marker.read_text()) if marker.exists() else 0\n"
        "    marker.write_text(str(count + 1))\n"
        "    writer = ctx.data_root / 'exclusive-writer'\n"
        "    if writer.exists():\n"
        "        raise RuntimeError('stateful writer already mounted')\n"
        "    writer.write_text(ctx.runtime.generation_id)\n"
        "    def cleanup():\n"
        "        writer.unlink()\n"
        "    await ctx.effect(lambda: cleanup, label='exclusive-writer')\n",
    )
    candidate_dir = _write_plugin(
        tmp_path / "plugins",
        "candidate_only",
        "api_version = 3\n"
        "name = 'candidate_only'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config): pass\n",
    )
    _ = stateful_dir
    workspace = tmp_path / "workspace"
    (workspace / "memory").mkdir(parents=True)
    (workspace / "memory" / "large-index").write_bytes(b"do-not-copy")
    (workspace / "sessions.db").write_bytes(b"do-not-copy")
    manager = _manager(tmp_path)
    await manager.load_all()
    stateful = manager.generation("stateful")
    assert stateful is not None
    marker = stateful.data_dir / "mount-count"
    writer = stateful.data_dir / "exclusive-writer"
    assert marker.read_text() == "1"
    assert writer.is_file()

    with (candidate_dir / "plugin.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# candidate revision\n")
    candidate = await manager.prepare_candidate("candidate_only")

    assert candidate is not None and candidate.runtime_snapshot is not None
    root = candidate.runtime_snapshot.composition_root
    assert root is not None
    assert [fiber.name for fiber in root.root_fiber.children] == ["candidate_only"]
    assert marker.read_text() == "1"
    assert writer.is_file()
    validation_root = candidate.validation_workspace
    assert validation_root is not None
    assert not (validation_root / "memory").exists()
    assert not (validation_root / "sessions.db").exists()
    attempt_root = validation_root.parent

    result = await manager.publish_prepared("candidate_only")

    assert result["publication_state"] == "committed"
    assert marker.read_text() == "2"
    assert writer.is_file()
    assert not attempt_root.exists()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_candidate_with_unknown_service_is_rejected_before_latest(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "invalid_dependency",
        "api_version = 3\n"
        "name = 'invalid_dependency'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config): pass\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.current_snapshot
    plugin_dir.joinpath("plugin.py").write_text(
        "from agent.plugin_composition import ServiceKey\n"
        "MISSING = ServiceKey('missing.service.v1')\n"
        "api_version = 3\n"
        "name = 'invalid_dependency'\n"
        "version = '2.0.0'\n"
        "inject = (MISSING,)\n"
        "async def apply(ctx, config): raise AssertionError('must stay pending')\n",
        encoding="utf-8",
    )

    candidate = await manager.prepare_candidate("invalid_dependency")

    assert candidate is None
    assert manager.current_snapshot is stable
    assert manager.prepared_generation("invalid_dependency") is None
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_candidate_cannot_remove_service_required_by_stable_plugin(
    tmp_path: Path,
) -> None:
    provider_dir = _write_plugin(
        tmp_path / "plugins",
        "provider",
        "from agent.plugin_composition import ServiceKey\n"
        "SHARED = ServiceKey('fixture.shared.v1')\n"
        "api_version = 3\n"
        "name = 'provider'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config): await ctx.provide(SHARED, object())\n",
    )
    _write_plugin(
        tmp_path / "plugins",
        "consumer",
        "from agent.plugin_composition import ServiceKey\n"
        "SHARED = ServiceKey('fixture.shared.v1')\n"
        "api_version = 3\n"
        "name = 'consumer'\n"
        "version = '1.0.0'\n"
        "inject = (SHARED,)\n"
        "async def apply(ctx, config): ctx.require(SHARED)\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.current_snapshot
    provider_dir.joinpath("plugin.py").write_text(
        "api_version = 3\n"
        "name = 'provider'\n"
        "version = '2.0.0'\n"
        "async def apply(ctx, config): pass\n",
        encoding="utf-8",
    )

    candidate = await manager.prepare_candidate("provider")

    assert candidate is None
    assert manager.current_snapshot is stable
    await manager.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_commit_fails", [False, True])
async def test_isolated_candidate_publish_drains_old_writer_before_new_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_commit_fails: bool,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "shared_writer",
        _shared_writer_source("v1"),
    )
    _write_static_v3_manifest(plugin_dir, "shared_writer", "v1")
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.generation("shared_writer")
    stable_snapshot = manager.current_snapshot
    assert stable is not None and stable_snapshot is not None
    assert stable_snapshot.composition_root is not None
    old_root_token = id(stable_snapshot.composition_root.instance_token)
    database = stable.data_dir / "writer.sqlite3"
    runtime_services = asyncio.create_task(manager.run_runtime_services())
    while stable.instance.module.writer_task is None:
        await asyncio.sleep(0)

    (plugin_dir / "plugin.py").write_text(
        _shared_writer_source("v2"),
        encoding="utf-8",
    )
    _write_static_v3_manifest(plugin_dir, "shared_writer", "v2")
    candidate = await manager.prepare_candidate("shared_writer")
    assert candidate is not None
    assert candidate.instance.module.writer_task is None
    assert candidate.runtime_snapshot is not None
    candidate_root = candidate.runtime_snapshot.composition_root
    assert candidate_root is not None
    if owner_commit_fails:

        def fail_owner_commit(*_args: object) -> None:
            raise RuntimeError("candidate owner commit failed")

        monkeypatch.setattr(
            manager,
            "_activate_published_generation",
            fail_owner_commit,
        )

    # 1. An accepted old Turn keeps the old writer alive while publication waits.
    old_lease = await manager.snapshot_store.acquire()
    publication = asyncio.create_task(manager.publish_prepared("shared_writer"))
    while stable_snapshot.accepting_leases:
        await asyncio.sleep(0)
    before_wait = cast(
        int,
        _sqlite_scalar(
            database,
            "SELECT COUNT(*) FROM writes "
            f"WHERE version = 'v1' AND root_token = {old_root_token}",
        ),
    )
    for _ in range(20):
        await asyncio.sleep(0)
    during_wait = cast(
        int,
        _sqlite_scalar(
            database,
            "SELECT COUNT(*) FROM writes "
            f"WHERE version = 'v1' AND root_token = {old_root_token}",
        ),
    )
    assert during_wait > before_wait
    waiting_admission = asyncio.create_task(manager.snapshot_store.acquire())
    await asyncio.sleep(0)
    assert not publication.done()
    assert not waiting_admission.done()

    # 2. Releasing the Turn lets STOPPING settle v1 before v2 receives STARTED.
    await old_lease.release()
    if owner_commit_fails:
        with pytest.raises(RuntimeError, match="candidate owner commit failed"):
            await publication
        result = None
    else:
        result = await publication
    new_lease = await waiting_admission
    await new_lease.release()
    if result is not None:
        assert result["publication_state"] == "committed"
    connection = sqlite3.connect(database)
    trace = [
        row[0] for row in connection.execute("SELECT event FROM trace ORDER BY seq")
    ]
    owners = connection.execute("SELECT version FROM owner").fetchall()
    old_writes = connection.execute(
        "SELECT COUNT(*) FROM writes WHERE version = 'v1' AND root_token = ?",
        (old_root_token,),
    ).fetchone()[0]
    connection.close()
    if owner_commit_fails:
        assert trace == [
            "start:v1",
            "stop:v1",
            "start:v1",
        ]
        assert owners == [("v1",)]
        assert candidate.instance.module.writer_task is None
        assert stable.instance.module.writer_task is not None
    else:
        assert trace == ["start:v1", "stop:v1", "start:v2"]
        assert owners == [("v2",)]
        assert stable.instance.module.writer_task is None

    # 3. The terminal old Root cannot write again after the new writer is open.
    for _ in range(20):
        await asyncio.sleep(0)
    assert (
        _sqlite_scalar(
            database,
            "SELECT COUNT(*) FROM writes "
            f"WHERE version = 'v1' AND root_token = {old_root_token}",
        )
        == old_writes
    )
    runtime_services.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runtime_services
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_formal_rebuild_rejects_candidate_topology_drift(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "shared_writer",
        _shared_writer_source("v1"),
    )
    _write_static_v3_manifest(plugin_dir, "shared_writer", "v1")
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.generation("shared_writer")
    stable_snapshot = manager.current_snapshot
    assert stable is not None and stable_snapshot is not None
    old_root = stable_snapshot.composition_root
    assert old_root is not None
    runtime_services = asyncio.create_task(manager.run_runtime_services())
    while old_root.instance_token not in manager._runtime_started_roots:
        await asyncio.sleep(0)

    mutant = _shared_writer_source("v2").replace(
        "    await ctx.on(RUNTIME_STOPPING, stopping)\n",
        "    await ctx.on(RUNTIME_STOPPING, stopping)\n"
        "    if 'plugin-validation' not in str(ctx.data_root):\n"
        "        await ctx.on(RUNTIME_STARTED, lambda _: None)\n",
    )
    (plugin_dir / "plugin.py").write_text(mutant, encoding="utf-8")
    _write_static_v3_manifest(plugin_dir, "shared_writer", "v2")
    candidate = await manager.prepare_candidate("shared_writer")
    assert candidate is not None

    with pytest.raises(RuntimeError, match="snapshot identity"):
        await manager.publish_prepared("shared_writer")

    replacement_root = stable_snapshot.composition_root
    assert replacement_root is not None and replacement_root is not old_root
    assert manager.current_snapshot is stable_snapshot
    assert manager.generation("shared_writer") is stable
    assert stable_snapshot.accepting_leases
    assert manager.prepared_generation("shared_writer") is None
    runtime_services.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runtime_services
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_channel_registry_redacts_candidate_credentials_before_import(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "channel_probe",
        _channel_plugin_source("1.0.0"),
    )
    manifest_path = plugin_dir / "akashic.plugin.toml"
    manifest_path.write_text(_channel_static_manifest("1.0.0"), encoding="utf-8")
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    config_path = data_dir / "config.local.toml"
    secret = "candidate-must-never-read-this-secret"
    original_config = f"app_id = 'app-1'\nappSecret = '{secret}'\n".encode()
    config_path.write_bytes(original_config)
    manager = _manager(tmp_path)

    await manager.load_all()

    stable = manager.current_snapshot
    generation = manager.generation("channel_probe")
    assert stable is not None and generation is not None
    assert stable.channel_registry is not None
    assert manager.stable_channel_catalog() is stable.channel_registry
    assert stable.channel_registry.descriptors[0].credential_paths == (
        "appSecret",
        "app_secret",
    )
    assert isinstance(generation.config.app_secret, CredentialRef)  # type: ignore[union-attr]
    assert config_path.read_bytes() == original_config
    runtime = _active_channel_generation(manager)
    assert runtime is not None and runtime.snapshot_id == stable.snapshot_id
    binding = runtime.channel("feishu")
    assert binding.admission_open is True
    assert generation.reload_tx_id is not None
    record = manager.reload_journal.get(generation.reload_tx_id)
    assert record.phase == "complete"
    evidence = repr(manager.reload_journal.events(generation.reload_tx_id))
    assert "channel_binding_reserved" in evidence
    assert secret not in evidence

    other_root = CompositionRoot("other-channel-root")
    await other_root.context.provide(
        CHANNELS,
        PluginChannels(other_root.instance_token),
    )

    async def register_other(ctx) -> None:
        await ctx.require(CHANNELS).register(
            ctx,
            ChannelDefinition(
                name="feishu",
                capabilities=frozenset(
                    {ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}
                ),
                factory_export="build_adapter",
                inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
                credential_paths=("appSecret", "app_secret"),
            ),
        )

    _ = await other_root.mount(
        register_other,
        name="channel_probe",
        runtime=PluginRuntime(
            plugin_id="channel_probe",
            generation_id="test-generation",
            plugin_dir=plugin_dir,
            data_dir=data_dir,
            workspace=tmp_path / "workspace",
            config=generation.config,
        ),
        inject=(CHANNELS,),
    )
    other_snapshot = RuntimeSnapshotCompiler().compile(
        {"channel_probe": generation},
        composition_root=other_root,
    )
    assert other_snapshot.channel_registry_identity == stable.channel_registry_identity
    other_snapshot.channel_registry = stable.channel_registry
    with pytest.raises(RuntimeError, match="不属于 exact Root"):
        RuntimeSnapshotStore().install(other_snapshot)
    await other_root.dispose()

    (plugin_dir / "plugin.py").write_text(
        _channel_plugin_source("2.0.0"),
        encoding="utf-8",
    )
    manifest_path.write_text(_channel_static_manifest("2.0.0"), encoding="utf-8")
    candidate = await manager.prepare_candidate("channel_probe")
    assert candidate is not None and candidate.runtime_snapshot is not None
    assert candidate.validation_workspace is not None
    candidate_root = candidate.runtime_snapshot.composition_root
    assert candidate_root is not None
    candidate_runtime = candidate_root.root_fiber.children[0].runtime
    assert candidate_runtime is not None
    assert isinstance(candidate_runtime.config.app_secret, CredentialRef)
    assert "config.local.toml" not in candidate.validation_data_inventory
    validation_root = candidate.validation_workspace.parent
    for path in validation_root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            assert secret.encode() not in path.read_bytes()
    assert config_path.read_bytes() == original_config

    await manager.discard_prepared("channel_probe")
    assert not validation_root.exists()
    assert manager.current_snapshot is stable
    assert config_path.read_bytes() == original_config

    promoted = await manager.prepare_candidate("channel_probe")
    assert promoted is not None
    held_stable = manager.snapshot_store.lease()
    publication = asyncio.create_task(manager.publish_prepared("channel_probe"))
    await asyncio.sleep(0)
    assert not publication.done()
    assert runtime.channel("feishu").admission_open
    await held_stable.release()
    result = await publication
    assert result["publication_state"] == "committed"
    current = manager.current_snapshot
    active_runtime = _active_channel_generation(manager)
    assert current is not None and current is not stable
    assert active_runtime is not None
    assert active_runtime.snapshot_id == current.snapshot_id
    assert active_runtime.channel("feishu").admission_open
    assert config_path.read_bytes() == original_config
    await manager.terminate_all()
    assert _active_channel_generation(manager) is None


@pytest.mark.asyncio
async def test_unrelated_snapshot_publication_rebinds_exact_channel_runtime(
    tmp_path: Path,
) -> None:
    """非 Channel 插件晋升后，入站 runtime 必须绑定新的 exact snapshot。"""

    # 1. 启动一个 Channel 与一个不贡献 Channel 的普通插件
    channel_dir = _write_plugin(
        tmp_path / "plugins",
        "channel_probe",
        _channel_plugin_source("1.0.0"),
    )
    (channel_dir / "akashic.plugin.toml").write_text(
        _channel_static_manifest("1.0.0"),
        encoding="utf-8",
    )
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    (data_dir / "config.local.toml").write_text(
        "app_id = 'app-1'\napp_secret = 'secret'\n",
        encoding="utf-8",
    )
    plain_dir = _write_plugin(
        tmp_path / "plugins",
        "plain_probe",
        "api_version = 3\n"
        "name = 'plain_probe'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config): pass\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    previous = manager.current_snapshot
    previous_runtime = _active_channel_generation(manager)
    assert previous is not None and previous_runtime is not None

    # 2. 只晋升普通插件，Channel catalog identity 保持不变
    (plain_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'plain_probe'\n"
        "version = '2.0.0'\n"
        "async def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    assert await manager.prepare_candidate("plain_probe") is not None
    result = await manager.publish_prepared("plain_probe")
    current = manager.current_snapshot
    current_runtime = _active_channel_generation(manager)

    # 3. 新入站只能租用新 snapshot，旧 binding 已完成排空
    assert result["publication_state"] == "committed"
    assert current is not None and current is not previous
    assert current.channel_registry_identity == previous.channel_registry_identity
    assert current_runtime is not None and current_runtime is not previous_runtime
    assert current_runtime.snapshot_id == current.snapshot_id
    assert current_runtime.channel("feishu").admission_open
    lease = manager.snapshot_store.lease(current_runtime.snapshot_id)
    await lease.release()
    with pytest.raises(RuntimeError, match="RuntimeSnapshot 不可(用|租用)"):
        manager.snapshot_store.lease(previous_runtime.snapshot_id)

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_channel_manager_binds_core_attachment_ports(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "channel_probe",
        _channel_plugin_source("1.0.0"),
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        _channel_static_manifest("1.0.0"),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    data_dir = workspace / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    (data_dir / "config.local.toml").write_text(
        "app_id = 'app-1'\napp_secret = 'secret'\n",
        encoding="utf-8",
    )
    session_store = SessionStore(workspace / "sessions.db")
    attachment_store = ChannelAttachmentArtifactStore(
        workspace=workspace,
        session_store=session_store,
    )
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
        channel_attachment_store=attachment_store,
    )
    try:
        await manager.load_all()
        runtime = _active_channel_generation(manager)
        assert runtime is not None
        state = cast(Any, manager.channel_generation_host)._bindings[
            (runtime.snapshot_id, "feishu")
        ]
        context = state.factory_context
        assert context is not None
        assert context.attachment_import is not None
        assert context.attachment_read is not None
        adapter = state.adapter
        assert adapter is not None
        assert adapter.ports.binding_token == state.binding_token
        assert adapter.ports.ingress is context.ingress
        assert adapter.admission_open is True

        ref = await context.attachment_import.import_bytes(
            b"manager-bound attachment",
            kind=AttachmentKind.FILE,
            filename="evidence.txt",
            media_type="text/plain",
        )
        lease = await context.attachment_read.acquire(ref)
        assert await lease.read_bytes(max_bytes=1024) == b"manager-bound attachment"
        await lease.aclose()
    finally:
        await manager.terminate_all()
        session_store.close()


@pytest.mark.asyncio
async def test_v3_channel_direct_push_uses_exact_stable_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "channel_probe",
        _channel_plugin_source("1.0.0"),
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        _channel_static_manifest("1.0.0"),
        encoding="utf-8",
    )
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    (data_dir / "config.local.toml").write_text(
        "app_id = 'app-1'\napp_secret = 'secret'\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    session_store = SessionStore(workspace / "sessions.db")
    attachment_store = ChannelAttachmentArtifactStore(
        workspace=workspace,
        session_store=session_store,
    )
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
        channel_attachment_store=attachment_store,
    )
    await manager.load_all()
    bus = MessageBus()
    bus.bind_channel_outbound_dispatcher(
        manager.channel_generation_host.dispatch_outbound
    )
    dispatch_task = asyncio.create_task(bus.dispatch_outbound())
    tool = MessagePushTool(chat_lane=bus.chat_lane)
    tool.bind_v3_channel_dispatcher(
        lambda message, passive: _dispatch_v3_channel_push(
            manager,
            bus,
            message,
            passive,
            attachment_store,
        )
    )

    source = await manager.snapshot_store.acquire()
    snapshot_token = bind_runtime_snapshot(source)

    async def reject_stable_reacquire(*args: object, **kwargs: object) -> object:
        raise AssertionError("direct push 必须复用当前 exact snapshot lease")

    monkeypatch.setattr(manager.snapshot_store, "acquire", reject_stable_reacquire)

    image = tmp_path / "image.png"
    image.write_bytes(b"channel image")
    try:
        delivered = json.loads(
            await asyncio.wait_for(
                tool.execute(
                    target_channel="feishu",
                    target_chat_id="ou_1",
                    message="hello",
                ),
                timeout=1,
            )
        )
        attached = json.loads(
            await asyncio.wait_for(
                tool.execute(
                    target_channel="feishu",
                    target_chat_id="ou_1",
                    image=str(image),
                ),
                timeout=1,
            )
        )
    finally:
        reset_runtime_snapshot(snapshot_token)
        await source.release()

    assert delivered["status"] == "delivered"
    assert delivered["retryable"] is False
    assert attached["status"] == "delivered"
    assert attached["retryable"] is False
    assert len(session_store.list_attachments()) == 1
    runtime = _active_channel_generation(manager)
    assert runtime is not None
    assert runtime.channel("feishu").in_flight == 0

    await bus.aclose()
    dispatch_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch_task
    await manager.terminate_all()
    session_store.close()


@pytest.mark.asyncio
async def test_v3_channel_direct_push_bus_close_settles_unknown_and_releases_binding(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "channel_probe",
        _channel_plugin_source("1.0.0", block_deliver=True),
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        _channel_static_manifest("1.0.0"),
        encoding="utf-8",
    )
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    (data_dir / "config.local.toml").write_text(
        "app_id = 'app-1'\napp_secret = 'secret'\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    generation = manager.generation("channel_probe")
    assert generation is not None
    module = cast(ComposablePlugin, generation.instance).module
    entered = cast(asyncio.Event, module.DELIVERY_ENTERED)
    bus = MessageBus()
    bus.bind_channel_outbound_dispatcher(
        manager.channel_generation_host.dispatch_outbound
    )
    dispatch_task = asyncio.create_task(bus.dispatch_outbound())
    tool = MessagePushTool(chat_lane=bus.chat_lane)
    tool.bind_v3_channel_dispatcher(
        lambda message, passive: _dispatch_v3_channel_push(
            manager,
            bus,
            message,
            passive,
        )
    )

    pending = asyncio.create_task(
        tool.execute(
            target_channel="feishu",
            target_chat_id="ou_1",
            message="hello",
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    await asyncio.wait_for(bus.aclose(), timeout=1)
    result = json.loads(await asyncio.wait_for(pending, timeout=1))

    assert result["status"] == "unknown"
    assert result["retryable"] is False
    runtime = _active_channel_generation(manager)
    assert runtime is not None
    assert runtime.channel("feishu").in_flight == 0
    assert dispatch_task.done()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_channel_formal_start_rejects_raw_config_drift_before_factory(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "channel_probe",
        _channel_plugin_source("1.0.0"),
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        _channel_static_manifest("1.0.0"),
        encoding="utf-8",
    )
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    config_path = data_dir / "config.local.toml"
    config_path.write_text(
        "app_id = 'app-1'\napp_secret = 'secret-before-seal'\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)

    class ProviderFactory:
        closed = 0

        async def create(self, credentials: Any) -> object:
            raise AssertionError("config drift 后不得创建 provider client")

        async def aclose(self) -> None:
            self.closed += 1

    provider = ProviderFactory()

    def drift_after_snapshot(snapshot: Any) -> Mapping[str, ProviderClientFactory]:
        config_path.write_text(
            "app_id = 'app-1'\napp_secret = 'secret-after-seal'\n",
            encoding="utf-8",
        )
        return {"feishu": cast(ProviderClientFactory, provider)}

    manager.bind_channel_provider_factory_resolver(drift_after_snapshot)
    with pytest.raises(RuntimeError, match="config revision 已漂移"):
        await manager.load_all()

    assert provider.closed == 1
    assert manager.current_snapshot is None
    assert _active_channel_generation(manager) is None
    record = manager.reload_journal.latest(plugin_id="channel_probe")
    assert record is not None and record.phase == "aborted"


@pytest.mark.asyncio
async def test_v3_channel_candidate_start_failure_restores_closed_stable_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "channel_probe",
        _channel_plugin_source("1.0.0"),
    )
    manifest_path = plugin_dir / "akashic.plugin.toml"
    manifest_path.write_text(_channel_static_manifest("1.0.0"), encoding="utf-8")
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    config_path = data_dir / "config.local.toml"
    config_path.write_text(
        "app_id = 'app-1'\napp_secret = 'formal-secret'\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)

    class ProviderFactory:
        async def create(self, credentials: Any) -> object:
            return object()

        async def aclose(self) -> None:
            return None

    manager.bind_channel_provider_factory_resolver(
        lambda snapshot: {
            descriptor.name: cast(ProviderClientFactory, ProviderFactory())
            for descriptor in cast(Any, snapshot.channel_registry).descriptors
        }
    )
    await manager.load_all()
    stable = manager.current_snapshot
    stable_runtime = _active_channel_generation(manager)
    assert stable is not None and stable_runtime is not None
    stable_token = stable_runtime.channel("feishu").binding_token

    (plugin_dir / "plugin.py").write_text(
        _channel_plugin_source("2.0.0"),
        encoding="utf-8",
    )
    manifest_path.write_text(_channel_static_manifest("2.0.0"), encoding="utf-8")
    candidate = await manager.prepare_candidate("channel_probe")
    assert candidate is not None
    original_start = manager.channel_generation_host.start_formal
    failed = False

    async def fail_candidate_once(snapshot: Any, factories: Any, **kwargs: Any):
        nonlocal failed
        if snapshot is not stable and not failed:
            failed = True
            assert manager.current_snapshot is stable
            assert manager.latest_snapshot is snapshot
            assert not stable.accepting_leases
            assert not snapshot.accepting_leases
            raise RuntimeError("candidate channel start failed")
        return await original_start(snapshot, factories, **kwargs)

    monkeypatch.setattr(
        manager.channel_generation_host,
        "start_formal",
        fail_candidate_once,
    )
    with pytest.raises(RuntimeError, match="candidate channel start failed"):
        await manager.publish_prepared("channel_probe")

    assert failed
    assert manager.current_snapshot is stable
    assert stable.accepting_leases
    restored = _active_channel_generation(manager)
    assert restored is not None and restored.snapshot_id == stable.snapshot_id
    assert restored.channel("feishu").admission_open
    assert restored.channel("feishu").binding_token != stable_token
    assert config_path.read_text(encoding="utf-8").endswith("formal-secret'\n")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_channel_old_restart_failure_keeps_durable_recovery_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "channel_probe",
        _channel_plugin_source("1.0.0"),
    )
    manifest_path = plugin_dir / "akashic.plugin.toml"
    manifest_path.write_text(_channel_static_manifest("1.0.0"), encoding="utf-8")
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    (data_dir / "config.local.toml").write_text(
        "app_id = 'app-1'\napp_secret = 'formal-secret'\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)
    endpoint_resume_calls = 0

    async def quiesce_endpoints() -> None:
        return None

    async def reject_unowned_resume() -> None:
        nonlocal endpoint_resume_calls
        endpoint_resume_calls += 1
        raise AssertionError("pure v3 Channel recovery 不拥有 endpoint admission")

    manager.bind_endpoint_admission(
        quiesce=quiesce_endpoints,
        resume=reject_unowned_resume,
    )

    class ProviderFactory:
        async def create(self, credentials: Any) -> object:
            return object()

        async def aclose(self) -> None:
            return None

    manager.bind_channel_provider_factory_resolver(
        lambda snapshot: {
            descriptor.name: cast(ProviderClientFactory, ProviderFactory())
            for descriptor in cast(Any, snapshot.channel_registry).descriptors
        }
    )
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None

    (plugin_dir / "plugin.py").write_text(
        _channel_plugin_source("2.0.0"),
        encoding="utf-8",
    )
    manifest_path.write_text(_channel_static_manifest("2.0.0"), encoding="utf-8")
    candidate = await manager.prepare_candidate("channel_probe")
    assert candidate is not None and candidate.reload_tx_id is not None
    original_start = manager.channel_generation_host.start_formal
    candidate_failed = False

    async def fail_candidate_and_rollback(
        snapshot: Any,
        factories: Any,
        **kwargs: Any,
    ):
        nonlocal candidate_failed
        if snapshot is not stable and not candidate_failed:
            candidate_failed = True
            raise RuntimeError("candidate channel start failed")
        if kwargs.get("boot_owner") == "plugin-manager-rollback":
            raise RuntimeError("rollback channel restart failed")
        return await original_start(snapshot, factories, **kwargs)

    monkeypatch.setattr(
        manager.channel_generation_host,
        "start_formal",
        fail_candidate_and_rollback,
    )
    with pytest.raises(RuntimeError, match="旧 owner 恢复失败"):
        await manager.publish_prepared("channel_probe")

    assert manager.current_snapshot is stable
    assert not stable.accepting_leases
    assert _active_channel_generation(manager) is None
    record = manager.reload_journal.get(candidate.reload_tx_id)
    assert record.phase == "degraded"
    assert record.failure_resource == (f"channel-publication:{candidate.generation_id}")

    recovered = await manager.retry_runtime_recovery("channel_probe")
    assert recovered["publication_state"] == "recovered"
    assert endpoint_resume_calls == 0
    assert stable.accepting_leases
    active = _active_channel_generation(manager)
    assert active is not None and active.channel("feishu").admission_open
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_channel_old_stop_failure_keeps_stable_closed_until_exact_retry(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "channel_probe",
        _channel_plugin_source("1.0.0", fail_stop=True),
    )
    manifest_path = plugin_dir / "akashic.plugin.toml"
    manifest_path.write_text(_channel_static_manifest("1.0.0"), encoding="utf-8")
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    (data_dir / "config.local.toml").write_text(
        "app_id = 'app-1'\napp_secret = 'formal-secret'\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)

    class ProviderFactory:
        async def create(self, credentials: Any) -> object:
            return object()

        async def aclose(self) -> None:
            return None

    manager.bind_channel_provider_factory_resolver(
        lambda snapshot: {
            descriptor.name: cast(ProviderClientFactory, ProviderFactory())
            for descriptor in cast(Any, snapshot.channel_registry).descriptors
        }
    )
    await manager.load_all()
    stable = manager.current_snapshot
    runtime = _active_channel_generation(manager)
    assert stable is not None and runtime is not None

    (plugin_dir / "plugin.py").write_text(
        _channel_plugin_source("2.0.0"),
        encoding="utf-8",
    )
    manifest_path.write_text(_channel_static_manifest("2.0.0"), encoding="utf-8")
    candidate = await manager.prepare_candidate("channel_probe")
    assert candidate is not None and candidate.reload_tx_id is not None
    with pytest.raises(RuntimeError, match="旧 owner 恢复失败"):
        await manager.publish_prepared("channel_probe")

    assert manager.current_snapshot is stable
    assert not stable.accepting_leases
    assert not runtime.channel("feishu").admission_open
    failure = manager.channel_generation_host.failure(
        runtime.snapshot_id,
        "feishu",
    )
    assert failure is not None
    record = manager.reload_journal.get(candidate.reload_tx_id)
    assert record.phase == "degraded"
    assert record.failure_resource is not None
    assert set(record.failure_resource.split(",")) == {
        f"channel-binding:{failure.binding_token}",
        f"channel-publication:{candidate.generation_id}",
    }

    state = next(
        value
        for key, value in cast(Any, manager.channel_generation_host)._bindings.items()
        if key[0] == runtime.snapshot_id
    )
    state.adapter.fail_stop = False
    recovered = await manager.retry_runtime_recovery("channel_probe")
    assert recovered["publication_state"] == "recovered"
    assert stable.accepting_leases
    active = _active_channel_generation(manager)
    assert active is not None and active.channel("feishu").admission_open
    assert manager.channel_generation_host.failure(runtime.snapshot_id) is None
    restored_state = next(
        value
        for key, value in cast(Any, manager.channel_generation_host)._bindings.items()
        if key[0] == active.snapshot_id
    )
    restored_state.adapter.fail_stop = False
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_channel_candidate_cleanup_failure_blocks_old_restore_until_retry(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "channel_probe",
        _channel_plugin_source("1.0.0"),
    )
    manifest_path = plugin_dir / "akashic.plugin.toml"
    manifest_path.write_text(_channel_static_manifest("1.0.0"), encoding="utf-8")
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    (data_dir / "config.local.toml").write_text(
        "app_id = 'app-1'\napp_secret = 'formal-secret'\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)

    class ProviderFactory:
        async def create(self, credentials: Any) -> object:
            return object()

        async def aclose(self) -> None:
            return None

    manager.bind_channel_provider_factory_resolver(
        lambda snapshot: {
            descriptor.name: cast(ProviderClientFactory, ProviderFactory())
            for descriptor in cast(Any, snapshot.channel_registry).descriptors
        }
    )
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None

    (plugin_dir / "plugin.py").write_text(
        _channel_plugin_source(
            "2.0.0",
            fail_start=True,
            fail_stop=True,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(_channel_static_manifest("2.0.0"), encoding="utf-8")
    candidate = await manager.prepare_candidate("channel_probe")
    assert (
        candidate is not None
        and candidate.reload_tx_id is not None
        and candidate.runtime_snapshot is not None
    )
    with pytest.raises(RuntimeError, match="旧 owner 恢复失败"):
        await manager.publish_prepared("channel_probe")

    assert manager.current_snapshot is stable
    assert not stable.accepting_leases
    assert _active_channel_generation(manager) is None
    failure = manager.channel_generation_host.failure(
        candidate.runtime_snapshot.snapshot_id,
        "feishu",
    )
    assert failure is not None
    state = next(
        value
        for key, value in cast(Any, manager.channel_generation_host)._bindings.items()
        if key[0] == candidate.runtime_snapshot.snapshot_id
    )
    state.adapter.fail_stop = False
    recovered = await manager.retry_runtime_recovery("channel_probe")
    assert recovered["publication_state"] == "recovered"
    assert stable.accepting_leases
    active = _active_channel_generation(manager)
    assert active is not None and active.channel("feishu").admission_open
    assert (
        manager.channel_generation_host.failure(candidate.runtime_snapshot.snapshot_id)
        is None
    )
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_channel_terminate_failure_retains_exact_owner_until_retry(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "channel_probe",
        _channel_plugin_source("1.0.0", fail_stop=True),
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        _channel_static_manifest("1.0.0"),
        encoding="utf-8",
    )
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    (data_dir / "config.local.toml").write_text(
        "app_id = 'app-1'\napp_secret = 'formal-secret'\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)

    class ProviderFactory:
        async def create(self, credentials: Any) -> object:
            return object()

        async def aclose(self) -> None:
            return None

    manager.bind_channel_provider_factory_resolver(
        lambda snapshot: {
            descriptor.name: cast(ProviderClientFactory, ProviderFactory())
            for descriptor in cast(Any, snapshot.channel_registry).descriptors
        }
    )
    await manager.load_all()
    stable = manager.current_snapshot
    runtime = _active_channel_generation(manager)
    assert stable is not None and runtime is not None

    with pytest.raises(RuntimeError, match="generation owner 已保留"):
        await manager.terminate_all()

    assert manager.current_snapshot is stable
    assert not stable.accepting_leases
    assert manager.generation("channel_probe") is not None
    failure = manager.channel_generation_host.failure(
        runtime.snapshot_id,
        "feishu",
    )
    assert failure is not None
    state = next(
        value
        for key, value in cast(Any, manager.channel_generation_host)._bindings.items()
        if key[0] == runtime.snapshot_id
    )
    state.adapter.fail_stop = False

    recovered = await manager.retry_runtime_recovery("channel_probe")
    assert recovered["publication_state"] == "recovered"
    active = _active_channel_generation(manager)
    assert active is not None and active.channel("feishu").admission_open
    restored_state = next(
        value
        for key, value in cast(Any, manager.channel_generation_host)._bindings.items()
        if key[0] == active.snapshot_id
    )
    restored_state.adapter.fail_stop = False
    await manager.terminate_all()


def test_v3_channel_secret_rejects_legacy_string_only_config_schema() -> None:
    class LegacyConfig(BaseModel):
        app_secret: str

    with pytest.raises(
        plugin_manager_module._PluginConfigError,  # pyright: ignore[reportPrivateUsage]
        match="app_secret",
    ):
        plugin_manager_module._validate_plugin_config_projection(  # pyright: ignore[reportPrivateUsage]
            {"app_secret": CredentialRef(("app_secret",))},
            LegacyConfig,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("manifest_paths", "config_text"),
    (
        (("app_secret",), "app_id = 'app-1'\nappSecret = 'secret'\n"),
        (
            ("app_secret", "appSecret"),
            "app_id = 'app-1'\napp_secret = 'one'\nappSecret = 'two'\n",
        ),
    ),
)
async def test_v3_channel_credential_aliases_fail_before_apply(
    tmp_path: Path,
    manifest_paths: tuple[str, ...],
    config_text: str,
) -> None:
    marker = tmp_path / "candidate-apply-ran"
    source = _channel_plugin_source("1.0.0").replace(
        "async def apply(ctx, config):\n",
        "async def apply(ctx, config):\n"
        f"    __import__('pathlib').Path({str(marker)!r}).write_text('bad')\n",
    )
    plugin_dir = _write_plugin(tmp_path / "plugins", "channel_probe", source)
    manifest = _channel_static_manifest("1.0.0").replace(
        "feishu = ['app_secret', 'appSecret']",
        f"feishu = {list(manifest_paths)!r}",
    )
    (plugin_dir / "akashic.plugin.toml").write_text(manifest, encoding="utf-8")
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    config_path = data_dir / "config.local.toml"
    original = config_text.encode()
    config_path.write_bytes(original)
    manager = _manager(tmp_path)

    await manager.load_all()

    assert manager.current_snapshot is None
    assert manager.generation("channel_probe") is None
    assert not marker.exists()
    assert config_path.read_bytes() == original
    assert not (tmp_path / "workspace" / "runtime" / "plugin-validation").exists()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_channel_credential_field_name_is_part_of_alias_admission(
    tmp_path: Path,
) -> None:
    validator_marker = tmp_path / "credential-before-validator-ran"
    apply_marker = tmp_path / "candidate-apply-ran"
    source = (
        "from pydantic import BaseModel, ConfigDict, Field, field_validator\n"
        "from agent.plugin_composition import (\n"
        "    CHANNELS, ChannelCapability, ChannelDefinition, CredentialRef, InboundIdentity,\n"
        ")\n"
        "api_version = 3\n"
        "name = 'channel_probe'\n"
        "version = '1.0.0'\n"
        "inject = (CHANNELS,)\n"
        "class Config(BaseModel):\n"
        "    model_config = ConfigDict(validate_by_name=True)\n"
        "    secret: CredentialRef = Field(validation_alias='appSecret')\n"
        "    @field_validator('secret', mode='before')\n"
        "    @classmethod\n"
        "    def observe_secret(cls, value):\n"
        f"        __import__('pathlib').Path({str(validator_marker)!r}).write_text(str(value))\n"
        "        return value\n"
        "async def apply(ctx, config):\n"
        f"    __import__('pathlib').Path({str(apply_marker)!r}).write_text('bad')\n"
        "    await ctx.require(CHANNELS).register(ctx, ChannelDefinition(\n"
        "        name='feishu',\n"
        "        capabilities=frozenset({ChannelCapability.OUTBOUND}),\n"
        "        factory_export='build_adapter',\n"
        "        inbound_identity=None,\n"
        "        credential_paths=('appSecret',),\n"
        "    ))\n"
    )
    plugin_dir = _write_plugin(tmp_path / "plugins", "channel_probe", source)
    manifest = _channel_static_manifest("1.0.0").replace(
        "feishu = ['app_secret', 'appSecret']",
        "feishu = ['appSecret']",
    )
    (plugin_dir / "akashic.plugin.toml").write_text(manifest, encoding="utf-8")
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    config_path = data_dir / "config.local.toml"
    original = b"secret = 'candidate-must-never-see-this'\n"
    config_path.write_bytes(original)
    manager = _manager(tmp_path)

    await manager.load_all()

    assert manager.current_snapshot is None
    assert manager.generation("channel_probe") is None
    assert not validator_marker.exists()
    assert not apply_marker.exists()
    assert config_path.read_bytes() == original
    assert not (tmp_path / "workspace" / "runtime" / "plugin-validation").exists()
    await manager.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_mutation,error",
    (
        (
            (
                "credential_paths=('appSecret', 'app_secret')",
                "credential_paths=('app_id',)",
            ),
            "credential 声明与静态 manifest 不一致",
        ),
        (
            ("inject = (CHANNELS,)", "inject = ()"),
            "静态 channel credential 没有对应 Root 声明",
        ),
    ),
)
async def test_v3_channel_manifest_and_root_declaration_must_match(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    source_mutation: tuple[str, str],
    error: str,
) -> None:
    source = _channel_plugin_source("1.0.0")
    if source_mutation[0] == "inject = (CHANNELS,)":
        source = source[: source.index("async def apply")] + (
            "async def apply(ctx, config):\n" "    pass\n"
        )
    source = source.replace(*source_mutation)
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "channel_probe",
        source,
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        _channel_static_manifest("1.0.0"),
        encoding="utf-8",
    )
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    config_path = data_dir / "config.local.toml"
    original_config = b"app_id = 'app-1'\nappSecret = 'secret'\n"
    config_path.write_bytes(original_config)
    manager = _manager(tmp_path)

    await manager.load_all()

    assert error in caplog.text
    assert manager.current_snapshot is None
    assert manager.generation("channel_probe") is None
    assert config_path.read_bytes() == original_config


@pytest.mark.asyncio
async def test_candidate_cannot_keep_channel_manifest_after_removing_declaration(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "channel_probe",
        _channel_plugin_source("1.0.0"),
    )
    manifest_path = plugin_dir / "akashic.plugin.toml"
    manifest_path.write_text(_channel_static_manifest("1.0.0"), encoding="utf-8")
    data_dir = tmp_path / "workspace" / "plugin-data" / "channel_probe-builtin"
    data_dir.mkdir(parents=True)
    (data_dir / "config.local.toml").write_text(
        "app_id = 'app-1'\nappSecret = 'secret'\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None and stable.channel_registry is not None

    source = _channel_plugin_source("2.0.0")
    source = source[: source.index("async def apply")] + (
        "async def apply(ctx, config):\n    pass\n"
    )
    source = source.replace("inject = (CHANNELS,)", "inject = ()")
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    manifest_path.write_text(_channel_static_manifest("2.0.0"), encoding="utf-8")

    candidate = await manager.prepare_candidate("channel_probe")

    assert candidate is None
    assert "候选验证失败: runtime_snapshot" in caplog.text
    assert manager.current_snapshot is stable
    assert manager.latest_snapshot is stable
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_namespace_loader_waits_for_service_not_scan_order(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "a_consumer",
        "from pydantic import BaseModel\n"
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'a_consumer'\n"
        "version = '1.0.0'\n"
        "VALUE = ServiceKey('fixture.value')\n"
        "inject = (VALUE,)\n"
        "observed = None\n"
        "disposed = False\n"
        "class Config(BaseModel):\n"
        "    suffix: str = 'default'\n"
        "async def apply(ctx, config):\n"
        "    global observed, disposed\n"
        "    observed = (ctx.require(VALUE), ctx.runtime.plugin_id, "
        "ctx.runtime.workspace.name, config.suffix)\n"
        "    def cleanup():\n"
        "        global disposed\n"
        "        disposed = True\n"
        "    await ctx.effect(lambda: cleanup, label='consumer')\n",
    )
    _write_plugin(
        tmp_path / "plugins",
        "z_provider",
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'z_provider'\n"
        "version = '1.0.0'\n"
        "VALUE = ServiceKey('fixture.value')\n"
        "async def apply(ctx, config):\n"
        "    await ctx.provide(VALUE, 'ready')\n",
    )
    config_dir = tmp_path / "workspace" / "plugin-data" / "a_consumer-builtin"
    config_dir.mkdir(parents=True)
    (config_dir / "config.local.toml").write_text(
        "suffix = 'configured'\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    consumer = manager.generation("a_consumer")
    snapshot = manager.current_snapshot
    assert consumer is not None and snapshot is not None
    assert isinstance(consumer.instance, ComposablePlugin)
    assert not hasattr(consumer.instance, "context")
    assert consumer.plugin_dir == tmp_path / "plugins" / "a_consumer"
    assert consumer.config.suffix == "configured"  # type: ignore[union-attr]
    assert consumer.instance.module.observed == (
        "ready",
        "a_consumer",
        "workspace",
        "configured",
    )
    assert snapshot.composition_root is not None
    assert snapshot.composition_topology is not None
    assert snapshot.composition_topology.services == (
        "core.commands",
        "fixture.value",
    )
    assert tuple(item.name for item in snapshot.composition_topology.fibers) == (
        "a_consumer",
        "z_provider",
    )

    await manager.terminate_all()

    assert consumer.instance.module.disposed is True


@pytest.mark.asyncio
async def test_v3_loader_publishes_declared_package_contributions(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "package_contributor",
        "api_version = 3\n"
        "name = 'package_contributor'\n"
        "version = '1.0.0'\n"
        "skill_roots = ('skills',)\n"
        "drift_skill_roots = ('drift/skills',)\n"
        "dashboard_module = 'dashboard.py'\n"
        "web_module = 'web_module.js'\n"
        "web_requires = ('web.root.v1',)\n"
        "web_provides = ()\n"
        "def apply(ctx, config): pass\n",
    )
    skill_dir = plugin_dir / "skills" / "package-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: package-skill\ndescription: package skill\n---\nnormal body\n",
        encoding="utf-8",
    )
    drift_skill_dir = plugin_dir / "drift" / "skills" / "package-drift"
    drift_skill_dir.mkdir(parents=True)
    (drift_skill_dir / "SKILL.md").write_text(
        "---\nname: package-drift\ndescription: package drift\n---\ndrift body\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard.py").write_text(
        "from agent.plugin_composition import DashboardContext\n"
        "def plugin_enabled(context):\n"
        "    return isinstance(context, DashboardContext) and not context.validation\n"
        "def register(app, context):\n"
        "    assert not hasattr(app.state, 'memory_admin')\n"
        "    assert not hasattr(app.state, 'memory_store')\n"
        "    (context.data_root / 'dashboard-context-ready').write_text(context.plugin_id)\n"
        "    @app.get('/api/dashboard/package-contributor')\n"
        "    def status(): return {'plugin': 'package_contributor'}\n"
        "    class Closeable:\n"
        "        def __init__(self, path): self.path = path\n"
        "        def close(self): self.path.write_text('closed')\n"
        "    return (\n"
        "        Closeable(context.data_root / 'dashboard-close-one'),\n"
        "        Closeable(context.data_root / 'dashboard-close-two'),\n"
        "    )\n",
        encoding="utf-8",
    )
    web_source = (
        "import React from 'react';\n"
        "import { jsx } from 'react/jsx-runtime';\n"
        "import { createRoot } from 'react-dom/client';\n"
        "import { currentTheme } from '@akashic/web-ui-v1';\n"
        "const helper = () => null, T = (ctx) => {\n"
        "  const label = 'import a connection'; // import is ordinary copy\n"
        "  const marker = /import/;\n"
        "  if (ctx) /export function activate/.test(label);\n"
        "  const api = {import() {}}; api.import();\n"
        "  return ctx.ui.inject('web.root.v1', (mount) =>\n"
        "    mount.register({id: 'fixture', render() {}}));\n"
        "};\n"
        "export { T as activate };\n"
    )
    (plugin_dir / "web_module.js").write_text(web_source, encoding="utf-8")
    (plugin_dir / "web_module.css").write_text(".fixture { display: block; }\n", encoding="utf-8")
    manager = _manager(tmp_path)

    await manager.load_all()

    generation = manager.generation("package_contributor")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    assert generation.contributions.skill_roots == ((plugin_dir / "skills").resolve(),)
    assert generation.contributions.drift_skill_roots == (
        (plugin_dir / "drift" / "skills").resolve(),
    )
    assert (
        generation.contributions.dashboard_module
        == (plugin_dir / "dashboard.py").resolve()
    )
    web_asset = generation.contributions.web_module
    assert web_asset is not None and web_asset.module == web_source
    assert web_asset.requires == ("web.root.v1",)
    assert web_asset.provides == ()
    assert len(web_asset.contract_sha256) == 64
    active = {item.plugin_id: item for item in manager.active_plugins()}
    assert active["package_contributor"].skill_roots == (
        (plugin_dir / "skills").resolve(),
    )
    assert active["package_contributor"].drift_skill_roots == (
        (plugin_dir / "drift" / "skills").resolve(),
    )
    catalog_id = snapshot.skill_catalog_generation_id
    assert catalog_id is not None
    catalog = manager._skill_host.get(catalog_id)
    assert catalog is not None
    assert snapshot.plugin_skill_index is not None
    assert snapshot.web_ui_catalog is not None
    assert [item.plugin_id for item in snapshot.web_ui_catalog.modules] == [
        "package_contributor"
    ]
    assert snapshot.web_ui_catalog.modules[0].asset is web_asset
    assert set(snapshot.plugin_skill_index.records) == {"package-skill"}
    assert set(catalog.drift.records) == {"package-drift"}
    assert snapshot.plugin_skill_index.records["package-skill"].root_dir != skill_dir

    dashboard_host = PluginDashboardHost(
        core_routes=(),
    )
    dashboard_host.prepare_snapshot(snapshot)
    assert len(snapshot.dashboard_bindings) == 1
    binding = snapshot.dashboard_bindings[0]
    assert isinstance(binding, DashboardBinding)
    assert binding.plugin_id == "package_contributor"
    assert binding.runtime_data_root == generation.data_dir.resolve()
    assert (generation.data_dir / "dashboard-context-ready").read_text() == (
        "package_contributor"
    )
    assert [route.path for route in binding.routes] == [
        "/api/dashboard/package-contributor"
    ]

    await manager.terminate_all()

    assert (generation.data_dir / "dashboard-close-one").is_file()
    assert (generation.data_dir / "dashboard-close-two").is_file()


@pytest.mark.asyncio
async def test_web_module_without_its_ui_provider_still_publishes(
    tmp_path: Path,
) -> None:
    """Keep runtime activation independent from a missing browser mount."""

    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "headless_capability",
        "api_version = 3\n"
        "name = 'headless_capability'\n"
        "version = '1.0.0'\n"
        "web_module = 'web_module.js'\n"
        "web_requires = ('optional.surface.v1',)\n"
        "async def apply(ctx, config): pass\n",
    )
    (plugin_dir / "web_module.js").write_text(
        "export function activate(ctx) {\n"
        "  return ctx.ui.inject('optional.surface.v1', () => () => {});\n"
        "}\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    snapshot = manager.current_snapshot
    assert snapshot is not None and snapshot.web_ui_catalog is not None
    assert tuple(item.plugin_id for item in snapshot.web_ui_catalog.modules) == (
        "headless_capability",
    )
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_dashboard_rejects_legacy_register_signature(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "legacy_dashboard_signature",
        "api_version = 3\n"
        "name = 'legacy_dashboard_signature'\n"
        "version = '1.0.0'\n"
        "dashboard_module = 'dashboard.py'\n"
        "def apply(ctx, config): pass\n",
    )
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, plugin_dir, workspace): return None\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    snapshot = manager.current_snapshot
    assert snapshot is not None
    dashboard_host = PluginDashboardHost(
        core_routes=(),
    )

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        dashboard_host.prepare_snapshot(snapshot)

    assert snapshot.dashboard_bindings == ()
    await manager.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dashboard_source", "message"),
    [
        (
            "plugin_enabled = 1\n" "def register(app, context): return None\n",
            "plugin_enabled 必须是可调用对象",
        ),
        (
            "async def plugin_enabled(context): return True\n"
            "def register(app, context): return None\n",
            "plugin_enabled 不支持 async",
        ),
        (
            "async def register(app, context): return None\n",
            "register 不支持 async",
        ),
        (
            "def register(app, context): return object()\n",
            "register 返回值不是 closeable",
        ),
    ],
)
async def test_v3_dashboard_rejects_invalid_callable_contracts(
    tmp_path: Path,
    dashboard_source: str,
    message: str,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "invalid_dashboard_contract",
        "api_version = 3\n"
        "name = 'invalid_dashboard_contract'\n"
        "version = '1.0.0'\n"
        "dashboard_module = 'dashboard.py'\n"
        "def apply(ctx, config): pass\n",
    )
    (plugin_dir / "dashboard.py").write_text(
        dashboard_source,
        encoding="utf-8",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    generation = manager.generation("invalid_dashboard_contract")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    dashboard_host = PluginDashboardHost(
        core_routes=(),
    )

    with pytest.raises(RuntimeError, match=message):
        dashboard_host.prepare_snapshot(snapshot)

    assert snapshot.dashboard_bindings == ()
    assert tuple(generation.data_dir.iterdir()) == ()
    assert f"{generation.module_path}.dashboard" not in sys.modules
    await manager.terminate_all()


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ("skill_roots = 'skills'", "skill_roots 必须是字符串序列"),
        ("drift_skill_roots = ('',)", "drift_skill_roots 必须只包含非空字符串"),
        ("skill_roots = ('skills', 'skills')", "skill_roots 不得重复"),
        (
            "workspace_roots = ('nested/root',)",
            "workspace_roots 必须是顶层目录名",
        ),
        (
            "workspace_roots = ('memes', 'memes')",
            "workspace_roots 不得重复",
        ),
        (
            "workspace_roots = ('plugin-data',)",
            "workspace_roots 不得声明 Core 保留目录 plugin-data",
        ),
        (
            "workspace_roots = ('runtime',)",
            "workspace_roots 不得声明 Core 保留目录 runtime",
        ),
        (
            "workspace_files = ('plugin-data/secret.db',)",
            "workspace_files 必须是 workspace 内相对文件路径",
        ),
        (
            "workspace_files = ('memory/../sessions.db',)",
            "workspace_files 必须是 workspace 内相对文件路径",
        ),
        ("dashboard_module = ''", "dashboard_module 必须是非空字符串或 None"),
        ("is_active = 1", "is_active 必须是可调用对象"),
    ],
)
def test_v3_namespace_rejects_invalid_package_contributions(
    declaration: str,
    message: str,
) -> None:
    from types import ModuleType

    module = ModuleType("invalid_v3_contribution")
    module.api_version = 3
    module.name = "invalid"
    module.version = "1.0.0"
    module.apply = lambda ctx, config: None
    exec(declaration, module.__dict__)

    with pytest.raises(ValueError, match=message):
        _ = ComposablePlugin.from_module(module)


def test_v3_namespace_freezes_package_contribution_lists() -> None:
    roots = ["skills"]
    module = ModuleType("frozen_v3_contribution")
    module.api_version = 3
    module.name = "frozen"
    module.version = "1.0.0"
    module.skill_roots = roots
    module.apply = lambda ctx, config: None

    plugin = ComposablePlugin.from_module(module)
    roots.append("mutated")

    assert plugin.skill_roots == ("skills",)


def test_v3_namespace_accepts_exact_nested_workspace_files() -> None:
    module = ModuleType("nested_workspace_file")
    module.api_version = 3
    module.name = "nested_workspace_file"
    module.version = "1.0.0"
    module.workspace_files = ("memory/MEMORY.md", "memory/SELF.md")
    module.apply = lambda ctx, config: None

    plugin = ComposablePlugin.from_module(module)

    assert plugin.workspace_files == ("memory/MEMORY.md", "memory/SELF.md")


@pytest.mark.parametrize(
    "declaration",
    [
        "def apply(ctx): pass",
        "def apply(): pass",
        "def apply(ctx, config, extra): pass",
        "def apply(ctx, config=None): pass",
        "def apply(*args): pass",
        "def apply(ctx, *, config): pass",
        "def apply(config, ctx): pass",
    ],
)
def test_v3_namespace_rejects_noncanonical_apply_signature(
    declaration: str,
) -> None:
    module = ModuleType("invalid_v3_apply")
    module.api_version = 3
    module.name = "invalid"
    module.version = "1.0.0"
    exec(declaration, module.__dict__)

    with pytest.raises(
        ValueError,
        match=r"apply 必须精确声明 apply\(ctx, config\)",
    ):
        _ = ComposablePlugin.from_module(module)


@pytest.mark.parametrize(
    "declaration",
    [
        "def apply(ctx, config): pass",
        "async def apply(ctx, config): pass",
        "def apply(ctx, config, /): pass",
    ],
)
def test_v3_namespace_accepts_canonical_apply_signature(declaration: str) -> None:
    module = ModuleType("valid_v3_apply")
    module.api_version = 3
    module.name = "valid"
    module.version = "1.0.0"
    exec(declaration, module.__dict__)

    plugin = ComposablePlugin.from_module(module)

    assert plugin.name == "valid"


@pytest.mark.asyncio
async def test_v3_manager_rejects_invalid_apply_before_plugin_data_creation(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "invalid_apply",
        "api_version = 3\n"
        "name = 'invalid_apply'\n"
        "version = '1.0.0'\n"
        "def apply(ctx): pass\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    assert manager.generation("invalid_apply") is None
    assert not (
        tmp_path / "workspace" / "plugin-data" / "invalid_apply-builtin"
    ).exists()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_manager_validates_plugin_data_path_before_config_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "invalid_apply",
        "api_version = 3\n"
        "name = 'invalid_apply'\n"
        "version = '1.0.0'\n"
        "def apply(ctx): pass\n",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external-plugin-data"
    external.mkdir()
    (workspace / "plugin-data").symlink_to(external, target_is_directory=True)
    config_revision_called = False

    def unexpected_config_revision(_path: Path) -> str:
        nonlocal config_revision_called
        config_revision_called = True
        raise AssertionError("config revision must not cross the plugin-data boundary")

    monkeypatch.setattr(
        plugin_manager_module,
        "_file_revision",
        unexpected_config_revision,
    )
    manager = _manager(tmp_path)

    with pytest.raises(ValueError, match="插件数据目录不能穿过符号链接"):
        await manager.load_all()

    assert config_revision_called is False
    assert tuple(external.iterdir()) == ()


@pytest.mark.asyncio
async def test_v3_loader_rejects_workspace_root_that_is_not_directory(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "invalid_workspace_root",
        "api_version = 3\n"
        "name = 'invalid_workspace_root'\n"
        "version = '1.0.0'\n"
        "workspace_roots = ('memes',)\n"
        "def apply(ctx, config): pass\n",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "memes").write_text("not a directory", encoding="utf-8")
    manager = _manager(tmp_path)

    await manager.load_all()

    assert manager.generation("invalid_workspace_root") is None
    gate = manager.latest_gate("invalid_workspace_root")
    assert gate is not None
    assert gate.status == "failed"
    assert "workspace root 不是目录" in gate.failure_reason


@pytest.mark.asyncio
async def test_v3_loader_rejects_workspace_root_symlink_outside_workspace(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "escaped_workspace_root",
        "api_version = 3\n"
        "name = 'escaped_workspace_root'\n"
        "version = '1.0.0'\n"
        "workspace_roots = ('memes',)\n"
        "def apply(ctx, config): pass\n",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside-memes"
    outside.mkdir()
    (workspace / "memes").symlink_to(outside, target_is_directory=True)
    manager = _manager(tmp_path)

    await manager.load_all()

    assert manager.generation("escaped_workspace_root") is None
    gate = manager.latest_gate("escaped_workspace_root")
    assert gate is not None
    assert gate.status == "failed"
    assert "workspace root 不能是符号链接" in gate.failure_reason
    assert tuple(outside.iterdir()) == ()


@pytest.mark.asyncio
async def test_v3_candidate_never_copies_workspace_root_symlink_target(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "candidate_workspace_root",
        "api_version = 3\n"
        "name = 'candidate_workspace_root'\n"
        "version = '1.0.0'\n"
        "workspace_roots = ('memes',)\n"
        "def apply(ctx, config): pass\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside-candidate-memes"
    outside.mkdir()
    marker = outside / "must-not-copy.txt"
    marker.write_text("outside", encoding="utf-8")
    (workspace / "memes").symlink_to(outside, target_is_directory=True)
    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'candidate_workspace_root'\n"
        "version = '2.0.0'\n"
        "workspace_roots = ('memes',)\n"
        "def apply(ctx, config): pass\n",
        encoding="utf-8",
    )

    candidate = await manager.prepare_candidate("candidate_workspace_root")

    assert candidate is None
    assert manager.current_snapshot is stable
    assert marker.read_text(encoding="utf-8") == "outside"
    assert not tuple((workspace / "plugin-data").glob("**/must-not-copy.txt"))
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_candidate_rejects_workspace_root_declaration_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "workspace_root_drift",
        "api_version = 3\n"
        "name = 'workspace_root_drift'\n"
        "version = '1.0.0'\n"
        "workspace_roots = ('memes',)\n"
        "def apply(ctx, config): pass\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None
    original_clone = manager._clone_candidate_composable

    def clone_with_drift(
        generation: PluginGeneration,
        *,
        candidate_owner: PluginGeneration,
        attempt_workspace: Path,
    ) -> tuple[ComposablePlugin, str, Path, object]:
        clone, module_path, data_dir, config = original_clone(
            generation,
            candidate_owner=candidate_owner,
            attempt_workspace=attempt_workspace,
        )
        clone.workspace_roots = ("drifted",)
        return clone, module_path, data_dir, config

    monkeypatch.setattr(manager, "_clone_candidate_composable", clone_with_drift)
    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'workspace_root_drift'\n"
        "version = '2.0.0'\n"
        "workspace_roots = ('memes',)\n"
        "def apply(ctx, config): pass\n",
        encoding="utf-8",
    )

    candidate = await manager.prepare_candidate("workspace_root_drift")

    assert candidate is None
    assert manager.current_snapshot is stable
    gate = manager.latest_gate("workspace_root_drift")
    assert gate is not None
    assert "workspace_roots 与 generation 冻结声明不一致" in gate.failure_reason
    assert not any("__candidate_" in name for name in sys.modules)
    await manager.terminate_all()


@pytest.mark.parametrize("value", [None, 1, "active"])
def test_v3_active_predicate_must_return_bool(value: object) -> None:
    from types import ModuleType

    module = ModuleType("invalid_v3_active_result")
    module.api_version = 3
    module.name = "invalid_active"
    module.version = "1.0.0"
    module.apply = lambda ctx, config: None
    module.is_active = lambda services: value
    plugin = ComposablePlugin.from_module(module)

    with pytest.raises(RuntimeError, match="is_active 必须返回 bool"):
        plugin.bind_static_services(ServiceView.freeze({}))


def test_v3_active_predicate_rejects_async_without_leaking_coroutine() -> None:
    from types import ModuleType

    module = ModuleType("async_v3_active_result")
    module.api_version = 3
    module.name = "async_active"
    module.version = "1.0.0"
    module.apply = lambda ctx, config: None

    async def active(services: ServiceView) -> bool:
        return True

    module.is_active = active
    plugin = ComposablePlugin.from_module(module)

    with pytest.raises(RuntimeError, match="is_active 不支持 async"):
        plugin.bind_static_services(ServiceView.freeze({}))


@pytest.mark.asyncio
async def test_inactive_v3_does_not_wait_for_declared_runtime_dependency(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "inactive_missing",
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'inactive_missing'\n"
        "version = '1.0.0'\n"
        "MISSING = ServiceKey('missing.runtime')\n"
        "inject = (MISSING,)\n"
        "def is_active(services): return False\n"
        "def apply(ctx, config): raise RuntimeError('inactive apply ran')\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    snapshot = manager.current_snapshot
    assert snapshot is not None and snapshot.composition_root is not None
    assert snapshot.composition_root.receipt().ready is True
    assert snapshot.composition_root.receipt().required_pending == ()
    assert snapshot.composition_topology is not None
    fiber = next(
        item
        for item in snapshot.composition_topology.fibers
        if item.name == "inactive_missing"
    )
    assert fiber.dependencies == ("missing.runtime",)
    assert fiber.static_active is False
    assert snapshot.active_generations() == ()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_package_contribution_path_cannot_escape_plugin_root(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    outside = tmp_path / "plugins" / "outside"
    outside.mkdir(parents=True)
    _write_plugin(
        tmp_path / "plugins",
        "escaped_contributor",
        "api_version = 3\n"
        "name = 'escaped_contributor'\n"
        "version = '1.0.0'\n"
        "skill_roots = ('../outside',)\n"
        "def apply(ctx, config): pass\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    assert "插件 能力目录 越界" in caplog.text
    assert manager.current_snapshot is None
    assert manager.generation("escaped_contributor") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "web_source",
    [
        "import React from './react.js';\nexport function activate() { return () => {}; }\n",
        "import React from 'preact';\nexport function activate() { return () => {}; }\n",
        "import { Button } from '@akashic/dashboard-ui';\n"
        "export function activate() { return () => {}; }\n",
        "import React from 'https://example.com/react.js';\n"
        "export function activate() { return () => {}; }\n",
        "export function activate() { import/**/('./late.js'); return () => {}; }\n",
        "export function activate() { `${import('./late.js')}`; return () => {}; }\n",
        "export { helper } from './helper.js';\nexport function activate() { return () => {}; }\n",
        "async function activate() { return () => {}; }\nexport { activate };\n",
        "export const activate = (async () => () => {});\n",
        "if (true) /export function activate/.test('copy');\n"
        "export const notActivate = () => {};\n",
    ],
)
async def test_v3_web_module_failure_never_publishes(
    tmp_path: Path,
    web_source: str,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "broken_web",
        "api_version = 3\n"
        "name = 'broken_web'\n"
        "version = '1.0.0'\n"
        "web_module = 'web_module.js'\n"
        "def apply(ctx, config): pass\n",
    )
    (plugin_dir / "web_module.js").write_text(
        web_source,
        encoding="utf-8",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    assert manager.current_snapshot is None
    assert manager.generation("broken_web") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stylesheet",
    [
        "@keyframes spin { to { transform: rotate(1turn); } }\n",
        "@keyframes broken_style-spin { to { transform: rotate(1turn); } }\n",
        "@font-face { font-family: shared; src: url(data:font/woff2;base64,AA); }\n",
    ],
)
async def test_v3_web_stylesheet_global_names_never_publish(
    tmp_path: Path,
    stylesheet: str,
) -> None:
    """Reject CSS names that @scope cannot isolate from sibling plugins."""

    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "broken_style",
        "api_version = 3\n"
        "name = 'broken_style'\n"
        "version = '1.0.0'\n"
        "web_module = 'web_module.js'\n"
        "def apply(ctx, config): pass\n",
    )
    (plugin_dir / "web_module.js").write_text(
        "export function activate() { return () => {}; }\n",
        encoding="utf-8",
    )
    (plugin_dir / "web_module.css").write_text(stylesheet, encoding="utf-8")
    manager = _manager(tmp_path)

    await manager.load_all()

    assert manager.current_snapshot is None
    assert manager.generation("broken_style") is None


@pytest.mark.asyncio
async def test_v3_package_contribution_rejects_duplicate_resolved_roots(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "duplicate_contributor",
        "api_version = 3\n"
        "name = 'duplicate_contributor'\n"
        "version = '1.0.0'\n"
        "skill_roots = ('skills', './skills')\n"
        "def apply(ctx, config): pass\n",
    )
    (plugin_dir / "skills").mkdir()
    manager = _manager(tmp_path)

    await manager.load_all()

    assert "插件能力目录重复" in caplog.text
    assert manager.current_snapshot is None
    assert manager.generation("duplicate_contributor") is None


@pytest.mark.asyncio
async def test_v3_loader_fails_loud_when_required_service_never_appears(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "waiting",
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'waiting'\n"
        "version = '1.0.0'\n"
        "inject = (ServiceKey('never.provided'),)\n"
        "def apply(ctx, config):\n"
        "    raise AssertionError('pending plugin must not apply')\n",
    )
    manager = _manager(tmp_path)

    with pytest.raises(RuntimeError, match="never.provided"):
        await manager.load_all()

    assert manager.current_snapshot is None
    assert manager.active_plugins() == []
    assert manager._snapshot_store.retained_snapshot_ids == ()
    assert manager._active_generations == {}
    assert manager._scopes == {}
    assert not (tmp_path / "workspace" / "plugin-data" / "waiting-builtin").exists()

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_stable_boot_publishes_one_complete_snapshot(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "independent",
        "api_version = 3\n"
        "name = 'independent'\n"
        "version = '1.0.0'\n"
        "activated = False\n"
        "def apply(ctx, config):\n"
        "    global activated\n"
        "    activated = True\n",
    )
    _write_plugin(
        tmp_path / "plugins",
        "consumer",
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'consumer'\n"
        "version = '1.0.0'\n"
        "VALUE = ServiceKey('fixture.batch')\n"
        "inject = (VALUE,)\n"
        "async def apply(ctx, config):\n"
        "    assert ctx.require(VALUE) == 'ready'\n",
    )
    _write_plugin(
        tmp_path / "plugins",
        "provider",
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'provider'\n"
        "version = '1.0.0'\n"
        "VALUE = ServiceKey('fixture.batch')\n"
        "async def apply(ctx, config):\n"
        "    await ctx.provide(VALUE, 'ready')\n",
    )
    manager = _manager(tmp_path)
    installed: list[object] = []
    original_install = manager._snapshot_store.install

    def record_install(snapshot: object) -> None:
        installed.append(snapshot)
        original_install(snapshot)  # type: ignore[arg-type]

    manager._snapshot_store.install = record_install  # type: ignore[method-assign]

    await manager.load_all()

    snapshot = manager.current_snapshot
    independent = manager.generation("independent")
    assert snapshot is not None and independent is not None
    assert len(installed) == 1
    assert set(snapshot.generations) == {"consumer", "independent", "provider"}
    assert snapshot.composition_root is not None
    assert snapshot.composition_topology is not None
    assert snapshot.composition_topology.services == (
        "core.commands",
        "fixture.batch",
    )
    assert independent.instance.module.activated is True
    catalog_id = snapshot.skill_catalog_generation_id
    assert catalog_id is not None
    assert manager._skill_host.get(catalog_id) is not None

    await manager.terminate_all()
    assert manager._skill_host.get(catalog_id) is None


@pytest.mark.asyncio
async def test_snapshot_sealing_runs_once_after_all_services_are_ready(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "sealing_probe",
        "from agent.plugin_composition import SNAPSHOT_SEALING\n"
        "api_version = 3\n"
        "name = 'sealing_probe'\n"
        "version = '1.0.0'\n"
        "events = []\n"
        "async def apply(ctx, config):\n"
        "    await ctx.on(SNAPSHOT_SEALING, lambda _event: events.append('sealed'))\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    generation = manager.generation("sealing_probe")
    assert generation is not None
    assert generation.instance.module.events == ["sealed"]
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_snapshot_sealing_rejects_bail_before_publication(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "sealing_bail",
        "from agent.plugin_composition import Bail, SNAPSHOT_SEALING\n"
        "api_version = 3\n"
        "name = 'sealing_bail'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config):\n"
        "    await ctx.on(SNAPSHOT_SEALING, lambda _event: Bail('stop'))\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    assert manager.current_snapshot is None
    assert manager.generation("sealing_bail") is None


@pytest.mark.asyncio
async def test_candidate_rebuilds_complete_explicit_service_component(
    tmp_path: Path,
) -> None:
    plugins = tmp_path / "plugins"
    driver_dir = _write_plugin(
        plugins,
        "a_driver",
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'a_driver'\n"
        "version = '1.0.0'\n"
        "DRIVERS = ServiceKey('fixture.drivers')\n"
        "inject = (DRIVERS,)\n"
        "async def apply(ctx, config):\n"
        "    assert ctx.require(DRIVERS) is not None\n",
    )
    _write_plugin(
        plugins,
        "b_models",
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'b_models'\n"
        "version = '1.0.0'\n"
        "DRIVERS = ServiceKey('fixture.drivers')\n"
        "CHAT = ServiceKey('fixture.chat')\n"
        "async def apply(ctx, config):\n"
        "    await ctx.provide(DRIVERS, object())\n"
        "    await ctx.provide(CHAT, object())\n",
    )
    _write_plugin(
        plugins,
        "c_driver",
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'c_driver'\n"
        "version = '1.0.0'\n"
        "DRIVERS = ServiceKey('fixture.drivers')\n"
        "inject = (DRIVERS,)\n"
        "async def apply(ctx, config):\n"
        "    assert ctx.require(DRIVERS) is not None\n",
    )
    _write_plugin(
        plugins,
        "d_consumer",
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'd_consumer'\n"
        "version = '1.0.0'\n"
        "CHAT = ServiceKey('fixture.chat')\n"
        "inject = (CHAT,)\n"
        "async def apply(ctx, config):\n"
        "    assert ctx.require(CHAT) is not None\n",
    )
    _write_plugin(
        plugins,
        "unrelated",
        "api_version = 3\n"
        "name = 'unrelated'\n"
        "version = '1.0.0'\n"
        "def apply(ctx, config): pass\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    (driver_dir / "plugin.py").write_text(
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'a_driver'\n"
        "version = '1.0.1'\n"
        "DRIVERS = ServiceKey('fixture.drivers')\n"
        "inject = (DRIVERS,)\n"
        "async def apply(ctx, config):\n"
        "    assert ctx.require(DRIVERS) is not None\n",
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("a_driver")
    assert candidate is not None and candidate.runtime_snapshot is not None
    root = candidate.runtime_snapshot.composition_root

    assert isinstance(root, CompositionOverlay)
    assert root.replaced_plugin_ids == frozenset(
        {"a_driver", "b_models", "c_driver", "d_consumer"}
    )
    await manager.publish_prepared("a_driver")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_cancelled_stable_batch_finishes_all_cleanup(tmp_path: Path) -> None:
    first_cleanup = tmp_path / "first-cleaned"
    blocking_started = tmp_path / "blocking-started"
    root_cleanup = tmp_path / "root-cleaned"
    _write_plugin(
        tmp_path / "plugins",
        "a_first",
        "from pathlib import Path\n"
        "api_version = 3\n"
        "name = 'a_first'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config):\n"
        f"    await ctx.effect(lambda: lambda: Path({str(first_cleanup)!r}).touch(), label='marker')\n",
    )
    _write_plugin(
        tmp_path / "plugins",
        "b_root",
        "from pathlib import Path\n"
        "api_version = 3\n"
        "name = 'b_root'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config):\n"
        f"    await ctx.effect(lambda: lambda: Path({str(root_cleanup)!r}).touch(), label='marker')\n",
    )
    _write_plugin(
        tmp_path / "plugins",
        "z_blocking",
        "import asyncio\n"
        "from pathlib import Path\n"
        "api_version = 3\n"
        "name = 'z_blocking'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config):\n"
        f"    Path({str(blocking_started)!r}).touch()\n"
        "    await asyncio.Event().wait()\n",
    )
    manager = _manager(tmp_path)
    original_discard = manager._discard_stable_batch
    discard_started = asyncio.Event()

    async def delayed_discard(*args: object, **kwargs: object) -> None:
        discard_started.set()
        await asyncio.sleep(0.05)
        await original_discard(*args, **kwargs)  # type: ignore[arg-type]

    manager._discard_stable_batch = delayed_discard  # type: ignore[method-assign]
    loading = asyncio.create_task(manager.load_all())
    while not blocking_started.exists():
        await asyncio.sleep(0)

    loading.cancel()
    await discard_started.wait()
    loading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loading

    assert first_cleanup.exists()
    assert root_cleanup.exists()
    assert manager.current_snapshot is None
    assert manager._snapshot_store.retained_snapshot_ids == ()
    assert manager._active_generations == {}
    assert manager._scopes == {}


@pytest.mark.asyncio
async def test_v3_reload_keeps_old_root_until_snapshot_lease_drains(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "reloadable",
        "api_version = 3\n"
        "name = 'reloadable'\n"
        "version = '1.0.0'\n"
        "marker = 'old'\n"
        "disposed = False\n"
        "async def apply(ctx, config):\n"
        "    def cleanup():\n"
        "        global disposed\n"
        "        disposed = True\n"
        "    await ctx.effect(lambda: cleanup, label=marker)\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    old_generation = manager.generation("reloadable")
    old_snapshot = manager.current_snapshot
    assert old_generation is not None and old_snapshot is not None
    lease = manager._snapshot_store.lease()

    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'reloadable'\n"
        "version = '1.0.0'\n"
        "marker = 'new'\n"
        "disposed = False\n"
        "async def apply(ctx, config):\n"
        "    def cleanup():\n"
        "        global disposed\n"
        "        disposed = True\n"
        "    await ctx.effect(lambda: cleanup, label=marker)\n",
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("reloadable")
    assert candidate is not None

    publication = asyncio.create_task(manager.publish_prepared("reloadable"))
    while old_snapshot.accepting_leases:
        await asyncio.sleep(0)
    assert not publication.done()
    assert old_generation.instance.module.disposed is False
    await lease.release()
    result = await publication

    assert result["publication_state"] == "committed"
    assert manager.current_snapshot is not old_snapshot
    active_root = manager.current_snapshot.composition_root
    assert active_root is not None
    active_runtime = active_root.root_fiber.children[0].runtime
    assert active_runtime is not None
    assert active_runtime.workspace == tmp_path / "workspace"
    assert candidate.validation_workspace is None
    assert old_generation.instance.module.disposed is True

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_direct_v3_rebuild_rejects_parent_ownership_drift(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "parent_drift",
        "api_version = 3\n"
        "name = 'parent_drift'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config):\n"
        "    pass\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    stable_snapshot = manager.current_snapshot
    assert stable_snapshot is not None

    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'parent_drift'\n"
        "version = '2.0.0'\n"
        "disposed = []\n"
        "async def apply(ctx, config):\n"
        "    validation = 'plugin-validation' in str(ctx.runtime.workspace)\n"
        "    async def apply_group(group_ctx):\n"
        "        if validation:\n"
        "            await group_ctx.mount(lambda _: None, name='worker')\n"
        "    await ctx.mount(apply_group, name='group')\n"
        "    if not validation:\n"
        "        await ctx.mount(lambda _: None, name='worker')\n"
        "    role = 'candidate' if validation else 'formal'\n"
        "    def cleanup():\n"
        "        disposed.append(role)\n"
        "    await ctx.effect(lambda: cleanup, label='parent-drift')\n",
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("parent_drift")
    assert candidate is not None and candidate.runtime_snapshot is not None
    candidate_root = candidate.runtime_snapshot.composition_root
    assert candidate_root is not None
    candidate_view = candidate_root.topology_view()
    assert tuple((item.name, item.parent) for item in candidate_view.fibers) == (
        ("group", "parent_drift"),
        ("parent_drift", None),
        ("worker", "group"),
    )
    attempt_workspace = candidate_root.root_fiber.children[0].runtime
    assert attempt_workspace is not None
    attempt_root = attempt_workspace.workspace.parent
    clone_modules = {
        module_name
        for module_name in sys.modules
        if module_name.startswith(f"{candidate.module_path}__candidate_")
    }
    assert clone_modules

    with pytest.raises(RuntimeError, match="snapshot identity 发生变化"):
        await manager.publish_prepared("parent_drift")

    assert manager.current_snapshot is stable_snapshot
    assert manager.prepared_generation("parent_drift") is None
    assert candidate.scope.closed is True
    assert candidate.instance.module.disposed == ["formal"]
    assert clone_modules.isdisjoint(sys.modules)
    assert not attempt_root.exists()

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_direct_v3_invariant_failure_never_applies_to_formal_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "isolated_reload",
        "api_version = 3\n"
        "name = 'isolated_reload'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config):\n"
        "    pass\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    stable_snapshot = manager.current_snapshot
    assert stable_snapshot is not None

    (plugin_dir / "plugin.py").write_text(
        "from pathlib import Path\n"
        "api_version = 3\n"
        "name = 'isolated_reload'\n"
        "version = '2.0.0'\n"
        "async def apply(ctx, config):\n"
        "    Path(ctx.data_root, 'apply-probe').write_text('candidate')\n",
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("isolated_reload")
    assert candidate is not None and candidate.validation_workspace is not None
    validation_root = candidate.validation_workspace.parent
    candidate_snapshot = candidate.runtime_snapshot
    assert candidate_snapshot is not None
    candidate_root = candidate_snapshot.composition_root
    assert candidate_root is not None
    candidate_runtime = candidate_root.root_fiber.children[0].runtime
    assert candidate_runtime is not None
    clone_modules = {
        module_name
        for module_name in sys.modules
        if module_name.startswith(f"{candidate.module_path}__candidate_")
    }
    assert clone_modules
    assert (candidate_runtime.data_dir / "apply-probe").is_file()
    first_attempt_root = candidate_runtime.workspace.parent

    original_invariants = manager._post_publish_invariants

    async def fail_invariant(*_args: object) -> None:
        raise RuntimeError("candidate invariant failed")

    monkeypatch.setattr(manager, "_post_publish_invariants", fail_invariant)
    with pytest.raises(RuntimeError, match="candidate invariant failed"):
        await manager.publish_prepared("isolated_reload")

    formal_probe = (
        tmp_path
        / "workspace"
        / "plugin-data"
        / "isolated_reload-builtin"
        / "apply-probe"
    )
    assert not formal_probe.exists()
    assert manager.current_snapshot is stable_snapshot
    assert manager.prepared_generation("isolated_reload") is None
    assert candidate.scope.closed is True
    assert clone_modules.isdisjoint(sys.modules)
    assert not validation_root.exists()
    assert not first_attempt_root.exists()

    monkeypatch.setattr(manager, "_post_publish_invariants", original_invariants)
    second = await manager.prepare_candidate("isolated_reload")
    assert second is not None and second.runtime_snapshot is not None
    second_root = second.runtime_snapshot.composition_root
    assert second_root is not None
    second_runtime = second_root.root_fiber.children[0].runtime
    assert second_runtime is not None
    second_attempt_root = second_runtime.workspace.parent
    second_clone_modules = {
        module_name
        for module_name in sys.modules
        if module_name.startswith(f"{second.module_path}__candidate_")
    }
    assert second_clone_modules

    published = await manager.publish_prepared("isolated_reload")

    assert published["publication_state"] == "committed"
    assert formal_probe.read_text(encoding="utf-8") == "candidate"
    assert second_clone_modules.isdisjoint(sys.modules)
    assert not second_attempt_root.exists()

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_cancelled_candidate_mount_cleans_partial_clones_and_data(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "a_first",
        "api_version = 3\n"
        "name = 'a_first'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config):\n"
        "    await ctx.effect(lambda: None, label='first')\n",
    )
    _write_plugin(
        tmp_path / "plugins",
        "z_blocker",
        "import asyncio\n"
        "api_version = 3\n"
        "name = 'z_blocker'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config):\n"
        "    if 'plugin-validation' not in str(ctx.runtime.workspace):\n"
        "        return\n"
        "    (ctx.runtime.workspace / 'blocker-entered').write_text('ready')\n"
        "    await asyncio.Event().wait()\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    stable_snapshot = manager.current_snapshot
    assert stable_snapshot is not None
    stable_root = stable_snapshot.composition_root
    validation_base = tmp_path / "workspace" / "runtime" / "plugin-validation"

    preparing = asyncio.create_task(manager.prepare_candidate("z_blocker"))
    marker: Path | None = None
    for _ in range(200):
        markers = list(validation_base.rglob("blocker-entered"))
        if markers:
            marker = markers[0]
            break
        await asyncio.sleep(0.01)
    if marker is None:
        preparing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await preparing
        pytest.fail("candidate Fiber did not enter apply")

    attempt_root = marker.parent.parent
    clone_modules = {
        module_name for module_name in sys.modules if "__candidate_" in module_name
    }
    assert len(clone_modules) == 1

    preparing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preparing

    assert manager.current_snapshot is stable_snapshot
    assert manager.current_snapshot.composition_root is stable_root
    assert manager.prepared_generation("z_blocker") is None
    assert clone_modules.isdisjoint(sys.modules)
    assert not attempt_root.exists()
    assert not validation_base.exists() or not any(validation_base.iterdir())

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_installed_v3_candidate_rebuilds_runtime_then_promotes(
    tmp_path: Path,
) -> None:
    plugin_base = tmp_path / "home" / "cache" / "lab" / "installed_v3"
    stable_root = plugin_base / ".artifacts" / "1.0.0-aaaa"
    latest_root = plugin_base / ".artifacts" / "2.0.0-bbbb"
    stable_root.mkdir(parents=True)
    latest_root.mkdir(parents=True)
    source = (
        "from pydantic import BaseModel\n"
        "from agent.plugin_composition import BACKGROUND_JOBS\n"
        "api_version = 3\n"
        "name = 'installed_v3'\n"
        "version = '1.0.0'\n"
        "skill_roots = ('skills',)\n"
        "drift_skill_roots = ('drift/skills',)\n"
        "dashboard_module = 'dashboard.py'\n"
        "inject = (BACKGROUND_JOBS,)\n"
        "class Config(BaseModel):\n"
        "    marker: str = 'default'\n"
        "applied = []\n"
        "disposed = []\n"
        "async def apply(ctx, config):\n"
        "    workspace = str(ctx.runtime.workspace)\n"
        "    writer = ctx.runtime.workspace / '.installed-v3-writer'\n"
        "    if writer.exists():\n"
        "        raise RuntimeError('installed writer already mounted')\n"
        "    writer.write_text(ctx.runtime.generation_id, encoding='utf-8')\n"
        "    applied.append((workspace, config.marker))\n"
        "    def cleanup():\n"
        "        disposed.append(workspace)\n"
        "        writer.unlink()\n"
        "    await ctx.effect(lambda: cleanup, label='runtime')\n"
    )
    (stable_root / "plugin.py").write_text(source, encoding="utf-8")
    (latest_root / "plugin.py").write_text(
        source.replace("version = '1.0.0'", "version = '2.0.0'"),
        encoding="utf-8",
    )
    _write_static_v3_manifest(stable_root, "installed_v3", "1.0.0")
    _write_static_v3_manifest(latest_root, "installed_v3", "2.0.0")
    for root, version in ((stable_root, "v1"), (latest_root, "v2")):
        skill_dir = root / "skills" / "installed-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\ndescription: installed {version}\n---\nbody {version}\n",
            encoding="utf-8",
        )
        drift_dir = root / "drift" / "skills" / "installed-drift"
        drift_dir.mkdir(parents=True)
        (drift_dir / "SKILL.md").write_text(
            f"---\ndescription: drift {version}\n---\ndrift {version}\n",
            encoding="utf-8",
        )
        (root / "dashboard.py").write_text(
            "def register(app, context): return None\n",
            encoding="utf-8",
        )
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    write_plugin_manifest(
        {"installed_v3@lab": True},
        plugins_home=tmp_path / "home",
    )
    config_dir = tmp_path / "workspace" / "plugin-data" / "installed_v3-lab"
    config_dir.mkdir(parents=True)
    (config_dir / "config.local.toml").write_text(
        "marker = 'configured'\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)
    manager.bind_activity_host(ActivityHost(()))
    await manager.load_all()
    stable = manager.generation("installed_v3@lab")
    stable_snapshot = manager.current_snapshot
    assert stable is not None and stable_snapshot is not None
    assert stable_snapshot.plugin_skill_index is not None
    assert (
        "body v1" in stable_snapshot.plugin_skill_index.get("installed-skill").content
    )  # type: ignore[union-attr]
    stable_lease = manager.snapshot_store.lease()

    write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
    result = (await manager.reconcile_changed())[0]
    candidate = manager.ready_candidate

    assert result["publication_state"] == "latest_ready"
    assert candidate is not None
    assert not hasattr(candidate.instance, "context")
    assert candidate.plugin_dir == latest_root
    assert candidate.config.marker == "configured"  # type: ignore[union-attr]
    candidate_snapshot = candidate.runtime_snapshot
    assert candidate_snapshot is not None
    assert candidate_snapshot.plugin_skill_index is not None
    assert (
        "body v2"
        in candidate_snapshot.plugin_skill_index.get("installed-skill").content
    )  # type: ignore[union-attr]
    assert (
        candidate.contributions.dashboard_module
        == (latest_root / "dashboard.py").resolve()
    )
    candidate_root = candidate_snapshot.composition_root
    stable_root_runtime = manager.current_snapshot.composition_root
    assert candidate_root is not None
    assert candidate_root is not stable_root_runtime
    candidate_runtime = candidate_root.root_fiber.children[0].runtime
    assert candidate_runtime is not None
    assert "plugin-validation" in str(candidate_runtime.workspace)
    assert candidate_runtime.config.marker == "configured"  # type: ignore[union-attr]
    assert candidate.validation_workspace is not None
    validation_root = candidate.validation_workspace.parent
    clone_modules = {
        module_name
        for module_name in sys.modules
        if module_name.startswith(f"{candidate.module_path}__candidate_")
    }
    assert clone_modules
    original_start_runtime = manager._start_runtime_snapshot
    formal_started_after_pointer = False

    async def observe_formal_start(snapshot) -> None:
        nonlocal formal_started_after_pointer
        if snapshot is not stable_snapshot:
            assert read_pointer(plugin_base, "stable") == latest_pointer
            assert read_pointer(plugin_base, "latest") == latest_pointer
            assert snapshot is manager.current_snapshot
            assert snapshot.accepting_leases
            formal_started_after_pointer = True
        await original_start_runtime(snapshot)

    manager._start_runtime_snapshot = observe_formal_start  # type: ignore[method-assign]
    promotion = asyncio.create_task(manager.switch_ready("installed_v3@lab"))
    while stable_snapshot.accepting_leases:
        await asyncio.sleep(0)
    assert not promotion.done()
    assert stable.instance.module.disposed == []
    await stable_lease.release()
    promoted = await promotion
    assert formal_started_after_pointer

    assert promoted["publication_state"] == "promoted"
    promoted_snapshot = manager.current_snapshot
    assert promoted_snapshot is not None
    assert promoted_snapshot.composition_root is not None
    assert promoted_snapshot.background_job_catalog is not None
    assert (
        promoted_snapshot.background_job_catalog.root_instance_token
        is promoted_snapshot.composition_root.instance_token
    )
    assert promoted_snapshot.plugin_skill_index is not None
    assert (
        "body v2" in promoted_snapshot.plugin_skill_index.get("installed-skill").content
    )  # type: ignore[union-attr]
    promoted_catalog_id = promoted_snapshot.skill_catalog_generation_id
    assert promoted_catalog_id is not None
    promoted_catalog = manager._skill_host.get(promoted_catalog_id)
    assert promoted_catalog is not None
    assert (
        "drift v2" in promoted_catalog.drift.get("installed-drift").content
    )  # type: ignore[union-attr]
    assert (
        promoted_snapshot.generations["installed_v3@lab"].contributions.dashboard_module
        == (latest_root / "dashboard.py").resolve()
    )
    assert candidate.instance.module.applied[-1] == (
        str(tmp_path / "workspace"),
        "configured",
    )
    assert clone_modules.isdisjoint(sys.modules)
    assert not validation_root.exists()
    assert stable.instance.module.disposed == [str(tmp_path / "workspace")]
    assert (tmp_path / "workspace" / ".installed-v3-writer").exists()

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_installed_v3_dashboard_uses_composition_runtime_until_promotion(
    tmp_path: Path,
) -> None:
    plugin_base = tmp_path / "home" / "cache" / "lab" / "dashboard_v3"
    stable_root = plugin_base / ".artifacts" / "1.0.0-aaaa"
    latest_root = plugin_base / ".artifacts" / "2.0.0-bbbb"
    stable_root.mkdir(parents=True)
    latest_root.mkdir(parents=True)
    source = (
        "api_version = 3\n"
        "name = 'dashboard_v3'\n"
        "version = '1.0.0'\n"
        "dashboard_module = 'dashboard.py'\n"
        "drift_skill_roots = ('drift/skills',)\n"
        "workspace_roots = ('memes',)\n"
        "def is_active(services): return True\n"
        "observed_workspace_root = None\n"
        "def apply(ctx, config):\n"
        "    global observed_workspace_root\n"
        "    observed_workspace_root = ctx.workspace_root('memes')\n"
    )
    (stable_root / "plugin.py").write_text(source, encoding="utf-8")
    (latest_root / "plugin.py").write_text(
        source.replace("version = '1.0.0'", "version = '2.0.0'"),
        encoding="utf-8",
    )
    _write_static_v3_manifest(stable_root, "dashboard_v3", "1.0.0")
    _write_static_v3_manifest(latest_root, "dashboard_v3", "2.0.0")
    (stable_root / "dashboard.py").write_text(
        "def register(app, context):\n"
        "    assert context.workspace_root('memes').is_dir()\n",
        encoding="utf-8",
    )
    (latest_root / "dashboard.py").write_text(
        "def register(app, context):\n"
        "    marker = 'candidate-registered' if context.validation else 'formal-registered'\n"
        "    (context.data_root / marker).write_text('ready')\n"
        "    shared = context.workspace_root('memes')\n"
        "    shared_marker = 'candidate-shared' if context.validation else 'formal-shared'\n"
        "    (shared / shared_marker).write_text('ready')\n"
        "    class Closeable:\n"
        "        def close(self):\n"
        "            (context.data_root / 'dashboard-v3-closed').write_text('closed')\n"
        "    return Closeable()\n",
        encoding="utf-8",
    )
    for artifact in (stable_root, latest_root):
        skill = artifact / "drift" / "skills" / "dashboard-v3-static"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("# static projection\n", encoding="utf-8")
    formal_memes = tmp_path / "workspace" / "memes"
    formal_memes.mkdir(parents=True)
    (formal_memes / "manifest.json").write_text("{}\n", encoding="utf-8")
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    write_plugin_manifest(
        {"dashboard_v3@lab": True},
        plugins_home=tmp_path / "home",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    PluginSkillLinker(
        workspace=tmp_path / "workspace",
        plugin_roots=manager.skill_projection_roots,
    ).sync(manager.active_plugins())
    skill_link = tmp_path / "workspace" / "drift" / "skills" / "dashboard-v3-static"
    assert skill_link.exists()
    stable_snapshot = manager.current_snapshot
    assert stable_snapshot is not None
    dashboard_host = PluginDashboardHost(
        core_routes=(),
    )
    dashboard_host.prepare_initial_snapshot(stable_snapshot)
    manager.bind_dashboard_preparer(
        dashboard_host.prepare_snapshot,
        validation_releaser=dashboard_host.release_validation,
    )

    write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
    first_change = (await manager.reconcile_changed())[0]
    first_gate = manager.latest_gate("dashboard_v3@lab")
    assert first_change["publication_state"] == "latest_ready", (
        first_change,
        None if first_gate is None else first_gate.failure_reason,
    )
    candidate = manager.ready_candidate
    assert candidate is not None and candidate.runtime_snapshot is not None
    assert "dashboard_v3@lab" in {
        item.plugin_id for item in candidate.runtime_snapshot.active_generations()
    }
    validation_workspace = candidate.validation_workspace
    assert validation_workspace is not None
    candidate_binding = candidate.runtime_snapshot.dashboard_bindings[0]
    assert isinstance(candidate_binding, DashboardBinding)
    assert candidate_binding.validation is True
    candidate_root = candidate.runtime_snapshot.composition_root
    assert candidate_root is not None
    candidate_runtime = candidate_root.plugin_runtime("dashboard_v3@lab")
    assert candidate_binding.runtime_workspace == candidate_runtime.workspace.resolve()
    candidate_data_root = candidate_binding.runtime_data_root
    assert candidate_data_root is not None
    assert candidate_data_root == candidate_runtime.data_dir.resolve()
    assert candidate_data_root.is_relative_to(validation_workspace.parent)
    assert (candidate_data_root / "candidate-registered").is_file()
    candidate_memes = candidate_runtime.workspace_root("memes")
    assert candidate_memes != formal_memes.resolve()
    assert (candidate_memes / "manifest.json").read_text() == "{}\n"
    assert (candidate_memes / "candidate-shared").is_file()
    assert not (formal_memes / "candidate-shared").exists()
    assert not (candidate_data_root / "formal-registered").exists()
    production_data_root = tmp_path / "workspace" / "plugin-data" / "dashboard_v3-lab"
    assert not (production_data_root / "candidate-registered").exists()
    assert not (production_data_root / "formal-registered").exists()
    assert not (formal_memes / "candidate-shared").exists()
    validation_root = validation_workspace.parent
    validation_module = candidate_binding.module_name

    await manager.drop_candidate("dashboard_v3@lab")

    assert manager.current_snapshot is stable_snapshot
    assert not validation_root.exists()
    assert validation_module not in sys.modules
    assert not (production_data_root / "candidate-registered").exists()
    assert not (production_data_root / "formal-registered").exists()

    write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
    assert (await manager.reconcile_changed())[0]["publication_state"] == "latest_ready"
    promoted_candidate = manager.ready_candidate
    assert promoted_candidate is not None
    promoted_validation_workspace = promoted_candidate.validation_workspace
    assert promoted_validation_workspace is not None
    promoted_validation_root = promoted_validation_workspace.parent

    promoted = await manager.switch_ready("dashboard_v3@lab")

    assert promoted["publication_state"] == "promoted"
    current = manager.current_snapshot
    assert current is not None
    formal_binding = current.dashboard_bindings[0]
    assert isinstance(formal_binding, DashboardBinding)
    assert formal_binding.validation is False
    assert formal_binding.runtime_workspace == (tmp_path / "workspace").resolve()
    assert formal_binding.runtime_data_root == production_data_root.resolve()
    assert (production_data_root / "formal-registered").is_file()
    assert (formal_memes / "formal-shared").is_file()
    assert not (formal_memes / "candidate-shared").exists()
    assert not (production_data_root / "candidate-registered").exists()
    assert skill_link.exists()
    assert not promoted_validation_root.exists()
    promoted_generation = manager.generation("dashboard_v3@lab")
    assert promoted_generation is not None
    assert promoted_generation.instance.module.observed_workspace_root == (
        formal_memes.resolve()
    )
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_dashboard_uses_exact_workspace_declarations(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "exact_workspace_root",
        "api_version = 3\n"
        "name = 'exact_workspace_root'\n"
        "version = '1.0.0'\n"
        "workspace_roots = ('memes',)\n"
        "workspace_files = ('sessions.db',)\n"
        "dashboard_module = 'dashboard.py'\n"
        "def apply(ctx, config): pass\n",
    )
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, context):\n"
        "    assert context.workspace_root('memes').name == 'memes'\n"
        "    assert context.workspace_file('sessions.db').name == 'sessions.db'\n",
        encoding="utf-8",
    )
    memes = tmp_path / "workspace" / "memes"
    memes.mkdir(parents=True)
    (tmp_path / "workspace" / "sessions.db").touch()
    manager = _manager(tmp_path)
    await manager.load_all()
    generation = manager.generation("exact_workspace_root")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    generation.instance.workspace_roots = ("drifted",)
    generation.instance.workspace_files = ("drifted.db",)
    dashboard_host = PluginDashboardHost(
        core_routes=(),
    )

    dashboard_host.prepare_snapshot(snapshot)

    assert len(snapshot.dashboard_bindings) == 1
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_inactive_v3_does_not_claim_active_plugin_skill_name(
    tmp_path: Path,
) -> None:
    plugin_root = tmp_path / "plugins"
    for name, active in (("inactive_owner", False), ("active_owner", True)):
        plugin = _write_plugin(
            plugin_root,
            name,
            "api_version = 3\n"
            f"name = '{name}'\n"
            "version = '1.0.0'\n"
            "drift_skill_roots = ('drift/skills',)\n"
            f"def is_active(services): return {active!r}\n"
            "def apply(ctx, config): pass\n",
        )
        skill = plugin / "drift" / "skills" / "shared-static-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    manager = _manager(tmp_path)

    await manager.load_all()
    PluginSkillLinker(
        workspace=tmp_path / "workspace",
        plugin_roots=manager.skill_projection_roots,
    ).sync(manager.active_plugins())

    snapshot = manager.current_snapshot
    assert snapshot is not None
    assert {item.plugin_id for item in snapshot.active_generations()} == {
        "active_owner"
    }
    link = tmp_path / "workspace" / "drift" / "skills" / "shared-static-skill"
    assert (
        link.resolve()
        == (
            plugin_root / "active_owner" / "drift" / "skills" / "shared-static-skill"
        ).resolve()
    )
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_builtin_v3_dashboard_candidate_clones_data_root_before_publish(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "dashboard_builtin_v3",
        "api_version = 3\n"
        "name = 'dashboard_builtin_v3'\n"
        "version = '1.0.0'\n"
        "from agent.plugin_composition import BACKGROUND_JOBS\n"
        "inject = (BACKGROUND_JOBS,)\n"
        "dashboard_module = 'dashboard.py'\n"
        "def apply(ctx, config): pass\n",
    )
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, context): return None\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)
    manager.bind_activity_host(ActivityHost(()))
    await manager.load_all()
    stable = manager.generation("dashboard_builtin_v3")
    stable_snapshot = manager.current_snapshot
    assert stable is not None and stable_snapshot is not None
    (stable.data_dir / "existing.txt").write_text("stable", encoding="utf-8")
    dashboard_host = PluginDashboardHost(
        core_routes=(),
    )
    dashboard_host.prepare_initial_snapshot(stable_snapshot)
    manager.bind_dashboard_preparer(
        dashboard_host.prepare_snapshot,
        validation_releaser=dashboard_host.release_validation,
    )
    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'dashboard_builtin_v3'\n"
        "version = '2.0.0'\n"
        "from agent.plugin_composition import BACKGROUND_JOBS\n"
        "inject = (BACKGROUND_JOBS,)\n"
        "dashboard_module = 'dashboard.py'\n"
        "def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, context):\n"
        "    assert (context.data_root / 'existing.txt').read_text() == 'stable'\n"
        "    marker = 'candidate.txt' if context.validation else 'formal.txt'\n"
        "    (context.data_root / marker).write_text('ready')\n",
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("dashboard_builtin_v3")
    assert candidate is not None and candidate.validation_workspace is not None
    validation_root = candidate.validation_workspace.parent

    result = await manager.publish_prepared("dashboard_builtin_v3")

    assert result["publication_state"] == "committed"
    current = manager.current_snapshot
    assert current is not None
    assert current.composition_root is not None
    assert current.background_job_catalog is not None
    assert (
        current.background_job_catalog.root_instance_token
        is current.composition_root.instance_token
    )
    binding = current.dashboard_bindings[0]
    assert isinstance(binding, DashboardBinding)
    assert binding.runtime_data_root == stable.data_dir.resolve()
    assert (stable.data_dir / "existing.txt").read_text() == "stable"
    assert (stable.data_dir / "formal.txt").is_file()
    assert not (stable.data_dir / "candidate.txt").exists()
    assert not validation_root.exists()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_installed_v3_candidate_health_blocks_promotion_until_recovered(
    tmp_path: Path,
) -> None:
    plugin_base = tmp_path / "home" / "cache" / "lab" / "installed_v3"
    stable_artifact = plugin_base / ".artifacts" / "1.0.0-aaaa"
    latest_artifact = plugin_base / ".artifacts" / "2.0.0-bbbb"
    stable_artifact.mkdir(parents=True)
    latest_artifact.mkdir(parents=True)
    source = (
        "api_version = 3\n"
        "name = 'installed_v3'\n"
        "version = '1.0.0'\n"
        "health = None\n"
        "async def apply(ctx, config):\n"
        "    global health\n"
        "    health = await ctx.health('worker', required=True)\n"
    )
    (stable_artifact / "plugin.py").write_text(source, encoding="utf-8")
    (latest_artifact / "plugin.py").write_text(
        source.replace("version = '1.0.0'", "version = '2.0.0'"),
        encoding="utf-8",
    )
    _write_static_v3_manifest(stable_artifact, "installed_v3", "1.0.0")
    _write_static_v3_manifest(latest_artifact, "installed_v3", "2.0.0")
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    write_plugin_manifest(
        {"installed_v3@lab": True},
        plugins_home=tmp_path / "home",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    stable_snapshot = manager.current_snapshot
    assert stable_snapshot is not None

    write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
    assert (await manager.reconcile_changed())[0]["publication_state"] == "latest_ready"
    candidate = manager.ready_candidate
    assert candidate is not None and candidate.runtime_snapshot is not None
    candidate_root = candidate.runtime_snapshot.composition_root
    assert candidate_root is not None
    clone_name = next(
        name
        for name in sys.modules
        if name.startswith(f"{candidate.module_path}__candidate_")
    )
    candidate_health = sys.modules[clone_name].health
    candidate_health.degrade("validation worker unavailable")

    with pytest.raises(RuntimeError, match="required_degraded"):
        await manager.switch_ready("installed_v3@lab")

    assert manager.current_snapshot is stable_snapshot
    assert manager.ready_candidate is candidate
    assert candidate_root.root_fiber.children[0].state.value == "active"

    candidate_health.recover()
    promoted = await manager.switch_ready("installed_v3@lab")

    assert promoted["publication_state"] == "promoted"
    assert manager.ready_candidate is None
    assert clone_name not in sys.modules
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_installed_v3_candidate_incident_overflow_blocks_promotion(
    tmp_path: Path,
) -> None:
    plugin_base = tmp_path / "home" / "cache" / "lab" / "installed_v3"
    stable_artifact = plugin_base / ".artifacts" / "1.0.0-aaaa"
    latest_artifact = plugin_base / ".artifacts" / "2.0.0-bbbb"
    stable_artifact.mkdir(parents=True)
    latest_artifact.mkdir(parents=True)
    source = (
        "api_version = 3\n"
        "name = 'installed_v3'\n"
        "version = '1.0.0'\n"
        "saved_ctx = None\n"
        "async def apply(ctx, config):\n"
        "    global saved_ctx\n"
        "    saved_ctx = ctx\n"
    )
    (stable_artifact / "plugin.py").write_text(source, encoding="utf-8")
    (latest_artifact / "plugin.py").write_text(
        source.replace("version = '1.0.0'", "version = '2.0.0'"),
        encoding="utf-8",
    )
    _write_static_v3_manifest(stable_artifact, "installed_v3", "1.0.0")
    _write_static_v3_manifest(latest_artifact, "installed_v3", "2.0.0")
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    write_plugin_manifest(
        {"installed_v3@lab": True},
        plugins_home=tmp_path / "home",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    stable_snapshot = manager.current_snapshot

    write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
    assert (await manager.reconcile_changed())[0]["publication_state"] == "latest_ready"
    candidate = manager.ready_candidate
    assert candidate is not None and candidate.runtime_snapshot is not None
    clone_name = next(
        name
        for name in sys.modules
        if name.startswith(f"{candidate.module_path}__candidate_")
    )
    candidate_context = sys.modules[clone_name].saved_ctx
    for index in range(1025):
        candidate_context.report_incident("probe", f"failure {index}")

    with pytest.raises(RuntimeError, match="incident_overflowed"):
        await manager.switch_ready("installed_v3@lab")

    assert manager.current_snapshot is stable_snapshot
    assert manager.ready_candidate is candidate
    dropped = await manager.drop_candidate("installed_v3@lab")
    assert dropped["publication_state"] == "discarded"
    assert clone_name not in sys.modules
    await manager.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_commit_fails", [False, True])
async def test_installed_v3_isolated_handoff_success_and_owner_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_commit_fails: bool,
) -> None:
    plugin_base = tmp_path / "home" / "cache" / "lab" / "installed_v3"
    stable_artifact = plugin_base / ".artifacts" / "1.0.0-aaaa"
    latest_artifact = plugin_base / ".artifacts" / "2.0.0-bbbb"
    stable_artifact.mkdir(parents=True)
    latest_artifact.mkdir(parents=True)
    source = (
        "api_version = 3\n"
        "name = 'installed_v3'\n"
        "version = '1.0.0'\n"
        "from agent.plugin_composition import RUNTIME_STARTED, RUNTIME_STOPPING\n"
        "disposed = False\n"
        "started_roots = []\n"
        "stopped_roots = []\n"
        "async def apply(ctx, config):\n"
            "    global disposed\n"
            "    disposed = False\n"
            "    token = id(ctx._root_instance_token())\n"
            "    async def started(_):\n"
            "        async with ctx.runtime_scope():\n"
            "            started_roots.append(token)\n"
            "    await ctx.on(RUNTIME_STARTED, started)\n"
        "    await ctx.on(RUNTIME_STOPPING, lambda _: stopped_roots.append(token))\n"
        "    def cleanup():\n"
        "        global disposed\n"
        "        disposed = True\n"
        "    await ctx.effect(lambda: cleanup, label='runtime')\n"
    )
    (stable_artifact / "plugin.py").write_text(source, encoding="utf-8")
    (latest_artifact / "plugin.py").write_text(
        source.replace("version = '1.0.0'", "version = '2.0.0'"),
        encoding="utf-8",
    )
    _write_static_v3_manifest(stable_artifact, "installed_v3", "1.0.0")
    _write_static_v3_manifest(latest_artifact, "installed_v3", "2.0.0")
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    write_plugin_manifest(
        {"installed_v3@lab": True},
        plugins_home=tmp_path / "home",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.generation("installed_v3@lab")
    stable_snapshot = manager.current_snapshot
    assert stable is not None and stable_snapshot is not None
    old_root = stable_snapshot.composition_root
    assert old_root is not None
    runtime_services = asyncio.create_task(manager.run_runtime_services())
    while old_root.instance_token not in manager._runtime_started_roots:
        await asyncio.sleep(0)

    write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
    assert (await manager.reconcile_changed())[0]["publication_state"] == "latest_ready"
    candidate = manager.ready_candidate
    assert candidate is not None and candidate.reload_tx_id is not None
    original_activate = manager._activate_published_generation

    if owner_commit_fails:
        original_recover = manager._recover_stable_root
        original_write_pointers = plugin_manager_module.write_pointers
        recovery_failed = False
        pointer_restore_failed = False

        def fail_owner_commit(*_args: object) -> None:
            raise RuntimeError("candidate owner commit failed")

        async def fail_recovery_once(*args: object, **kwargs: object) -> None:
            nonlocal recovery_failed
            if not recovery_failed:
                recovery_failed = True
                raise RuntimeError("stable Root rebuild failed")
            await original_recover(*args, **kwargs)  # type: ignore[arg-type]

        def fail_pointer_restore_once(*args: object, **kwargs: object):
            nonlocal pointer_restore_failed
            if (
                not pointer_restore_failed
                and kwargs.get("stable") == stable_pointer
                and kwargs.get("latest") == latest_pointer
            ):
                pointer_restore_failed = True
                raise RuntimeError("stable pointer restore failed")
            return original_write_pointers(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            manager,
            "_activate_published_generation",
            fail_owner_commit,
        )
        monkeypatch.setattr(manager, "_recover_stable_root", fail_recovery_once)
        monkeypatch.setattr(
            plugin_manager_module,
            "write_pointers",
            fail_pointer_restore_once,
        )
        with pytest.raises(RuntimeError, match="formal recovery"):
            await manager.switch_ready("installed_v3@lab")
        record = manager.reload_journal.get(candidate.reload_tx_id)
        assert record.phase == "degraded"
        assert record.recovery_target == "base"
        assert read_pointer(plugin_base, "stable") == latest_pointer
        assert read_pointer(plugin_base, "latest") == latest_pointer
        monkeypatch.setattr(manager, "_recover_stable_root", original_recover)
        recovered = await manager.retry_runtime_recovery("installed_v3@lab")
        assert recovered["publication_state"] == "recovered"
    else:
        result = await manager.switch_ready("installed_v3@lab")
        assert result["publication_state"] == "promoted"
        promoted_snapshot = manager.current_snapshot
        assert promoted_snapshot is not None
        promoted_root = promoted_snapshot.composition_root
        assert promoted_root is not None and promoted_root is not old_root
        assert promoted_root.instance_token in manager._runtime_started_roots
        assert old_root.instance_token not in manager._runtime_started_roots
        assert manager.generation("installed_v3@lab") is candidate
        assert manager.ready_candidate is None
        assert candidate.instance.module.started_roots == [
            id(promoted_root.instance_token)
        ]
        assert candidate.instance.module.stopped_roots == []
        assert stable.instance.module.stopped_roots == [id(old_root.instance_token)]
        assert read_pointer(plugin_base, "stable") == latest_pointer
        assert read_pointer(plugin_base, "latest") == latest_pointer
        runtime_services.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runtime_services
        await manager.terminate_all()
        return

    assert manager.current_snapshot is stable_snapshot
    assert manager.generation("installed_v3@lab") is stable
    assert manager.ready_candidate is None
    assert manager.latest_snapshot is stable_snapshot
    replacement_root = stable_snapshot.composition_root
    assert replacement_root is not None and replacement_root is not old_root
    assert replacement_root.instance_token in manager._runtime_started_roots
    assert old_root.instance_token not in manager._runtime_started_roots
    assert candidate.instance.module.disposed is True
    assert candidate.scope.closed is True
    assert read_pointer(plugin_base, "stable") == stable_pointer
    assert read_pointer(plugin_base, "latest") == latest_pointer
    assert stable.instance.module.disposed is False
    assert stable.instance.module.started_roots == [
        id(old_root.instance_token),
        id(replacement_root.instance_token),
    ]
    assert stable.instance.module.stopped_roots == [id(old_root.instance_token)]
    # A candidate that never became public must not receive runtime lifecycle.
    assert candidate.instance.module.started_roots == []
    assert candidate.instance.module.stopped_roots == []
    monkeypatch.setattr(manager, "_activate_published_generation", original_activate)
    runtime_services.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runtime_services
    await manager.terminate_all()
