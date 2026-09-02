from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import asyncio
import json
import os
import shutil
import subprocess
import sys

import pytest

from agent.plugin_composition import (
    AddConnection,
    AddModel,
    CapabilitySources,
    CompositionRoot,
    CONVERSATION_SEMANTIC_INTEREST,
    COMMANDS,
    EMBEDDING_MEMORY_PLUGIN,
    EMBEDDINGS,
    EmbeddingSpaceDescriptor,
    INTERACTION_UNDO,
    SNAPSHOT_SEALING,
    RUNTIME_STOPPING,
    RUNTIME_STARTED,
    TOOL_CATALOG,
    UI_SLOTS,
    ModelCapabilities,
    ModelKind,
    SetDefaultModel,
    PluginRuntime,
    PluginCommands,
    PluginTools,
    PluginUiSlots,
    SnapshotSealing,
    RuntimeStopping,
    RuntimeStarted,
)
from agent.plugin_composition.interaction_undo import InteractionUndoService
from agent.plugin_composition.diagnostics import CorePluginDiagnostics
from agent.lifecycle.types import PromptRenderCtx
from agent.lifecycle.composition import PROMPT_RENDER_EVENT
from agent.plugins.manager import PluginManager
from agent.plugins.install import (
    finalize_uninstall_plugin,
    install_git_plugin,
    set_installed_plugin_enabled,
)
from agent.plugins.snapshot import (
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    get_current_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.tools.registry import ToolRegistry
from agent.turn_events.after_turn import AFTER_TURN_COMMITTED
from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted
from core.memory.engine import MemoryQueryResult
from plugins.akasha.engine import AkashaMemoryEngine
from plugins.akasha.plugin import _AkashaRuntimeHandle, _inject_memory
from plugins.akasha import plugin as akasha_plugin
from plugins.models.store import ModelsStore
from session.manager import SessionManager


class _QueryRuntimeStub:
    def __init__(self, result: MemoryQueryResult | None = None) -> None:
        self.query = AsyncMock(return_value=result)


class _RepositoryAkashaImportBlocker:
    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> None:
        _ = path, target
        if fullname == "plugins.akasha" or fullname.startswith("plugins.akasha."):
            raise ModuleNotFoundError(f"repository plugin import blocked: {fullname}")
        return None


def _commit_plugin(repo: Path) -> None:
    for args in (
        ("init",),
        ("config", "user.name", "test"),
        ("config", "user.email", "test@example.com"),
        ("add", "."),
        ("commit", "-m", "init"),
    ):
        result = subprocess.run(
            ("git", *args),
            cwd=repo,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        assert result.returncode == 0, result.stderr


def _diagnostics() -> CorePluginDiagnostics:
    return CorePluginDiagnostics(
        plugin_id="akasha",
        generation_id="test-generation",
        fiber="memory-claim-test",
    )


@pytest.mark.asyncio
async def test_memory_plugin_claim_is_declarative_and_exclusive() -> None:
    root = CompositionRoot("memory-claim")

    async def first_memory(ctx) -> None:
        _ = await ctx.provide(EMBEDDING_MEMORY_PLUGIN, object())

    await root.mount(first_memory, name="first-memory")

    async def second_memory(ctx) -> None:
        _ = await ctx.provide(EMBEDDING_MEMORY_PLUGIN, object())

    await root.mount(second_memory, name="akasha")

    receipt = root.receipt()
    assert receipt.ready is False
    assert receipt.required_pending == ("akasha",)
    assert any(
        incident.owner == "akasha"
        and "DUPLICATE_SERVICE" in incident.message
        and "plugin.claim.embedding_memory" in incident.message
        for incident in receipt.incidents
    )
    await root.dispose()


def _manager(
    tmp_path: Path,
    *plugin_names: str,
) -> tuple[PluginManager, SessionManager]:
    plugin_root = Path(__file__).resolve().parents[1] / "plugins"
    workspace = tmp_path / "workspace"
    sessions = SessionManager(workspace)
    return (
        PluginManager(
            [plugin_root / name for name in plugin_names],
            event_bus=EventBus(),
            tool_registry=ToolRegistry(),
            workspace=workspace,
            session_manager=sessions,
            installed_cache_root=tmp_path / "plugin-home" / "cache",
        ),
        sessions,
    )


@pytest.mark.asyncio
async def test_akasha_starts_as_an_ordinary_memory_provider(tmp_path: Path) -> None:
    manager, sessions = _manager(tmp_path, "akasha", "models", "openai_compatible")
    workspace = tmp_path / "workspace"
    store = ModelsStore(
        workspace / "model-registry.sqlite3",
        backup_dir=workspace / "runtime" / "model-backups",
        writable=True,
    )
    revision = store.add_connection(
        AddConnection(
            expected_revision=0,
            connection_id="fixture-connection",
            name="Fixture",
            driver_id="openai-compatible",
            endpoint="http://127.0.0.1:9/v1",
            auth_identity="fixture-account",
            credential={"driver": "api_key", "access_token": "fixture"},
        )
    )
    revision = store.add_model(
        AddModel(
            expected_revision=revision,
            model_id="fixture-embedding",
            connection_id="fixture-connection",
            kind=ModelKind.EMBEDDING,
            model="fixture-embedding",
            capabilities=ModelCapabilities(
                embedding_dimensions=32,
                embedding_normalization="none",
            ),
            capability_sources=CapabilitySources(
                embedding_dimensions="fixture",
                embedding_normalization="fixture",
            ),
        )
    )
    _ = store.set_default(
        SetDefaultModel(
            expected_revision=revision,
            role=None,
            model_id="fixture-embedding",
        )
    )
    try:
        await manager.load_all()
        assert {item.plugin_id for item in manager.active_plugins()} == {
            "akasha",
            "models",
            "openai-compatible",
        }
        assert manager.current_snapshot is not None
        topology = manager.current_snapshot.composition_topology
        assert topology is not None
        assert "plugin.claim.embedding_memory" in topology.services
    finally:
        await manager.terminate_all()
        sessions.close()


@pytest.mark.asyncio
async def test_akasha_installs_without_repository_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Akasha remains a normal installable artifact when its source tree is absent."""

    repo = tmp_path / "akasha-repo"
    shutil.copytree(Path("plugins/akasha"), repo)
    shutil.rmtree(repo / "__pycache__", ignore_errors=True)
    _commit_plugin(repo)
    workspace = tmp_path / "workspace"
    installed = install_git_plugin(
        workspace=workspace,
        source=str(repo),
        marketplace="ordinary-test",
        plugins_home=tmp_path / "plugin-home",
    )
    for module_name in tuple(sys.modules):
        if module_name == "plugins.akasha" or module_name.startswith(
            "plugins.akasha."
        ):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(
        sys,
        "meta_path",
        [_RepositoryAkashaImportBlocker(), *sys.meta_path],
    )
    plugin_root = Path(__file__).resolve().parents[1] / "plugins"

    def installed_manager() -> tuple[PluginManager, SessionManager]:
        sessions = SessionManager(workspace)
        return (
            PluginManager(
                [plugin_root / "models", plugin_root / "openai_compatible"],
                event_bus=EventBus(),
                workspace=workspace,
                session_manager=sessions,
                installed_cache_root=tmp_path / "plugin-home" / "cache",
            ),
            sessions,
        )

    manager, sessions = installed_manager()
    try:
        await manager.load_all()
        generation = manager.generation("akasha@ordinary-test")
        assert generation is not None
        assert generation.plugin_dir == installed.installed_path
        assert generation.source_type == "installed"
        package = generation.instance.module.__package__
        assert package
        installed_modules = [
            module
            for module_name, module in sys.modules.items()
            if module_name == package or module_name.startswith(f"{package}.")
        ]
        assert installed_modules
        for module in installed_modules:
            module_file = module.__file__
            if module_file is not None:
                assert Path(module_file).resolve().is_relative_to(
                    installed.installed_path
                )
    finally:
        await manager.terminate_all()
        sessions.close()

    set_installed_plugin_enabled(
        "akasha@ordinary-test",
        enabled=False,
        plugins_home=tmp_path / "plugin-home",
    )
    _ = finalize_uninstall_plugin(
        "akasha@ordinary-test",
        workspace=workspace,
        plugins_home=tmp_path / "plugin-home",
    )
    without, sessions = installed_manager()
    try:
        await without.load_all()
        assert without.generation("akasha@ordinary-test") is None
    finally:
        await without.terminate_all()
        sessions.close()

    _ = install_git_plugin(
        workspace=workspace,
        source=str(repo),
        marketplace="ordinary-test",
        plugins_home=tmp_path / "plugin-home",
    )
    restored, sessions = installed_manager()
    try:
        await restored.load_all()
        assert restored.generation("akasha@ordinary-test") is not None
    finally:
        await restored.terminate_all()
        sessions.close()


@pytest.mark.asyncio
async def test_akasha_without_embedding_degrades_prompt_and_wake_scoring(
    tmp_path: Path,
) -> None:
    """An optional memory lane must not break a normal Turn or Wake maintenance."""

    manager, sessions = _manager(tmp_path, "akasha", "models", "openai_compatible")
    try:
        await manager.load_all()
        snapshot = manager.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        root = snapshot.composition_root
        embedding_health = next(
            item for item in root.receipt().health if item.name == "embedding"
        )
        assert embedding_health.required is False
        assert embedding_health.healthy is False
        assert embedding_health.reason

        prompt = PromptRenderCtx(
            session_key="web:one",
            channel="web",
            chat_id="one",
            content="hello",
            media=None,
            timestamp=datetime(2026, 8, 29, tzinfo=UTC),
            history=[],
            skill_names=[],
            disabled_sections=set(),
            turn_injection_prompt="",
        )
        lease = await manager._snapshot_store.acquire()
        token = bind_runtime_snapshot(lease)
        try:
            await root.context.serial(PROMPT_RENDER_EVENT, prompt)
            assert snapshot.tool_registry is not None
            snapshot.tool_registry.set_context(turn_id="turn:memory-unavailable")
            result = await snapshot.tool_registry.execute(
                "recall_memory",
                {"query": "hello"},
                raise_errors=True,
            )
            feedback_result = await snapshot.tool_registry.execute(
                "remember_memory",
                {
                    "message_ids": ["current_user_message"],
                    "reason": "remember this correction",
                },
                raise_errors=True,
            )
        finally:
            reset_runtime_snapshot(token)
            await lease.release()
        assert prompt.system_sections_bottom == []
        assert isinstance(result, str)
        payload = json.loads(result)
        assert payload["count"] == 0
        assert payload["items"] == []
        assert payload["error"] == "memory_unavailable"
        assert payload["reason"] == embedding_health.reason
        assert isinstance(feedback_result, str)
        assert json.loads(feedback_result) == {
            "status": "not_staged",
            "error": "memory_unavailable",
            "reason": embedding_health.reason,
        }

        semantic = root.context.require(CONVERSATION_SEMANTIC_INTEREST)
        scores = await semantic.score(
            ("due content",),
            cutoff="2026-08-29T00:00:00+00:00",
        )
        assert scores == (0.0,)
    finally:
        await manager.terminate_all()
        sessions.close()


@pytest.mark.asyncio
async def test_akasha_stops_using_an_old_embedding_after_default_changes(
    tmp_path: Path,
) -> None:
    """A live kernel must not keep writing its old space after settings change."""

    selected = {"identity": "space-a"}
    projected: list[str] = []

    class Runtime(AkashaMemoryEngine):
        closeables: tuple[object, ...] = ()

        def __init__(self) -> None:
            pass

        @property
        def embedding_api(self):
            return type("EmbeddingApi", (), {"model_id": "space-a"})()

        async def project_committed_turn(self, event: TurnCommitted) -> None:
            projected.append(event.turn_id)

    handle = _AkashaRuntimeHandle()
    handle.configure(
        Runtime,
        embedding_identity=lambda: selected["identity"],
    )
    assert handle.available() is True
    selected["identity"] = "space-b"
    assert handle.available() is False
    assert handle.model_id == ""
    await handle.project_committed_turn(
        TurnCommitted(
            session_key="test:one",
            channel="test",
            chat_id="one",
            input_message="hello",
            persisted_user_message="hello",
            assistant_response="world",
            tools_used=[],
            turn_id="turn:changed",
        )
    )
    assert projected == []

    prompt = PromptRenderCtx(
        session_key="web:one",
        channel="web",
        chat_id="one",
        content="hello",
        media=None,
        timestamp=datetime(2026, 8, 29, tzinfo=UTC),
        history=[],
        skill_names=[],
        disabled_sections=set(),
        turn_injection_prompt="",
    )
    await _inject_memory(prompt, handle, _diagnostics())
    assert prompt.system_sections_bottom == []

    selected["identity"] = "space-a"
    assert handle.available() is True


@pytest.mark.asyncio
async def test_akasha_post_commit_worker_uses_the_source_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The detached projector must bind the exact lease captured by its event."""

    projected = asyncio.Event()

    class Embeddings:
        def describe(self, *, model_id: str | None = None):
            _ = model_id
            return type("Descriptor", (), {"identity": "test-space"})()

        async def assert_bound(self) -> None:
            assert get_current_runtime_snapshot() is not None

    embeddings = Embeddings()

    class Runtime:
        closeables: tuple[object, ...] = ()
        embedding_api = type("EmbeddingApi", (), {"model_id": "test-space"})()

        async def project_committed_turn(self, event: TurnCommitted) -> None:
            assert event.turn_id == "turn:queued"
            await embeddings.assert_bound()
            projected.set()

    runtime = Runtime()
    monkeypatch.setattr(akasha_plugin, "_build_runtime", lambda **_: runtime)

    async def skip_tools(*_args: object) -> None:
        return None

    monkeypatch.setattr(akasha_plugin, "_register_tools", skip_tools)
    root = CompositionRoot("akasha-post-commit-scope")
    store = RuntimeSnapshotStore()
    root._bind_runtime_scope_acquirer(
        lambda: store.acquire_composition_root(root)
    )
    _ = await root.context.provide(EMBEDDINGS, embeddings)
    _ = await root.context.provide(COMMANDS, PluginCommands())
    _ = await root.context.provide(TOOL_CATALOG, PluginTools(root.instance_token))
    _ = await root.context.provide(UI_SLOTS, PluginUiSlots())
    _ = await root.context.provide(
        INTERACTION_UNDO,
        InteractionUndoService.candidate_validation(),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sessions.db").touch()
    _ = await root.mount(
        lambda ctx: akasha_plugin.apply(ctx, None),
        name="akasha",
        inject=(COMMANDS, EMBEDDINGS, INTERACTION_UNDO, TOOL_CATALOG, UI_SLOTS),
        runtime=PluginRuntime(
            plugin_id="akasha",
            generation_id="akasha:test",
            plugin_dir=Path("plugins/akasha").resolve(),
            data_dir=tmp_path / "plugin-data",
            workspace=workspace,
            config=None,
            workspace_roots=("memory",),
            workspace_files=("sessions.db",),
        ),
    )
    assert root.receipt().ready, [item.message for item in root.receipt().incidents]
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    store.install(snapshot)
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        await root.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
        root.context.emit(
            AFTER_TURN_COMMITTED,
            TurnCommitted(
                session_key="test:one",
                channel="test",
                chat_id="one",
                input_message="hello",
                persisted_user_message="hello",
                assistant_response="world",
                tools_used=[],
                turn_id="turn:queued",
            ),
        )
        await asyncio.wait_for(projected.wait(), timeout=1)
        await asyncio.sleep(0)
        assert snapshot.lease_count == 1
        await root.context.serial(RUNTIME_STOPPING, RuntimeStopping())
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
        await root.dispose()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_fails", [False, True])
async def test_akasha_reindex_worker_runs_after_public_start_and_retains_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repair_fails: bool,
) -> None:
    """Hot-start repair uses the public Root and clears intent only on success."""

    attempted = asyncio.Event()
    finished: list[Path] = []

    class Embeddings:
        def describe(self, *, model_id: str | None = None):
            _ = model_id
            return SimpleNamespace(identity="test-space", model_id="embedding")

    class Runtime:
        closeables: tuple[object, ...] = ()
        embedding_api = SimpleNamespace(model_id="test-space")

    async def fake_reindex(**kwargs: object):
        runtime_scope = kwargs["runtime_scope"]
        async with runtime_scope():  # type: ignore[operator]
            assert get_current_runtime_snapshot() is not None
        attempted.set()
        if repair_fails:
            raise RuntimeError("injected repair failure")
        return SimpleNamespace(embedded_messages=2)

    monkeypatch.setattr(akasha_plugin, "_build_runtime", lambda **_: Runtime())
    monkeypatch.setattr(akasha_plugin, "_register_tools", AsyncMock())
    monkeypatch.setattr(akasha_plugin, "load_request", lambda _root: object())
    monkeypatch.setattr(akasha_plugin, "reindex", fake_reindex)
    monkeypatch.setattr(
        akasha_plugin,
        "finish_request",
        lambda root: finished.append(root),
    )

    root = CompositionRoot("akasha-reindex-hot-start")
    store = RuntimeSnapshotStore()
    root._bind_runtime_scope_acquirer(lambda: store.acquire_composition_root(root))
    _ = await root.context.provide(EMBEDDINGS, Embeddings())
    _ = await root.context.provide(COMMANDS, PluginCommands())
    _ = await root.context.provide(TOOL_CATALOG, PluginTools(root.instance_token))
    _ = await root.context.provide(UI_SLOTS, PluginUiSlots())
    _ = await root.context.provide(
        INTERACTION_UNDO,
        InteractionUndoService.candidate_validation(),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sessions.db").touch()
    _ = await root.mount(
        lambda ctx: akasha_plugin.apply(ctx, None),
        name="akasha",
        inject=(COMMANDS, EMBEDDINGS, INTERACTION_UNDO, TOOL_CATALOG, UI_SLOTS),
        runtime=PluginRuntime(
            plugin_id="akasha",
            generation_id="akasha:repair",
            plugin_dir=Path("plugins/akasha").resolve(),
            data_dir=tmp_path / "plugin-data",
            workspace=workspace,
            config=None,
            workspace_roots=("memory",),
            workspace_files=("sessions.db",),
        ),
    )
    await root.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    store.install(snapshot)
    await root.context.serial(RUNTIME_STARTED, RuntimeStarted())
    await asyncio.wait_for(attempted.wait(), timeout=1)
    await asyncio.sleep(0)

    assert finished == ([] if repair_fails else [tmp_path / "plugin-data"])
    if repair_fails:
        assert any(
            incident.kind == "akasha.reindex_failed"
            for incident in root.receipt().incidents
        )
    await root.context.serial(RUNTIME_STOPPING, RuntimeStopping())
    await root.dispose()
    await store.close()


@pytest.mark.asyncio
async def test_akasha_reindex_cancel_retains_request_for_fresh_root_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root retirement keeps repair intent and releases its exact scope."""

    descriptor = EmbeddingSpaceDescriptor(
        plugin_snapshot_id="snapshot",
        model_revision=1,
        model_id="embedding",
        connection_id="connection",
        driver_id="driver",
        driver_contract_version="1",
        auth_identity="account",
        connection_fingerprint="endpoint",
        model="embedding",
        dimensions=3,
        normalization="none",
        capability_digest="caps",
    )
    entered = asyncio.Event()
    block_first = asyncio.Event()
    request_finished = asyncio.Event()
    calls = 0
    bound_snapshots: list[object] = []
    data_root = tmp_path / "plugin-data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sessions.db").touch()
    _ = akasha_plugin.save_request(data_root, descriptor)

    class Embeddings:
        def describe(self, *, model_id: str | None = None):
            _ = model_id
            return descriptor

    class Runtime:
        closeables: tuple[object, ...] = ()
        embedding_api = SimpleNamespace(model_id=descriptor.identity)

    async def fake_reindex(**kwargs: object):
        nonlocal calls
        calls += 1
        runtime_scope = kwargs["runtime_scope"]
        async with runtime_scope():  # type: ignore[operator]
            snapshot = get_current_runtime_snapshot()
            assert snapshot is not None
            bound_snapshots.append(snapshot)
            if calls == 1:
                entered.set()
                await block_first.wait()
        return SimpleNamespace(embedded_messages=2)

    real_finish_request = akasha_plugin.finish_request

    def finish_request(root: Path) -> None:
        real_finish_request(root)
        request_finished.set()

    monkeypatch.setattr(akasha_plugin, "_build_runtime", lambda **_: Runtime())
    monkeypatch.setattr(akasha_plugin, "_register_tools", AsyncMock())
    monkeypatch.setattr(akasha_plugin, "reindex", fake_reindex)
    monkeypatch.setattr(akasha_plugin, "finish_request", finish_request)

    async def mount_root(generation: str):
        root = CompositionRoot(generation)
        store = RuntimeSnapshotStore()
        root._bind_runtime_scope_acquirer(
            lambda: store.acquire_composition_root(root)
        )
        _ = await root.context.provide(EMBEDDINGS, Embeddings())
        _ = await root.context.provide(COMMANDS, PluginCommands())
        _ = await root.context.provide(
            TOOL_CATALOG,
            PluginTools(root.instance_token),
        )
        _ = await root.context.provide(UI_SLOTS, PluginUiSlots())
        _ = await root.context.provide(
            INTERACTION_UNDO,
            InteractionUndoService.candidate_validation(),
        )
        _ = await root.mount(
            lambda ctx: akasha_plugin.apply(ctx, None),
            name="akasha",
            inject=(
                COMMANDS,
                EMBEDDINGS,
                INTERACTION_UNDO,
                TOOL_CATALOG,
                UI_SLOTS,
            ),
            runtime=PluginRuntime(
                plugin_id="akasha",
                generation_id=generation,
                plugin_dir=Path("plugins/akasha").resolve(),
                data_dir=data_root,
                workspace=workspace,
                config=None,
                workspace_roots=("memory",),
                workspace_files=("sessions.db",),
            ),
        )
        await root.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
        snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
        store.install(snapshot)
        await root.context.serial(RUNTIME_STARTED, RuntimeStarted())
        return root, store, snapshot

    first_root, first_store, first_snapshot = await mount_root("akasha:first")
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert akasha_plugin.load_request(data_root) is not None
    assert first_snapshot.lease_count == 1

    await first_root.dispose()
    assert first_snapshot.lease_count == 0
    assert akasha_plugin.load_request(data_root) is not None
    assert not request_finished.is_set()
    await first_store.close()

    second_root, second_store, second_snapshot = await mount_root("akasha:second")
    await asyncio.wait_for(request_finished.wait(), timeout=1)
    assert akasha_plugin.load_request(data_root) is None
    assert calls == 2
    assert bound_snapshots == [first_snapshot, second_snapshot]
    await second_root.context.serial(RUNTIME_STOPPING, RuntimeStopping())
    await second_root.dispose()
    await second_store.close()


@pytest.mark.asyncio
async def test_akasha_injects_recall_as_an_ordinary_prompt_section() -> None:
    runtime = _QueryRuntimeStub(
        MemoryQueryResult(
            text_block="embedded recall",
            records=[],
            raw={},
        )
    )
    event = PromptRenderCtx(
        session_key="web:one",
        channel="web",
        chat_id="one",
        content="hello",
        media=None,
        timestamp=datetime(2026, 8, 25, tzinfo=UTC),
        history=[],
        skill_names=[],
        disabled_sections=set(),
        turn_injection_prompt="",
    )

    await _inject_memory(event, runtime, _diagnostics())

    assert [(item.name, item.content) for item in event.system_sections_bottom] == [
        ("memory", "embedded recall")
    ]


@pytest.mark.asyncio
async def test_akasha_prompt_section_obeys_generic_disable_switch() -> None:
    runtime = _QueryRuntimeStub()
    event = PromptRenderCtx(
        session_key="scheduler:one",
        channel="scheduler",
        chat_id="one",
        content="tick",
        media=None,
        timestamp=datetime(2026, 8, 25, tzinfo=UTC),
        history=[],
        skill_names=[],
        disabled_sections={"memory"},
        turn_injection_prompt="",
    )

    await _inject_memory(event, runtime, _diagnostics())

    runtime.query.assert_not_awaited()
    assert event.system_sections_bottom == []
