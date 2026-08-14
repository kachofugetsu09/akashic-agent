from __future__ import annotations

# pyright: reportPrivateUsage=false

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from agent.control.errors import ThreadBusyError
from agent.control.service import ControlService
from agent.plugin_composition import (
    AGENT_INPUT,
    AgentInputService,
    CompositionError,
    CompositionRoot,
    Context,
    FiberState,
    PluginRuntime,
)
from agent.plugins.composable import ComposablePlugin
from agent.plugins.artifacts import ArtifactPointer, write_pointers
from agent.plugins.manager import PluginManager
from agent.plugins.manifest import write_plugin_manifest
from agent.plugins.registry import plugin_registry
from bootstrap.plugin_agent_input import ControlAgentInput
from bus.event_bus import EventBus


@pytest.fixture(autouse=True)
def _clean_registry():
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    yield
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()


class _AcceptsDisposedContext(AgentInputService):
    def _require_active_owner(self, ctx: Context) -> str:
        return ctx.runtime.plugin_id


class _ControlHandle:
    id = "turn-1"


class _ControlProbe:
    def __init__(self) -> None:
        self.thread_calls: list[tuple[dict[str, object], str]] = []
        self.turn_calls: list[tuple[str, str, dict[str, object], str | None, bool]] = []
        self.turn_error: BaseException | None = None

    def start_thread(
        self,
        metadata: dict[str, object],
        runtime: str = "stable",
        plugin_rollout_capability: str = "",
    ) -> dict[str, object]:
        del plugin_rollout_capability
        self.thread_calls.append((dict(metadata), runtime))
        return {"id": "session-1"}

    async def start_turn(
        self,
        thread_id: str,
        input_text: str,
        metadata: dict[str, object],
        runtime: str | None = None,
        attached: bool = True,
    ) -> _ControlHandle:
        if self.turn_error is not None:
            raise self.turn_error
        self.turn_calls.append(
            (thread_id, input_text, dict(metadata), runtime, attached)
        )
        return _ControlHandle()


@pytest.mark.asyncio
async def test_agent_input_service_copies_json_and_returns_admission_receipts(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("agent-input-service")
    calls: list[tuple[object, ...]] = []

    async def create_session(
        plugin_id: str,
        metadata: Mapping[str, object],
    ) -> str:
        calls.append(("create", plugin_id, dict(metadata)))
        return "session-1"

    async def submit(
        plugin_id: str,
        session_id: str,
        content: str,
        metadata: Mapping[str, object],
    ) -> str:
        calls.append(("submit", plugin_id, session_id, content, dict(metadata)))
        return "turn-1"

    service = AgentInputService(
        root,
        create_session=create_session,
        submit=submit,
    )
    _ = await root.context.provide(AGENT_INPUT, service)
    plugin_ctx: Context | None = None

    async def plugin(ctx: Context) -> None:
        nonlocal plugin_ctx
        plugin_ctx = ctx

    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    _ = await root.mount(
        plugin,
        name="input-probe",
        inject=(AGENT_INPUT,),
        runtime=_runtime(plugin_dir),
    )
    assert plugin_ctx is not None
    source_metadata: dict[str, object] = {"nested": {"values": [1, 2]}}

    session = await service.create_session(plugin_ctx, metadata=source_metadata)
    cast(dict[str, object], source_metadata["nested"])["values"] = [9]
    receipt = await service.submit(
        plugin_ctx,
        session.id,
        "wake",
        metadata={"event": "issue#1"},
    )

    assert session.id == "session-1"
    assert receipt.session_id == "session-1"
    assert receipt.turn_id == "turn-1"
    assert calls == [
        ("create", "plugin", {"nested": {"values": [1, 2]}}),
        ("submit", "plugin", "session-1", "wake", {"event": "issue#1"}),
    ]
    await root.dispose()


@pytest.mark.asyncio
async def test_agent_input_rejects_loading_foreign_and_invalid_boundaries(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("agent-input-boundary")

    async def create_session(
        plugin_id: str,
        metadata: Mapping[str, object],
    ) -> str:
        del plugin_id, metadata
        return "session-1"

    async def submit(
        plugin_id: str,
        session_id: str,
        content: str,
        metadata: Mapping[str, object],
    ) -> str:
        del plugin_id, session_id, content, metadata
        return "turn-1"

    service = AgentInputService(
        root,
        create_session=create_session,
        submit=submit,
    )
    _ = await root.context.provide(AGENT_INPUT, service)

    async def drives_during_apply(ctx: Context) -> None:
        _ = await service.create_session(ctx)

    failed = await root.mount(
        drives_during_apply,
        name="loading-driver",
        inject=(AGENT_INPUT,),
        runtime=_runtime(tmp_path / "loading"),
    )
    assert failed.state is FiberState.FAILED
    assert any("loading 状态不能提交" in error for error in root.receipt().errors)

    other_root = CompositionRoot("foreign-agent-input")
    foreign_ctx: Context | None = None

    async def foreign(ctx: Context) -> None:
        nonlocal foreign_ctx
        foreign_ctx = ctx

    _ = await other_root.mount(
        foreign,
        name="foreign",
        runtime=_runtime(tmp_path / "foreign"),
    )
    assert foreign_ctx is not None
    with pytest.raises(CompositionError) as caught:
        _ = await service.create_session(foreign_ctx)
    assert caught.value.code == "FOREIGN_AGENT_INPUT_CONTEXT"

    active_ctx: Context | None = None

    async def active(ctx: Context) -> None:
        nonlocal active_ctx
        active_ctx = ctx

    _ = await root.mount(
        active,
        name="active",
        inject=(AGENT_INPUT,),
        runtime=_runtime(tmp_path / "active"),
    )
    assert active_ctx is not None
    with pytest.raises(ValueError, match="lossless JSON"):
        _ = await service.create_session(active_ctx, metadata={"bad": float("nan")})
    with pytest.raises(ValueError, match="lossless JSON"):
        _ = await service.create_session(active_ctx, metadata={"bad": (1, 2)})
    with pytest.raises(TypeError, match="key 必须是字符串"):
        _ = await service.create_session(
            active_ctx,
            metadata={"nested": cast(object, {1: "bad"})},
        )
    with pytest.raises(ValueError, match="content 长度"):
        _ = await service.submit(active_ctx, "session-1", "")

    await other_root.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_agent_input_context_oracle_kills_disposed_owner_mutant(
    tmp_path: Path,
) -> None:
    correct = await _disposed_owner_fixture(
        tmp_path / "correct",
        AgentInputService,
    )
    mutant = await _disposed_owner_fixture(
        tmp_path / "mutant",
        _AcceptsDisposedContext,
    )

    assert correct == (False, 0)
    assert mutant == (True, 1)


@pytest.mark.asyncio
async def test_manager_injects_agent_input_and_rejects_candidate_attempt(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "plugins" / "input_probe"
    plugin_dir.mkdir(parents=True)
    source = _namespace_plugin_source("1.0.0")
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.generation("input_probe")
    snapshot = manager.current_snapshot
    assert stable is not None and snapshot is not None
    assert isinstance(stable.instance, ComposablePlugin)
    assert snapshot.composition_topology is not None
    assert "core.agent_input" in snapshot.composition_topology.services

    with pytest.raises(RuntimeError, match="backend 尚未绑定"):
        _ = await stable.instance.module.create({"source": "probe"})

    calls: list[tuple[object, ...]] = []

    async def create_session(
        plugin_id: str,
        metadata: Mapping[str, object],
    ) -> str:
        calls.append(("create", plugin_id, dict(metadata)))
        return "session-1"

    async def submit(
        plugin_id: str,
        session_id: str,
        content: str,
        metadata: Mapping[str, object],
    ) -> str:
        calls.append(("submit", plugin_id, session_id, content, dict(metadata)))
        return "turn-1"

    manager.bind_agent_input(create_session=create_session, submit=submit)
    session = await stable.instance.module.create({"source": "probe"})
    receipt = await stable.instance.module.submit(session.id, "wake")
    assert receipt.turn_id == "turn-1"
    assert calls == [
        ("create", "input_probe", {"source": "probe"}),
        ("submit", "input_probe", "session-1", "wake", {"kind": "fixture"}),
    ]

    (plugin_dir / "plugin.py").write_text(
        _namespace_plugin_source("2.0.0"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("input_probe")
    assert candidate is not None and isinstance(candidate.instance, ComposablePlugin)
    assert await candidate.instance.module.try_candidate_create() == "denied"
    candidate_snapshot = candidate.runtime_snapshot
    assert candidate_snapshot is not None
    candidate_root = candidate_snapshot.composition_root
    assert candidate_root is not None
    candidate_receipt = candidate_root.receipt()
    assert candidate_receipt.ready is False
    assert [
        (effect.kind, effect.target, effect.outcome)
        for effect in candidate_receipt.external_effects
    ] == [("agent-input", "input_probe:new-session", "denied")]

    await manager.discard_prepared("input_probe", preserve_latest=True)
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_candidate_denied_agent_input_cannot_promote(tmp_path: Path) -> None:
    plugin_base = _write_installed_artifact(
        tmp_path,
        "1.0.0-aaaa",
        _namespace_plugin_source("1.0.0"),
    )
    _ = _write_installed_artifact(
        tmp_path,
        "2.0.0-bbbb",
        _namespace_plugin_source("2.0.0"),
    )
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    _ = write_pointers(
        plugin_base,
        stable=stable_pointer,
        latest=stable_pointer,
    )
    write_plugin_manifest(
        {"input_probe@lab": True},
        plugins_home=tmp_path / "home",
    )
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    stable = manager.generation("input_probe@lab")
    stable_snapshot = manager.current_snapshot
    assert stable is not None and stable_snapshot is not None

    async def create_session(
        plugin_id: str,
        metadata: Mapping[str, object],
    ) -> str:
        del plugin_id, metadata
        return "session-1"

    async def submit(
        plugin_id: str,
        session_id: str,
        content: str,
        metadata: Mapping[str, object],
    ) -> str:
        del plugin_id, session_id, content, metadata
        return "turn-1"

    manager.bind_agent_input(create_session=create_session, submit=submit)
    _ = write_pointers(
        plugin_base,
        stable=stable_pointer,
        latest=latest_pointer,
    )
    result = (await manager.reconcile_changed())[0]
    candidate = manager.ready_candidate
    latest_snapshot = manager.latest_snapshot
    assert result["publication_state"] == "latest_ready"
    assert candidate is not None and latest_snapshot is not None
    assert isinstance(candidate.instance, ComposablePlugin)

    assert await candidate.instance.module.try_candidate_create() == "denied"
    with pytest.raises(RuntimeError, match="组合验证回执未就绪"):
        _ = await manager.switch_ready("input_probe@lab")

    assert manager.generation("input_probe@lab") is stable
    assert manager.current_snapshot is stable_snapshot
    assert manager.latest_snapshot is latest_snapshot
    await manager.drop_candidate("input_probe@lab")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_control_agent_input_stamps_owner_and_preserves_busy() -> None:
    control = _ControlProbe()
    adapter = ControlAgentInput(cast(ControlService, control))

    session_id = await adapter.create_session("watcher", {"repo": "owner/repo"})
    turn_id = await adapter.submit(
        "watcher",
        session_id,
        "inspect issue",
        {"event": "issue#1"},
    )

    assert session_id == "session-1"
    assert turn_id == "turn-1"
    assert control.thread_calls == [
        ({"repo": "owner/repo", "_pluginInputPluginId": "watcher"}, "stable")
    ]
    assert control.turn_calls == [
        (
            "session-1",
            "inspect issue",
            {
                "inboundMetadata": {
                    "event": "issue#1",
                    "_pluginInputPluginId": "watcher",
                }
            },
            "stable",
            False,
        )
    ]

    with pytest.raises(ValueError, match="Core 保留字段"):
        _ = await adapter.create_session("watcher", {"_pluginInputFake": True})
    control.turn_error = ThreadBusyError("busy")
    with pytest.raises(ThreadBusyError, match="busy"):
        _ = await adapter.submit("watcher", "session-1", "again", {})


async def _disposed_owner_fixture(
    root_dir: Path,
    service_type: type[AgentInputService],
) -> tuple[bool, int]:
    """Run one post-dispose call against the correct and intentionally bad service."""

    root = CompositionRoot(f"disposed:{service_type.__name__}")
    calls = 0

    async def create_session(
        plugin_id: str,
        metadata: Mapping[str, object],
    ) -> str:
        nonlocal calls
        del plugin_id, metadata
        calls += 1
        return "session-1"

    async def submit(
        plugin_id: str,
        session_id: str,
        content: str,
        metadata: Mapping[str, object],
    ) -> str:
        del plugin_id, session_id, content, metadata
        return "turn-1"

    service = service_type(
        root,
        create_session=create_session,
        submit=submit,
    )
    _ = await root.context.provide(AGENT_INPUT, service)
    plugin_ctx: Context | None = None

    async def plugin(ctx: Context) -> None:
        nonlocal plugin_ctx
        plugin_ctx = ctx

    root_dir.mkdir(parents=True)
    fiber = await root.mount(
        plugin,
        name="disposed-probe",
        inject=(AGENT_INPUT,),
        runtime=_runtime(root_dir),
    )
    assert plugin_ctx is not None
    await fiber.dispose()
    accepted = True
    try:
        _ = await service.create_session(plugin_ctx)
    except CompositionError:
        accepted = False
    await root.dispose()
    return accepted, calls


def _manager(tmp_path: Path) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


def _runtime(plugin_dir: Path) -> PluginRuntime:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    return PluginRuntime(
        plugin_id=plugin_dir.name,
        plugin_dir=plugin_dir,
        data_dir=plugin_dir / "data",
        workspace=plugin_dir / "workspace",
        config=object(),
    )


def _namespace_plugin_source(version: str) -> str:
    return (
        "from agent.plugin_composition import AGENT_INPUT\n"
        "api_version = 3\n"
        "name = 'input_probe'\n"
        f"version = {version!r}\n"
        "inject = (AGENT_INPUT,)\n"
        "plugin_ctx = None\n"
        "async def apply(ctx, config):\n"
        "    global plugin_ctx\n"
        "    del config\n"
        "    plugin_ctx = ctx\n"
        "async def create(metadata):\n"
        "    service = plugin_ctx.require(AGENT_INPUT)\n"
        "    return await service.create_session(plugin_ctx, metadata=metadata)\n"
        "async def submit(session_id, content):\n"
        "    service = plugin_ctx.require(AGENT_INPUT)\n"
        "    return await service.submit(\n"
        "        plugin_ctx, session_id, content, metadata={'kind': 'fixture'}\n"
        "    )\n"
        "async def try_candidate_create():\n"
        "    try:\n"
        "        await create({'source': 'candidate'})\n"
        "    except PermissionError:\n"
        "        return 'denied'\n"
        "    return 'accepted'\n"
    )


def _write_installed_artifact(
    tmp_path: Path,
    artifact_id: str,
    source: str,
) -> Path:
    plugin_base = tmp_path / "home" / "cache" / "lab" / "input_probe"
    artifact = plugin_base / ".artifacts" / artifact_id
    artifact.mkdir(parents=True)
    _ = (artifact / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_base
