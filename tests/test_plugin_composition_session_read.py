from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import pytest

from agent.plugin_composition import SessionReadService, SessionReadSnapshot
from agent.plugins.composable import ComposablePlugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from session.manager import SessionManager


def _database_snapshot(workspace: Path) -> dict[str, tuple[int, str]]:
    return {
        path.name: (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in workspace.glob("sessions.db*")
        if path.is_file()
    }


def test_session_read_returns_detached_existing_snapshot() -> None:
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "original"}],
        }
    ]
    session = SimpleNamespace(messages=messages, last_consolidated=1)
    compaction = SimpleNamespace(generation=1, consolidated_through_seq=1)
    service = SessionReadService(cast(Any, lambda _key: (session, compaction)))

    snapshot = service.read("mobile:one")

    assert snapshot is not None
    assert snapshot.session_key == "mobile:one"
    assert snapshot.compaction_generation == 1
    assert snapshot.consolidated_through_seq == 1
    assert isinstance(snapshot.messages[0], MappingProxyType)
    messages[0]["role"] = "assistant"
    nested = cast(list[dict[str, object]], snapshot.messages[0]["content"])
    nested[0]["text"] = "snapshot-only"
    assert snapshot.messages[0]["role"] == "user"
    original = cast(list[dict[str, object]], messages[0]["content"])
    assert original[0]["text"] == "original"


def test_session_read_missing_and_candidate_boundaries_fail_loud() -> None:
    calls: list[str] = []

    def missing(session_key: str):
        calls.append(session_key)
        raise KeyError(session_key)

    formal = SessionReadService(missing)
    candidate = SessionReadService.candidate_validation()
    assert formal.formal is True
    assert candidate.formal is False
    assert formal.read("mobile:missing") is None
    assert calls == ["mobile:missing"]
    with pytest.raises(RuntimeError, match="candidate 验证期禁止"):
        candidate.read("mobile:existing")

    inconsistent = SimpleNamespace(messages=[], last_consolidated=2)
    active = SimpleNamespace(generation=1, consolidated_through_seq=3)
    with pytest.raises(RuntimeError, match="active compaction generation 不一致"):
        SessionReadService(cast(Any, lambda _key: (inconsistent, active))).read(
            "mobile:inconsistent"
        )


@pytest.mark.asyncio
async def test_manager_injects_formal_session_read_without_persistence_write(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    sessions = SessionManager(workspace)
    session = sessions.get_or_create("mobile:existing")
    session.messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    sessions.save(session)
    sessions.invalidate(session.key)
    before = _database_snapshot(workspace)
    plugin_dir = tmp_path / "plugins" / "session_read_probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "from agent.plugin_composition import SESSION_READ\n"
        "api_version = 3\n"
        "name = 'session_read_probe'\n"
        "version = '1.0.0'\n"
        "inject = (SESSION_READ,)\n"
        "snapshot = None\n"
        "async def apply(ctx, config):\n"
        "    global snapshot\n"
        "    snapshot = ctx.require(SESSION_READ).read('mobile:existing')\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    try:
        await manager.load_all()

        generation = manager.generation("session_read_probe")
        current = manager.current_snapshot
        assert generation is not None and current is not None
        assert isinstance(generation.instance, ComposablePlugin)
        snapshot = cast(SessionReadSnapshot, generation.instance.module.snapshot)
        assert tuple(message["role"] for message in snapshot.messages) == (
            "user",
            "assistant",
        )
        assert snapshot.compaction_generation is None
        assert snapshot.consolidated_through_seq is None
        assert _database_snapshot(workspace) == before
        assert current.composition_topology is not None
        assert "core.session_read" in current.composition_topology.services

        root = current.composition_root
        assert root is not None
        await manager.terminate_all()
        assert root.receipt().services == ()
        assert root.receipt().effects == ()
    finally:
        sessions.close()


@pytest.mark.asyncio
async def test_candidate_session_read_fails_without_touching_stable_session(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    sessions = SessionManager(workspace)
    session = sessions.get_or_create("mobile:existing")
    session.messages = [{"role": "user", "content": "protected"}]
    sessions.save(session)
    plugin_dir = tmp_path / "plugins" / "session_read_probe"
    plugin_dir.mkdir(parents=True)
    plugin_path = plugin_dir / "plugin.py"
    plugin_path.write_text(
        "from agent.plugin_composition import SESSION_READ\n"
        "api_version = 3\n"
        "name = 'session_read_probe'\n"
        "version = '1.0.0'\n"
        "inject = (SESSION_READ,)\n"
        "async def apply(ctx, config):\n"
        "    ctx.require(SESSION_READ)\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    try:
        await manager.load_all()
        stable = manager.current_snapshot
        assert stable is not None
        before = _database_snapshot(workspace)
        plugin_path.write_text(
            "from agent.plugin_composition import SESSION_READ\n"
            "api_version = 3\n"
            "name = 'session_read_probe'\n"
            "version = '2.0.0'\n"
            "inject = (SESSION_READ,)\n"
            "async def apply(ctx, config):\n"
            "    ctx.require(SESSION_READ).read('mobile:existing')\n",
            encoding="utf-8",
        )

        candidate = await manager.prepare_candidate("session_read_probe")

        assert candidate is None
        assert manager.current_snapshot is stable
        assert manager.prepared_generation("session_read_probe") is None
        assert _database_snapshot(workspace) == before
    finally:
        await manager.terminate_all()
        sessions.close()
