from __future__ import annotations

# pyright: reportPrivateUsage=false

import hashlib
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import pytest

from agent.plugin_composition import SessionReadService, SessionReadSnapshot
from agent.plugins.composable import ComposablePlugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from session.manager import SessionManager


class _SessionSource:
    def __init__(self, session: object | None) -> None:
        self.session = session
        self.existing_calls: list[str] = []
        self.create_calls: list[str] = []

    def get_existing(self, session_key: str) -> Any:
        self.existing_calls.append(session_key)
        if self.session is None:
            raise KeyError(session_key)
        return self.session

    def get_or_create(self, session_key: str) -> Any:
        self.create_calls.append(session_key)
        return SimpleNamespace(messages=[], last_consolidated=0)


def _database_snapshot(workspace: Path) -> dict[str, tuple[int, int, str]]:
    return {
        path.name: (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in workspace.glob("sessions.db*")
        if path.is_file()
    }


def test_session_read_returns_detached_existing_snapshot() -> None:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "original"}],
        }
    ]
    session = SimpleNamespace(messages=messages, last_consolidated=1)
    source = _SessionSource(session)

    snapshot = SessionReadService(source.get_existing).read("mobile:one")

    assert snapshot is not None
    assert snapshot.session_key == "mobile:one"
    assert snapshot.last_consolidated == 1
    assert isinstance(snapshot.messages[0], MappingProxyType)
    with pytest.raises(TypeError):
        cast(dict[str, object], snapshot.messages[0])["role"] = "mutated"
    messages[0]["role"] = "assistant"
    cast(list[dict[str, object]], snapshot.messages[0]["content"])[0][
        "text"
    ] = "snapshot-only"
    assert snapshot.messages[0]["role"] == "user"
    assert cast(list[dict[str, object]], messages[0]["content"])[0]["text"] == (
        "original"
    )
    assert source.existing_calls == ["mobile:one"]
    assert source.create_calls == []


def test_missing_session_oracle_kills_create_on_read_mutant() -> None:
    correct = _read_missing(lambda source: SessionReadService(source.get_existing))
    mutant = _read_missing(lambda source: SessionReadService(source.get_or_create))

    assert correct == (None, ["mobile:missing"], [])
    assert isinstance(mutant[0], SessionReadSnapshot)
    assert mutant[1:] == ([], ["mobile:missing"])


@pytest.mark.asyncio
async def test_namespace_loader_injects_read_only_session_service(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    sessions = SessionManager(workspace)
    session = sessions.get_or_create("mobile:existing")
    session.messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    session.last_consolidated = 2
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

    await manager.load_all()

    generation = manager.generation("session_read_probe")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    assert isinstance(generation.instance, ComposablePlugin)
    view = cast(SessionReadSnapshot, generation.instance.module.snapshot)
    assert tuple(message["role"] for message in view.messages) == (
        "user",
        "assistant",
    )
    assert _database_snapshot(workspace) == before
    assert snapshot.composition_topology is not None
    assert "core.session_read" in snapshot.composition_topology.services

    root = snapshot.composition_root
    assert root is not None
    await manager.terminate_all()
    assert root.receipt().services == ()
    assert root.receipt().effects == ()


def _read_missing(
    factory: Callable[[_SessionSource], SessionReadService],
) -> tuple[SessionReadSnapshot | None, list[str], list[str]]:
    source = _SessionSource(None)
    result = factory(source).read("mobile:missing")
    return result, source.existing_calls, source.create_calls
