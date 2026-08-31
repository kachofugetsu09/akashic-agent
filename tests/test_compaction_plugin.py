from __future__ import annotations

import asyncio
import ast
import hashlib
from pathlib import Path

import pytest

from agent.plugin_composition import PROVIDER_REQUEST_PROJECTION
from agent.plugin_composition import ProviderTurnInput, SessionCompactionStorage
from agent.control.context import running_turn_id
from agent.plugins.composable import ComposablePlugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from plugins.compaction import plugin as compaction_plugin
from session.manager import SessionManager
from plugins.compaction.receipts import SqliteCompactionReceipts
from agent.model_runtime.compaction_migration_v1 import (
    compaction_scope_id,
    compaction_source_ref,
)


@pytest.mark.asyncio
async def test_compaction_mounts_as_ordinary_service_with_exact_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    sessions = SessionManager(workspace)
    manager = PluginManager(
        plugin_dirs=[Path(compaction_plugin.__file__).parent],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "cache",
    )

    await manager.load_all()

    snapshot = manager.current_snapshot
    assert snapshot is not None and snapshot.composition_root is not None
    generation = snapshot.generations["compaction"]
    plugin = generation.instance
    assert isinstance(plugin, ComposablePlugin)
    assert plugin.workspace_roots == ()
    assert plugin.workspace_files == ("memory/consolidation_writes.db",)
    assert (
        snapshot.composition_root.context.get(PROVIDER_REQUEST_PROJECTION) is not None
    )
    await manager.terminate_all()
    sessions.close()


@pytest.mark.asyncio
async def test_compaction_candidate_copies_only_receipt_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    memory = workspace / "memory"
    memory.mkdir(parents=True)
    receipt = memory / "consolidation_writes.db"
    receipts = SqliteCompactionReceipts(receipt)
    receipts.write("source:1", {"version": 3})
    secret = memory / "SECRET.md"
    secret.write_text("must not project", encoding="utf-8")
    formal_digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    sessions = SessionManager(workspace)
    manager = PluginManager(
        plugin_dirs=[Path(compaction_plugin.__file__).parent],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()

    candidate = await manager.prepare_candidate("compaction")

    assert candidate is not None and candidate.validation_workspace is not None
    assert not tuple(candidate.validation_workspace.rglob("SECRET.md"))
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == formal_digest
    await manager.discard_prepared("compaction")
    await manager.terminate_all()
    sessions.close()


@pytest.mark.asyncio
async def test_compaction_candidate_invocation_cannot_touch_formal_session(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    sessions = SessionManager(workspace)
    session = sessions.get_or_create("session")
    session.add_message("user", "kept", control_turn_id="turn-1")
    session.add_message("assistant", "kept", control_turn_id="turn-1")
    sessions.save(session)
    manager = PluginManager(
        plugin_dirs=[Path(compaction_plugin.__file__).parent],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    before_db = hashlib.sha256((workspace / "sessions.db").read_bytes()).hexdigest()
    candidate = await manager.prepare_candidate("compaction")
    assert candidate is not None and candidate.runtime_snapshot is not None
    root = candidate.runtime_snapshot.composition_root
    assert root is not None
    turn_token = running_turn_id.set("turn:candidate")
    try:
        grant = session.issue_projection_grant("turn:candidate")
        with pytest.raises(RuntimeError, match="candidate 验证期禁止"):
            _ = await root.context.require(PROVIDER_REQUEST_PROJECTION).open_turn(
                ProviderTurnInput(
                    session_key=session.key,
                    session_created_at=session.created_at.isoformat(),
                    history_units=(),
                    access_grant=grant,
                )
            )
    finally:
        running_turn_id.reset(turn_token)
    assert hashlib.sha256((workspace / "sessions.db").read_bytes()).hexdigest() == before_db
    await manager.discard_prepared("compaction")
    await manager.terminate_all()
    sessions.close()


def test_session_projection_grant_is_turn_and_session_scoped(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)
    for key in ("session-a", "session-b"):
        session = sessions.get_or_create(key)
        session.add_message("user", key, control_turn_id=f"turn:{key}")
        session.add_message("assistant", key, control_turn_id=f"turn:{key}")
        sessions.save(session)
    session_a = sessions.get_existing("session-a")
    token = running_turn_id.set("turn:scope")
    try:
        grant = session_a.issue_projection_grant("turn:scope")
        storage = SessionCompactionStorage(sessions).scope(grant)
        assert storage.history_units("session-a")
        with pytest.raises(PermissionError, match="scope 不匹配"):
            _ = storage.history_units("session-b")
    finally:
        running_turn_id.reset(token)
    with pytest.raises(PermissionError, match="scope 不匹配"):
        _ = storage.history_units("session-a")
    sessions.close()


@pytest.mark.asyncio
async def test_session_projection_grant_is_revoked_for_inherited_child_context(
    tmp_path: Path,
) -> None:
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("session")
    session.add_message("user", "kept", control_turn_id="turn:kept")
    session.add_message("assistant", "kept", control_turn_id="turn:kept")
    sessions.save(session)
    token = running_turn_id.set("turn:lease")
    grant = session.issue_projection_grant("turn:lease")
    storage = SessionCompactionStorage(sessions).scope(grant)
    release = asyncio.Event()

    async def delayed_read() -> None:
        await release.wait()
        with pytest.raises(PermissionError, match="scope 不匹配"):
            _ = storage.history_units(session.key)

    child = asyncio.create_task(delayed_read())
    session.revoke_projection_grant(grant)
    running_turn_id.reset(token)
    release.set()
    await child
    sessions.close()


@pytest.mark.asyncio
async def test_compaction_can_be_disabled_without_a_required_core_service(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    sessions = SessionManager(workspace)
    manager = PluginManager(
        plugin_dirs=[Path(compaction_plugin.__file__).parent],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "cache",
        disabled_builtin_plugins=frozenset({"compaction"}),
    )

    await manager.load_all()

    snapshot = manager.current_snapshot
    assert snapshot is None
    await manager.terminate_all()
    sessions.close()


def test_compaction_plugin_does_not_import_privileged_runtime_owners() -> None:
    source = Path(compaction_plugin.__file__).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported.isdisjoint(
        {
            "agent.plugins.manager",
            "agent.looping.core",
            "core.memory.markdown",
            "core.memory.runtime",
            "session.manager",
            "session.store",
        }
    )
    assert "MEMORY.md" not in source
    assert "SELF.md" not in source
    assert "PENDING.md" not in source


def test_core_and_session_do_not_import_concrete_compaction_plugin() -> None:
    root = Path(__file__).parents[1]
    paths = [
        root / "agent/core",
        root / "agent/looping",
        root / "agent/plugin_composition",
        root / "session/manager.py",
    ]
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for item in paths
        for path in ([item] if item.is_file() else item.rglob("*.py"))
    )
    assert "plugins.compaction" not in sources
    assert "ContextCompactionConfig" not in sources


def test_historical_compaction_identity_is_frozen_without_plugin_import() -> None:
    identity_module = (
        Path(__file__).parents[1] / "agent/model_runtime/compaction_migration_v1.py"
    ).read_text(encoding="utf-8")
    assert "plugins.compaction" not in identity_module
    scope = compaction_scope_id("session", "2026-08-31T00:00:00+00:00")
    assert scope == "session@0a59cea82a844ffa"
    assert (
        compaction_source_ref(scope, 1)
        == "context-compaction:session@0a59cea82a844ffa:1:823ea2ad93fa664f"
    )
