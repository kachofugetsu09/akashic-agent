from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from agent.plugin_composition import InteractionUndoService
from agent.plugins.composable import ComposablePlugin
from agent.plugins.interaction_undo import InteractionUndoCoordinator
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from session.manager import SessionManager


def _seed_interaction(
    manager: SessionManager,
    *,
    session_key: str = "cli:undo",
    turn_id: str = "turn:undo",
) -> tuple[str, ...]:
    now = datetime.now(UTC).isoformat()
    rows = manager.control_store.persist_session(
        session_key,
        created_at=now,
        updated_at=now,
        metadata={},
        messages=[
            {
                "role": "user",
                "content": "question",
                "timestamp": now,
                "extra": {
                    "control_turn_id": turn_id,
                    "turn_input_ordinal": 0,
                },
            },
            {
                "role": "assistant",
                "content": "answer",
                "timestamp": now,
                "extra": {
                    "control_turn_id": turn_id,
                    "turn_terminal": True,
                    "turn_input_count": 1,
                },
            },
        ],
    )
    return tuple(str(row["id"]) for row in rows)


@pytest.mark.asyncio
async def test_candidate_interaction_undo_has_no_destructive_owner() -> None:
    service = InteractionUndoService.candidate_validation()

    with pytest.raises(RuntimeError, match="candidate 验证期禁止"):
        await service.undo_latest("cli:undo")


@pytest.mark.asyncio
async def test_undo_runs_bound_source_fence_and_invalidates_session(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    message_ids = _seed_interaction(sessions)
    _ = sessions.get_existing("cli:undo")
    coordinator = InteractionUndoCoordinator(cast(Any, sessions))
    service = InteractionUndoService(coordinator.undo_latest)
    fenced: list[str] = []

    async def fence(control_turn_id, delete_source):
        fenced.append(control_turn_id)
        return delete_source()

    cleanup = service.bind_source_fence(fence)
    result = await service.undo_latest("cli:undo")

    assert result is not None
    assert result.message_ids == message_ids
    assert result.reconciliation_pending is False
    assert fenced == ["turn:undo"]
    assert sessions.get_existing("cli:undo").messages == []
    cleanup()


@pytest.mark.asyncio
async def test_manager_keeps_external_plugin_undo_contract(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    sessions = SessionManager(workspace)
    message_ids = _seed_interaction(sessions)
    plugin_dir = tmp_path / "plugins" / "plugin_undo"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "from agent.plugin_composition import INTERACTION_UNDO\n"
        "api_version = 3\n"
        "name = 'plugin_undo'\n"
        "version = '2.0.0'\n"
        "inject = (INTERACTION_UNDO,)\n"
        "service = None\n"
        "async def apply(ctx, config):\n"
        "    global service\n"
        "    service = ctx.require(INTERACTION_UNDO)\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "cache",
    )

    try:
        await manager.load_all()
        generation = manager.generation("plugin_undo")
        assert generation is not None
        assert isinstance(generation.instance, ComposablePlugin)
        service = generation.instance.module.service
        assert isinstance(service, InteractionUndoService)
        result = await service.undo_latest("cli:undo")
        assert result is not None and result.message_ids == message_ids
    finally:
        await manager.terminate_all()
