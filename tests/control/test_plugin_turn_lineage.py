from __future__ import annotations

from pathlib import Path

import pytest

from agent.control.models import TurnRequest
from agent.control.runtime import ConversationRuntime
from agent.control.service import ControlService
from session.manager import SessionManager


@pytest.mark.asyncio
async def test_attached_programmatic_child_automatically_binds_frozen_latest(
    tmp_path: Path,
) -> None:
    observed: list[TurnRequest] = []

    async def execute(request: TurnRequest) -> str:
        observed.append(request)
        return "ok"

    sessions = SessionManager(tmp_path)
    runtime = ConversationRuntime(sessions.control_store, execute)

    def binding(owner_turn_id: str, attached: bool) -> dict[str, str] | None:
        assert owner_turn_id == "parent-1"
        if not attached:
            return None
        return {
            "runtime": "latest",
            "ownerTurnId": owner_turn_id,
            "pluginId": "fitbit@github",
            "generationId": "gen-2",
            "sourceRevision": "rev-2",
        }

    service = ControlService(
        runtime,
        sessions,
        tmp_path,
        plugin_child_binding=binding,
    )
    thread = service.start_thread(
        {"_pluginRolloutOwnerTurnId": "parent-1"},
    )
    handle = await service.start_turn(str(thread["id"]), "verify", {}, attached=True)
    await handle.result()

    assert observed[0].metadata["runtime"] == "latest"
    assert observed[0].metadata["_pluginRolloutGenerationId"] == "gen-2"
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_detached_programmatic_child_does_not_inherit_candidate(
    tmp_path: Path,
) -> None:
    observed: list[TurnRequest] = []

    async def execute(request: TurnRequest) -> str:
        observed.append(request)
        return "ok"

    sessions = SessionManager(tmp_path)
    runtime = ConversationRuntime(sessions.control_store, execute)
    service = ControlService(
        runtime,
        sessions,
        tmp_path,
        plugin_child_binding=lambda _owner, _attached: None,
    )
    thread = service.start_thread(
        {"_pluginRolloutOwnerTurnId": "parent-1"},
    )
    handle = await service.start_turn(str(thread["id"]), "verify", {}, attached=False)
    await handle.result()

    assert observed[0].metadata["runtime"] == "stable"
    assert "_pluginRolloutGenerationId" not in observed[0].metadata
    await runtime.shutdown()
