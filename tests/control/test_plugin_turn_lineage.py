from __future__ import annotations

from pathlib import Path

import pytest

from agent.control.models import TurnRequest
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from agent.control.service import ControlService
from session.manager import SessionManager


@pytest.mark.asyncio
async def test_attached_programmatic_child_automatically_binds_frozen_latest(
    tmp_path: Path,
) -> None:
    observed: list[TurnRequest] = []

    async def execute(request: TurnRequest) -> ControlExecutionResult:
        observed.append(request)
        return ControlExecutionResult(response="ok")

    sessions = SessionManager(tmp_path)
    runtime = ConversationRuntime(sessions.control_store, execute)

    consumed = False

    def binding(capability: str, consume: bool) -> dict[str, str] | None:
        nonlocal consumed
        assert capability == "opaque-child-capability"
        if consume and consumed:
            return None
        consumed = consume
        return {
            "runtime": "latest",
            "ownerTurnId": "parent-1",
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
    thread = service.start_thread({}, plugin_rollout_capability="opaque-child-capability")
    handle = await service.start_turn(
        str(thread["id"]),
        "verify",
        {},
        attached=True,
    )
    await handle.result()

    assert observed[0].metadata["runtime"] == "latest"
    assert observed[0].metadata["_pluginRolloutGenerationId"] == "gen-2"
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_detached_programmatic_child_capability_is_rejected(
    tmp_path: Path,
) -> None:
    observed: list[TurnRequest] = []

    async def execute(request: TurnRequest) -> ControlExecutionResult:
        observed.append(request)
        return ControlExecutionResult(response="ok")

    sessions = SessionManager(tmp_path)
    runtime = ConversationRuntime(sessions.control_store, execute)
    service = ControlService(
        runtime,
        sessions,
        tmp_path,
        plugin_child_binding=lambda _capability, _consume: {
            "runtime": "latest",
            "ownerTurnId": "parent-1",
            "pluginId": "fitbit@github",
            "generationId": "gen-2",
            "sourceRevision": "rev-2",
        },
    )
    thread = service.start_thread({}, plugin_rollout_capability="opaque")
    with pytest.raises(ValueError, match="必须 attached"):
        await service.start_turn(str(thread["id"]), "verify", {}, attached=False)

    assert observed == []
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_detached_programmatic_child_cannot_request_latest(
    tmp_path: Path,
) -> None:
    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        return ControlExecutionResult(response="unused")

    sessions = SessionManager(tmp_path)
    runtime = ConversationRuntime(sessions.control_store, execute)
    service = ControlService(
        runtime,
        sessions,
        tmp_path,
        plugin_child_binding=lambda _capability, _consume: None,
    )
    thread = service.start_thread({})

    with pytest.raises(ValueError, match="attached 插件验证子 turn"):
        await service.start_turn(
            str(thread["id"]),
            "verify",
            {},
            runtime="latest",
            attached=False,
        )
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_rollout_metadata_forgery_and_capability_replay_are_rejected(
    tmp_path: Path,
) -> None:
    sessions = SessionManager(tmp_path)

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        return ControlExecutionResult(response="unused")

    runtime = ConversationRuntime(sessions.control_store, execute)
    consumed = False

    def binding(_capability: str, consume: bool) -> dict[str, str] | None:
        nonlocal consumed
        if not consume and consumed:
            return None
        if consume:
            return {
                "runtime": "latest",
                "ownerTurnId": "parent-1",
                "pluginId": "fitbit@github",
                "generationId": "gen-2",
                "sourceRevision": "rev-2",
            }
        consumed = True
        return {
            "runtime": "latest",
            "ownerTurnId": "parent-1",
            "pluginId": "fitbit@github",
            "generationId": "gen-2",
            "sourceRevision": "rev-2",
        }

    service = ControlService(runtime, sessions, tmp_path, plugin_child_binding=binding)
    with pytest.raises(ValueError, match="Core 保留"):
        service.start_thread({"_pluginRolloutOwnerTurnId": "parent-1"})
    thread = service.start_thread({}, plugin_rollout_capability="opaque")
    with pytest.raises(ValueError, match="已经使用"):
        service.start_thread({}, plugin_rollout_capability="opaque")
    with pytest.raises(ValueError, match="Core 保留"):
        await service.start_turn(
            str(thread["id"]),
            "verify",
            {"_pluginRolloutGenerationId": "gen-2"},
        )
    await runtime.shutdown()
