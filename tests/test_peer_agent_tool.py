from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.peer_agent.card_resolver import AgentCard
from agent.peer_agent.poller import PeerAgentPoller, PeerTaskDuplicateError
from agent.peer_agent.process_manager import PeerProcessManager, PeerReady
from agent.peer_agent.tool import PeerAgentTool
from core.net.http import HttpRequester


def _as_process_manager(fake: object) -> PeerProcessManager:
    return cast(PeerProcessManager, fake)


def _as_poller(fake: object) -> PeerAgentPoller:
    return cast(PeerAgentPoller, fake)


def _as_requester(fake: object) -> HttpRequester:
    return cast(HttpRequester, fake)


def _build_tool(
    response_payload: object,
    *,
    ready: PeerReady = PeerReady(started_by_call=False),
    has_pending: bool = False,
    register_side_effect: BaseException | None = None,
) -> tuple[PeerAgentTool, SimpleNamespace, MagicMock]:
    response = SimpleNamespace(
        raise_for_status=MagicMock(),
        json=lambda: response_payload,
    )
    requester = SimpleNamespace(post=AsyncMock(return_value=response))
    process_manager = SimpleNamespace(
        ensure_ready=AsyncMock(return_value=ready),
        terminate=AsyncMock(),
    )
    poller = SimpleNamespace(
        register=MagicMock(side_effect=register_side_effect),
        has_pending=MagicMock(return_value=has_pending),
    )

    @asynccontextmanager
    async def submission_lease(agent_name: str):
        yield

    poller.submission_lease = submission_lease
    tool = PeerAgentTool(
        AgentCard(name="research", url="http://peer.test"),
        _as_process_manager(process_manager),
        _as_poller(poller),
        _as_requester(requester),
    )
    return tool, process_manager, poller.register


@pytest.mark.asyncio
async def test_peer_agent_registers_server_task_id() -> None:
    tool, _, register = _build_tool({"result": {"id": "task-123"}})

    raw_result = await tool.execute(
        goal="research topic",
        channel="telegram",
        chat_id="42",
    )

    result = json.loads(raw_result)
    assert result["status"] == "submitted"
    assert result["task_id"] == "task-123"
    register.assert_called_once_with(
        task_id="task-123",
        agent_name="research",
        agent_url="http://peer.test",
        channel="telegram",
        chat_id="42",
        goal="research topic",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_payload",
    [
        [],
        {},
        {"result": None},
        {"result": {}},
        {"result": {"id": ""}},
        {"result": {"id": 123}},
    ],
)
async def test_peer_agent_rejects_response_without_server_task_id(
    response_payload: object,
) -> None:
    tool, _, register = _build_tool(response_payload)

    with pytest.raises(RuntimeError, match="A2A message/send"):
        await tool.execute(goal="research topic")
    register.assert_not_called()


@pytest.mark.asyncio
async def test_programming_error_is_not_converted_to_json_error() -> None:
    tool, process_manager, register = _build_tool({"result": {"id": "task-123"}})
    tool._requester.post.side_effect = AssertionError("requester invariant")

    with pytest.raises(AssertionError, match="requester invariant"):
        await tool.execute(goal="research topic")

    process_manager.terminate.assert_not_awaited()
    register.assert_not_called()


@pytest.mark.asyncio
async def test_cold_start_submit_failure_reaps_new_process() -> None:
    tool, process_manager, register = _build_tool(
        {"result": {"id": "task-123"}},
        ready=PeerReady(started_by_call=True),
    )
    tool._requester.post.side_effect = RuntimeError("submit failed")

    with pytest.raises(RuntimeError, match="submit failed"):
        await tool.execute(goal="research topic")

    process_manager.terminate.assert_awaited_once_with("research")
    register.assert_not_called()


@pytest.mark.asyncio
async def test_external_healthy_peer_submit_failure_is_not_terminated() -> None:
    tool, process_manager, register = _build_tool({"result": {"id": "task-123"}})
    tool._requester.post.side_effect = RuntimeError("submit failed")

    with pytest.raises(RuntimeError, match="submit failed"):
        await tool.execute(goal="research topic")

    process_manager.terminate.assert_not_awaited()
    register.assert_not_called()


@pytest.mark.asyncio
async def test_submit_failure_keeps_shared_process_when_old_pending_exists() -> None:
    tool, process_manager, register = _build_tool(
        {"result": {"id": "task-123"}},
        ready=PeerReady(started_by_call=True),
        has_pending=True,
    )
    tool._requester.post.side_effect = RuntimeError("submit failed")

    with pytest.raises(RuntimeError, match="submit failed"):
        await tool.execute(goal="research topic")

    process_manager.terminate.assert_not_awaited()
    register.assert_not_called()


@pytest.mark.asyncio
async def test_submit_cancellation_reaps_new_process_and_propagates() -> None:
    tool, process_manager, register = _build_tool(
        {"result": {"id": "task-123"}},
        ready=PeerReady(started_by_call=True),
    )
    submit_started = asyncio.Event()
    never_complete = asyncio.Event()

    async def submit(*args: object, **kwargs: object) -> object:
        submit_started.set()
        await never_complete.wait()
        return SimpleNamespace()

    tool._requester.post = submit
    task = asyncio.create_task(tool.execute(goal="research topic"))
    await asyncio.wait_for(submit_started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    process_manager.terminate.assert_awaited_once_with("research")
    register.assert_not_called()


@pytest.mark.asyncio
async def test_submit_and_cleanup_failures_are_both_preserved() -> None:
    tool, process_manager, register = _build_tool(
        {"result": {"id": "task-123"}},
        ready=PeerReady(started_by_call=True),
    )
    tool._requester.post.side_effect = RuntimeError("submit failed")
    process_manager.terminate.side_effect = OSError("cleanup failed")

    with pytest.raises(BaseExceptionGroup) as captured:
        await tool.execute(goal="research topic")

    assert [str(error) for error in captured.value.exceptions] == [
        "submit failed",
        "cleanup failed",
    ]
    register.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_task_id_is_explicit_and_preserves_existing_pending() -> None:
    tool, process_manager, register = _build_tool(
        {"result": {"id": "task-123"}},
        ready=PeerReady(started_by_call=True),
        has_pending=True,
        register_side_effect=PeerTaskDuplicateError("duplicate task"),
    )

    with pytest.raises(PeerTaskDuplicateError, match="duplicate task"):
        await tool.execute(goal="research topic")

    process_manager.terminate.assert_not_awaited()
    register.assert_called_once()
