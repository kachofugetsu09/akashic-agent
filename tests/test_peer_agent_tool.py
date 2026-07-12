from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.peer_agent.card_resolver import AgentCard
from agent.peer_agent.tool import PeerAgentTool


def _build_tool(response_payload: object) -> tuple[PeerAgentTool, MagicMock]:
    response = SimpleNamespace(
        raise_for_status=MagicMock(),
        json=lambda: response_payload,
    )
    requester = SimpleNamespace(post=AsyncMock(return_value=response))
    process_manager = SimpleNamespace(ensure_ready=AsyncMock())
    poller = SimpleNamespace(register=MagicMock())
    tool = PeerAgentTool(
        AgentCard(name="research", url="http://peer.test"),
        process_manager,
        poller,
        requester,
    )
    return tool, poller.register


@pytest.mark.asyncio
async def test_peer_agent_registers_server_task_id() -> None:
    tool, register = _build_tool({"result": {"id": "task-123"}})

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
    tool, register = _build_tool(response_payload)

    raw_result = await tool.execute(goal="research topic")

    result = json.loads(raw_result)
    assert "error" in result
    assert "A2A message/send" in result["error"]
    register.assert_not_called()
