from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.looping.core import AgentLoop
from bus.events import InboundMessage, OutboundMessage


@pytest.mark.asyncio
async def test_react_routes_passive_message_to_pipeline():
    loop = AgentLoop.__new__(AgentLoop)
    loop._passive_pipeline = SimpleNamespace(
        run=AsyncMock(
            return_value=OutboundMessage(
                channel="cli",
                chat_id="1",
                content="final",
            )
        )
    )
    msg = InboundMessage(channel="cli", sender="hua", chat_id="1", content="hi")

    out = await loop._react(msg, "cli:1")

    assert out.content == "final"
    loop._passive_pipeline.run.assert_awaited_once_with(
        msg,
        "cli:1",
        dispatch_outbound=True,
        command_admitted=False,
    )
