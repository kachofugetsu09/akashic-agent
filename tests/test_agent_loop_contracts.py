from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.looping.core import AgentLoop
from agent.looping.ports import LLMConfig
from bus.events import InboundMessage, OutboundMessage


@pytest.mark.asyncio
async def test_run_cleans_active_state_before_inbound_completion_failure() -> None:
    item = InboundMessage(
        channel="cli",
        sender="user",
        chat_id="1",
        content="hello",
    )
    bus = SimpleNamespace(
        consume_inbound=AsyncMock(return_value=item),
        publish_outbound=AsyncMock(),
        complete_inbound=AsyncMock(side_effect=RuntimeError("ack failed")),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop._llm_config = LLMConfig()
    loop.bus = bus
    loop._active_tasks = {}
    loop._active_turn_states = {}
    loop._process_with_runtime_admission = AsyncMock(
        return_value=OutboundMessage(channel="cli", chat_id="1", content="ok")
    )

    with pytest.raises(RuntimeError, match="ack failed"):
        await loop.run()

    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}
