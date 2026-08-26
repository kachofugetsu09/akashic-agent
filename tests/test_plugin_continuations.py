from __future__ import annotations

import asyncio
from datetime import timedelta

from agent.plugin_composition.continuations import PluginContinuations
from bus.events import InboundMessage


def test_continuation_default_timestamp_is_aware_utc() -> None:
    published: list[object] = []

    async def publish(message: object) -> None:
        published.append(message)

    asyncio.run(
        PluginContinuations(publish).submit(
            channel="akashic",
            chat_id="session-id",
            sender="spawn",
            content="done",
        )
    )

    message = published[0]
    assert isinstance(message, InboundMessage)
    assert message.timestamp.utcoffset() == timedelta(0)
