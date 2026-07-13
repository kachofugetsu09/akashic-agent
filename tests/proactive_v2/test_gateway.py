from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock

import pytest

from plugins.default_proactive.gateway import DataGateway


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_channel", ["alert", "content", "context"])
async def test_gateway_propagates_source_snapshot_failure(
    failed_channel: str,
) -> None:
    failing = AsyncMock(side_effect=RuntimeError(f"{failed_channel} unavailable"))
    deps = {
        "alert_fn": AsyncMock(return_value=[]),
        "feed_fn": AsyncMock(return_value=[]),
        "context_fn": AsyncMock(return_value=[]),
    }
    source_name = "feed" if failed_channel == "content" else failed_channel
    deps[f"{source_name}_fn"] = failing

    with pytest.raises(RuntimeError, match=f"{failed_channel} unavailable"):
        await DataGateway(**deps).run()


@pytest.mark.asyncio
async def test_gateway_keeps_explicit_web_fetch_error_as_empty_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="plugins.default_proactive.gateway")
    web_fetch = AsyncMock()
    web_fetch.execute.return_value = json.dumps(
        {"error": "HTTP 404", "url": "https://example.com"}
    )
    gateway = DataGateway(
        feed_fn=AsyncMock(
            return_value=[
                {
                    "id": "item-1",
                    "ack_server": "feed",
                    "url": "https://example.com",
                }
            ]
        ),
        web_fetch_tool=web_fetch,
    )

    result = await gateway.run()

    assert result.content_meta[0]["id"] == "feed:item-1"
    assert result.content_store == {"feed:item-1": ""}
    assert "https://example.com" in caplog.text
    assert "HTTP 404" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", ["[]", json.dumps({"text": 123})])
async def test_gateway_rejects_corrupt_web_fetch_payload(payload: str) -> None:
    web_fetch = AsyncMock()
    web_fetch.execute.return_value = payload
    gateway = DataGateway(
        feed_fn=AsyncMock(
            return_value=[
                {
                    "id": "item-1",
                    "ack_server": "feed",
                    "url": "https://example.com",
                }
            ]
        ),
        web_fetch_tool=web_fetch,
    )

    with pytest.raises(RuntimeError, match="web_fetch 返回值"):
        await gateway.run()
