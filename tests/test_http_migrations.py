import json
from pathlib import Path

import httpx
import pytest

from agent.tools.web_fetch import WebFetchTool
from infra.channels.qq_channel import _download_to_temp
from core.net.http import (
    HttpRequester,
    RequestBudget,
    RetryPolicy,
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
    get_default_shared_http_resources,
)
from memory2.embedder import Embedder


def _build_requester(handler) -> HttpRequester:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpRequester(
        client=client,
        retry_policy=RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0),
        default_timeout_s=1.0,
        default_budget=RequestBudget(total_timeout_s=2.0),
    )


@pytest.mark.asyncio
async def test_default_shared_http_resources_requires_explicit_configuration():
    clear_default_shared_http_resources()

    with pytest.raises(RuntimeError, match="not configured"):
        get_default_shared_http_resources()

    resources = SharedHttpResources()
    try:
        configure_default_shared_http_resources(resources)
        assert get_default_shared_http_resources() is resources
    finally:
        clear_default_shared_http_resources(resources)
        await resources.aclose()


@pytest.mark.asyncio
async def test_web_fetch_tool_uses_injected_requester():
    async def _handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"].startswith("text/plain")
        return httpx.Response(
            200,
            request=request,
            text="hello from shared requester",
            headers={"content-type": "text/plain; charset=utf-8"},
        )

    requester = _build_requester(_handler)
    try:
        tool = WebFetchTool(requester)
        payload = json.loads(
            await tool.execute(url="https://example.com/data.txt", format="text")
        )
        assert payload["status"] == 200
        assert payload["text"] == "hello from shared requester"
    finally:
        await requester.client.aclose()


@pytest.mark.asyncio
async def test_download_to_temp_uses_injected_requester(tmp_path: Path):
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=b"fake-image-bytes",
            headers={"content-type": "image/png"},
        )

    requester = _build_requester(_handler)
    try:
        paths = await _download_to_temp(["https://example.com/image.png"], requester)
        assert len(paths) == 1
        path = Path(paths[0])
        assert path.suffix == ".png"
        assert path.read_bytes() == b"fake-image-bytes"
    finally:
        for raw_path in paths if "paths" in locals() else []:
            Path(raw_path).unlink(missing_ok=True)
        await requester.client.aclose()


@pytest.mark.asyncio
async def test_embedder_uses_injected_requester():
    def _handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["input"] == ["first", "second"]
        return httpx.Response(
            200,
            request=request,
            json={
                "data": [
                    {"index": 1, "embedding": [0.2, 0.3]},
                    {"index": 0, "embedding": [0.0, 0.1]},
                ]
            },
        )

    requester = _build_requester(_handler)
    try:
        embedder = Embedder(
            base_url="https://embeddings.example.com/v1",
            api_key="test-key",
            requester=requester,
        )
        vectors = await embedder.embed_batch(["first", "second"])
        assert vectors == [[0.0, 0.1], [0.2, 0.3]]
    finally:
        await requester.client.aclose()
