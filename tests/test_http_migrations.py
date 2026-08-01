import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest

from agent.tools import web_fetch as web_fetch_module
from agent.tools.web_fetch import WebFetchTool
from agent.tools.base import ToolExecutionContext
from agent.tools.web_fetch_spill import (
    INLINE_MAX_BYTES,
    WebFetchSpillStore,
)
from infra.channels.base import AttachmentStore
from infra.channels.qq_channel import (
    MAX_QQ_IMAGE_BYTES,
    _download_to_temp,
)
from core.net.http import (
    HttpRequester,
    RequestBudget,
    RetryPolicy,
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
    get_default_shared_http_resources,
)
from agent.tool_bundles import build_readonly_research_tools
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
async def test_web_fetch_spills_large_response_with_execution_owner(tmp_path: Path):
    body = b"x" * (INLINE_MAX_BYTES + 128)

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=body,
            headers={"content-type": "text/plain"},
        )

    requester = _build_requester(_handler)
    store = WebFetchSpillStore(root=tmp_path / "spill")
    tool = WebFetchTool(
        requester,
        spill_store=store,
        context_provider=lambda: ToolExecutionContext(
            execution_id="turn-1", turn_id="turn-1"
        ),
    )
    try:
        payload = json.loads(
            await tool.execute(
                url="https://example.com/data.txt",
                format="text",
                execution_id="turn-1",
            )
        )
        spill_path = Path(payload["file_path"])
        assert payload["format"] == "file"
        assert payload["execution_id"] == "turn-1"
        assert spill_path.read_bytes() == body
        assert spill_path.stat().st_mode & 0o777 == 0o600
        assert (tmp_path / "spill").stat().st_mode & 0o777 == 0o700

        cleanup = tool.release("turn-1")
        assert cleanup.status == "released"
        assert not spill_path.exists()
        assert tool._execution_turns == {}
    finally:
        await requester.client.aclose()


@pytest.mark.asyncio
async def test_web_fetch_does_not_create_ownerless_spill(tmp_path: Path):
    body = b"x" * (INLINE_MAX_BYTES + 1)

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=body)

    requester = _build_requester(_handler)
    store = WebFetchSpillStore(root=tmp_path / "spill")
    tool = WebFetchTool(requester, spill_store=store)
    try:
        payload = json.loads(
            await tool.execute(
                url="https://example.com/data.txt",
                execution_id="model-forged-id",
            )
        )
        assert payload["classification"] == "unit_failed"
        assert "execution owner" in payload["error"]
        assert not list((tmp_path / "spill").glob("*.spill"))
    finally:
        await requester.client.aclose()


def test_readonly_research_bundle_puts_spill_under_allowed_workspace(tmp_path: Path):
    tools = build_readonly_research_tools(
        fetch_requester=cast(HttpRequester, SimpleNamespace()),
        allowed_dir=tmp_path,
    )
    web_fetch = next(tool for tool in tools if tool.name == "web_fetch")
    assert isinstance(web_fetch, WebFetchTool)
    assert web_fetch._spill_store is not None
    assert web_fetch._spill_store.root == (tmp_path / ".tmp" / "web-fetch")


@pytest.mark.asyncio
async def test_web_fetch_cancel_cleans_partial_spill(tmp_path: Path):
    class _Response:
        status_code = 200
        headers = {"content-type": "text/plain"}
        encoding = "utf-8"
        url = "https://example.com/data.txt"

        async def aiter_bytes(self, *, chunk_size: int):
            _ = chunk_size
            yield b"x" * (INLINE_MAX_BYTES + 1)
            raise asyncio.CancelledError()

    class _Requester:
        @asynccontextmanager
        async def stream(self, method: str, url: str, **kwargs: object):
            _ = (method, url, kwargs)
            yield _Response()

    store = WebFetchSpillStore(root=tmp_path / "spill")
    tool = WebFetchTool(
        cast(HttpRequester, _Requester()),
        spill_store=store,
        context_provider=lambda: ToolExecutionContext(
            execution_id="e1", turn_id="t1"
        ),
    )
    with pytest.raises(asyncio.CancelledError):
        await tool.execute(url="https://example.com/data.txt")
    assert not list((tmp_path / "spill").glob("*.spill"))


@pytest.mark.asyncio
async def test_web_fetch_spill_limit_cleanup_failure_keeps_turn_owner_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class _Response:
        status_code = 200
        headers = {"content-type": "text/plain"}
        encoding = "utf-8"
        url = "https://example.com/data.txt"

        async def aiter_bytes(self, *, chunk_size: int):
            _ = chunk_size
            yield b"x" * (INLINE_MAX_BYTES + 1)
            yield b"y"

    class _Requester:
        @asynccontextmanager
        async def stream(self, method: str, url: str, **kwargs: object):
            _ = (method, url, kwargs)
            yield _Response()

    monkeypatch.setattr(web_fetch_module, "SPILL_MAX_FILE_BYTES", INLINE_MAX_BYTES + 1)
    original_unlink = Path.unlink
    unlink_attempts = 0

    def fail_first_spill_unlink(path: Path, *, missing_ok: bool = False) -> None:
        nonlocal unlink_attempts
        if path.suffix == ".spill" and unlink_attempts == 0:
            unlink_attempts += 1
            raise OSError("unlink denied")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_first_spill_unlink)
    store = WebFetchSpillStore(root=tmp_path / "spill")
    tool = WebFetchTool(
        cast(HttpRequester, _Requester()),
        spill_store=store,
        context_provider=lambda: ToolExecutionContext(
            execution_id="execution-1", turn_id="turn-1"
        ),
    )

    payload = json.loads(await tool.execute(url="https://example.com/data.txt"))

    assert payload["classification"] == "operation_rejected"
    assert payload["cleanup_classification"] == "cleanup_degraded"
    diagnostic = store.diagnostics("execution-1")
    assert diagnostic is not None
    assert diagnostic.status == "cleanup_degraded"
    assert diagnostic.path is not None
    assert Path(diagnostic.path).exists()
    assert tool._execution_turns == {"execution-1": "turn-1"}

    retry = tool.release_turn("turn-1")

    assert [item.status for item in retry] == ["released"]
    assert not Path(diagnostic.path).exists()
    assert tool._execution_turns == {}
    assert store.diagnostics("execution-1") is not None
    assert store.diagnostics("execution-1").status == "released"


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
        paths = await _download_to_temp(
            ["https://example.com/image.png"],
            requester,
            AttachmentStore(tmp_path / "uploads"),
        )
        assert len(paths) == 1
        path = Path(paths[0])
        assert path.suffix == ".png"
        assert path.read_bytes() == b"fake-image-bytes"
    finally:
        for raw_path in paths if "paths" in locals() else []:
            Path(raw_path).unlink(missing_ok=True)
        await requester.client.aclose()


@pytest.mark.asyncio
async def test_qq_download_stream_enforces_item_limit_and_degrades(tmp_path: Path):
    class _Response:
        status_code = 200
        headers = {"content-type": "image/png"}

        async def aiter_bytes(self, *, chunk_size: int):
            _ = chunk_size
            yield b"x" * (MAX_QQ_IMAGE_BYTES + 1)

    class _Requester:
        @asynccontextmanager
        async def stream(self, method: str, url: str, **kwargs: object):
            _ = (method, kwargs)
            if url.endswith("bad.png"):
                raise RuntimeError("upstream unavailable")
            yield _Response()

    diagnostics: list[str] = []
    paths = await _download_to_temp(
        ["https://example.com/bad.png", "https://example.com/large.png"],
        cast(HttpRequester, _Requester()),
        AttachmentStore(tmp_path / "uploads"),
        diagnostics,
    )

    assert paths == []
    assert len(diagnostics) == 2
    assert not list((tmp_path / "uploads").glob("akashic_qq_*"))


@pytest.mark.asyncio
async def test_embedder_uses_injected_requester():
    def _handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["input"] == ["first", "second"]
        assert "dimensions" not in payload
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


@pytest.mark.asyncio
async def test_embedder_sends_configured_output_dimension():
    def _handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["dimensions"] == 768
        return httpx.Response(
            200,
            request=request,
            json={"data": [{"index": 0, "embedding": [0.1] * 768}]},
        )

    requester = _build_requester(_handler)
    try:
        embedder = Embedder(
            base_url="https://embeddings.example.com/v1",
            api_key="test-key",
            model="text-embedding-v4",
            output_dimensionality=768,
            requester=requester,
        )
        vectors = await embedder.embed_batch(["first"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 768
        assert vectors[0][0] == 0.1
        assert embedder.model_id == "text-embedding-v4"
    finally:
        await requester.client.aclose()


@pytest.mark.asyncio
async def test_embedder_rejects_malformed_provider_embedding_payload():
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={"data": [{"index": 0, "embedding": [0.1, "bad"]}]},
        )

    requester = _build_requester(_handler)
    try:
        embedder = Embedder(
            base_url="https://embeddings.example.com/v1",
            api_key="test-key",
            requester=requester,
        )
        with pytest.raises(ValueError, match="embedding provider"):
            await embedder.embed_batch(["first"])
    finally:
        await requester.client.aclose()


@pytest.mark.asyncio
async def test_embedder_rejects_configured_embedding_dimension_mismatch():
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={"data": [{"index": 0, "embedding": [0.1]}]},
        )

    requester = _build_requester(_handler)
    try:
        embedder = Embedder(
            base_url="https://embeddings.example.com/v1",
            api_key="test-key",
            output_dimensionality=2,
            requester=requester,
        )
        with pytest.raises(ValueError, match="维度错误"):
            await embedder.embed_batch(["first"])
    finally:
        await requester.client.aclose()


@pytest.mark.asyncio
async def test_embedder_rejects_inconsistent_batch_dimensions_without_configured_size():
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 1, "embedding": [0.3, 0.4, 0.5]},
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
        with pytest.raises(ValueError, match="维度不一致"):
            await embedder.embed_batch(["first", "second"])
    finally:
        await requester.client.aclose()
