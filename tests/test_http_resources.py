import asyncio
from typing import Any, cast

import httpx
import pytest

from core.net.http import (
    AddressPolicyError,
    HttpRequester,
    RequestBudget,
    RetryPolicy,
    SafeExternalTransport,
    SharedHttpResources,
    _HttpcoreResponseStream,
    _PinnedNetworkBackend,
    _public_addresses,
)


@pytest.mark.asyncio
async def test_http_requester_retries_timeout_then_succeeds():
    calls = {"count": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(200, request=request, text="ok")

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    requester = HttpRequester(
        client=client,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0),
        default_timeout_s=1.0,
        default_budget=RequestBudget(total_timeout_s=2.0),
        sleep=lambda _: asyncio.sleep(0),
    )

    response = await requester.get("https://example.com")

    assert response.status_code == 200
    assert calls["count"] == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_http_requester_retries_retryable_status_then_succeeds():
    calls = {"count": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, request=request, text="retry")
        return httpx.Response(200, request=request, text="ok")

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    requester = HttpRequester(
        client=client,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0),
        default_timeout_s=1.0,
        default_budget=RequestBudget(total_timeout_s=2.0),
        sleep=lambda _: asyncio.sleep(0),
    )

    response = await requester.get("https://example.com")

    assert response.status_code == 200
    assert calls["count"] == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_http_requester_stream_revalidates_redirect_hop() -> None:
    calls: list[str] = []

    def _resolver(host: str, port: int, **_: object):
        ip = "8.8.8.8" if host == "public.example" else "127.0.0.1"
        return [(2, 1, 6, "", (ip, port, 0, 0))]

    def _handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            302,
            request=request,
            headers={"location": "https://private.example/next"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    requester = HttpRequester(
        client=client,
        retry_policy=RetryPolicy(max_attempts=1),
        default_timeout_s=1.0,
        default_budget=RequestBudget(total_timeout_s=2.0),
        resolver=_resolver,
    )

    with pytest.raises(AddressPolicyError, match="非公开连接地址"):
        async with requester.stream(
            "GET",
            "https://public.example/start",
            validate_redirects=True,
        ):
            raise AssertionError("blocked redirect must not yield a response")

    assert calls == ["https://public.example/start"]
    await client.aclose()


@pytest.mark.asyncio
async def test_http_requester_stream_follows_multiple_hops_before_retry_budget() -> None:
    calls: list[str] = []

    def _resolver(host: str, port: int, **_: object):
        return [(2, 1, 6, "", ("8.8.8.8", port, 0, 0))]

    def _handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/start":
            return httpx.Response(302, request=request, headers={"location": "/next"})
        if request.url.path == "/next":
            return httpx.Response(302, request=request, headers={"location": "/done"})
        return httpx.Response(200, request=request, content=b"done")

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    requester = HttpRequester(
        client=client,
        retry_policy=RetryPolicy(max_attempts=1),
        default_timeout_s=1.0,
        default_budget=RequestBudget(total_timeout_s=2.0),
        resolver=_resolver,
    )

    async with requester.stream(
        "GET",
        "https://public.example/start",
        validate_redirects=True,
    ) as response:
        assert response.status_code == 200
        assert await response.aread() == b"done"

    assert calls == ["/start", "/next", "/done"]
    await client.aclose()


def test_dns_multi_answer_rejects_private_and_rebind_is_rechecked() -> None:
    answers = [(2, 1, 6, "", ("8.8.8.8", 443, 0, 0))]

    def _resolver(host: str, port: int, **kwargs: object):
        _ = (host, port, kwargs)
        return answers

    assert _public_addresses("example.com", 443, _resolver) == ("8.8.8.8",)
    answers[:] = [(2, 1, 6, "", ("127.0.0.1", 443, 0, 0))]
    with pytest.raises(AddressPolicyError, match="非公开连接地址"):
        _public_addresses("example.com", 443, _resolver)


@pytest.mark.asyncio
async def test_pinned_backend_connects_approved_ipv6_and_ipv4_targets() -> None:
    calls: list[str] = []

    class _Backend:
        async def connect_tcp(self, host: str, port: int, **kwargs: object):
            _ = (port, kwargs)
            calls.append(host)
            return object()

        async def connect_unix_socket(self, path: str, **kwargs: object):
            _ = (path, kwargs)
            return object()

        async def sleep(self, seconds: float):
            _ = seconds

    backend = _PinnedNetworkBackend(("2001:4860:4860::8888", "8.8.8.8"))
    backend._backend = _Backend()
    await backend.connect_tcp("example.com", 443)
    await backend.connect_tcp("example.com", 443)
    assert calls == ["2001:4860:4860::8888", "8.8.8.8"]


@pytest.mark.asyncio
async def test_safe_external_transport_rejects_literal_private_target() -> None:
    transport = SafeExternalTransport()
    request = httpx.Request("GET", "http://127.0.0.1/")
    with pytest.raises(AddressPolicyError, match="非公开连接地址"):
        await transport.handle_async_request(request)


@pytest.mark.asyncio
async def test_httpcore_response_stream_awaits_response_close() -> None:
    class _CoreResponse:
        closed = 0

        async def aiter_stream(self):
            yield b"ok"

        async def aclose(self) -> None:
            self.closed += 1

        def close(self) -> None:
            raise AssertionError("async response must use aclose")

    class _Pool:
        closed = 0

        async def aclose(self) -> None:
            self.closed += 1

    core_response = _CoreResponse()
    pool = _Pool()
    stream = _HttpcoreResponseStream(
        cast(Any, core_response),
        cast(Any, pool),
    )

    await stream.aclose()

    assert core_response.closed == 1
    assert pool.closed == 1


@pytest.mark.asyncio
async def test_local_service_keeps_local_network_transport() -> None:
    resources = SharedHttpResources()

    try:
        assert resources.local_service.safe_transport is None
        assert not isinstance(
            resources.local_service.client._transport,
            SafeExternalTransport,
        )
    finally:
        await resources.aclose()


@pytest.mark.asyncio
async def test_web_fetch_profile_allows_private_targets_without_weakening_external_profiles() -> None:
    resources = SharedHttpResources()

    try:
        resources.web_fetch.validate_external_url("http://127.0.0.1:8080/status")
        resources.web_fetch.validate_external_url("http://router.local/status")
        assert resources.web_fetch.allow_private_targets is True
        assert resources.web_fetch.safe_transport is None
        assert resources.external_default.allow_private_targets is False
        assert isinstance(resources.external_default.client._transport, SafeExternalTransport)
    finally:
        await resources.aclose()


@pytest.mark.asyncio
async def test_shared_http_resources_aclose_is_idempotent():
    resources = SharedHttpResources()

    await resources.aclose()
    await resources.aclose()

    assert resources.closed is True


@pytest.mark.asyncio
async def test_shared_http_resources_aclose_preserves_order_and_all_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = SharedHttpResources()
    close_order: list[str] = []
    errors = {
        "feed_fetcher": RuntimeError("feed cleanup failed"),
        "local_service": RuntimeError("local cleanup failed"),
    }

    for profile, requester in (
        ("external_default", resources.external_default),
        ("feed_fetcher", resources.feed_fetcher),
        ("local_service", resources.local_service),
        ("web_fetch", resources.web_fetch),
    ):
        async def _close(*, _profile: str = profile) -> None:
            close_order.append(_profile)
            if _profile in errors:
                raise errors[_profile]

        monkeypatch.setattr(requester.client, "aclose", _close)

    with pytest.raises(ExceptionGroup) as caught:
        await resources.aclose()

    assert close_order == [
        "web_fetch",
        "local_service",
        "feed_fetcher",
        "external_default",
    ]
    assert caught.value.exceptions == (
        errors["local_service"],
        errors["feed_fetcher"],
    )
    assert resources.closed is True
