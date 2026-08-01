import asyncio

import httpx
import pytest

from core.net.http import (
    AddressPolicyError,
    HttpRequester,
    RequestBudget,
    RetryPolicy,
    SharedHttpResources,
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
    ):
        async def _close(*, _profile: str = profile) -> None:
            close_order.append(_profile)
            if _profile in errors:
                raise errors[_profile]

        monkeypatch.setattr(requester.client, "aclose", _close)

    with pytest.raises(ExceptionGroup) as caught:
        await resources.aclose()

    assert close_order == ["local_service", "feed_fetcher", "external_default"]
    assert caught.value.exceptions == (
        errors["local_service"],
        errors["feed_fetcher"],
    )
    assert resources.closed is True
