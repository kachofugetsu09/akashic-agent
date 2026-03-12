from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

HttpProfile = Literal["external_default", "feed_fetcher", "local_service"]


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    retry_statuses: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
    base_delay_s: float = 0.3
    max_delay_s: float = 1.5
    jitter_ratio: float = 0.2


@dataclass(frozen=True)
class RequestBudget:
    total_timeout_s: float


@dataclass
class HttpRequester:
    client: httpx.AsyncClient
    retry_policy: RetryPolicy
    default_timeout_s: float
    default_budget: RequestBudget
    sleep: Any = asyncio.sleep

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        content: bytes | str | None = None,
        json: Any = None,
        follow_redirects: bool = False,
        timeout_s: float | None = None,
        budget: RequestBudget | None = None,
    ) -> httpx.Response:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (
            budget.total_timeout_s
            if budget is not None
            else self.default_budget.total_timeout_s
        )
        attempts = max(1, self.retry_policy.max_attempts)
        last_error: Exception | None = None
        response: httpx.Response | None = None
        method = method.upper()

        for attempt in range(1, attempts + 1):
            remaining = max(0.0, deadline - loop.time())
            if remaining <= 0:
                break
            try:
                response = await self.client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    content=content,
                    json=json,
                    follow_redirects=follow_redirects,
                    timeout=min(timeout_s or self.default_timeout_s, remaining),
                )
                if not self._should_retry_response(response, attempt, attempts):
                    return response
                await response.aread()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if not self._should_retry_exception(exc, attempt, attempts):
                    raise

            sleep_s = min(
                self._backoff_seconds(attempt), max(0.0, deadline - loop.time())
            )
            if sleep_s <= 0:
                continue
            await self.sleep(sleep_s)

        if last_error is not None:
            raise last_error
        if response is None:
            raise httpx.TimeoutException("request budget exhausted")
        return response

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", url, **kwargs)

    def _should_retry_response(
        self,
        response: httpx.Response,
        attempt: int,
        attempts: int,
    ) -> bool:
        return (
            attempt < attempts
            and response.status_code in self.retry_policy.retry_statuses
        )

    @staticmethod
    def _should_retry_exception(
        exc: Exception,
        attempt: int,
        attempts: int,
    ) -> bool:
        return attempt < attempts and isinstance(
            exc,
            (httpx.TimeoutException, httpx.TransportError),
        )

    def _backoff_seconds(self, attempt: int) -> float:
        delay = min(
            self.retry_policy.max_delay_s,
            self.retry_policy.base_delay_s * (2 ** max(0, attempt - 1)),
        )
        jitter = delay * self.retry_policy.jitter_ratio
        return max(0.0, delay + random.uniform(-jitter, jitter))


@dataclass
class SharedHttpResources:
    external_default: HttpRequester = field(init=False)
    feed_fetcher: HttpRequester = field(init=False)
    local_service: HttpRequester = field(init=False)
    _clients: list[httpx.AsyncClient] = field(init=False, default_factory=list)
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        external_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
        )
        feed_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        local_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        self._clients = [external_client, feed_client, local_client]
        self.external_default = HttpRequester(
            client=external_client,
            retry_policy=RetryPolicy(max_attempts=3),
            default_timeout_s=30.0,
            default_budget=RequestBudget(total_timeout_s=45.0),
        )
        self.feed_fetcher = HttpRequester(
            client=feed_client,
            retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.2, max_delay_s=0.8),
            default_timeout_s=15.0,
            default_budget=RequestBudget(total_timeout_s=20.0),
        )
        self.local_service = HttpRequester(
            client=local_client,
            retry_policy=RetryPolicy(
                max_attempts=2, base_delay_s=0.15, max_delay_s=0.3
            ),
            default_timeout_s=5.0,
            default_budget=RequestBudget(total_timeout_s=8.0),
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        first_error: Exception | None = None
        for client in reversed(self._clients):
            try:
                await client.aclose()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self._closed = True
        if first_error is not None:
            raise first_error

    @property
    def closed(self) -> bool:
        return self._closed


_default_shared_http_resources: SharedHttpResources | None = None


def configure_default_shared_http_resources(
    resources: SharedHttpResources,
) -> None:
    global _default_shared_http_resources
    _default_shared_http_resources = resources


def clear_default_shared_http_resources(
    resources: SharedHttpResources | None = None,
) -> None:
    global _default_shared_http_resources
    if resources is None or _default_shared_http_resources is resources:
        _default_shared_http_resources = None


def get_default_shared_http_resources() -> SharedHttpResources:
    resources = _default_shared_http_resources
    if resources is None:
        raise RuntimeError("shared http resources not configured")
    return resources


def get_default_http_requester(profile: HttpProfile) -> HttpRequester:
    resources = get_default_shared_http_resources()
    return getattr(resources, profile)
