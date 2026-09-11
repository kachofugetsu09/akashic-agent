"""HTTP 请求的公开结构合同。

模型 driver 与 web 工具需要构造 HTTP 客户端、重试策略与预算。这些是 httpx 的
通用包装（不持有 Core 状态），因此由合同层拥有；Core 的**共享默认连接池**
（`SharedHttpResources`/`get_default_http_requester`）留在 `core/net/http.py`，
因为它是 Core 拥有的全局资源。
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)

HttpProfile = Literal["external_default", "feed_fetcher", "local_service"]


class HttpClient:
    """首个请求才建客户端；持有者显式关闭，配置检查不创建网络资源。"""

    def __init__(self, create: Callable[[], httpx.AsyncClient]) -> None:
        self._create = create
        self._client: httpx.AsyncClient | None = None
        self._closed = False

    def client(self) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError("HTTP 连接已关闭")
        if self._client is None:
            self._client = self._create()
        return self._client

    async def aclose(self) -> None:
        self._closed = True
        if self._client is not None:
            await self._client.aclose()


async def finish_response(lines: AsyncIterator[str]) -> None:
    """协议已确认成功后收尾 HTTP 正文；异常尾流最多占用 10 ms。"""
    try:
        async with asyncio.timeout(0.01):
            async for _ in lines:
                pass
    except (TimeoutError, httpx.HTTPError) as error:
        # 结果已经完整；放弃连接复用，stream 退出时关闭该连接。
        logger.debug("响应已完成，尾流未结束，关闭连接: %s", type(error).__name__)


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
                _ = await response.aread()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt >= attempts:
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

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        content: bytes | str | None = None,
        json: Any = None,
        timeout_s: float | None = None,
        budget: RequestBudget | None = None,
        validate_redirects: bool = False,
        max_redirects: int = 5,
    ) -> AsyncIterator[httpx.Response]:
        """逐跳请求并以流式 response 暴露 body，调用方负责消费有界内容。"""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + (
            budget.total_timeout_s
            if budget is not None
            else self.default_budget.total_timeout_s
        )
        attempts = max(1, self.retry_policy.max_attempts)
        current_url = url
        redirects = 0
        method = method.upper()
        attempt = 1
        while True:
            remaining = max(0.0, deadline - loop.time())
            if remaining <= 0:
                raise httpx.TimeoutException("request budget exhausted")
            try:
                async with self.client.stream(
                    method,
                    current_url,
                    headers=headers,
                    params=params,
                    content=content,
                    json=json,
                    follow_redirects=False,
                    timeout=min(timeout_s or self.default_timeout_s, remaining),
                ) as response:
                    if (
                        response.status_code in self.retry_policy.retry_statuses
                        and attempt < attempts
                    ):
                        attempt += 1
                    elif (
                        validate_redirects
                        and 300 <= response.status_code < 400
                        and response.headers.get("location")
                    ):
                        if redirects >= max(0, max_redirects):
                            raise httpx.TooManyRedirects(
                                f"redirect hop limit exceeded: {max_redirects}"
                            )
                        location = response.headers["location"]
                        current_url = urljoin(current_url, location)
                        redirects += 1
                        continue
                    else:
                        yield response
                        return
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= attempts:
                    raise
                attempt += 1

            sleep_s = min(
                self._backoff_seconds(attempt), max(0.0, deadline - loop.time())
            )
            if sleep_s > 0:
                await self.sleep(sleep_s)

        raise httpx.TimeoutException("request budget exhausted")

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

    def _backoff_seconds(self, attempt: int) -> float:
        delay = min(
            self.retry_policy.max_delay_s,
            self.retry_policy.base_delay_s * (2 ** max(0, attempt - 1)),
        )
        jitter = delay * self.retry_policy.jitter_ratio
        return max(0.0, delay + random.uniform(-jitter, jitter))

