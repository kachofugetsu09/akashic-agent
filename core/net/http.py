from __future__ import annotations

import asyncio
import random
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urljoin

import httpx

# 通用 HTTP 客户端/重试/预算的拥有者已移到结构合同层；这里按原路径再导出，
# 既有 Core 调用点与类型身份不变。共享默认连接池仍由本模块拥有。
from agent.plugin_contracts.http import (  # noqa: E402,F401  (再导出)
    HttpClient,
    HttpProfile,
    HttpRequester,
    RequestBudget,
    RetryPolicy,
    finish_response,
)


logger = logging.getLogger(__name__)














@dataclass
class SharedHttpResources:
    external_default: HttpRequester = field(init=False)
    feed_fetcher: HttpRequester = field(init=False)
    local_service: HttpRequester = field(init=False)
    _clients: list[httpx.AsyncClient] = field(
        init=False,
        default_factory=list[httpx.AsyncClient],
    )
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
        """按逆序关闭共享客户端，并汇总所有清理失败。"""

        if self._closed:
            return

        # 1. 按既有关闭顺序关闭全部客户端
        errors: list[Exception] = []
        for client in reversed(self._clients):
            try:
                await client.aclose()
            except Exception as exc:
                errors.append(exc)

        # 2. 标记生命周期终止并暴露全部清理失败
        self._closed = True
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup("shared HTTP client cleanup failed", errors)

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
