from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, cast

import httpx

from plugins.models.contract import (
    CredentialHandle,
    DriverConnectionDescriptor,
    QuotaReading,
    QuotaWindow,
    quota_window_label,
)
from plugins.models.contract import (
    AuthenticationError,
    TransportError,
)

from .auth import headers

_PLANS = {"free": "Free", "plus": "Plus", "pro": "Pro", "team": "Team", "business": "Business",
          "enterprise": "Enterprise", "edu": "Edu"}


async def read_quota(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> QuotaReading:
    """读取 ChatGPT 订阅的 Codex 窗口（与 Codex CLI 的 /status 同一来源）。"""

    payload = await _usage_request(descriptor, credential)
    limits = _mapping(payload.get("rate_limit"))
    windows: list[QuotaWindow] = []
    if limits is not None:
        for key in ("primary_window", "secondary_window"):
            window = _window(limits.get(key))
            if window is not None:
                windows.append(window)
    # 短窗口在前：5 小时、每周。
    windows.sort(key=lambda window: window.window_seconds or 0)
    plan = _string(payload.get("plan_type"))
    return QuotaReading(
        plan=_PLANS.get(plan, plan.capitalize()),
        account=_string(payload.get("email")),
        windows=tuple(windows),
        note=_credits(payload.get("credits")),
    )


async def _usage_request(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> Mapping[str, object]:
    # 1. 额度在 backend-api/wham 下，与模型接口同主机：去掉连接地址末尾的 /codex。
    base = descriptor.endpoint.rstrip("/").removesuffix("/codex")
    rejected: str | None = None
    for attempt in range(2):
        token, request_headers = await headers(credential, rejected_access_token=rejected)
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
                response = await client.get(f"{base}/wham/usage",
                                            headers={**request_headers, "Accept": "application/json"})
        except asyncio.CancelledError:
            raise
        except httpx.TimeoutException as exc:
            raise TransportError("读取 Codex 额度超时").exception() from exc
        except httpx.TransportError as exc:
            raise TransportError("读取 Codex 额度时连接失败").exception() from exc
        # 2. 令牌过期时刷新一次再读，和模型目录同一套轮换。
        if response.status_code == 401 and attempt == 0:
            rejected = token
            continue
        if response.status_code in {401, 403}:
            raise AuthenticationError("Codex 登录已失效，请重新登录").exception()
        if response.status_code >= 400:
            raise TransportError(f"读取 Codex 额度失败 (HTTP {response.status_code})").exception()
        try:
            value: Any = response.json()
        except json.JSONDecodeError as exc:
            raise TransportError("Codex 额度返回了无效 JSON").exception() from exc
        mapping = _mapping(value)
        if mapping is None:
            raise TransportError("Codex 额度响应必须是对象").exception()
        return mapping
    raise AuthenticationError("Codex 登录已失效，请重新登录").exception()


def _window(value: object) -> QuotaWindow | None:
    raw = _mapping(value)
    if raw is None:
        return None
    used = raw.get("used_percent")
    if not isinstance(used, (int, float)) or isinstance(used, bool):
        return None
    seconds = raw.get("limit_window_seconds")
    span = seconds if isinstance(seconds, int) and not isinstance(seconds, bool) and seconds > 0 else None
    reset = raw.get("reset_at")
    resets_at = (datetime.fromtimestamp(reset, timezone.utc).isoformat()
                 if isinstance(reset, (int, float)) and not isinstance(reset, bool) and reset > 0 else None)
    return QuotaWindow(quota_window_label(span), float(used), resets_at, span)


# 订阅之外另购的 credits：只在有余量且不是无限时提一句。
def _credits(value: object) -> str:
    raw = _mapping(value)
    if raw is None or raw.get("unlimited") is True or raw.get("has_credits") is not True:
        return ""
    try:
        balance = float(str(raw.get("balance") or "0"))
    except ValueError:
        return ""
    return f"另有 {balance:g} credits" if balance > 0 else ""


def _mapping(value: object) -> Mapping[str, object] | None:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else None


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
