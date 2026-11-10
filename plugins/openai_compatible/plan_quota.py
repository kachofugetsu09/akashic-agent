"""按服务主机读取 API Key 套餐的额度窗口；认不出的主机没有额度（与 magpie 的 planQuotaSourceOf 同法）。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

import httpx

from plugins.models.contract import (
    AuthenticationError,
    CredentialHandle,
    DriverConnectionDescriptor,
    ModelError,
    QuotaReading,
    QuotaWindow,
    TransportError,
    quota_window_label,
)

_TIMEOUT_SECONDS = 15.0
# Command Code 的套餐接口有时要 15–20 秒，等不到就只显示额度。
_PLAN_TIMEOUT_SECONDS = 6.0


@dataclass(frozen=True, slots=True)
class PlanSource:
    vendor: str
    path: str
    keys_url: str
    read: Callable[[Mapping[str, Any], Mapping[str, Any] | None, "PlanSource", bool], QuotaReading | None]
    # 智谱的监控接口要裸 key 作 Authorization，不带 Bearer。
    bearer: bool = False
    # 套餐名和续期日另有接口时与额度同时读；它慢或失败只少显示套餐名。
    plan_path: str = ""


async def read_plan_quota(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> QuotaReading | None:
    # 1. 主机不在表里就没有额度可读，不发任何请求。
    parts = urlsplit(descriptor.endpoint)
    source = PLAN_SOURCES.get((parts.hostname or "").lower())
    if source is None:
        return None
    if credential.connection_id != descriptor.connection_id or credential.auth_identity != descriptor.auth_identity:
        raise AuthenticationError("credential scope does not match").exception()
    token = _token(await credential.read())
    # 2. 额度接口挂在主机根下，与连接填的推理路径无关。
    authorization = f"Bearer {token}" if source.bearer else token
    url = urlunsplit((parts.scheme, parts.netloc, source.path, "", ""))
    plan_url = urlunsplit((parts.scheme, parts.netloc, source.plan_path, "", "")) if source.plan_path else ""
    payload, plan = await asyncio.gather(_get_json(url, authorization), _get_plan(plan_url, authorization))
    # 3. Coding Plan 地址上读不到窗口才算错；按量付费的 key 本来就没有套餐。
    return source.read(payload, plan, source, "/api/coding/" in parts.path)


async def _get_plan(url: str, authorization: str) -> Mapping[str, Any] | None:
    if not url:
        return None
    try:
        async with asyncio.timeout(_PLAN_TIMEOUT_SECONDS):
            return await _get_json(url, authorization)
    except TimeoutError:
        return None
    except RuntimeError as error:
        if ModelError.matches(error):
            return None
        raise


def _token(payload: Mapping[str, str]) -> str:
    token = payload.get("access_token") or payload.get("api_key") or payload.get("token") or ""
    if not token.strip():
        raise AuthenticationError("credential does not contain an API token").exception()
    return token.strip()


async def _get_json(url: str, authorization: str) -> Mapping[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS, follow_redirects=False) as client:
            response = await client.get(url, headers={"Authorization": authorization, "Accept": "application/json"})
    except asyncio.CancelledError:
        raise
    except httpx.HTTPError as error:
        raise TransportError(f"额度接口无法访问：{type(error).__name__}").exception() from error
    if response.status_code in (401, 403):
        raise AuthenticationError(f"额度接口拒绝了这把 key（HTTP {response.status_code}）").exception()
    if response.status_code >= 400:
        raise TransportError(f"额度接口返回 HTTP {response.status_code}").exception()
    try:
        payload: object = response.json()
    except ValueError as error:
        raise TransportError("额度接口返回的不是 JSON").exception() from error
    if not isinstance(payload, Mapping):
        raise TransportError("额度接口返回的不是对象").exception()
    return cast(Mapping[str, Any], payload)


# 读智谱 / Z.ai 的 GLM Coding Plan：
#   {"success":true,"data":{"level":"pro","limits":[
#     {"type":"TOKENS_LIMIT","unit":3,"number":5,"percentage":12,"nextResetTime":1758800000000},
#     {"type":"TOKENS_LIMIT","unit":6,"number":1,"percentage":40,"nextResetTime":...},
#     {"type":"TIME_LIMIT","unit":5,"number":1,"percentage":3,"nextResetTime":...}]}}
# unit 3 是小时、6 是周；TIME_LIMIT 是每月的 MCP 工具调用次数，用完不影响模型。
def _read_glm(payload: Mapping[str, Any], _plan: Mapping[str, Any] | None, source: PlanSource,
              coding: bool) -> QuotaReading | None:
    # 1. 拒绝以 HTTP 200 包一层返回：code 401 或 1000–1005 是 key 本身被拒。
    if payload.get("success") is False:
        code = _int(payload.get("code"))
        message = str(payload.get("msg") or "").strip()
        if code == 401 or 1000 <= code <= 1005:
            raise AuthenticationError(f"{source.vendor} 拒绝了这把 key：{message}。请到 {source.keys_url} 检查。").exception()
        raise TransportError(message or f"{source.vendor} 没有返回套餐").exception()
    data = payload.get("data")
    limits = cast(Mapping[str, Any], data).get("limits") if isinstance(data, Mapping) else None
    # 2. 逐个窗口换成通用的已用比例与重置时间。
    rows = cast(list[object], limits) if isinstance(limits, list) else []
    windows = tuple(window for item in rows
                    if isinstance(item, Mapping) and (window := _glm_window(cast(Mapping[str, Any], item))))
    if not windows:
        if coding:
            raise TransportError(f"{source.vendor} 没有返回 GLM Coding Plan 的额度窗口").exception()
        return None
    level = str(cast(Mapping[str, Any], data).get("level") or "").strip()
    return QuotaReading(plan=f"Coding {level.title()}" if level else "Coding Plan", windows=windows)


def _glm_window(item: Mapping[str, Any]) -> QuotaWindow | None:
    percent = item.get("percentage")
    if not isinstance(percent, (int, float)) or isinstance(percent, bool):
        return None
    reset = _int(item.get("nextResetTime"))
    resets_at = datetime.fromtimestamp(reset / 1000, timezone.utc).isoformat() if reset > 0 else None
    unit, number = _int(item.get("unit")), _int(item.get("number"))
    if str(item.get("type") or "").upper() == "TIME_LIMIT":
        return QuotaWindow("MCP 每月", float(percent), resets_at, None)
    # 团队套餐的 5 小时窗口不带 number。
    seconds = (number if number > 0 else 5) * 3600 if unit == 3 else 604800 if unit == 6 else None
    return QuotaWindow(quota_window_label(seconds), float(percent), resets_at, seconds)


# 读 Command Code 的 /alpha/billing/credits（magpie 社区插件 opencode-commandcode-auth 同法）：
#   {"credits":{"monthlyCredits":21.93,"purchasedCredits":0,"freeCredits":0},
#    "windowLimits":{"fiveHour":{"used":0,"cap":14,"resetAt":0},"weekly":{"used":2.4,"cap":35,"resetAt":1793637017830}}}
# 套餐来自 /alpha/billing/subscriptions 的 planId（individual-goat 等），本月额度按套餐的月度美元算已用比例。
def _read_commandcode(payload: Mapping[str, Any], plan: Mapping[str, Any] | None, _source: PlanSource,
                      _coding: bool) -> QuotaReading | None:
    # 1. 5 小时与每周窗口按 used / cap 换成已用比例；resetAt 为 0 是窗口还没开始计时。
    limits = _mapping(payload.get("windowLimits"))
    windows: list[QuotaWindow] = []
    for key, seconds in (("fiveHour", 5 * 3600), ("weekly", 604800)):
        item = _mapping(limits.get(key))
        used, cap = _float(item.get("used")), _float(item.get("cap"))
        if cap > 0:
            windows.append(QuotaWindow(quota_window_label(seconds), min(100.0, max(0.0, used) * 100 / cap),
                                       _iso(item.get("resetAt")), seconds))
    # 2. 套餐读到了就补上本月额度这一格，重置时间是本期结束。
    sub = _mapping(plan.get("data") if plan is not None else None)
    plan_id = "" if sub.get("status") in ("canceled", "incomplete_expired") else str(sub.get("planId") or "")
    name, monthly = next(((label, dollars) for prefix, label, dollars in _COMMANDCODE_PLANS
                          if plan_id.lower().replace("_", "-").startswith(prefix)), ("", 0.0))
    amounts = _mapping(payload.get("credits"))
    month_left = max(0.0, _float(amounts.get("monthlyCredits")))
    left = month_left + max(0.0, _float(amounts.get("purchasedCredits"))) + max(0.0, _float(amounts.get("freeCredits")))
    if monthly > 0:
        pool = max(monthly, month_left) + left - month_left
        windows.insert(min(1, len(windows)), QuotaWindow("本月额度", min(100.0, (pool - left) * 100 / pool),
                                                         _iso(sub.get("currentPeriodEnd")), None))
    if not windows and not amounts:
        raise TransportError("Command Code 没有返回额度").exception()
    return QuotaReading(plan=name or plan_id, windows=tuple(windows), balance=f"${left:.2f}" if amounts else "")


# planId 前缀 → 套餐名与每月美元额度（Command Code CLI 的 getPlanInfo），长前缀在前。
_COMMANDCODE_PLANS = (
    ("individual-provider", "Provider", 15.0),
    ("individual-pro-v1", "Pro", 80.0),
    ("individual-goat", "GOAT", 70.0),
    ("individual-ultra", "Ultra", 300.0),
    ("individual-max", "Max", 150.0),
    ("individual-pro", "Pro", 30.0),
    ("individual-go", "Go", 10.0),
    ("teams-pro", "Teams Pro", 40.0),
)


def _mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value) if isinstance(value, str) and value.strip() else 0.0
    except ValueError:
        return 0.0


# 毫秒或秒的时间戳、或 ISO 文本，统一成 UTC ISO；没有就是 None。
def _iso(value: object) -> str | None:
    if isinstance(value, str) and "T" in value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        except ValueError:
            return None
    stamp = _float(value)
    if stamp <= 0:
        return None
    return datetime.fromtimestamp(stamp / 1000 if stamp >= 1e12 else stamp, timezone.utc).isoformat()


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return 0


# 主机 → 额度来源；以后其他按 key 售卖的套餐（Kimi Code、MiniMax 等）在这里加一行。
PLAN_SOURCES: dict[str, PlanSource] = {
    "open.bigmodel.cn": PlanSource("智谱", "/api/monitor/usage/quota/limit",
                                   "https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys", _read_glm),
    "api.z.ai": PlanSource("Z.ai", "/api/monitor/usage/quota/limit",
                           "https://z.ai/manage-apikey/apikey-list", _read_glm),
    "api.commandcode.ai": PlanSource("Command Code", "/alpha/billing/credits", "https://commandcode.ai/settings/keys",
                                     _read_commandcode, bearer=True, plan_path="/alpha/billing/subscriptions"),
}

