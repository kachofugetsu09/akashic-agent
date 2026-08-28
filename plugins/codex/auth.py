from __future__ import annotations

import base64
import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import httpx

from agent.plugin_composition import (
    AuthenticationError,
    CredentialHandle,
    RateLimitError,
    TransportError,
)

CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_AUTH_BASE = "https://auth.openai.com"
CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"
CODEX_CLIENT_VERSION = "0.144.1"
_REFRESH_SKEW_SECONDS = 120


async def start_auth(input: Mapping[str, Any]) -> Mapping[str, Any]:
    """Begin one device login and separate private state from public challenge."""

    unknown = sorted(set(input) - {"auth_base", "api_base"})
    if unknown:
        raise ValueError(f"unsupported Codex auth input: {', '.join(unknown)}")
    auth_base = (_string(input.get("auth_base")) or CODEX_AUTH_BASE).rstrip("/")
    api_base = (_string(input.get("api_base")) or CODEX_API_BASE).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{auth_base}/api/accounts/deviceauth/usercode",
                json={"client_id": CODEX_CLIENT_ID},
            )
    except asyncio.CancelledError:
        raise
    except httpx.TimeoutException as exc:
        raise TransportError("连接 Codex 登录服务超时") from exc
    except httpx.TransportError as exc:
        raise TransportError("连接 Codex 登录服务失败") from exc
    _require_auth_response(response, "获取 Codex device code 失败")
    data = _json_object(response)
    user_code = _required(data.get("user_code"), "user_code")
    device_auth_id = _required(data.get("device_auth_id"), "device_auth_id")
    interval = max(3, _integer(data.get("interval"), default=5))
    return {
        "state": {
            "auth_base": auth_base,
            "api_base": api_base,
            "device_auth_id": device_auth_id,
            "user_code": user_code,
            "interval": interval,
        },
        "challenge": {
            "user_code": user_code,
            "verification_uri": f"{auth_base}/codex/device",
            "interval": interval,
        },
    }


async def finish_auth(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Poll device auth once; models owns repetition and the final CAS commit."""

    auth_base = _required(state.get("auth_base"), "auth_base")
    api_base = _required(state.get("api_base"), "api_base")
    device_auth_id = _required(state.get("device_auth_id"), "device_auth_id")
    user_code = _required(state.get("user_code"), "user_code")
    interval = max(3, _integer(state.get("interval"), default=5))
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{auth_base.rstrip('/')}/api/accounts/deviceauth/token",
                json={"device_auth_id": device_auth_id, "user_code": user_code},
            )
    except asyncio.CancelledError:
        raise
    except httpx.TimeoutException as exc:
        raise TransportError("连接 Codex 登录服务超时") from exc
    except httpx.TransportError as exc:
        raise TransportError("连接 Codex 登录服务失败") from exc
    if response.status_code in {403, 404}:
        return {
            "status": "pending",
            "state": dict(state),
            "challenge": {
                "user_code": user_code,
                "verification_uri": f"{auth_base.rstrip('/')}/codex/device",
                "interval": interval,
            },
        }
    _require_auth_response(response, "Codex 登录轮询失败")
    data = _json_object(response)
    credential = await _exchange_code(
        auth_base,
        api_base,
        _required(data.get("authorization_code"), "authorization_code"),
        _required(data.get("code_verifier"), "code_verifier"),
    )
    return {
        "status": "complete",
        "name": "Codex",
        "endpoint": api_base,
        "auth_identity": credential["account_id"],
        "credential": credential,
        "driver_config": {},
    }


async def headers(
    handle: CredentialHandle,
    *,
    rejected_access_token: str | None = None,
) -> tuple[str, Mapping[str, str]]:
    """Return valid headers and collapse concurrent refresh into one rotation."""

    current = _credential(await handle.read())
    needs_refresh = _expires_soon(current)
    if rejected_access_token is not None and current["access_token"] == rejected_access_token:
        needs_refresh = True
    if needs_refresh:
        async with handle.exclusive():
            current = _credential(await handle.read())
            rejected_still_current = (
                rejected_access_token is not None
                and current["access_token"] == rejected_access_token
            )
            if _expires_soon(current) or rejected_still_current:
                current = await _refresh(current)
                await handle.refresh(current)
    result = {"Authorization": f"Bearer {current['access_token']}"}
    if current["account_id"]:
        result["ChatGPT-Account-ID"] = current["account_id"]
    return current["access_token"], result


async def _exchange_code(
    auth_base: str,
    api_base: str,
    code: str,
    verifier: str,
) -> dict[str, str]:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"{auth_base.rstrip('/')}/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": f"{auth_base.rstrip('/')}/deviceauth/callback",
                    "client_id": CODEX_CLIENT_ID,
                    "code_verifier": verifier,
                },
            )
    except asyncio.CancelledError:
        raise
    except httpx.TimeoutException as exc:
        raise TransportError("连接 Codex token 服务超时") from exc
    except httpx.TransportError as exc:
        raise TransportError("连接 Codex token 服务失败") from exc
    _require_auth_response(response, "Codex token 交换失败")
    return _credential_from_token(
        _json_object(response),
        auth_base=auth_base,
        api_base=api_base,
    )


async def _refresh(current: Mapping[str, str]) -> dict[str, str]:
    refresh_token = current.get("refresh_token", "")
    if not refresh_token:
        raise AuthenticationError("Codex refresh token 缺失，请重新登录")
    auth_base = current.get("auth_base") or CODEX_AUTH_BASE
    api_base = current.get("api_base") or CODEX_API_BASE
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"{auth_base.rstrip('/')}/oauth/token",
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": CODEX_CLIENT_ID,
                },
            )
    except asyncio.CancelledError:
        raise
    except httpx.TimeoutException as exc:
        raise TransportError("连接 Codex token 服务超时") from exc
    except httpx.TransportError as exc:
        raise TransportError("连接 Codex token 服务失败") from exc
    _require_auth_response(response, "Codex token 刷新失败，请重新登录")
    return _credential_from_token(
        _json_object(response),
        fallback_account_id=current.get("account_id", ""),
        fallback_refresh_token=refresh_token,
        auth_base=auth_base,
        api_base=api_base,
    )


def _credential_from_token(
    data: Mapping[str, Any],
    *,
    fallback_account_id: str = "",
    fallback_refresh_token: str = "",
    auth_base: str = CODEX_AUTH_BASE,
    api_base: str = CODEX_API_BASE,
) -> dict[str, str]:
    access_token = _string(data.get("access_token"))
    refresh_token = _string(data.get("refresh_token")) or fallback_refresh_token
    if not access_token or not refresh_token:
        raise AuthenticationError("Codex token 响应缺少必要字段")
    id_token = _string(data.get("id_token"))
    account_id = _account_id_from_jwt(id_token) if id_token else fallback_account_id
    if not account_id:
        raise AuthenticationError("Codex token 响应缺少账号标识")
    expires_in = _integer(data.get("expires_in"), default=3600)
    now = datetime.now(timezone.utc)
    return {
        "driver": "codex",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "account_id": account_id,
        "expires_at": (now + timedelta(seconds=expires_in)).isoformat(),
        "updated_at": now.isoformat(),
        "auth_base": auth_base,
        "api_base": api_base,
    }


def _credential(raw: Mapping[str, str]) -> dict[str, str]:
    result = dict(raw)
    if result.get("driver") != "codex":
        raise AuthenticationError("Codex 引用了非 Codex 凭据")
    if not result.get("access_token") or not result.get("account_id"):
        raise AuthenticationError("Codex 凭据缺少 access_token 或 account_id")
    return result


def _expires_soon(credential: Mapping[str, str]) -> bool:
    encoded = credential.get("expires_at", "")
    if not encoded:
        return True
    try:
        expires = datetime.fromisoformat(encoded.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuthenticationError("Codex expires_at 无效") from exc
    return expires <= datetime.now(timezone.utc) + timedelta(seconds=_REFRESH_SKEW_SECONDS)


def _account_id_from_jwt(token: str) -> str:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        auth = claims["https://api.openai.com/auth"]
        account_id = auth["chatgpt_account_id"]
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AuthenticationError("Codex token 缺少 chatgpt_account_id") from exc
    return _required(account_id, "chatgpt_account_id")


def _require_auth_response(response: httpx.Response, message: str) -> None:
    if response.status_code == 429:
        raise RateLimitError(message)
    if response.status_code >= 500:
        raise TransportError(f"{message} (HTTP {response.status_code})")
    if response.status_code >= 400:
        raise AuthenticationError(f"{message} (HTTP {response.status_code})")


def _json_object(response: httpx.Response) -> Mapping[str, Any]:
    try:
        value: Any = response.json()
    except json.JSONDecodeError as exc:
        raise TransportError("Codex 返回了无效 JSON") from exc
    if not isinstance(value, dict):
        raise TransportError("Codex JSON 响应必须是对象")
    return value


def _required(value: object, name: str) -> str:
    result = _string(value)
    if not result:
        raise AuthenticationError(f"Codex 响应缺少 {name}")
    return result


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _integer(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuthenticationError("Codex 响应包含无效整数")
    return value
