from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from agent.model_runtime.errors import AuthenticationError, RateLimitError

from .store import Credential, CredentialStore

CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_AUTH_BASE = "https://auth.openai.com"
CODEX_TOKEN_URL = f"{CODEX_AUTH_BASE}/oauth/token"
CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"
CODEX_CLIENT_VERSION = "0.144.1"
_REFRESH_SKEW_SECONDS = 120


@dataclass(frozen=True)
class DeviceCode:
    user_code: str
    device_auth_id: str
    verification_uri: str
    interval: int


class CodexAuthDriver:
    """执行 Codex device-code 登录并提供可刷新的请求头。"""

    def __init__(self, store: CredentialStore, credential_id: str) -> None:
        self.store = store
        self.credential_id = credential_id

    def begin_device_login(self) -> DeviceCode:
        response = httpx.post(
            f"{CODEX_AUTH_BASE}/api/accounts/deviceauth/usercode",
            json={"client_id": CODEX_CLIENT_ID},
            timeout=15,
        )
        if response.status_code == 429:
            raise RateLimitError("Codex 登录请求被限流，请稍后重试")
        self._require_success(response, "获取 Codex device code 失败")
        data = response.json()
        try:
            return DeviceCode(
                user_code=str(data["user_code"]),
                device_auth_id=str(data["device_auth_id"]),
                verification_uri=f"{CODEX_AUTH_BASE}/codex/device",
                interval=max(3, int(data.get("interval", 5))),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AuthenticationError("Codex device code 响应结构无效") from exc

    def complete_device_login(
        self,
        code: DeviceCode,
        *,
        timeout_seconds: int = 900,
    ) -> Credential:
        """轮询授权结果、交换 token 并保存独立凭据。"""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            time.sleep(code.interval)
            response = httpx.post(
                f"{CODEX_AUTH_BASE}/api/accounts/deviceauth/token",
                json={"device_auth_id": code.device_auth_id, "user_code": code.user_code},
                timeout=15,
            )
            if response.status_code in {403, 404}:
                continue
            self._require_success(response, "Codex 登录轮询失败")
            credential = self._exchange_code(response.json())
            self.store.put(self.credential_id, credential)
            return credential
        raise AuthenticationError("Codex 登录等待超时")

    def headers(self, *, force_refresh: bool = False) -> dict[str, str]:
        credential = self.store.get(self.credential_id)
        self._validate_credential(credential)
        if force_refresh or self._expires_soon(credential):
            credential = self.refresh()
        headers = {"Authorization": f"Bearer {credential.access_token}"}
        if credential.account_id:
            headers["ChatGPT-Account-ID"] = credential.account_id
        return headers

    def refresh(self) -> Credential:
        """在跨进程锁内刷新并原子保存 rotation token。"""
        with self.store.locked():
            current = self.store.get(self.credential_id)
            if not current.refresh_token:
                raise AuthenticationError("Codex refresh token 缺失，请重新登录")
            response = httpx.post(
                CODEX_TOKEN_URL,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": current.refresh_token,
                    "client_id": CODEX_CLIENT_ID,
                },
                timeout=20,
            )
            if response.status_code == 429:
                raise RateLimitError("Codex token 刷新被限流")
            self._require_success(response, "Codex token 刷新失败，请重新登录")
            refreshed = self._credential_from_token(
                response.json(),
                fallback_account_id=current.account_id,
                fallback_refresh_token=current.refresh_token,
            )
            self.store.replace_locked(self.credential_id, refreshed)
            return refreshed

    def _exchange_code(self, data: dict) -> Credential:
        try:
            authorization_code = str(data["authorization_code"])
            code_verifier = str(data["code_verifier"])
        except (KeyError, TypeError) as exc:
            raise AuthenticationError("Codex 授权响应结构无效") from exc
        response = httpx.post(
            CODEX_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": f"{CODEX_AUTH_BASE}/deviceauth/callback",
                "client_id": CODEX_CLIENT_ID,
                "code_verifier": code_verifier,
            },
            timeout=20,
        )
        self._require_success(response, "Codex token 交换失败")
        return self._credential_from_token(response.json())

    @staticmethod
    def _credential_from_token(
        data: dict,
        fallback_account_id: str = "",
        fallback_refresh_token: str = "",
    ) -> Credential:
        access_token = str(data.get("access_token") or "")
        refresh_token = str(data.get("refresh_token") or fallback_refresh_token)
        if not access_token or not refresh_token:
            raise AuthenticationError("Codex token 响应缺少必要字段")
        id_token = str(data.get("id_token") or "")
        if id_token:
            resolved_account_id = _account_id_from_jwt(id_token)
        elif fallback_account_id:
            resolved_account_id = fallback_account_id
        else:
            raise AuthenticationError("Codex token 响应缺少 id_token")
        expires_in = int(data.get("expires_in") or 3600)
        now = datetime.now(timezone.utc)
        return Credential(
            driver="codex",
            access_token=access_token,
            refresh_token=refresh_token,
            account_id=resolved_account_id,
            expires_at=(now + timedelta(seconds=expires_in)).isoformat(),
            updated_at=now.isoformat(),
        )

    @staticmethod
    def _expires_soon(credential: Credential) -> bool:
        if not credential.expires_at:
            return True
        expires = datetime.fromisoformat(credential.expires_at.replace("Z", "+00:00"))
        return expires <= datetime.now(timezone.utc) + timedelta(seconds=_REFRESH_SKEW_SECONDS)

    @staticmethod
    def _validate_credential(credential: Credential) -> None:
        if credential.driver != "codex":
            raise AuthenticationError("Codex auth 引用了非 Codex 凭据")
        if not credential.access_token or not credential.account_id:
            raise AuthenticationError("Codex 凭据缺少 access_token 或 account_id")

    @staticmethod
    def _require_success(response: httpx.Response, message: str) -> None:
        if response.status_code < 400:
            return
        raise AuthenticationError(f"{message} (HTTP {response.status_code})")


def _account_id_from_jwt(token: str) -> str:
    """只解析账号路由声明，不验证已由 OAuth 端点签发的 token。"""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError, json.JSONDecodeError) as exc:
        raise AuthenticationError("Codex access token 不是有效 JWT") from exc
    auth_claims = claims.get("https://api.openai.com/auth", {})
    account_id = auth_claims.get("chatgpt_account_id")
    if not isinstance(account_id, str) or not account_id:
        raise AuthenticationError("Codex token 缺少 chatgpt_account_id")
    return account_id
