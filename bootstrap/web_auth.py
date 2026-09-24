from __future__ import annotations

import asyncio
import base64
import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

PASSWORD_HASH_ENV = "AKASHIC_WEB_PASSWORD_HASH"
SESSION_COOKIE = "akashic_session"
SESSION_TTL_SECONDS = 30 * 24 * 3600
SESSION_REFRESH_SECONDS = 24 * 3600
WEBSOCKET_UNAUTHORIZED = 4401

_SCRYPT_N = 1 << 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_THROTTLE_WINDOW_SECONDS = 600.0
_THROTTLE_MAX_FAILURES = 8
_FAILURE_DELAY_SECONDS = 0.4
_PUBLIC_PATHS = frozenset({"/login", "/logout", "/favicon.ico"})
_PUBLIC_PREFIXES = ("/api/auth/",)


# 生成可写入环境变量的 scrypt 口令摘要。
def hash_password(password: str) -> str:
    if not password:
        raise ValueError("访问密码不能为空")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
    )
    return ":".join((
        "scrypt", str(_SCRYPT_N), str(_SCRYPT_R), str(_SCRYPT_P),
        _b64encode(salt), _b64encode(digest),
    ))


# 按摘要自带的参数重算并常量时间比较。
def verify_password(password: str, encoded: str) -> bool:
    n, r, p, salt, expected = _parse_password_hash(encoded)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p)
    return hmac.compare_digest(digest, expected)


def _parse_password_hash(encoded: str) -> tuple[int, int, int, bytes, bytes]:
    parts = encoded.split(":")
    if len(parts) != 6 or parts[0] != "scrypt":
        raise ValueError(f"{PASSWORD_HASH_ENV} 不是有效的 scrypt 摘要")
    try:
        return (
            int(parts[1]), int(parts[2]), int(parts[3]),
            _b64decode(parts[4]), _b64decode(parts[5]),
        )
    except ValueError as error:
        raise ValueError(f"{PASSWORD_HASH_ENV} 不是有效的 scrypt 摘要") from error


def _json_object(raw: bytes) -> dict[str, object] | None:
    try:
        payload: object = json.loads(raw)
    except ValueError:
        return None
    return cast(dict[str, object], payload) if isinstance(payload, dict) else None


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass(frozen=True, slots=True)
class SessionClaims:
    issued_at: int
    expires_at: int


@dataclass(frozen=True, slots=True)
class WebAuth:
    """单人访问口令；签名密钥由口令摘要派生，改密码即令全部登录失效。"""

    password_hash: str

    def __post_init__(self) -> None:
        _ = _parse_password_hash(self.password_hash)

    @classmethod
    def from_env(cls) -> WebAuth | None:
        encoded = os.environ.get(PASSWORD_HASH_ENV, "").strip()
        return cls(encoded) if encoded else None

    def check_password(self, password: str) -> bool:
        return verify_password(password, self.password_hash)

    def issue(self, now: int) -> str:
        payload = json.dumps(
            {"v": 1, "iat": now, "exp": now + SESSION_TTL_SECONDS, "n": secrets.token_urlsafe(6)},
            separators=(",", ":"),
        ).encode("utf-8")
        body = _b64encode(payload)
        return f"{body}.{self._sign(body)}"

    # 只接受签名正确、版本匹配且未过期的会话。
    def verify(self, token: str, now: int) -> SessionClaims | None:
        body, _, signature = token.partition(".")
        if not body or not hmac.compare_digest(signature, self._sign(body)):
            return None
        payload = _json_object(_b64decode(body))
        if payload is None or payload.get("v") != 1:
            return None
        issued_at, expires_at = payload.get("iat"), payload.get("exp")
        if not isinstance(issued_at, int) or not isinstance(expires_at, int) or expires_at <= now:
            return None
        return SessionClaims(issued_at, expires_at)

    def _sign(self, body: str) -> str:
        key = hmac.new(self.password_hash.encode("utf-8"), b"akashic-web-session-v1", hashlib.sha256).digest()
        return _b64encode(hmac.new(key, body.encode("ascii"), hashlib.sha256).digest())


class LoginThrottle:
    """按来源地址限制失败次数；只存在于进程内存，重启即清空。"""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._failures: dict[str, deque[float]] = {}

    def retry_after(self, client: str) -> int:
        window = self._window(client)
        if len(window) < _THROTTLE_MAX_FAILURES:
            return 0
        return max(1, int(window[0] + _THROTTLE_WINDOW_SECONDS - self._clock()) + 1)

    def record_failure(self, client: str) -> None:
        self._window(client).append(self._clock())

    def reset(self, client: str) -> None:
        _ = self._failures.pop(client, None)

    def _window(self, client: str) -> deque[float]:
        window = self._failures.setdefault(client, deque())
        horizon = self._clock() - _THROTTLE_WINDOW_SECONDS
        while window and window[0] <= horizon:
            _ = window.popleft()
        return window


class WebAuthMiddleware:
    """在唯一公共入口拦截未登录的页面、接口和 WebSocket。"""

    def __init__(self, app: ASGIApp, auth: WebAuth, clock: Callable[[], float] = time.time) -> None:
        self._app = app
        self._auth = auth
        self._clock = clock

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"} or _is_public(scope["path"]):
            await self._app(scope, receive, send)
            return
        now = int(self._clock())
        claims = self._claims(scope, now)
        if claims is None:
            await _reject(scope, send)
            return
        if scope["type"] == "http" and now - claims.issued_at >= SESSION_REFRESH_SECONDS:
            send = _with_cookie(send, _session_cookie(self._auth.issue(now), secure=_is_secure(scope)))
        await self._app(scope, receive, send)

    def _claims(self, scope: Scope, now: int) -> SessionClaims | None:
        token = _cookie_value(Headers(scope=scope), SESSION_COOKIE)
        return None if token is None else self._auth.verify(token, now)


def _is_public(path: str) -> bool:
    return path in _PUBLIC_PATHS or path.startswith(_PUBLIC_PREFIXES)


# 页面导航跳到登录页；接口返回 401；WebSocket 在握手阶段直接关闭。
async def _reject(scope: Scope, send: Send) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": WEBSOCKET_UNAUTHORIZED})
        return
    headers = Headers(scope=scope)
    if scope["method"] == "GET" and "text/html" in headers.get("accept", ""):
        target = scope["path"] + (f"?{scope['query_string'].decode('latin-1')}" if scope["query_string"] else "")
        response: Response = RedirectResponse(f"/login?next={quote(target, safe='')}", status_code=303)
    else:
        response = JSONResponse(
            status_code=401,
            content={"code": "unauthenticated", "message": "需要登录"},
            headers={"Cache-Control": "no-store"},
        )
    await response(scope, _receive_nothing, send)


async def _receive_nothing() -> Message:
    return {"type": "http.disconnect"}


def _with_cookie(send: Send, cookie: str) -> Send:
    async def wrapped(message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = list(message.get("headers", []))
            headers.append((b"set-cookie", cookie.encode("latin-1")))
            message = {**message, "headers": headers}
        await send(message)
    return wrapped


def _cookie_value(headers: Headers, name: str) -> str | None:
    for chunk in headers.get("cookie", "").split(";"):
        key, _, value = chunk.strip().partition("=")
        if key == name and value:
            return value
    return None


def _is_secure(scope: Scope) -> bool:
    forwarded = Headers(scope=scope).get("x-forwarded-proto", "")
    return scope.get("scheme") in {"https", "wss"} or forwarded.split(",")[0].strip() == "https"


def _session_cookie(token: str, *, secure: bool, max_age: int = SESSION_TTL_SECONDS) -> str:
    attributes = [f"{SESSION_COOKIE}={token}", "Path=/", f"Max-Age={max_age}", "HttpOnly", "SameSite=Lax"]
    if secure:
        attributes.append("Secure")
    return "; ".join(attributes)


# 登录后只允许跳回本站路径，避免 next 参数把用户带去外站。
def safe_next(raw: str | None) -> str:
    if not raw or not raw.startswith("/") or raw.startswith("//") or "\\" in raw:
        return "/"
    return raw


# 注册登录、退出和状态接口；未启用口令时这些页面只报告无需登录。
def include_auth_routes(app: FastAPI, auth: WebAuth | None, clock: Callable[[], float] = time.time) -> None:
    throttle = LoginThrottle()

    @app.get("/api/auth/status")
    async def auth_status(request: Request) -> JSONResponse:
        token = request.cookies.get(SESSION_COOKIE)
        authenticated = auth is None or (token is not None and auth.verify(token, int(clock())) is not None)
        return JSONResponse(
            {"auth_required": auth is not None, "authenticated": authenticated},
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/login", response_model=None)
    async def login_page(request: Request) -> Response:
        if auth is None:
            return RedirectResponse(safe_next(request.query_params.get("next")), status_code=303)
        return _html_page(LOGIN_PAGE)

    @app.get("/logout", response_model=None)
    async def logout_page() -> Response:
        return _html_page(LOGOUT_PAGE) if auth is not None else RedirectResponse("/", status_code=303)

    @app.post("/api/auth/login")
    async def login(request: Request) -> JSONResponse:
        return await _login(request, auth, throttle, clock)

    @app.post("/api/auth/logout")
    async def logout(request: Request) -> JSONResponse:
        response = JSONResponse({"status": "signed_out"}, headers={"Cache-Control": "no-store"})
        response.headers.append("set-cookie", _session_cookie("", secure=_is_secure(request.scope), max_age=0))
        return response


# 校验来源、限流和口令，成功后签发会话 Cookie。
async def _login(
    request: Request,
    auth: WebAuth | None,
    throttle: LoginThrottle,
    clock: Callable[[], float],
) -> JSONResponse:
    # 1. 未启用口令或跨站提交都不签发会话
    if auth is None:
        return _auth_error(409, "auth_disabled", "服务器未启用登录")
    origin = request.headers.get("origin")
    if origin is not None and origin != f"{request.url.scheme}://{request.url.netloc}":
        return _auth_error(403, "origin_rejected", "请求来源无效")
    # 2. 连续失败过多时先拒绝，不再计算口令
    client = request.client.host if request.client is not None else "unknown"
    retry_after = throttle.retry_after(client)
    if retry_after:
        return _auth_error(429, "too_many_attempts", "尝试次数过多", retry_after=retry_after)
    # 3. 口令错误时固定延迟并记录失败
    password = await _read_password(request)
    if password is None or not await asyncio.to_thread(auth.check_password, password):
        throttle.record_failure(client)
        await asyncio.sleep(_FAILURE_DELAY_SECONDS)
        return _auth_error(401, "invalid_password", "密码不正确")
    # 4. 成功后清除失败记录并写入 HttpOnly 会话
    throttle.reset(client)
    response = JSONResponse({"status": "signed_in"}, headers={"Cache-Control": "no-store"})
    response.headers.append("set-cookie", _session_cookie(auth.issue(int(clock())), secure=_is_secure(request.scope)))
    return response


async def _read_password(request: Request) -> str | None:
    raw = await request.body()
    if request.headers.get("content-type", "").startswith("application/json"):
        payload = _json_object(raw)
        value = None if payload is None else payload.get("password")
    else:
        value = parse_qs(raw.decode("utf-8", "replace")).get("password", [None])[0]
    return value if isinstance(value, str) and value else None


def _auth_error(status: int, code: str, message: str, *, retry_after: int | None = None) -> JSONResponse:
    content: dict[str, object] = {"code": code, "message": message}
    headers = {"Cache-Control": "no-store"}
    if retry_after is not None:
        content["retry_after"] = retry_after
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(status_code=status, content=content, headers=headers)


def _html_page(body: str) -> HTMLResponse:
    return HTMLResponse(
        body,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


_PAGE_STYLE = """
:root{color-scheme:light dark;--paper:#f5f4ed;--sheet:#faf9f5;--ink:#141413;--ink-2:#504e49;
--rule:#e5e3d8;--rule-strong:#6b6a64;--action:#1b365d;--on-action:#fafaf9;--error:#9b2c2c;
--error-soft:#f5d5d2}
@media (prefers-color-scheme:dark){:root{--paper:#141413;--sheet:#0f0f0e;--ink:#e8e7e3;--ink-2:#b0aea7;
--rule:#30302e;--rule-strong:#8f8d86;--action:#a8bdd9;--on-action:#0d1f38;--error:#e8b4b0;
--error-soft:#6e1f1f}}
*{box-sizing:border-box}
html,body{margin:0;min-height:100%;background:var(--paper);color:var(--ink);
font:16px/1.6 "LXGW WenKai GB Screen","Noto Serif SC","Songti SC",serif}
main{min-height:100dvh;display:flex;flex-direction:column;justify-content:center;
padding:max(24px,env(safe-area-inset-top)) 24px max(24px,env(safe-area-inset-bottom));
max-width:400px;margin:0 auto}
h1{font-size:30px;font-weight:600;letter-spacing:.02em;margin:0 0 4px}
.lead{margin:0 0 32px;color:var(--ink-2)}
label{display:block;font-size:14px;color:var(--ink-2);margin-bottom:8px}
.field{display:flex;align-items:center;border-bottom:1.5px solid var(--rule-strong);transition:border-color .15s}
.field:focus-within{border-color:var(--action)}
.field.invalid{border-color:var(--error)}
input{flex:1;min-width:0;border:0;background:transparent;color:var(--ink);font:inherit;font-size:18px;
padding:10px 0;outline:none}
.reveal{border:0;background:none;color:var(--ink-2);font:inherit;font-size:14px;padding:10px 0 10px 12px;
cursor:pointer;min-height:44px}
.message{min-height:24px;margin:10px 0 0;font-size:14px;color:var(--error)}
.primary{width:100%;min-height:48px;margin-top:24px;border:0;border-radius:6px;background:var(--action);
color:var(--on-action);font:inherit;font-size:17px;cursor:pointer;transition:opacity .15s}
.primary:disabled{opacity:.55;cursor:default}
.primary:focus-visible,.reveal:focus-visible,.secondary:focus-visible{outline:2px solid var(--action);outline-offset:3px}
.secondary{width:100%;min-height:44px;margin-top:12px;border:1px solid var(--rule);border-radius:6px;
background:none;color:var(--ink);font:inherit;cursor:pointer}
.note{margin-top:28px;padding-top:16px;border-top:1px solid var(--rule);font-size:13px;color:var(--ink-2)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

LOGIN_PAGE = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="light dark"><title>登录 · Akashic</title>
<style>{_PAGE_STYLE}</style></head>
<body><main>
<h1>Akashic</h1>
<p class="lead">输入访问密码以继续。</p>
<form id="form" novalidate>
<label for="password">访问密码</label>
<div class="field" id="field">
<input id="password" name="password" type="password" autocomplete="current-password" autofocus required
 aria-describedby="message">
<button type="button" class="reveal" id="reveal" aria-controls="password" aria-pressed="false">显示</button>
</div>
<p class="message" id="message" role="alert" aria-live="polite"></p>
<button class="primary" id="submit" type="submit">登录</button>
</form>
<p class="note">登录后此设备会保持登录 30 天。更换访问密码会让所有设备退出。</p>
</main>
<script>
(() => {{
  const form = document.getElementById("form");
  const input = document.getElementById("password");
  const field = document.getElementById("field");
  const message = document.getElementById("message");
  const submit = document.getElementById("submit");
  const reveal = document.getElementById("reveal");
  const next = new URLSearchParams(location.search).get("next") || "/";
  let lockedUntil = 0;
  const safeNext = (value) => value.startsWith("/") && !value.startsWith("//") ? value : "/";
  const show = (text) => {{ message.textContent = text; field.classList.toggle("invalid", Boolean(text)); }};
  reveal.addEventListener("click", () => {{
    const visible = input.type === "text";
    input.type = visible ? "password" : "text";
    reveal.textContent = visible ? "显示" : "隐藏";
    reveal.setAttribute("aria-pressed", String(!visible));
    input.focus();
  }});
  input.addEventListener("input", () => {{ if (Date.now() >= lockedUntil) show(""); }});
  const countdown = () => {{
    const left = Math.ceil((lockedUntil - Date.now()) / 1000);
    if (left <= 0) {{ submit.disabled = false; show(""); return; }}
    submit.disabled = true;
    show(`尝试次数过多，请 ${{left}} 秒后再试。`);
    setTimeout(countdown, 1000);
  }};
  form.addEventListener("submit", async (event) => {{
    event.preventDefault();
    if (!input.value) {{ show("请输入访问密码。"); input.focus(); return; }}
    submit.disabled = true;
    submit.textContent = "正在验证…";
    show("");
    try {{
      const response = await fetch("/api/auth/login", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ password: input.value }}),
        credentials: "same-origin",
      }});
      if (response.ok) {{ submit.textContent = "已登录，正在进入…"; location.replace(safeNext(next)); return; }}
      const payload = await response.json().catch(() => ({{}}));
      if (response.status === 429) {{ lockedUntil = Date.now() + (payload.retry_after || 60) * 1000; countdown(); }}
      else if (response.status === 401) {{ show("密码不正确，请重新输入。"); input.select(); }}
      else {{ show(payload.message || "登录失败，请稍后再试。"); }}
    }} catch {{
      show("无法连接服务器，请检查网络后重试。");
    }}
    submit.textContent = "登录";
    if (Date.now() >= lockedUntil) submit.disabled = false;
  }});
}})();
</script>
</body></html>
"""

LOGOUT_PAGE = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="light dark"><title>退出登录 · Akashic</title>
<style>{_PAGE_STYLE}</style></head>
<body><main>
<h1>退出登录</h1>
<p class="lead">退出后，这个浏览器需要重新输入访问密码。其他设备不受影响。</p>
<button class="primary" id="confirm" type="button">退出登录</button>
<button class="secondary" id="cancel" type="button">返回 Akashic</button>
<p class="message" id="message" role="alert" aria-live="polite"></p>
</main>
<script>
(() => {{
  const confirm = document.getElementById("confirm");
  document.getElementById("cancel").addEventListener("click", () => location.replace("/"));
  confirm.addEventListener("click", async () => {{
    confirm.disabled = true;
    confirm.textContent = "正在退出…";
    try {{
      await fetch("/api/auth/logout", {{ method: "POST", credentials: "same-origin" }});
      location.replace("/login");
    }} catch {{
      document.getElementById("message").textContent = "无法连接服务器，请稍后再试。";
      confirm.disabled = false;
      confirm.textContent = "退出登录";
    }}
  }});
}})();
</script>
</body></html>
"""


# 交互式生成摘要：python -m bootstrap.web_auth
def main() -> int:
    first = getpass.getpass("设置 Akashic 访问密码：")
    second = getpass.getpass("再次输入：")
    if not first or first != second:
        print("两次输入不一致或为空", file=sys.stderr)
        return 1
    print(f"{PASSWORD_HASH_ENV}={hash_password(first)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
