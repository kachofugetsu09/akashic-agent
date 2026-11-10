"""入口守卫与登录接口：Web Shell 每个请求先问 /_guard/check，登录页与接口在 /_auth 下。"""
from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path
import secrets
import time
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from core.common.file_io import run_file_io

from .state import COOKIE, REMEMBER_S, USERNAME, OwnerAuth

logger = logging.getLogger(__name__)

# 未登录也放行：登录页与接口自身，以及只返回就绪状态的两个健康检查。
_PUBLIC = {"/api/shell/state", "/api/chat/health"}
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_PAGE_POLICY = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")


class _Throttle:
    """按来源退避，并给全部来源设总上限；不锁定账号，避免陌生人把主人锁在门外。"""

    def __init__(self) -> None:
        self._failures: dict[str, tuple[int, float]] = {}
        self._recent: list[float] = []

    # 放行的那一刻就记一次尝试（成功再清零），并发请求不能在失败记账前整批越过上限。
    def admit(self, key: str) -> int:
        now = time.monotonic()
        self._recent = [moment for moment in self._recent if now - moment < 60]
        count, until = self._failures.get(key, (0, 0.0))
        if len(self._recent) >= 60:
            until = max(until, self._recent[0] + 60)
        if until > now:
            return int(until - now + 0.999)
        count += 1
        self._failures[key] = (count, now + (0 if count < 5 else min(900, 30 * 2 ** (count - 5))))
        self._recent.append(now)
        return 0

    def clear(self, key: str) -> None:
        self._failures.pop(key, None)


def _same_origin(headers: Mapping[str, str], *, required: bool) -> bool:
    origin, host = headers.get("origin"), headers.get("x-akashic-host")
    if not origin:
        return not required
    parsed = urlsplit(origin)
    return parsed.scheme in {"http", "https"} and bool(host) and parsed.netloc.casefold() == str(host).casefold()


# 只用外壳写入的对端地址；外壳只采信本机反代给的 X-Forwarded-For，远端伪造无效。
def _peer(headers: Mapping[str, str]) -> str:
    return headers.get("x-akashic-peer", "") or "unknown"


def create_app(auth: OwnerAuth, page: Path) -> FastAPI:
    """本机 Bearer 读同一份密钥文件；能读到它的进程就是主人。"""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    throttle = _Throttle()
    html = page.read_text(encoding="utf-8")

    # 返回 "local"、会话 token 或 None；Bearer 只认本机密钥，不回退到 cookie。
    async def identity(request: Request) -> str | None:
        bearer = request.headers.get("authorization", "")
        if bearer.startswith("Bearer "):
            local = await run_file_io(auth.local_secret)
            return "local" if secrets.compare_digest(bearer[7:].encode(), local.encode()) else None
        token = request.cookies.get(COOKIE)
        if token and await run_file_io(lambda: auth.session(token)) is not None:
            return token
        return None

    def signed_in(request: Request, token: str, *, remember: bool) -> JSONResponse:
        response = JSONResponse({"mustChange": auth.must_change()})
        https = request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(COOKIE, token, max_age=REMEMBER_S if remember else None, path="/",
                            httponly=True, samesite="lax", secure=https)
        return response

    @app.get("/_guard/check")
    async def check(request: Request) -> Response:
        uri = request.headers.get("x-akashic-uri", "/")
        method = request.headers.get("x-akashic-method", "GET")
        path = urlsplit(uri).path
        if path in _PUBLIC or path == "/_auth" or path.startswith("/_auth/"):
            return Response(status_code=204)
        websocket = request.headers.get("x-akashic-upgrade") == "websocket"
        page_request = not websocket and method in {"GET", "HEAD"} and "text/html" in request.headers.get("accept", "")
        login = RedirectResponse("/_auth/login?next=" + quote(uri, safe=""), status_code=303)
        who = await identity(request)
        if who is None:
            return login if page_request else JSONResponse({"code": "unauthorized", "message": "需要登录"}, 401)
        if who == "local":
            return Response(status_code=204)
        if (websocket or method not in _SAFE_METHODS) and not _same_origin(request.headers, required=websocket):
            return JSONResponse({"code": "cross_origin", "message": "来源与 Host 不一致"}, 403)
        if await run_file_io(auth.must_change):
            return login if page_request else JSONResponse(
                {"code": "password_change_required", "message": "需要先设置自己的密码"}, 403)
        return Response(status_code=204)

    @app.get("/_auth/login")
    async def login_page() -> HTMLResponse:
        return HTMLResponse(html, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                           "Content-Security-Policy": _PAGE_POLICY})

    @app.get("/_auth/state")
    async def state(request: Request) -> JSONResponse:
        who = await identity(request)
        return JSONResponse({"signedIn": who is not None,
                             "mustChange": who is not None and await run_file_io(auth.must_change)})

    @app.post("/_auth/login")
    async def login(request: Request) -> JSONResponse:
        key = _peer(request.headers)
        if not _same_origin(request.headers, required=False):
            return JSONResponse({"code": "cross_origin"}, 403)
        if wait := throttle.admit(key):
            return JSONResponse({"code": "throttled", "retryAfter": wait}, 429)
        body = await request.json()
        if not await run_file_io(lambda: auth.check_password(str(body.get("username", "")),
                                                             str(body.get("password", "")))):
            logger.warning("[web.auth] 密码错误 来源=%s", key)
            return JSONResponse({"code": "invalid_credentials"}, 401)
        throttle.clear(key)
        remember = body.get("remember") is not False
        token = await run_file_io(lambda: auth.create_session(
            remember=remember, label=request.headers.get("user-agent", "")))
        logger.info("[web.auth] 登录成功 来源=%s 记住=%s", key, remember)
        return signed_in(request, token, remember=remember)

    @app.post("/_auth/ticket")
    async def ticket(request: Request) -> JSONResponse:
        body = await request.json()
        if not _same_origin(request.headers, required=False) or not await run_file_io(
                lambda: auth.consume_ticket(str(body.get("ticket", "")))):
            return JSONResponse({"code": "invalid_ticket"}, 401)
        token = await run_file_io(lambda: auth.create_session(
            remember=True, label=request.headers.get("user-agent", "")))
        logger.info("[web.auth] 本机票据登录成功 来源=%s", _peer(request.headers))
        return signed_in(request, token, remember=True)

    @app.post("/_auth/password")
    async def password(request: Request) -> JSONResponse:
        token = request.cookies.get(COOKIE, "")
        if not token or not _same_origin(request.headers, required=True) or await run_file_io(
                lambda: auth.session(token)) is None:
            return JSONResponse({"code": "unauthorized"}, 401)
        body = await request.json()
        if not await run_file_io(auth.must_change) and not await run_file_io(
                lambda: auth.check_password(USERNAME, str(body.get("current", "")))):
            return JSONResponse({"code": "invalid_credentials"}, 401)
        try:
            await run_file_io(lambda: auth.set_password(str(body.get("password", "")), keep=token))
        except ValueError as error:
            return JSONResponse({"code": "weak_password", "message": str(error)}, 400)
        logger.info("[web.auth] 密码已更新，其他设备已退出")
        return JSONResponse({"mustChange": False})

    @app.post("/_auth/logout")
    async def logout(request: Request) -> JSONResponse:
        if token := request.cookies.get(COOKIE):
            await run_file_io(lambda: auth.revoke(token))
        response = JSONResponse({"signedIn": False})
        response.delete_cookie(COOKIE, path="/")
        return response

    return app
