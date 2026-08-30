from __future__ import annotations

import asyncio
import logging
import stat
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from contextlib import suppress
from pathlib import Path

import httpx
import uvicorn
import websockets
from websockets.asyncio.client import ClientConnection
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect

from bootstrap.settings_api import SettingsServer, create_settings_app
from bootstrap.web_runtime import chat_socket_path, dashboard_socket_path

_REQUEST_HEADERS_EXCLUDED = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_RESPONSE_HEADERS_ALLOWED = {
    "accept-ranges",
    "cache-control",
    "content-disposition",
    "content-length",
    "content-range",
    "content-type",
    "etag",
    "last-modified",
    "x-akashic-web-stale",
}

logger = logging.getLogger(__name__)


class _ProxyStreamingResponse(StreamingResponse):
    """Own the upstream response until the browser stream ends."""

    def __init__(
        self,
        content: AsyncIterable[bytes],
        *,
        status_code: int,
        headers: Mapping[str, str],
        close: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(content, status_code=status_code, headers=headers)
        self._close = close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        except ClientDisconnect:
            pass
        finally:
            await self._close()

_WEB_CONTENT_SECURITY_POLICY = "; ".join((
    "default-src 'self'",
    "script-src 'self' blob: 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    "connect-src 'self'",
    "frame-src 'self'",
    "media-src 'none'",
    "worker-src 'none'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
))


def create_web_shell_app(
    config_path: Path,
    workspace: Path,
) -> FastAPI:
    """Serve the only public Web entry and relay ready Gateway capabilities."""

    chat_socket = chat_socket_path(workspace)
    dashboard_socket = dashboard_socket_path(workspace)
    dashboard_static = Path(__file__).resolve().parent.parent / "static" / "dashboard"
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    @app.get("/dashboard")
    @app.get("/dashboard/")
    async def dashboard_shell_index() -> Response:
        index_file = dashboard_static / "index.html"
        if not index_file.exists():
            return Response(
                content="Dashboard 前端尚未构建，请先运行 `npm run build`。",
                media_type="text/plain; charset=utf-8",
                status_code=503,
            )
        return Response(
            content=index_file.read_text(encoding="utf-8"),
            media_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": _WEB_CONTENT_SECURITY_POLICY,
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/api/shell/state")
    async def shell_state() -> dict[str, object]:
        chat_ready = await _runtime_ready(chat_socket, "/api/chat/health")
        return {
            "status": (
                "ready"
                if chat_ready
                else "starting" if config_path.exists() else "needs_setup"
            ),
            "configured": config_path.exists(),
            "chatReady": chat_ready,
        }

    @app.api_route(
        "/api/chat/{proxy_path:path}",
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy_chat(proxy_path: str, request: Request) -> Response:
        if proxy_path.startswith("model-settings"):
            return JSONResponse(
                status_code=404,
                content={"code": "not_found", "message": "接口不存在"},
            )
        return await _proxy_http(request, chat_socket, f"/api/chat/{proxy_path}")

    @app.api_route(
        "/api/settings/model/{proxy_path:path}",
        methods=["GET", "POST", "OPTIONS"],
    )
    async def proxy_model_settings(proxy_path: str, request: Request) -> Response:
        rejection = _reject_settings_request(request)
        if rejection is not None:
            return rejection
        response = await _proxy_http(
            request,
            chat_socket,
            f"/api/chat/model-settings/{proxy_path}",
        )
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.api_route("/api/settings/state", methods=["GET", "HEAD", "OPTIONS"])
    @app.api_route("/api/settings/models", methods=["POST", "OPTIONS"])
    @app.api_route("/api/settings/apply", methods=["POST", "OPTIONS"])
    @app.api_route("/api/settings/roles", methods=["POST", "OPTIONS"])
    @app.api_route("/api/settings/embedding-models", methods=["POST", "OPTIONS"])
    @app.api_route("/api/settings/codex-login", methods=["POST", "OPTIONS"])
    @app.api_route(
        "/api/settings/codex-login/{login_id}",
        methods=["GET", "HEAD", "OPTIONS"],
    )
    async def retired_model_settings(login_id: str | None = None) -> Response:
        _ = login_id
        return JSONResponse(
            status_code=410,
            content={
                "code": "model_settings_moved",
                "message": "模型设置已迁移到插件控制接口",
            },
        )

    @app.websocket("/ws")
    async def proxy_chat_websocket(websocket: WebSocket) -> None:
        await _proxy_websocket(websocket, chat_socket, "/ws")

    @app.api_route(
        "/api/dashboard/{proxy_path:path}",
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy_dashboard_api(proxy_path: str, request: Request) -> Response:
        return await _proxy_http(
            request,
            dashboard_socket,
            f"/api/dashboard/{proxy_path}",
        )

    app.mount(
        "/dashboard/assets",
        StaticFiles(directory=dashboard_static, check_dir=False),
        name="dashboard-shell-assets",
    )

    app.mount(
        "/",
        create_settings_app(),
        name="web-shell-static-and-settings",
    )
    return app


def _reject_settings_request(request: Request) -> JSONResponse | None:
    """Keep authenticated mutations at the public 2236 boundary."""

    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return None
    expected = f"http://{request.url.netloc}"
    if (
        request.headers.get("origin", "") == expected
        and request.headers.get("x-akasic-csrf") == "1"
    ):
        return None
    return JSONResponse(
        status_code=403,
        content={"code": "csrf_rejected", "message": "请求来源无效"},
    )


def create_web_shell_server(
    config_path: Path,
    workspace: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 2236,
) -> SettingsServer:
    config = uvicorn.Config(
        create_web_shell_app(config_path, workspace),
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    return SettingsServer(config)


async def _runtime_ready(socket_path: Path, health_path: str) -> bool:
    if not _is_socket(socket_path):
        return False
    transport = httpx.AsyncHTTPTransport(uds=str(socket_path))
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://akashic-runtime",
            timeout=0.5,
        ) as client:
            response = await client.get(health_path)
            return response.status_code == 200
    except httpx.HTTPError:
        return False


async def _proxy_http(
    request: Request,
    socket_path: Path,
    target_path: str,
) -> Response:
    """Relay one HTTP request to a workspace-owned Unix socket."""

    # 1. Refuse stale or unavailable runtimes with an explicit readiness result.
    if not _is_socket(socket_path):
        logger.warning(
            "[web_shell.proxy] http backend unavailable socket=%s target=%s",
            socket_path,
            target_path,
        )
        return _runtime_unavailable()
    client = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=str(socket_path)),
        base_url="http://akashic-runtime",
        timeout=httpx.Timeout(30.0, read=None),
    )
    query = request.url.query
    target = f"{target_path}?{query}" if query else target_path
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in _REQUEST_HEADERS_EXCLUDED
    }

    # 2. Stream request and response bodies without turning attachments into RAM copies.
    try:
        logger.debug(
            "[web_shell.proxy] http relay start socket=%s target=%s method=%s",
            socket_path,
            target_path,
            request.method,
        )
        upstream_request = client.build_request(
            request.method,
            target,
            headers=headers,
            content=request.stream(),
        )
        upstream = await client.send(upstream_request, stream=True)
    except ClientDisconnect:
        await client.aclose()
        return Response(status_code=499)
    except httpx.HTTPError:
        await client.aclose()
        return _runtime_unavailable()
    response_headers = {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() in _RESPONSE_HEADERS_ALLOWED
    }

    async def close_upstream() -> None:
        try:
            await upstream.aclose()
        finally:
            await client.aclose()

    return _ProxyStreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=response_headers,
        close=close_upstream,
    )


async def _proxy_websocket(
    websocket: WebSocket,
    socket_path: Path,
    target_path: str,
) -> None:
    """Relay one browser WebSocket while preserving disconnect semantics."""

    # 1. Reject before accepting when no Gateway owns the runtime socket.
    if not _is_socket(socket_path):
        logger.warning(
            "[web_shell.proxy] ws reject, upstream unavailable socket=%s target=%s",
            socket_path,
            target_path,
        )
        await websocket.close(code=1013, reason="Gateway 尚未就绪")
        return
    origin = websocket.headers.get("origin")
    websocket_id = f"ws-{id(websocket):x}"
    try:
        logger.debug(
            "[web_shell.proxy] ws connect start ws_id=%s socket=%s target=%s origin=%s",
            websocket_id,
            socket_path,
            target_path,
            origin,
        )
        async with websockets.unix_connect(
            str(socket_path),
            uri=f"ws://akashic-runtime{target_path}",
            origin=origin,
            max_size=None,
        ) as upstream:
            try:
                await websocket.accept()
            except OSError as error:
                logger.info(
                    "[web_shell.proxy] browser disconnected before accept "
                    "ws_id=%s err=%r",
                    websocket_id,
                    error,
                )
                return
            logger.info(
                "[web_shell.proxy] ws connected ws_id=%s socket=%s target=%s",
                websocket_id,
                socket_path,
                target_path,
            )

            # 2. Stop both directions as soon as either peer disconnects.
            browser_to_gateway = asyncio.create_task(
                _relay_browser_messages(websocket, upstream)
            )
            gateway_to_browser = asyncio.create_task(
                _relay_gateway_messages(upstream, websocket)
            )
            done, pending = await asyncio.wait(
                (browser_to_gateway, gateway_to_browser),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                _ = task.cancel()
            for task in pending:
                with suppress(asyncio.CancelledError):
                    await task
            for task in done:
                task_name = (
                    "browser->gateway"
                    if task is browser_to_gateway
                    else "gateway->browser"
                )
                exception = task.exception()
                if exception is not None:
                    logger.warning(
                        "[web_shell.proxy] ws task failed ws_id=%s task=%s err=%r",
                        websocket_id,
                        task_name,
                        exception,
                    )
            logger.info(
                "[web_shell.proxy] ws flow complete ws_id=%s socket=%s target=%s",
                websocket_id,
                socket_path,
                target_path,
            )
    except (OSError, websockets.WebSocketException) as error:
        logger.warning(
            "[web_shell.proxy] ws connect/relay failed ws_id=%s socket=%s target=%s err=%r",
            websocket_id,
            socket_path,
            target_path,
            error,
        )
        with suppress(OSError, RuntimeError, WebSocketDisconnect):
            await websocket.close(code=1013, reason="Gateway 连接不可用")


async def _relay_browser_messages(
    websocket: WebSocket,
    upstream: ClientConnection,
) -> None:
    while True:
        try:
            message = await websocket.receive()
        except Exception as error:
            logger.debug(
                "[web_shell.proxy] browser->gateway receive failed ws=%s err=%r",
                f"ws-{id(websocket):x}",
                error,
            )
            raise
        if message["type"] == "websocket.disconnect":
            logger.info(
                "[web_shell.proxy] browser->gateway disconnect ws=%s",
                f"ws-{id(websocket):x}",
            )
            await upstream.close()
            return
        if message.get("text") is not None:
            await upstream.send(message["text"])
            continue
        if message.get("bytes") is not None:
            await upstream.send(message["bytes"])
            continue
        logger.debug(
            "[web_shell.proxy] browser->gateway unsupported frame ws=%s type=%s",
            f"ws-{id(websocket):x}",
            message["type"],
        )


async def _relay_gateway_messages(
    upstream: ClientConnection,
    websocket: WebSocket,
) -> None:
    try:
        async for message in upstream:
            if isinstance(message, str):
                await websocket.send_text(message)
            else:
                await websocket.send_bytes(message)
    except Exception as error:
        logger.debug(
            "[web_shell.proxy] gateway->browser closed ws=%s err=%r",
            f"ws-{id(websocket):x}",
            error,
        )
        raise
    logger.debug(
        "[web_shell.proxy] gateway->browser stream closed ws=%s",
        f"ws-{id(websocket):x}",
    )


def _is_socket(path: Path) -> bool:
    try:
        return stat.S_ISSOCK(path.stat().st_mode)
    except FileNotFoundError:
        return False


def _runtime_unavailable() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"code": "gateway_unavailable", "message": "Gateway 尚未就绪"},
        headers={"Retry-After": "1"},
    )
