from __future__ import annotations

import asyncio
import logging
import stat
import socket
import threading
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import uvicorn
import websockets
from websockets.asyncio.client import ClientConnection
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect, WebSocketState

from agent.plugin_composition.endpoints import Endpoint, load_endpoint_plan
from core.common.file_io import run_file_io

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
    "content-security-policy",
    "pragma",
    "referrer-policy",
    "x-content-type-options",
    "etag",
    "last-modified",
    "location",
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
    # 浏览器模块的媒体缓冲与解码 worker 不取得外部网络来源。
    "media-src blob:",
    "worker-src blob:",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
))


class WebShellServer(uvicorn.Server):
    """在线程和进程入口之间发布确定的监听启动结果。"""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.startup_event = threading.Event()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        try:
            await super().startup(sockets=sockets)
        finally:
            self.startup_event.set()


def create_web_shell_app(workspace: Path) -> FastAPI:
    """按 provider 发布的路由前缀转发；runtime 缺席时由外壳响应。"""
    plan = workspace / "runtime/endpoints.json"
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    async def endpoint(path: str) -> Endpoint | None:
        return _matching_endpoint(await run_file_io(lambda: load_endpoint_plan(plan)), path)

    @app.api_route("/{path:path}",
                   methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy(path: str, request: Request) -> Response:
        try:
            target = await endpoint("/" + path)
        except (OSError, ValueError) as error:
            logger.error("Web Shell endpoint plan 无法读取", exc_info=error)
            return _runtime_unavailable(html="text/html" in request.headers.get("accept", ""),
                                        code="endpoint_plan_unavailable", message="Runtime 路由信息不可用")
        if target is None:
            return _runtime_unavailable(html="text/html" in request.headers.get("accept", ""))
        response = await _proxy_http(request, Path(target.address),
                                     request.scope["raw_path"].decode("ascii"))
        response.headers.setdefault("Content-Security-Policy", _WEB_CONTENT_SECURITY_POLICY)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    @app.websocket("/{path:path}")
    async def proxy_socket(path: str, websocket: WebSocket) -> None:
        try:
            target = await endpoint("/" + path)
        except (OSError, ValueError) as error:
            logger.error("Web Shell endpoint plan 无法读取", exc_info=error)
            await websocket.close(code=1013, reason="Runtime 端点不可用")
            return
        if target is None:
            await websocket.close(code=1013, reason="Runtime 尚未就绪")
            return
        await _proxy_websocket(websocket, Path(target.address),
                               websocket.scope["raw_path"].decode("ascii"))

    return app


def _matching_endpoint(endpoints: tuple[Endpoint, ...], path: str) -> Endpoint | None:
    """使用最长完整路径前缀；不把 /panel-other 误送给 /panel。"""
    matches = [
        (len(prefix), endpoint)
        for endpoint in endpoints if endpoint.protocol == "http+unix"
        for prefix in endpoint.routes
        if prefix == "/" or path == prefix or path.startswith(prefix + "/")
    ]
    return max(matches, key=lambda match: match[0])[1] if matches else None


def create_web_shell_server(
    workspace: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 2236,
) -> WebShellServer:
    config = uvicorn.Config(
        create_web_shell_app(workspace),
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        # 流式代理没有读取期限；停止时不能让旧浏览器请求无限阻止 Core 关闭。
        timeout_graceful_shutdown=10,
    )
    return WebShellServer(config)


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
        return _runtime_unavailable(html="text/html" in request.headers.get("accept", ""))
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
    except asyncio.CancelledError:
        # 停止可能发生在 upstream 响应头之前，此时还没有 Response owner 收尾。
        await client.aclose()
        raise
    except httpx.HTTPError:
        await client.aclose()
        return _runtime_unavailable(html="text/html" in request.headers.get("accept", ""))
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

    # 查询参数属于上游协议，按原始编码透传。
    query = bytes(websocket.scope.get("query_string", b""))
    if query:
        target_path = f"{target_path}?{query.decode('ascii')}"

    # 1. Reject before accepting when no provider owns the runtime socket.
    if not _is_socket(socket_path):
        logger.warning(
            "[web_shell.proxy] ws reject, upstream unavailable socket=%s target=%s",
            socket_path,
            target_path,
        )
        await websocket.close(code=1013, reason="Runtime 尚未就绪")
        return
    origin = websocket.headers.get("origin")
    host = websocket.headers.get("host", "akashic-runtime")
    try:
        parsed_host = urlsplit(f"//{host}")
        _ = parsed_host.port
    except ValueError:
        await websocket.close(code=1008, reason="Host 无效")
        return
    if (
        not parsed_host.hostname
        or parsed_host.username is not None
        or parsed_host.password is not None
        or parsed_host.path
        or parsed_host.query
        or parsed_host.fragment
    ):
        await websocket.close(code=1008, reason="Host 无效")
        return
    requested_protocols = [
        protocol.strip()
        for protocol in websocket.headers.get("sec-websocket-protocol", "").split(",")
        if protocol.strip()
    ]
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
            uri=f"ws://{host}{target_path}",
            origin=origin,
            subprotocols=requested_protocols or None,
            max_size=None,
        ) as upstream:
            try:
                await websocket.accept(subprotocol=upstream.subprotocol)
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
            await websocket.close(code=1013, reason="Runtime 连接不可用")


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
    except websockets.ConnectionClosed:
        pass  # 对端关闭已经结束流；下面转发其真实关闭码。
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

    if websocket.client_state is WebSocketState.DISCONNECTED:
        return
    code = upstream.close_code
    assert code is not None
    if code in {1005, 1006, 1015}:
        await websocket.close(code=1013, reason="Runtime 连接不可用")
    else:
        await websocket.close(code=code, reason=upstream.close_reason or "")


def _is_socket(path: Path) -> bool:
    try:
        return stat.S_ISSOCK(path.stat().st_mode)
    except FileNotFoundError:
        return False


def _runtime_unavailable(*, html: bool = False, code: str = "runtime_unavailable",
                         message: str = "Runtime 尚未就绪") -> Response:
    if html:
        return HTMLResponse("<!doctype html><meta charset=utf-8><title>Akashic</title>"
                            f"<main><h1>{message}</h1><p>服务恢复后刷新此页面。</p></main>",
                            status_code=503, headers={"Retry-After": "1", "Cache-Control": "no-store"})
    return JSONResponse(status_code=503,
                        content={"code": code, "message": message},
                        headers={"Retry-After": "1"})
