from __future__ import annotations

import asyncio
import json
import logging
import stat
import socket
import threading
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from http.cookiejar import CookieJar, DefaultCookiePolicy
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

import httpx
import uvicorn
import websockets
from websockets.asyncio.client import ClientConnection
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect, WebSocketState

from agent.plugin_composition.endpoints import GUARD_PROTOCOL, Endpoint, load_endpoint_plan_requirements
from core.common.file_io import run_file_io

# 外壳写给守卫与上游的保留头；客户端同名头一律丢弃，其余 x-akashic-*（如前端模块身份）照常透传。
_SHELL_HEADERS = frozenset({"x-akashic-method", "x-akashic-uri", "x-akashic-upgrade", "x-akashic-peer", "x-akashic-host"})
# 已建立的 WebSocket 每隔这么久重新问一次守卫；凭据失效后在此期限内断开。
_WS_RECHECK_SECONDS = 5.0
# 守卫暂不可用（超时、5xx、守卫缺席、plan 读不到）时，从首次复检失败起最多保持这么久，之后以 1013 断开让浏览器重连。
# 一次循环停顿无论多长只让在途复检失败一次；要连续失败跨过宽限才断开，未经确认的转发最长约
# 宽限 + 复检间隔 + 守卫超时（约 45 秒），不会无限放行。
_WS_GUARD_GRACE_SECONDS = 30.0
_GUARD_RESPONSE_HEADERS = {"content-type", "location", "cache-control", "retry-after", "www-authenticate"}

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
    # 入口守卫插件的登录接口经代理下发会话 cookie。
    "set-cookie",
    "x-akashic-web-stale",
}

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class _HttpClient:
    client: httpx.AsyncClient
    requests: int = 0


class _HttpClients:
    """Web Shell 独占端点连接池；已撤回端点只排空已有响应。"""

    def __init__(self) -> None:
        self._current: dict[Endpoint, _HttpClient] = {}
        self._retired: set[_HttpClient] = set()
        self._lock = asyncio.Lock()

    async def update(self, endpoints: tuple[Endpoint, ...]) -> None:
        """按实际发布计划退休旧池；长流结束前不关闭其底层连接。"""
        async with self._lock:
            for endpoint, entry in tuple(self._current.items()):
                if endpoint not in endpoints:
                    del self._current[endpoint]
                    self._retired.add(entry)
            for entry in tuple(self._retired):
                if entry.requests == 0:
                    await entry.client.aclose()
                    self._retired.remove(entry)

    @asynccontextmanager
    async def open(self, endpoint: Endpoint) -> AsyncIterator[httpx.AsyncClient]:
        """请求持有借用直到响应结束；Cookie 与鉴权只来自当前浏览器请求。"""
        # 1. 完整端点身份含 owner、generation 和地址，不跨换代复用旧 socket。
        async with self._lock:
            entry = self._current.get(endpoint)
            if entry is None:
                entry = _HttpClient(httpx.AsyncClient(
                    # 只限制闲置连接；长流不能占满一个新增的全局并发配额。
                    transport=httpx.AsyncHTTPTransport(uds=endpoint.address,
                        limits=httpx.Limits(max_connections=None, max_keepalive_connections=20)),
                    base_url="http://akashic-runtime", timeout=httpx.Timeout(30.0, read=None),
                    cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=())),
                ))
                self._current[endpoint] = entry
            entry.requests += 1
        try:
            yield entry.client
        finally:
            # 2. 最后一个响应负责完成旧池的排空。
            async with self._lock:
                entry.requests -= 1
                if entry in self._retired and entry.requests == 0:
                    await entry.client.aclose()
                    self._retired.remove(entry)

    async def aclose(self) -> None:
        async with self._lock:
            for entry in (*self._current.values(), *self._retired):
                await entry.client.aclose()
            self._current.clear()
            self._retired.clear()


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
    # 终端图片（sixel）的解码器是 WebAssembly；只放行编译，不放行 eval。
    "script-src 'self' blob: 'unsafe-inline' 'wasm-unsafe-eval'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    # 聊天上传前读取本页生成的附件 Blob，文本附件预览也读 data/blob。
    "connect-src 'self' data: blob:",
    "frame-src 'self'",
    # 浏览器模块的媒体缓冲与解码 worker 不取得外部网络来源；代码高亮 worker 是同源的构建产物。
    # 聊天里已发送的视频是同源制品地址。
    "media-src 'self' blob:",
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
))


class WebShellServer(uvicorn.Server):
    """在线程和进程入口之间发布确定的监听启动结果。"""

    def __init__(self, config: uvicorn.Config) -> None:
        self._exit_loop: asyncio.AbstractEventLoop | None = None
        self._exit_event: asyncio.Event | None = None
        self._exit_requested = False
        super().__init__(config)
        self.startup_event = threading.Event()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        try:
            await super().startup(sockets=sockets)
        finally:
            self.startup_event.set()

    @property
    def should_exit(self) -> bool:
        return self._exit_requested

    # uvicorn 只轮询该标志；置位时同时通知主循环，可能来自信号处理器。
    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self._exit_requested = value
        loop, event = self._exit_loop, self._exit_event
        if value and loop is not None and event is not None and not loop.is_closed():
            _ = loop.call_soon_threadsafe(event.set)

    # 未请求退出时保持 uvicorn 每 0.1s 一次的 on_tick 节拍。
    async def main_loop(self) -> None:
        self._exit_event = asyncio.Event()
        self._exit_loop = asyncio.get_running_loop()
        counter = 0
        should_exit = await self.on_tick(counter)
        while not should_exit:
            counter = (counter + 1) % 864000
            try:
                _ = await asyncio.wait_for(self._exit_event.wait(), timeout=0.1)
            except TimeoutError:
                pass
            should_exit = await self.on_tick(counter)


class _IngressGuard:
    """plan 要求守卫（选中插件静态声明）时守卫缺席即拒绝；等待就绪的外壳在 Core 就绪前也拒绝。"""

    def __init__(self, workspace: Path, *, wait_for_ready: bool) -> None:
        self._ready = workspace / ".runtime-ready.json"
        self._wait_for_ready = wait_for_ready

    def _ready_now(self) -> bool:
        if not self._wait_for_ready:
            return True
        try:
            ready = json.loads(self._ready.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return False
        return isinstance(ready, dict) and ready.get("state") == "ready"

    # 返回要询问的守卫；None 表示不要求守卫且允许直通；False 表示必须拒绝。
    def select(self, endpoints: tuple[Endpoint, ...], requires: tuple[str, ...]) -> Endpoint | None | bool:
        guards = [item for item in endpoints if item.protocol == GUARD_PROTOCOL]
        if len(guards) > 1 or not self._ready_now():
            return False
        if guards:
            return guards[0]
        return False if GUARD_PROTOCOL in requires else None


async def _guard_verdict(clients: _HttpClients, guard: Endpoint, headers: Mapping[str, str], *,
                         method: str, uri: str, upgrade: str, peer: str) -> Response | None:
    """把原请求的方法、路径与头交给守卫；2xx 放行，其余响应原样作为拒绝结果。"""
    forwarded = {name: value for name, value in headers.items()
                 if name.lower() not in _REQUEST_HEADERS_EXCLUDED and name.lower() not in _SHELL_HEADERS}
    # 代理不转 Host；原值以 x-akashic-host 交给守卫做同源判断。
    forwarded.update({"x-akashic-method": method, "x-akashic-uri": uri, "x-akashic-upgrade": upgrade,
                      "x-akashic-peer": peer, "x-akashic-host": headers.get("host", "")})
    try:
        async with clients.open(guard) as client:
            verdict = await client.get("/_guard/check", headers=forwarded, timeout=10)
    except httpx.HTTPError as error:
        logger.error("Web Shell 入口守卫不可用", exc_info=error)
        return _runtime_unavailable(code="guard_unavailable", message="入口守卫不可用")
    if 200 <= verdict.status_code < 300:
        return None
    return Response(verdict.content[:65536], status_code=verdict.status_code,
                    headers={name: value for name, value in verdict.headers.items()
                             if name.lower() in _GUARD_RESPONSE_HEADERS})


def create_web_shell_app(workspace: Path, *, wait_for_ready: bool = False) -> FastAPI:
    """按 provider 发布的路由前缀转发；runtime 缺席时由外壳响应。"""
    plan = workspace / "runtime/endpoints.json"
    clients = _HttpClients()
    ingress = _IngressGuard(workspace, wait_for_ready=wait_for_ready)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await clients.aclose()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    # 先过入口守卫再匹配路由；拒绝时不暴露任何插件端点。
    # 先过入口守卫再匹配路由；守卫与路由使用同一规范路径。
    async def endpoint(raw: str, query: str, headers: Mapping[str, str], *, method: str,
                       upgrade: str, peer: str) -> Endpoint | Response | None:
        endpoints, requires = await run_file_io(lambda: load_endpoint_plan_requirements(plan))
        await clients.update(endpoints)
        guard = await run_file_io(lambda: ingress.select(endpoints, requires))
        if guard is False:
            return _runtime_unavailable(html="text/html" in headers.get("accept", ""),
                                        code="guard_unavailable", message="入口守卫尚未就绪")
        if isinstance(guard, Endpoint):
            verdict = await _guard_verdict(clients, guard, headers, method=method,
                                           uri=raw + ("?" + query if query else ""), upgrade=upgrade, peer=peer)
            if verdict is not None:
                return verdict
        return _matching_endpoint(endpoints, unquote(raw))

    @app.api_route("/{path:path}",
                   methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy(path: str, request: Request) -> Response:
        raw = _canonical_path(request.scope["raw_path"])
        if raw is None:
            return JSONResponse({"code": "bad_request_target", "message": "请求路径不规范"}, 400)
        try:
            target = await endpoint(raw, request.url.query, request.headers, method=request.method,
                                    upgrade="", peer=request.client.host if request.client else "")
        except (OSError, ValueError) as error:
            logger.error("Web Shell endpoint plan 无法读取", exc_info=error)
            return _runtime_unavailable(html="text/html" in request.headers.get("accept", ""),
                                        code="endpoint_plan_unavailable", message="Runtime 路由信息不可用")
        if isinstance(target, Response):
            return target
        if target is None:
            return _runtime_unavailable(html="text/html" in request.headers.get("accept", ""))
        response = await _proxy_http(request, clients, target, raw)
        response.headers.setdefault("Content-Security-Policy", _WEB_CONTENT_SECURITY_POLICY)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    @app.websocket("/{path:path}")
    async def proxy_socket(path: str, websocket: WebSocket) -> None:
        raw = _canonical_path(websocket.scope["raw_path"])
        if raw is None:
            await websocket.close(code=1008, reason="请求路径不规范")
            return
        query = websocket.scope.get("query_string", b"").decode("latin-1")

        async def admit() -> Endpoint | Response | None:
            return await endpoint(raw, query, websocket.headers, method="GET", upgrade="websocket",
                                  peer=websocket.client.host if websocket.client else "")
        try:
            target = await admit()
        except (OSError, ValueError) as error:
            logger.error("Web Shell endpoint plan 无法读取", exc_info=error)
            await websocket.close(code=1013, reason="Runtime 端点不可用")
            return
        if isinstance(target, Response):
            await websocket.close(code=1008 if target.status_code < 500 else 1013, reason="未通过入口守卫")
            return
        if target is None:
            await websocket.close(code=1013, reason="Runtime 尚未就绪")
            return
        await _relay_while_admitted(websocket, _proxy_websocket(websocket, Path(target.address), raw), admit)

    return app


_Recheck = Literal["admitted", "denied", "unavailable"]


# 复检结论与握手一致：守卫的非 2xx、<500 响应或路由已撤回算拒绝；5xx 与 plan 读取失败算守卫暂不可用。
async def _recheck(admit: Callable[[], Awaitable[Endpoint | Response | None]]) -> _Recheck:
    try:
        target = await admit()
    except (OSError, ValueError) as error:
        logger.warning("Web Shell 复检时 endpoint plan 无法读取", exc_info=error)
        return "unavailable"
    if isinstance(target, Endpoint):
        return "admitted"
    if isinstance(target, Response) and target.status_code >= 500:
        return "unavailable"
    return "denied"


async def _relay_while_admitted(websocket: WebSocket, relay: Awaitable[None],
                                admit: Callable[[], Awaitable[Endpoint | Response | None]]) -> None:
    """长连接不止握手时授权一次：定期重问守卫；明确拒绝即 1008 断开，守卫持续不可用超过宽限则 1013 断开。"""
    task = asyncio.ensure_future(relay)
    unavailable_since: float | None = None
    try:
        while not (await asyncio.wait({task}, timeout=_WS_RECHECK_SECONDS))[0]:
            outcome = await _recheck(admit)
            if outcome == "admitted":
                unavailable_since = None
                continue
            if outcome == "denied":
                await _close_relay(websocket, task, 1008, "授权已失效")
                return
            # 守卫答不上来不等于撤权：保持转发，下一轮再问；连续失败跨过宽限才断开。
            now = time.monotonic()
            if unavailable_since is None:
                unavailable_since = now
                logger.warning("Web Shell 复检暂时无法确认授权，保持连接；持续 %.0f 秒后的下一次失败才断开", _WS_GUARD_GRACE_SECONDS)
            if now - unavailable_since >= _WS_GUARD_GRACE_SECONDS:
                logger.warning("Web Shell 入口守卫持续不可用 %.0f 秒，以 1013 断开", now - unavailable_since)
                await _close_relay(websocket, task, 1013, "入口守卫不可用")
                return
        await task
    finally:
        if not task.done():
            _ = task.cancel()
            with suppress(asyncio.CancelledError):
                await task


# 先以关闭码告知浏览器，再停止转发；反过来会被转发清理抢先以 1000 关闭。
async def _close_relay(websocket: WebSocket, task: asyncio.Future[None], code: int, reason: str) -> None:
    if websocket.client_state is WebSocketState.CONNECTED:
        with suppress(RuntimeError, OSError):
            await websocket.close(code=code, reason=reason)
    _ = task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def _canonical_path(raw_path: bytes) -> str | None:
    """返回原样转发的 ASCII 路径；网络路径、点段、编码斜杠与反斜杠返回 None。"""
    try:
        raw = raw_path.decode("ascii")
    except UnicodeDecodeError:
        return None
    lowered = raw.lower()
    if not raw.startswith("/") or raw.startswith("//") or "\\" in raw or "%2f" in lowered or "%5c" in lowered:
        return None
    if any(segment in {".", ".."} for segment in unquote(raw).split("/")):
        return None
    return raw


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
    wait_for_ready: bool = False,
) -> WebShellServer:
    config = uvicorn.Config(
        create_web_shell_app(workspace, wait_for_ready=wait_for_ready),
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
    clients: _HttpClients,
    endpoint: Endpoint,
    target_path: str,
) -> Response:
    """Relay one HTTP request to a workspace-owned Unix socket."""

    # 1. Refuse stale or unavailable runtimes with an explicit readiness result.
    socket_path = Path(endpoint.address)
    if not _is_socket(socket_path):
        logger.warning(
            "[web_shell.proxy] http backend unavailable socket=%s target=%s",
            socket_path,
            target_path,
        )
        return _runtime_unavailable(html="text/html" in request.headers.get("accept", ""))
    query = request.url.query
    target = f"{target_path}?{query}" if query else target_path
    # 保留头只能由外壳写入；对端地址与原 Host 供插件退避和同源判断。
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in _REQUEST_HEADERS_EXCLUDED and name.lower() not in _SHELL_HEADERS
    }
    headers["x-akashic-peer"] = request.client.host if request.client else ""
    headers["x-akashic-host"] = request.headers.get("host", "")

    # 2. 借用跨越整个流式响应；异常、取消和浏览器断连都先关闭响应再归还借用。
    async with AsyncExitStack() as lifetime:
        client = await lifetime.enter_async_context(clients.open(endpoint))
        try:
            logger.debug("[web_shell.proxy] http relay start socket=%s target=%s method=%s",
                         socket_path, target_path, request.method)
            upstream_request = client.build_request(request.method, target, headers=headers,
                                                    content=request.stream())
            upstream = await client.send(upstream_request, stream=True)
        except ClientDisconnect:
            return Response(status_code=499)
        except httpx.HTTPError:
            return _runtime_unavailable(html="text/html" in request.headers.get("accept", ""))
        lifetime.push_async_callback(upstream.aclose)
        response_headers = {name: value for name, value in upstream.headers.items()
                            if name.lower() in _RESPONSE_HEADERS_ALLOWED}
        return _ProxyStreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code,
                                       headers=response_headers, close=lifetime.pop_all().aclose)


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
            await _relay_both_directions(websocket, upstream, websocket_id)
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


# 任一方向结束即停止另一方向；外层取消（复检断开、外壳停止）时同样取消并等待两个方向，异常都在这里取走。
async def _relay_both_directions(websocket: WebSocket, upstream: ClientConnection, websocket_id: str) -> None:
    tasks = {
        "browser->gateway": asyncio.create_task(_relay_browser_messages(websocket, upstream)),
        "gateway->browser": asyncio.create_task(_relay_gateway_messages(upstream, websocket)),
    }
    try:
        _ = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks.values():
            _ = task.cancel()
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for name, result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.warning("[web_shell.proxy] ws task failed ws_id=%s task=%s err=%r",
                               websocket_id, name, result)


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
        # 外壳已向浏览器发出关闭帧（如复检断开）后，在途的 send 失败属于正常收尾。
        if websocket.application_state is WebSocketState.DISCONNECTED:
            return
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

    if WebSocketState.DISCONNECTED in (websocket.client_state, websocket.application_state):
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
