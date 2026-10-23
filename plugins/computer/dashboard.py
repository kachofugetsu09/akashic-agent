from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from typing import cast
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake
from websockets.typing import Subprotocol

from agent.plugin_composition import DashboardContext

from .stream import STREAM_CONFIG, IceServer, StreamAccess, StreamMode, lan_candidate, relay_only, signaling_hello

inject = (STREAM_CONFIG,)


def _websocket_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("Computer display endpoint is not an HTTP URL")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunsplit((scheme, parsed.netloc, "/", "", ""))


# 只读观看时只放行画面协商：分辨率、帧率、关键帧和帧确认；键鼠、剪贴板与上传一律丢弃。
_VIEW_ONLY_PREFIXES = ("SETTINGS,", "r,", "s,", "_arg_fps,", "_crf,", "_f,", "START_VIDEO", "STOP_VIDEO",
                       "REQUEST_KEYFRAME", "CLIENT_FRAME_ACK", "LOST_FRAME", "_gz,")


def _view_only_allows(text: str) -> bool:
    return text.startswith(_VIEW_ONLY_PREFIXES)


def _state_event(line: str) -> dict[str, object] | None:
    if not line.startswith("data: "):
        return None
    value: object = json.loads(line[6:])
    if not isinstance(value, dict):
        return None
    state = cast(dict[str, object], value)
    return state if state.get("type") == "state" else None


# Agent 没有进行中的回合和操作、也没人接管时，观看者可以直接操作。
def _agent_idle(state: dict[str, object]) -> bool:
    browser = state.get("browser")
    ready = isinstance(browser, dict) and cast(dict[str, object], browser).get("state") == "ready"
    return ready and state.get("turns") == [] and state.get("active") is False and state.get("control") is False


def _gateway_error(response: httpx.Response) -> str:
    value: object = response.json()
    message = cast(dict[str, object], value).get("error") if isinstance(value, dict) else None
    return message if isinstance(message, str) else f"Computer returned {response.status_code}"


def register(app: FastAPI, context: DashboardContext) -> httpx.Client:
    """Expose the exact generation's private Computer endpoint to its web tab."""

    gateway = context.workload_url("computer", "gateway")
    stream_http = context.workload_url("computer", "stream")
    stream = _websocket_url(stream_http).rstrip("/") + "/api/websockets"
    signaling = _websocket_url(stream_http).rstrip("/") + "/api/webrtc/signaling/"
    client = httpx.Client(base_url=gateway, timeout=125.0)
    access = StreamAccess(gateway, stream_http)

    def forward(
        method: str, path: str, payload: object | None = None
    ) -> httpx.Response:
        try:
            response = client.request(method, path, json=payload)
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if response.status_code == 409:
            # Agent 正在使用浏览器：原样告诉面板，由它提示先接管。
            raise HTTPException(status_code=409, detail="Agent is using the browser")
        if response.status_code == 400:
            raise HTTPException(status_code=400, detail=_gateway_error(response))
        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=_gateway_error(response),
            )
        return response

    @app.get("/api/dashboard/computer/activity")
    def activity() -> Response:
        result = forward("GET", "/activity")
        return Response(result.content, media_type="application/json")

    @app.post("/api/dashboard/computer/wake")
    def wake() -> Response:
        result = forward("POST", "/wake", {})
        return Response(result.content, media_type="application/json")

    @app.post("/api/dashboard/computer/restart")
    def restart() -> Response:
        result = forward("POST", "/restart", {})
        return Response(result.content, media_type="application/json")

    @app.post("/api/dashboard/computer/show")
    def show(payload: dict[str, str]) -> Response:
        result = forward("POST", "/view/show", {"target": payload.get("target", "")})
        return Response(result.content, media_type="application/json")

    @app.post("/api/dashboard/computer/navigate")
    def navigate(payload: dict[str, str]) -> Response:
        body = {key: payload[key] for key in ("target", "action", "url", "control") if key in payload}
        result = forward("POST", "/view/navigate", body)
        return Response(result.content, media_type="application/json")

    @app.post("/api/dashboard/computer/open")
    def open_page(payload: dict[str, str]) -> Response:
        result = forward("POST", "/view/open", {key: payload[key] for key in ("url", "control") if key in payload})
        return Response(result.content, media_type="application/json")

    @app.post("/api/dashboard/computer/close")
    def close_page(payload: dict[str, str]) -> Response:
        result = forward("POST", "/view/close", {key: payload[key] for key in ("target", "control") if key in payload})
        return Response(result.content, media_type="application/json")

    @app.get("/api/dashboard/computer/stream-client")
    def stream_client() -> Response:
        result = forward("GET", stream_http.rstrip("/") + "/selkies-core.js")
        return Response(result.content, media_type="text/javascript", headers={"Cache-Control": "no-store"})

    @app.get("/api/dashboard/computer/stream-source/{name}")
    def stream_source(name: str) -> Response:
        if name not in {"selkies-ws-core.js", "selkies-wr-core.js", "util.js", "LICENSE"}:
            raise HTTPException(status_code=404, detail="Unknown Computer stream source")
        result = forward("GET", stream_http.rstrip("/") + "/source/" + name)
        return Response(result.content, media_type="text/plain")

    @app.websocket("/api/dashboard/computer/events")
    async def computer_events(socket: WebSocket) -> None:
        """当前 generation 只读订阅电脑状态与操作位置，不获取输入或唤醒权限。"""
        timeout = httpx.Timeout(10.0, read=None)
        try:
            async with httpx.AsyncClient(timeout=timeout) as reader:
                async with reader.stream("GET", gateway.rstrip("/") + "/events") as response:
                    response.raise_for_status()
                    await socket.accept()

                    async def receive_events() -> None:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                await socket.send_text(line[6:])

                    async def receive_browser() -> None:
                        message = await socket.receive()
                        if message["type"] != "websocket.disconnect":
                            await socket.close(code=1008, reason="Computer events are read-only")

                    tasks = {asyncio.create_task(receive_events()), asyncio.create_task(receive_browser())}
                    try:
                        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            await task
                    finally:
                        for task in tasks:
                            task.cancel()
                        for task in tasks:
                            with suppress(asyncio.CancelledError):
                                await task
        except WebSocketDisconnect:
            return
        except httpx.HTTPError:
            if socket.client_state is WebSocketState.CONNECTING:
                await socket.accept()
            if socket.client_state is WebSocketState.CONNECTED:
                await socket.close(code=1013, reason="Computer events are unavailable")

    @app.post("/api/dashboard/computer/stream-session")
    async def stream_session(prefer: str = "") -> dict[str, object]:
        """选定这次连接的传输：能直连或有中继就用 WebRTC；面板回退时改用 WebSocket。"""
        settings = context.require(STREAM_CONFIG)
        servers: list[IceServer] = []
        if settings.webrtc and prefer != "websockets":
            servers = await settings.ice_servers()
        # 中继凭据没拿到又没有局域网地址，WebRTC 注定连不上，直接用网页连接。
        webrtc = bool(servers) or (settings.config.lan_address is not None and prefer != "websockets")
        mode: StreamMode = "webrtc" if webrtc else "websockets"
        try:
            if webrtc:
                await access.relay(relay_only(servers))
            await access.use(mode)
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return {"mode": mode, "rtcConfig": {"iceServers": servers}}

    @app.post("/api/dashboard/computer/touch")
    def touch_view() -> Response:
        # WebRTC 的键鼠不经 Core，由面板在人直接操作时续期。
        result = forward("POST", "/view/touch", {})
        return Response(result.content, media_type="application/json")

    @app.websocket("/api/dashboard/computer/stream")
    async def computer_display(socket: WebSocket) -> None:
        await _serve(_View(socket, gateway, access, "websockets"))

    @app.websocket("/api/dashboard/computer/signaling")
    async def computer_signaling(socket: WebSocket) -> None:
        await _serve(_View(socket, gateway, access, "webrtc"))

    async def _serve(view: _View) -> None:
        """把当前 generation 的画面连接转发到私有流服务；接管时先把目标标签提到最前。"""
        socket = view.socket
        if view.control_id and (len(view.control_id) > 128 or not view.control_id.isascii()):
            await socket.close(code=1008, reason="Invalid control identity")
            return
        try:
            # 1. 先取得接管和会话令牌，流服务才接受这条连接的输入。
            if view.control_id:
                await view.control("take")
                view.controlled = True
            # WebSocket 的输入经下面的转发过滤；WebRTC 的键鼠直达流服务，只能靠令牌权限挡住。
            view.token = await access.open(bool(view.control_id) or view.mode == "websockets")
            url = f"{stream}?role=controller&token={view.token}" if view.mode == "websockets" else signaling
            # 2. 画面连接总以主控端身份接入，屏幕才会跟随面板尺寸；只读由令牌和转发过滤保证。
            upstream_context = connect(url, subprotocols=view.protocols(), compression=None, open_timeout=10,
                                       close_timeout=5, max_size=None, proxy=None)
            async with upstream_context as upstream:
                await socket.accept(subprotocol=upstream.subprotocol)
                await _pump(view, upstream)
            # 3. 别的窗口换了传输方式，等同于它接管了画面。
            if socket.client_state is WebSocketState.CONNECTED and view.switched:
                await socket.close(code=4001, reason="superseded by another window")
        except WebSocketDisconnect:
            return
        except (OSError, TimeoutError, InvalidHandshake, httpx.HTTPError):
            if socket.client_state is WebSocketState.CONNECTING:
                await socket.accept()
            if socket.client_state is WebSocketState.CONNECTED:
                await socket.close(code=1013, reason="Computer display or control is unavailable")
        finally:
            await view.close()

    async def _pump(view: _View, upstream: ClientConnection) -> None:
        if view.mode == "websockets":
            tasks = {_display_up(view, upstream), _display_down(view, upstream)}
        else:
            lan = context.require(STREAM_CONFIG).config.lan_address
            tasks = {_signal_up(view, upstream), _signal_down(view, upstream, lan)}
        if view.control_id:
            tasks.add(view.renew_control())
        else:
            tasks.add(view.watch_idle(upstream))
            if view.mode == "websockets":
                tasks.add(view.renew_while_used())
        running = {asyncio.create_task(task) for task in tasks}
        done, pending = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            _ = task.cancel()
        for task in pending:
            with suppress(asyncio.CancelledError):
                await task
        for task in done:
            await task

    return client


class _View:
    """一条画面连接：接管身份、会话令牌，以及没接管时随 Agent 空闲与否开关的输入权限。"""

    def __init__(self, socket: WebSocket, gateway: str, access: StreamAccess, mode: StreamMode) -> None:
        self.socket = socket
        self.mode: StreamMode = mode
        self.target = socket.query_params.get("target", "")
        self.control_id = socket.query_params.get("control", "")
        self.reader = httpx.AsyncClient(base_url=gateway, timeout=40.0)
        self.access = access
        self.switches = access.switches
        self.token = ""
        self.controlled = False
        # 没接管时，Agent 空闲就可以直接操作；状态来自网关事件流，收回时先松开按住的键和鼠标。
        self.idle = False
        self.held_keys: set[str] = set()
        self.last_pointer: list[str] = []
        self.touched = 0.0
        self.renew_wanted = asyncio.Event()

    @property
    def switched(self) -> bool:
        return self.access.switches != self.switches

    def protocols(self) -> list[Subprotocol] | None:
        requested = {item.strip() for item in self.socket.headers.get("sec-websocket-protocol", "").split(",")}
        return [Subprotocol("binary")] if "binary" in requested else None

    async def control(self, action: str) -> None:
        result = await self.reader.post("/control/" + action, json={"id": self.control_id, "target": self.target})
        _ = result.raise_for_status()

    async def renew_control(self) -> None:
        while True:
            await asyncio.sleep(5)
            await self.control("renew")

    async def watch_idle(self, upstream: ClientConnection) -> None:
        timeout = httpx.Timeout(10.0, read=None)
        async with self.reader.stream("GET", "/events", timeout=timeout) as response:
            _ = response.raise_for_status()
            async for line in response.aiter_lines():
                if (state := _state_event(line)) is None:
                    continue
                was_idle, self.idle = self.idle, _agent_idle(state)
                # 先停止转发再松键，最后收回令牌：发送期间到达的按下不会再被转发，松键也不会被流服务丢弃。
                if self.mode == "webrtc":
                    await self.access.allow(self.token, self.idle)
                elif was_idle and not self.idle:
                    await self._release(upstream)

    async def _release(self, upstream: ClientConnection) -> None:
        releasing = tuple(self.held_keys)
        self.held_keys.clear()
        for key in releasing:
            await upstream.send("ku," + key)
        if len(self.last_pointer) == 4 and self.last_pointer[3] != "0":
            await upstream.send(",".join([*self.last_pointer[:3], "0", "0"]))
            self.last_pointer.clear()

    def touch(self) -> None:
        # 人在直接操作时续期，浏览器不会在用户使用中途休眠；这里只做标记，请求由后台任务发，不拖慢输入转发。
        if time.monotonic() - self.touched < 30:
            return
        self.touched = time.monotonic()
        self.renew_wanted.set()

    async def renew_while_used(self) -> None:
        # 续期失败和以前一样结束这条画面连接（关闭码 1013）。
        while True:
            _ = await self.renew_wanted.wait()
            self.renew_wanted.clear()
            result = await self.reader.post("/view/touch", json={})
            _ = result.raise_for_status()

    def allowed(self, text: str) -> bool:
        return bool(self.control_id) or _view_only_allows(text) or self.idle

    async def close(self) -> None:
        try:
            if self.controlled:
                await self.control("release")
            if self.token:
                with suppress(httpx.HTTPError):
                    await self.access.close(self.token)
        finally:
            await self.reader.aclose()


async def _display_up(view: _View, upstream: ClientConnection) -> None:
    try:
        while True:
            message = await view.socket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                if view.control_id or view.idle:
                    await upstream.send(message["bytes"])
            elif (text := message.get("text")) is not None and view.allowed(text):
                if text.startswith("kd,"):
                    view.held_keys.add(text.split(",")[1])
                elif text.startswith("ku,"):
                    view.held_keys.discard(text.split(",")[1])
                elif text == "kr":
                    view.held_keys.clear()
                if text.startswith("m,"):
                    view.last_pointer[:] = text.split(",")[:4]
                await upstream.send(text)
                if view.idle and not view.control_id and not _view_only_allows(text):
                    view.touch()
    except WebSocketDisconnect:
        return


async def _display_down(view: _View, upstream: ClientConnection) -> None:
    try:
        async for message in upstream:
            if isinstance(message, bytes):
                await view.socket.send_bytes(message)
            else:
                await view.socket.send_text(message)
    except ConnectionClosed:
        return


async def _signal_up(view: _View, upstream: ClientConnection) -> None:
    try:
        while True:
            message = await view.socket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if (text := message.get("text")) is not None:
                await upstream.send(signaling_hello(text, view.token))
    except WebSocketDisconnect:
        return


async def _signal_down(view: _View, upstream: ClientConnection, lan: str | None) -> None:
    try:
        async for message in upstream:
            text = message if isinstance(message, str) else message.decode()
            await view.socket.send_text(text)
            if (extra := lan_candidate(text, lan)) is not None:
                await view.socket.send_text(extra)
    except ConnectionClosed:
        pass
    # 别的窗口换了传输时流服务以 4000 关掉信令，改报顶替（4001）；其余应用关闭码原样交给面板。
    code = upstream.close_code
    if view.switched:
        await view.socket.close(code=4001, reason="superseded by another window")
    elif code is not None and 4000 <= code < 5000:
        await view.socket.close(code=code, reason=upstream.close_reason or "")
