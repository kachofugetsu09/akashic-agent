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
    client = httpx.Client(base_url=gateway, timeout=125.0)

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
        if name not in {"selkies-ws-core.js", "util.js", "LICENSE"}:
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

    @app.websocket("/api/dashboard/computer/stream")
    async def computer_display(socket: WebSocket) -> None:
        """把当前 generation 的浏览器画面连接转发到私有显示服务；接管时先把目标标签提到最前。"""

        target = socket.query_params.get("target", "")
        control_id = socket.query_params.get("control", "")
        if control_id and (len(control_id) > 128 or not control_id.isascii()):
            await socket.close(code=1008, reason="Invalid control identity")
            return
        reader = httpx.AsyncClient(base_url=gateway, timeout=40.0)
        async def control_request(action: str) -> None:
            result = await reader.post("/control/" + action, json={"id": control_id, "target": target})
            result.raise_for_status()

        requested = {
            item.strip()
            for item in socket.headers.get("sec-websocket-protocol", "").split(",")
            if item.strip()
        }
        protocols = [Subprotocol("binary")] if "binary" in requested else None
        controlled = False
        # 没接管时，Agent 空闲就可以直接操作；状态来自网关事件流，收回时先松开按住的键和鼠标。
        idle = False
        held_keys: set[str] = set()
        last_pointer: list[str] = []
        touched = 0.0
        renew_wanted = asyncio.Event()

        async def watch_idle(upstream: ClientConnection) -> None:
            nonlocal idle
            timeout = httpx.Timeout(10.0, read=None)
            async with reader.stream("GET", "/events", timeout=timeout) as response:
                _ = response.raise_for_status()
                async for line in response.aiter_lines():
                    if (state := _state_event(line)) is None:
                        continue
                    was_idle, idle = idle, _agent_idle(state)
                    # 先收回输入再松键：发送期间到达的按下不会再被转发。
                    if was_idle and not idle:
                        releasing = tuple(held_keys)
                        held_keys.clear()
                        for key in releasing:
                            await upstream.send("ku," + key)
                        if len(last_pointer) == 4 and last_pointer[3] != "0":
                            await upstream.send(",".join([*last_pointer[:3], "0", "0"]))
                            last_pointer.clear()

        def touch() -> None:
            # 人在直接操作时续期，浏览器不会在用户使用中途休眠；这里只做标记，请求由后台任务发，不拖慢输入转发。
            nonlocal touched
            if time.monotonic() - touched < 30:
                return
            touched = time.monotonic()
            renew_wanted.set()

        async def renew_while_used() -> None:
            # 续期失败和以前一样结束这条画面连接（关闭码 1013）。
            while True:
                _ = await renew_wanted.wait()
                renew_wanted.clear()
                result = await reader.post("/view/touch", json={})
                _ = result.raise_for_status()

        def allowed(text: str) -> bool:
            if control_id or _view_only_allows(text):
                return True
            return idle

        try:
            if control_id:
                await control_request("take")
                controlled = True
            upstream_context = connect(
                # 画面连接总以主控端身份接入，屏幕才会跟随面板尺寸；只读由下面的转发过滤保证。
                stream + "?role=controller",
                subprotocols=protocols,
                compression=None,
                open_timeout=10,
                close_timeout=5,
                max_size=None,
                proxy=None,
            )
            async with upstream_context as upstream:
                await socket.accept(subprotocol=upstream.subprotocol)

                async def send_to_display() -> None:
                    try:
                        while True:
                            message = await socket.receive()
                            if message["type"] == "websocket.disconnect":
                                return
                            if message.get("bytes") is not None:
                                if control_id or idle:
                                    await upstream.send(message["bytes"])
                            elif (text := message.get("text")) is not None and allowed(text):
                                if text.startswith("kd,"):
                                    held_keys.add(text.split(",")[1])
                                elif text.startswith("ku,"):
                                    held_keys.discard(text.split(",")[1])
                                elif text == "kr":
                                    held_keys.clear()
                                if text.startswith("m,"):
                                    last_pointer[:] = text.split(",")[:4]
                                await upstream.send(text)
                                if idle and not control_id and not _view_only_allows(text):
                                    touch()
                    except WebSocketDisconnect:
                        return

                async def send_to_browser() -> None:
                    try:
                        async for message in upstream:
                            if isinstance(message, bytes):
                                await socket.send_bytes(message)
                            else:
                                await socket.send_text(message)
                    except ConnectionClosed:
                        return

                async def renew_control() -> None:
                    while True:
                        await asyncio.sleep(5)
                        await control_request("renew")

                tasks = {
                    asyncio.create_task(send_to_display()),
                    asyncio.create_task(send_to_browser()),
                }
                if control_id:
                    tasks.add(asyncio.create_task(renew_control()))
                else:
                    tasks.add(asyncio.create_task(watch_idle(upstream)))
                    tasks.add(asyncio.create_task(renew_while_used()))
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in pending:
                    with suppress(asyncio.CancelledError):
                        await task
                for task in done:
                    await task
        except WebSocketDisconnect:
            return
        except (OSError, TimeoutError, InvalidHandshake, httpx.HTTPError):
            if socket.client_state is WebSocketState.CONNECTING:
                await socket.accept()
            if socket.client_state is WebSocketState.CONNECTED:
                await socket.close(code=1013, reason="Computer display or control is unavailable")
        finally:
            try:
                if controlled:
                    await control_request("release")
            finally:
                await reader.aclose()

    return client
