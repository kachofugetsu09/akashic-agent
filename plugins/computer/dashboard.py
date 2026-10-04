from __future__ import annotations

import asyncio
from contextlib import suppress
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake
from websockets.typing import Subprotocol

from agent.plugin_composition import DashboardContext


def _websocket_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("Computer display endpoint is not an HTTP URL")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunsplit((scheme, parsed.netloc, "/", "", ""))


def register(app: FastAPI, context: DashboardContext) -> httpx.Client:
    """Expose the exact generation's private Computer endpoint to its web tab."""

    gateway = context.workload_url("computer", "gateway")
    display = _websocket_url(context.workload_url("computer", "display"))
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
        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Computer returned {response.status_code}",
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

    @app.post("/api/dashboard/computer/touch")
    def touch() -> Response:
        result = forward("POST", "/touch", {})
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

    @app.websocket("/api/dashboard/computer/cursor")
    async def computer_cursor(socket: WebSocket) -> None:
        """当前 generation 只读订阅操作位置，不获取输入或唤醒权限。"""
        timeout = httpx.Timeout(10.0, read=None)
        try:
            async with httpx.AsyncClient(timeout=timeout) as reader:
                async with reader.stream("GET", gateway.rstrip("/") + "/cursor") as response:
                    response.raise_for_status()
                    await socket.accept()

                    async def receive_cursor() -> None:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                await socket.send_text(line[6:])

                    async def receive_browser() -> None:
                        message = await socket.receive()
                        if message["type"] != "websocket.disconnect":
                            await socket.close(code=1008, reason="Cursor is read-only")

                    tasks = {asyncio.create_task(receive_cursor()), asyncio.create_task(receive_browser())}
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
                await socket.close(code=1013, reason="Computer cursor is unavailable")

    @app.websocket("/api/dashboard/computer/stream")
    @app.websocket("/api/dashboard/computer/display")
    async def computer_display(socket: WebSocket) -> None:
        """把当前 generation 的浏览器连接转发到私有显示服务。"""

        requested = {
            item.strip()
            for item in socket.headers.get("sec-websocket-protocol", "").split(",")
            if item.strip()
        }
        protocols = [Subprotocol("binary")] if "binary" in requested else None
        try:
            upstream_context = connect(
                stream if socket.url.path.endswith("/stream") else display,
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
                                await upstream.send(message["bytes"])
                            elif message.get("text") is not None:
                                await upstream.send(message["text"])
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

                tasks = {
                    asyncio.create_task(send_to_display()),
                    asyncio.create_task(send_to_browser()),
                }
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
        except (OSError, TimeoutError, InvalidHandshake):
            if socket.client_state is WebSocketState.CONNECTING:
                await socket.accept()
            if socket.client_state is WebSocketState.CONNECTED:
                await socket.close(code=1013, reason="Computer display is unavailable")

    return client
