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

    @app.get("/api/dashboard/computer/targets")
    def targets() -> Response:
        result = forward("GET", "/targets")
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

    @app.websocket("/api/dashboard/computer/browser-stream")
    @app.websocket("/api/dashboard/computer/stream")
    async def computer_display(socket: WebSocket) -> None:
        """把当前 generation 的浏览器连接转发到私有显示服务。"""

        browser_view = socket.url.path.endswith("/browser-stream")
        target = socket.query_params.get("target", "") if browser_view else "desktop"
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
        try:
            if control_id:
                await control_request("take")
                controlled = True
            if browser_view:
                await socket.accept()

                async def send_frames() -> None:
                    while True:
                        response = await reader.get("/view/frame", params={"target": target})
                        response.raise_for_status()
                        await socket.send_bytes(response.content)
                        await asyncio.sleep(0.15)

                async def receive_input() -> None:
                    while True:
                        message = await socket.receive_json()
                        if not control_id:
                            await socket.close(code=1008, reason="Browser view is read-only")
                            return
                        response = await reader.post("/view/input", json={
                            "target": target, "owner": control_id, "input": message})
                        response.raise_for_status()

                async def renew_browser_control() -> None:
                    while True:
                        await asyncio.sleep(5)
                        await control_request("renew")

                tasks = {asyncio.create_task(send_frames()), asyncio.create_task(receive_input())}
                if control_id:
                    tasks.add(asyncio.create_task(renew_browser_control()))
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
                return
            upstream_context = connect(
                stream + ("?role=controller" if control_id else "?role=viewer"),
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
