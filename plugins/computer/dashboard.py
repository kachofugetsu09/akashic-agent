from __future__ import annotations

import asyncio
from contextlib import suppress
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

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

    @app.websocket("/api/dashboard/computer/display")
    async def computer_display(socket: WebSocket) -> None:
        """Relay one generation-bound browser session to the private RFB bridge."""

        requested = {
            item.strip()
            for item in socket.headers.get("sec-websocket-protocol", "").split(",")
            if item.strip()
        }
        protocols = ["binary"] if "binary" in requested else None
        try:
            upstream_context = connect(
                display,
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
