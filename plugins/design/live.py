"""设计画布的实时通道：Agent 的设计工具调用经 WebSocket 交给用户浏览器里打开的画布执行，结果原路返回。"""
from __future__ import annotations

import asyncio
import json
import secrets
from contextlib import suppress
from dataclasses import dataclass, field
from typing import cast

from fastapi import WebSocket, WebSocketDisconnect

from agent.plugin_composition import ServiceKey

# 第一次调用要在浏览器里加载编辑器（约 12 MB 脚本、WASM 与字体），之后每次调用只是一次画布编辑。
CALL_TIMEOUT = 90.0
MAX_MESSAGE = 4 * 1024 * 1024


class CanvasUnavailable(Exception):
    """会话没有打开的网页画布，或画布在调用途中断开。"""


@dataclass(eq=False)
class _Canvas:
    socket: WebSocket
    session: str
    visible: bool = False
    pending: dict[str, asyncio.Future[dict[str, object]]] = field(default_factory=lambda: {})


class DesignLive:
    """登记每个会话里连着的画布；调用优先交给正在显示的那个，其次最近连上的。"""

    def __init__(self) -> None:
        self._canvases: dict[str, list[_Canvas]] = {}

    # 一条画布连接的整个生命周期：登记、收回执、断开时让还在等的调用立即失败。
    async def serve(self, socket: WebSocket, session: str) -> None:
        await socket.accept()
        canvas = _Canvas(socket, session)
        self._canvases.setdefault(session, []).append(canvas)
        try:
            while True:
                text = await socket.receive_text()
                if len(text) > MAX_MESSAGE:
                    await socket.close(code=1009, reason="design message too large")
                    return
                self._receive(canvas, text)
        except WebSocketDisconnect:
            return
        finally:
            peers = self._canvases.get(session, [])
            if canvas in peers:
                peers.remove(canvas)
            if not peers:
                _ = self._canvases.pop(session, None)
            for future in canvas.pending.values():
                if not future.done():
                    future.set_exception(CanvasUnavailable("设计画布在执行途中关闭了"))

    def _receive(self, canvas: _Canvas, text: str) -> None:
        try:
            parsed: object = json.loads(text)
        except ValueError:
            return
        if not isinstance(parsed, dict):
            return
        message = cast(dict[str, object], parsed)
        call_id = message.get("id")
        if message.get("type") == "visible":
            canvas.visible = message.get("visible") is True
        elif message.get("type") == "result" and isinstance(call_id, str):
            future = canvas.pending.pop(call_id, None)
            if future is not None and not future.done():
                future.set_result(message)

    def connected(self, session: str) -> bool:
        return bool(self._canvases.get(session))

    # 1. 选画布；2. 发出调用并等回执；超时或断开都按画布不可用报给 Agent
    async def call(self, session: str, turn: str, tool: str, arguments: dict[str, object]) -> dict[str, object]:
        peers = self._canvases.get(session, [])
        if not peers:
            raise CanvasUnavailable("这个会话没有打开的网页设计画布")
        canvas = next((peer for peer in reversed(peers) if peer.visible), peers[-1])
        call_id = secrets.token_hex(8)
        future = asyncio.get_running_loop().create_future()
        canvas.pending[call_id] = future
        try:
            await canvas.socket.send_text(json.dumps(
                {"type": "call", "id": call_id, "turn": turn, "tool": tool, "arguments": arguments}, ensure_ascii=False))
            return await asyncio.wait_for(future, CALL_TIMEOUT)
        except TimeoutError as error:
            raise CanvasUnavailable(f"设计画布 {CALL_TIMEOUT:.0f} 秒内没有回应") from error
        except (RuntimeError, WebSocketDisconnect) as error:
            raise CanvasUnavailable("设计画布连接已断开") from error
        finally:
            _ = canvas.pending.pop(call_id, None)

    # 文档在别处保存后通知同一会话里的其他画布：没有未保存修改的画布直接换成新版本。
    async def changed(self, session: str, document: str, revision: int) -> None:
        text = json.dumps({"type": "changed", "document": document, "revision": revision})
        for canvas in list(self._canvases.get(session, [])):
            with suppress(RuntimeError, WebSocketDisconnect):
                await canvas.socket.send_text(text)


DESIGN_LIVE = ServiceKey[DesignLive]("design.live.v1")
