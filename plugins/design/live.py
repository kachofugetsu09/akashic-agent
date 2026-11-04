"""设计画布的实时通道：把存储里的新修订和 Agent 正在改哪份文档推给同一会话里打开的网页画布。"""
from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect

from agent.plugin_composition import ServiceKey

MAX_MESSAGE = 64 * 1024


@dataclass(eq=False)
class _Canvas:
    socket: WebSocket
    session: str


class DesignLive:
    """登记每个会话里连着的画布；只往外推通知，画布不在时什么也不发生。"""

    def __init__(self) -> None:
        self._canvases: dict[str, list[_Canvas]] = {}

    # 一条画布连接的整个生命周期：登记，读到断开为止（画布不发消息，读只为及时发现断开）。
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
        except WebSocketDisconnect:
            return
        finally:
            peers = self._canvases.get(session, [])
            if canvas in peers:
                peers.remove(canvas)
            if not peers:
                _ = self._canvases.pop(session, None)

    # 文档在别处保存后通知同一会话里的画布：画布把新修订合并进来。
    async def changed(self, session: str, document: str, revision: int) -> None:
        await self._send(session, {"type": "changed", "document": document, "revision": revision})

    # Agent 开始改某份文档：画布据此展开标签（每轮一次）并切到那份文档；view 为 preview 时切到网页预览。
    async def agent(self, session: str, turn: str, document: str, view: str = "canvas") -> None:
        await self._send(session, {"type": "agent", "turn": turn, "document": document, "view": view})

    async def _send(self, session: str, message: dict[str, object]) -> None:
        text = json.dumps(message, ensure_ascii=False)
        for canvas in list(self._canvases.get(session, [])):
            with suppress(RuntimeError, WebSocketDisconnect):
                await canvas.socket.send_text(text)


DESIGN_LIVE = ServiceKey[DesignLive]("design.live.v1")
