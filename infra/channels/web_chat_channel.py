from __future__ import annotations

import asyncio
import logging
import mimetypes
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from infra.channels.base import AttachmentStore
from infra.channels.contract import ChannelContext

logger = logging.getLogger(__name__)


class WebChatChannel:
    def __init__(self, channel_name: str = "web") -> None:
        self.name = channel_name
        self._ctx: ChannelContext | None = None
        self._attachments = AttachmentStore()
        self._connections: dict[str, set[WebSocket]] = {}
        self._active_turn_ids: dict[str, str] = {}
        self._media_paths: set[str] = set()
        self._connection_lock = asyncio.Lock()
        self._events_bound = False
        self._outbound_bound = False

    async def start(self, ctx: ChannelContext) -> None:
        self._ctx = ctx
        self._attachments = ctx.attachment_store
        if not self._outbound_bound:
            ctx.bus.subscribe_outbound(self.name, self._on_response)
            self._outbound_bound = True
        if not self._events_bound:
            ctx.event_bus.on(TurnStarted, self._on_turn_started)
            ctx.event_bus.on(StreamDeltaReady, self._on_stream_delta)
            ctx.event_bus.on(ToolCallStarted, self._on_tool_call_started)
            ctx.event_bus.on(ToolCallCompleted, self._on_tool_call_completed)
            self._events_bound = True
        ctx.push_tool.register_channel(
            self.name,
            text=self.send,
            stream_text=self.send_stream,
            file=self.send_file,
            image=self.send_image,
        )

    async def stop(self) -> None:
        async with self._connection_lock:
            sockets = [
                socket
                for sockets in self._connections.values()
                for socket in sockets
            ]
            self._connections.clear()
        for socket in sockets:
            if socket.application_state == WebSocketState.CONNECTED:
                await socket.close()

    async def handle_websocket(self, websocket: WebSocket) -> None:
        await websocket.accept()
        session_keys: set[str] = set()
        try:
            while True:
                payload = await websocket.receive_json()
                if not isinstance(payload, dict):
                    await self._send_error(websocket, "", "消息格式必须是 JSON object")
                    continue
                session_key = await self._handle_client_frame(
                    websocket,
                    cast(dict[str, Any], payload),
                )
                if session_key:
                    session_keys.add(session_key)
        except WebSocketDisconnect:
            pass
        finally:
            await self._remove_connection(websocket, session_keys)

    def save_upload(self, data: bytes, filename: str) -> dict[str, str]:
        suffix = Path(filename).suffix
        if not suffix:
            guessed = mimetypes.guess_extension(mimetypes.guess_type(filename)[0] or "")
            suffix = guessed or ".bin"
        path = self._attachments.write_bytes(data, prefix="web_", suffix=suffix)
        return {
            "filename": filename,
            "upload_path": str(path),
            "upload_url": f"/api/chat/media?path={str(path)}",
        }

    def upload_roots(self) -> list[Path]:
        return [
            self._attachments.root,
            Path("/tmp") / "akashic_uploads",
        ]

    def remember_media(self, paths: list[str]) -> None:
        for path in paths:
            text = str(path or "").strip()
            if not text or text.startswith(("http://", "https://")):
                continue
            try:
                self._media_paths.add(str(Path(text).expanduser().resolve()))
            except OSError:
                continue

    def has_media(self, path: Path) -> bool:
        return str(path.resolve()) in self._media_paths

    async def send(self, chat_id: str, message: str) -> None:
        session_key = self._session_key(chat_id)
        await self._broadcast(session_key, {
            "type": "message.final",
            "session_id": session_key,
            "turn_id": "",
            "content": message,
            "media": [],
            "metadata": {"source": "message_push"},
        })

    async def send_stream(self, chat_id: str, message: str) -> None:
        await self.send(chat_id, message)

    async def send_file(
        self,
        chat_id: str,
        file_path: str,
        name: str | None = None,
    ) -> None:
        session_key = self._session_key(chat_id)
        content = name or Path(file_path).name
        self.remember_media([file_path])
        await self._broadcast(session_key, {
            "type": "message.final",
            "session_id": session_key,
            "turn_id": "",
            "content": content,
            "media": [file_path],
            "metadata": {"source": "message_push", "kind": "file"},
        })

    async def send_image(self, chat_id: str, image: str) -> None:
        session_key = self._session_key(chat_id)
        self.remember_media([image])
        await self._broadcast(session_key, {
            "type": "message.final",
            "session_id": session_key,
            "turn_id": "",
            "content": "",
            "media": [image],
            "metadata": {"source": "message_push", "kind": "image"},
        })

    async def _handle_client_frame(
        self,
        websocket: WebSocket,
        payload: dict[str, Any],
    ) -> str:
        frame_type = str(payload.get("type") or "")
        request_id = str(payload.get("request_id") or "")
        if frame_type == "session.create":
            return await self._create_session(websocket, request_id)
        if frame_type == "message.send":
            return await self._send_user_message(websocket, request_id, payload)
        if frame_type == "turn.stop":
            return await self._stop_turn(websocket, request_id, payload)
        if frame_type == "ping":
            await websocket.send_json({"type": "pong", "request_id": request_id})
            return ""
        await self._send_error(websocket, request_id, f"未知消息类型: {frame_type}")
        return ""

    async def _create_session(self, websocket: WebSocket, request_id: str) -> str:
        chat_id = uuid4().hex
        session_key = self._session_key(chat_id)
        await self._add_connection(session_key, websocket)
        await websocket.send_json({
            "type": "session.created",
            "request_id": request_id,
            "session_id": session_key,
        })
        return session_key

    async def _send_user_message(
        self,
        websocket: WebSocket,
        request_id: str,
        payload: dict[str, Any],
    ) -> str:
        ctx = self._require_ctx()
        session_key = self._normalize_session_id(payload.get("session_id"))
        if not session_key:
            session_key = self._session_key(uuid4().hex)
        text = str(payload.get("text") or "")
        media = [
            str(item)
            for item in payload.get("media", [])
            if isinstance(item, str) and item.strip()
        ]
        if not text.strip() and not media:
            await self._send_error(websocket, request_id, "text 和 media 不能同时为空")
            return session_key
        await self._add_connection(session_key, websocket)
        chat_id = self._chat_id(session_key)
        await ctx.bus.publish_inbound(
            InboundMessage(
                channel=self.name,
                sender="web",
                chat_id=chat_id,
                content=text,
                media=media,
                metadata={"client_request_id": request_id},
            )
        )
        return session_key

    async def _stop_turn(
        self,
        websocket: WebSocket,
        request_id: str,
        payload: dict[str, Any],
    ) -> str:
        ctx = self._require_ctx()
        session_key = self._normalize_session_id(payload.get("session_id"))
        if not session_key:
            await self._send_error(websocket, request_id, "session_id 缺失或无效")
            return ""
        if ctx.interrupt_controller is None:
            await self._send_error(websocket, request_id, "当前未启用中断功能")
            return session_key
        result = ctx.interrupt_controller.request_interrupt(
            session_key=session_key,
            sender="web",
            command="/stop",
        )
        await websocket.send_json({
            "type": "turn.interrupted",
            "request_id": request_id,
            "session_id": session_key,
            "message": result.message,
            "status": result.status,
        })
        return session_key

    async def _on_turn_started(self, event: TurnStarted) -> None:
        if event.channel != self.name:
            return
        turn_id = self._turn_id(event.session_key, event.timestamp.timestamp())
        self._active_turn_ids[event.session_key] = turn_id
        await self._broadcast(event.session_key, {
            "type": "turn.started",
            "session_id": event.session_key,
            "turn_id": turn_id,
            "content": event.content,
        })

    async def _on_stream_delta(self, event: StreamDeltaReady) -> None:
        if event.channel != self.name:
            return
        if event.thinking_delta:
            await self._broadcast(event.session_key, {
                "type": "react.thinking.delta",
                "session_id": event.session_key,
                "turn_id": self._current_turn_id(event.session_key),
                "delta": event.thinking_delta,
            })
        if event.content_delta:
            await self._broadcast(event.session_key, {
                "type": "answer.delta",
                "session_id": event.session_key,
                "turn_id": self._current_turn_id(event.session_key),
                "delta": event.content_delta,
            })

    async def _on_tool_call_started(self, event: ToolCallStarted) -> None:
        if event.channel != self.name:
            return
        await self._broadcast(event.session_key, {
            "type": "react.tool.started",
            "session_id": event.session_key,
            "turn_id": self._current_turn_id(event.session_key),
            "call_id": event.call_id,
            "tool_name": event.tool_name,
            "arguments": event.arguments,
        })

    async def _on_tool_call_completed(self, event: ToolCallCompleted) -> None:
        if event.channel != self.name:
            return
        await self._broadcast(event.session_key, {
            "type": "react.tool.completed",
            "session_id": event.session_key,
            "turn_id": self._current_turn_id(event.session_key),
            "call_id": event.call_id,
            "tool_name": event.tool_name,
            "status": event.status,
            "result_preview": event.result_preview,
        })

    async def _on_response(self, msg: OutboundMessage) -> None:
        session_key = self._session_key(msg.chat_id)
        media = list(msg.media or [])
        metadata = dict(msg.metadata or {})
        self.remember_media(media)
        await self._broadcast(session_key, {
            "type": "message.final",
            "session_id": session_key,
            "turn_id": self._current_turn_id(session_key),
            "content": msg.content,
            "thinking": msg.thinking or "",
            "media": media,
            "duration_ms": metadata.get("turn_duration_ms"),
            "metadata": metadata,
        })
        _ = self._active_turn_ids.pop(session_key, None)

    async def _add_connection(self, session_key: str, websocket: WebSocket) -> None:
        async with self._connection_lock:
            self._connections.setdefault(session_key, set()).add(websocket)

    async def _remove_connection(
        self,
        websocket: WebSocket,
        session_keys: set[str],
    ) -> None:
        async with self._connection_lock:
            for session_key in session_keys:
                sockets = self._connections.get(session_key)
                if sockets is None:
                    continue
                sockets.discard(websocket)
                if not sockets:
                    _ = self._connections.pop(session_key, None)

    async def _broadcast(self, session_key: str, frame: dict[str, Any]) -> None:
        async with self._connection_lock:
            sockets = list(self._connections.get(session_key, set()))
        if not sockets:
            return
        stale: list[WebSocket] = []
        for socket in sockets:
            if socket.application_state != WebSocketState.CONNECTED:
                stale.append(socket)
                continue
            try:
                await socket.send_json(frame)
            except Exception as e:
                logger.warning("[web_chat] 发送 WebSocket frame 失败: %s", e)
                stale.append(socket)
        if stale:
            async with self._connection_lock:
                current = self._connections.get(session_key)
                if current is not None:
                    for socket in stale:
                        current.discard(socket)

    async def _send_error(
        self,
        websocket: WebSocket,
        request_id: str,
        message: str,
    ) -> None:
        await websocket.send_json({
            "type": "error",
            "request_id": request_id,
            "message": message,
        })

    def _normalize_session_id(self, value: object) -> str:
        text = str(value or "").strip()
        if not text.startswith(f"{self.name}:"):
            return ""
        chat_id = text[len(self.name) + 1:]
        if not chat_id:
            return ""
        return text

    def _session_key(self, chat_id: str) -> str:
        text = str(chat_id).strip()
        if text.startswith(f"{self.name}:"):
            return text
        return f"{self.name}:{text}"

    def _chat_id(self, session_key: str) -> str:
        return session_key[len(self.name) + 1:]

    def _turn_id(self, session_key: str, seed: float) -> str:
        return f"{session_key}:{seed:.6f}"

    def _current_turn_id(self, session_key: str) -> str:
        return self._active_turn_ids.get(session_key, session_key)

    def _require_ctx(self) -> ChannelContext:
        if self._ctx is None:
            raise RuntimeError("WebChatChannel 尚未启动")
        return self._ctx
