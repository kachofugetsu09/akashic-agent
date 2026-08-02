from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
from pathlib import Path
from collections.abc import AsyncIterable
from typing import Any, cast
from uuid import uuid4

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from bus.events import (
    ChannelMessage,
    DeliveryReceipt,
    DeliveryStatus,
    InboundMessage,
    OutboundMessage,
    channel_message_from_outbound,
)
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from infra.channels.base import AttachmentStore
from infra.channels.contract import ChannelContext
from infra.channels.reply_context import build_reply_inbound_text

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class UploadTooLargeError(ValueError):
    """上传内容超过单文件上限。"""


class WebChatChannel:
    def __init__(self, channel_name: str = "web") -> None:
        self.name = channel_name
        self._ctx: ChannelContext | None = None
        self._attachments: AttachmentStore | None = None
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
            deliver=self._deliver_message,
        )

    def bind_attachment_store(self, store: AttachmentStore) -> None:
        """在 channel 启动前为独立 Chat API 绑定显式附件目录。"""

        if self._attachments is None:
            self._attachments = store

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
        if len(data) > MAX_UPLOAD_BYTES:
            raise UploadTooLargeError("上传内容超过 50MB 限制")
        suffix = Path(filename).suffix
        if not suffix:
            guessed = mimetypes.guess_extension(mimetypes.guess_type(filename)[0] or "")
            suffix = guessed or ".bin"
        path = self._require_attachment_store().write_bytes(data, prefix="web_", suffix=suffix)
        return self._upload_result(filename, path)

    async def save_upload_stream(
        self,
        chunks: AsyncIterable[bytes],
        filename: str,
        *,
        max_bytes: int = MAX_UPLOAD_BYTES,
    ) -> dict[str, str]:
        """在分配正式附件前有界读取，并以 fsync + replace 原子发布。"""

        return await self._save_upload_stream(chunks, filename, max_bytes=max_bytes)

    async def _save_upload_stream(
        self,
        chunks: AsyncIterable[bytes],
        filename: str,
        *,
        max_bytes: int,
    ) -> dict[str, str]:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        suffix = Path(filename).suffix
        if not suffix:
            guessed = mimetypes.guess_extension(mimetypes.guess_type(filename)[0] or "")
            suffix = guessed or ".bin"
        store = self._require_attachment_store()
        staging = store.create_staging_path(prefix=".web_", suffix=".part")
        total = 0
        try:
            fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                async for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise TypeError("上传 chunk 必须是 bytes")
                    total += len(chunk)
                    if total > max_bytes:
                        raise UploadTooLargeError(f"上传内容超过 {max_bytes // (1024 * 1024)}MB 限制")
                    handle.write(chunk)
                if total == 0:
                    raise ValueError("上传内容不能为空")
                handle.flush()
                os.fsync(handle.fileno())
            path = store.publish_staging(staging, prefix="web_", suffix=suffix)
        except BaseException as exc:
            try:
                staging.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise BaseExceptionGroup(
                    "Web 上传 staging 清理失败",
                    [exc, cleanup_error],
                ) from exc
            raise
        return self._upload_result(filename, path)

    @staticmethod
    def _upload_result(filename: str, path: Path) -> dict[str, str]:
        return {
            "filename": filename,
            "upload_path": str(path),
            "upload_url": f"/api/chat/media?path={str(path)}",
        }

    def upload_roots(self) -> list[Path]:
        return [self._require_attachment_store().root]

    def _require_attachment_store(self) -> AttachmentStore:
        if self._attachments is None:
            raise RuntimeError("WebChatChannel 尚未绑定附件目录")
        return self._attachments

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

    async def _deliver_message(self, message: ChannelMessage) -> DeliveryReceipt:
        """把完整渠道消息映射为一个 Web final frame。"""

        session_key = self._session_key(message.chat_id)
        media = [attachment.source for attachment in message.attachments]
        self.remember_media(media)
        metadata = dict(message.metadata)
        passive = metadata.pop("_channel_commit_role", None) == "passive"
        if not passive:
            metadata.setdefault("source", "message_push")
        delivered = await self._broadcast(session_key, {
            "type": "message.final",
            "session_id": session_key,
            "turn_id": message.control_turn_id or self._current_turn_id(session_key),
            "content": message.content,
            "thinking": message.thinking or "",
            "media": media,
            "duration_ms": metadata.get("turn_duration_ms"),
            "metadata": metadata,
        })
        if delivered == 0:
            return DeliveryReceipt(
                DeliveryStatus.FAILED,
                detail="Web 会话没有可用连接",
            )
        return DeliveryReceipt(
            DeliveryStatus.SUCCESS,
            canonical_media=tuple(media),
        )

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
        if "text" not in payload:
            text = ""
        else:
            raw_text = payload["text"]
            if not isinstance(raw_text, str):
                await self._send_error(websocket, request_id, "text 必须是字符串")
                return session_key
            text = raw_text
        raw_media = payload.get("media", [])
        if not isinstance(raw_media, list):
            await self._send_error(websocket, request_id, "media 必须是数组")
            return session_key
        media: list[str] = []
        for item in raw_media:
            if not isinstance(item, str):
                await self._send_error(websocket, request_id, "media 必须是字符串数组")
                return session_key
            if item.strip():
                media.append(item)
        if not text.strip() and not media:
            await self._send_error(websocket, request_id, "text 和 media 不能同时为空")
            return session_key
        reply_to_message_id = payload.get("reply_to_message_id")
        metadata: dict[str, object] = {"client_request_id": request_id}
        inbound_content = text
        if reply_to_message_id is not None:
            if not isinstance(reply_to_message_id, str) or not reply_to_message_id.strip():
                await self._send_error(
                    websocket,
                    request_id,
                    "reply_to_message_id 必须是非空字符串",
                )
                return session_key
            target = ctx.session_manager.control_store.get_message(
                reply_to_message_id.strip()
            )
            if target is None:
                await self._send_error(websocket, request_id, "被引用的消息不存在")
                return session_key
            if target["session_key"] != session_key:
                await self._send_error(websocket, request_id, "不能引用其他会话的消息")
                return session_key
            reply_role = str(target["role"])
            if reply_role not in {"user", "assistant"}:
                raise RuntimeError(
                    f"被引用消息角色无效: {target['id']} {reply_role}"
                )
            reply_content = _reply_source_text(target)
            reply_preview = " ".join(reply_content.split())[:512]
            metadata.update({
                "display_content": text,
                "reply_to_message_id": str(target["id"]),
                "reply_role": reply_role,
                "reply_preview": reply_preview,
            })
            inbound_content = build_reply_inbound_text(
                text,
                reply_content,
                sender_label="你" if reply_role == "user" else "Akashic",
            )
        await self._add_connection(session_key, websocket)
        chat_id = self._chat_id(session_key)
        await ctx.bus.publish_inbound(
            InboundMessage(
                channel=self.name,
                sender="web",
                chat_id=chat_id,
                content=inbound_content,
                media=media,
                metadata=metadata,
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
        outbound = channel_message_from_outbound(msg)
        outbound.metadata["_channel_commit_role"] = "passive"
        receipt = await self._deliver_message(outbound)
        if not receipt.succeeded:
            raise RuntimeError(receipt.detail or "Web 消息提交失败")
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

    async def _broadcast(self, session_key: str, frame: dict[str, Any]) -> int:
        async with self._connection_lock:
            sockets = list(self._connections.get(session_key, set()))
        if not sockets:
            return 0
        stale: list[WebSocket] = []
        delivered = 0
        for socket in sockets:
            if socket.application_state != WebSocketState.CONNECTED:
                stale.append(socket)
                continue
            try:
                await socket.send_json(frame)
                delivered += 1
            except Exception as e:
                logger.warning("[web_chat] 发送 WebSocket frame 失败: %s", e)
                stale.append(socket)
        if stale:
            async with self._connection_lock:
                current = self._connections.get(session_key)
                if current is not None:
                    for socket in stale:
                        current.discard(socket)
        return delivered

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


def _reply_source_text(target: dict[str, Any]) -> str:
    content = str(target["content"])
    if content.strip():
        return content
    media = target.get("media")
    if isinstance(media, list) and media:
        return "[附件]"
    return "[无文字消息]"
