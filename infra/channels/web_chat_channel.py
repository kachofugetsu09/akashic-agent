from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import AsyncIterable, Mapping
from typing import Any, cast
from uuid import uuid4

from agent.plugin_composition.channels import (
    AttachmentKind as V3AttachmentKind,
    AttachmentReadLease,
    AttachmentRef,
    ChannelInboundMessage,
    ChannelAttachmentReadPort,
    ChannelCommitRole,
    ChannelFactoryContext,
    ChannelRuntimePorts,
    ChannelReady,
    DeliveryStatus as V3DeliveryStatus,
    InboundIdentity,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    RawInbound,
    StopReceipt,
)
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from bus.events import (
    ChannelMessage,
    DeliveryReceipt,
    DeliveryStatus,
)
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnOutputCompleted,
    TurnStarted,
)
from infra.channels.base import AttachmentStore
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from infra.channels.contract import ChannelContext
from infra.channels.reply_context import build_reply_inbound_text

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_WEB_INBOUND_ID_LENGTH = 256


class UploadTooLargeError(ValueError):
    """上传内容超过单文件上限。"""


class WebNativeChannelAdapter:
    """把一个已启动的 Web provider owner 暴露为 v3 native adapter。"""

    def __init__(
        self,
        channel: WebChatChannel,
        context: ChannelFactoryContext,
    ) -> None:
        self._channel = channel
        self._binding_token = context.binding_token
        self._attachment_read = context.attachment_read
        self._started = False
        self._stopped = False
        self._runtime: ChannelRuntimePorts | None = None
        self._admission_open = False
        self._in_flight = 0
        self._drain_event = asyncio.Event()
        self._drain_event.set()
        self._stop_receipt: StopReceipt | None = None
        channel._register_v3_adapter(self)

    @property
    def binding_token(self) -> str:
        return self._binding_token

    @property
    def admission_open(self) -> bool:
        return self._admission_open

    async def start(self) -> ChannelReady:
        """返回 binding readiness，不重复启动 Web provider owner。"""

        if self._stopped:
            raise RuntimeError("Web native channel binding 已停止")
        if self._started:
            raise RuntimeError("Web native channel binding 重复 start")
        self._started = True
        return ChannelReady(self._binding_token)

    def attach_runtime(self, ports: ChannelRuntimePorts) -> None:
        """把 provider callback 固定到一个不可替换的 exact Core binding。"""

        if not isinstance(ports, ChannelRuntimePorts):
            raise TypeError("Web runtime ports 类型无效")
        if ports.binding_token != self._binding_token:
            raise RuntimeError("Web runtime ports binding token 不匹配")
        if ports.ingress is None:
            raise RuntimeError("Web v3 ingress 缺少 Core ingress")
        if self._stopped:
            raise RuntimeError("Web native channel binding 已停止")
        if self._runtime is not None:
            raise RuntimeError("Web v3 ingress runtime 不允许替换")
        self._runtime = ports

    def open_admission(self) -> None:
        """在 stable publication 完成后打开当前 exact Web ingress。"""

        if not self._started:
            raise RuntimeError("Web native channel 尚未 start")
        if self._runtime is None:
            raise RuntimeError("Web v3 ingress 尚未 attach")
        if self._admission_open:
            raise RuntimeError("Web v3 ingress 已打开")
        self._channel._open_v3_binding(self)
        self._admission_open = True

    def close_admission(self) -> None:
        """同步拒绝新 Web callback，已开始的 callback 继续使用旧 binding。"""

        if not self._admission_open:
            return
        self._admission_open = False
        self._channel._close_v3_binding(self)

    def begin_inbound(self) -> ChannelRuntimePorts:
        """在 provider callback 的第一个 await 前锁定当前 exact binding。"""

        runtime = self._runtime
        if (
            self._stopped
            or not self._started
            or not self._admission_open
            or runtime is None
            or runtime.ingress is None
        ):
            raise RuntimeError("Web v3 ingress admission 已关闭")
        self._in_flight += 1
        self._drain_event.clear()
        return runtime

    def _finish_inbound(self) -> None:
        if self._in_flight <= 0:
            raise RuntimeError("Web v3 ingress in-flight 计数下溢")
        self._in_flight -= 1
        if self._in_flight == 0:
            self._drain_event.set()

    async def admit_captured(
        self,
        runtime: ChannelRuntimePorts,
        raw: RawInbound,
    ) -> bool:
        """把已获准的 Web callback 投递到其开始时捕获的 Core ingress。"""

        if not isinstance(raw, RawInbound):
            raise TypeError("Web v3 ingress 只接受 RawInbound")
        if runtime is not self._runtime or runtime.ingress is None:
            raise RuntimeError("Web v3 ingress runtime 已失效")
        return await runtime.ingress.admit(raw)

    async def deliver(self, request: ProviderDeliveryRequest) -> ProviderDeliveryReceipt:
        """把 exact v3 request 投影为 Web final frame。"""

        if not self._started:
            raise RuntimeError("Web native channel 尚未 start")
        if request.binding_token != self._binding_token:
            raise RuntimeError("Web native channel binding token 不匹配")
        return await self._channel.deliver_v3(
            request,
            attachment_read=self._attachment_read,
        )

    async def stop(self) -> StopReceipt:
        """停止 adapter binding，但不关闭由 ChannelHost 持有的 Web provider。"""

        if self._stop_receipt is not None:
            return self._stop_receipt
        self.close_admission()
        await self._drain_event.wait()
        self._started = False
        self._stopped = True
        self._runtime = None
        self._channel._unregister_v3_adapter(self)
        self._stop_receipt = StopReceipt(self._binding_token, resources_closed=True)
        return self._stop_receipt


class WebChatChannel:
    v3_inbound_identity = InboundIdentity.PROVIDER_MESSAGE_ID

    def __init__(self, channel_name: str = "akashic") -> None:
        self.name = channel_name
        self._ctx: ChannelContext | None = None
        self._attachments: AttachmentStore | None = None
        self._artifact_store: ChannelAttachmentArtifactStore | None = None
        self._connections: dict[str, set[WebSocket]] = {}
        self._pending_terminal: dict[str, dict[str, Any]] = {}
        self._media_paths: set[str] = set()
        self._connection_lock = asyncio.Lock()
        self._events_bound = False
        self._v3_adapters: dict[str, WebNativeChannelAdapter] = {}

    @staticmethod
    def _socket_id(websocket: WebSocket) -> str:
        return f"ws-{id(websocket):x}"

    def _connection_count(self, session_key: str) -> int:
        return len(self._connections.get(session_key, set()))

    async def start(self, ctx: ChannelContext) -> None:
        self._ctx = ctx
        self._attachments = ctx.attachment_store
        if not self._events_bound:
            ctx.event_bus.on(TurnStarted, self._on_turn_started)
            ctx.event_bus.on(StreamDeltaReady, self._on_stream_delta)
            ctx.event_bus.on(ToolCallStarted, self._on_tool_call_started)
            ctx.event_bus.on(ToolCallCompleted, self._on_tool_call_completed)
            ctx.event_bus.on(TurnOutputCompleted, self._on_output_completed)
            self._events_bound = True

    def bind_attachment_store(self, store: AttachmentStore) -> None:
        """在 channel 启动前为独立 Chat API 绑定显式附件目录。"""

        if self._attachments is None:
            self._attachments = store

    def bind_artifact_store(self, store: ChannelAttachmentArtifactStore) -> None:
        """绑定 Core-owned artifact store，供 Web upload/read API 使用。"""

        if not isinstance(store, ChannelAttachmentArtifactStore):
            raise TypeError("Web artifact store 类型无效")
        if self._artifact_store is not None and self._artifact_store is not store:
            raise RuntimeError("Web artifact store 不允许在运行中替换")
        self._artifact_store = store

    @property
    def artifact_store(self) -> ChannelAttachmentArtifactStore | None:
        """Return the currently bound Core artifact owner, if the API is ready."""

        return self._artifact_store

    def build_v3_adapter(self, context: ChannelFactoryContext) -> WebNativeChannelAdapter:
        """为一个 exact Core binding 创建不接管 provider 生命周期的 native adapter。"""

        if not isinstance(context, ChannelFactoryContext):
            raise TypeError("Web native adapter context 类型无效")
        return WebNativeChannelAdapter(self, context)

    def _register_v3_adapter(self, adapter: WebNativeChannelAdapter) -> None:
        current = self._v3_adapters.get(adapter.binding_token)
        if current is not None and current is not adapter:
            raise RuntimeError("Web v3 binding token 已注册")
        self._v3_adapters[adapter.binding_token] = adapter

    def _unregister_v3_adapter(self, adapter: WebNativeChannelAdapter) -> None:
        if self._v3_adapters.get(adapter.binding_token) is adapter:
            self._v3_adapters.pop(adapter.binding_token, None)

    def _open_v3_binding(self, adapter: WebNativeChannelAdapter) -> None:
        if self._v3_adapters.get(adapter.binding_token) is not adapter:
            raise RuntimeError("Web v3 binding 未注册")
        if any(
            current is not adapter and current.admission_open
            for current in self._v3_adapters.values()
        ):
            raise RuntimeError("Web v3 不允许同时打开多个 binding")

    def _close_v3_binding(self, adapter: WebNativeChannelAdapter) -> None:
        if self._v3_adapters.get(adapter.binding_token) is not adapter:
            raise RuntimeError("Web v3 binding 未注册")

    def _begin_v3_inbound(self) -> tuple[WebNativeChannelAdapter, ChannelRuntimePorts]:
        """在处理 Web frame 前捕获唯一打开的 exact Core binding。"""

        adapters = tuple(self._v3_adapters.values())
        for adapter in adapters:
            if adapter.admission_open:
                return adapter, adapter.begin_inbound()
        raise RuntimeError("Web v3 ingress admission 已关闭")

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
        socket_id = self._socket_id(websocket)
        logger.info("[web_chat] websocket opened id=%s", socket_id)
        await websocket.accept()
        try:
            while True:
                payload = await websocket.receive_json()
                if not isinstance(payload, dict):
                    await self._send_error(websocket, "", "消息格式必须是 JSON object")
                    continue
                await self._handle_client_frame(
                    websocket,
                    cast(dict[str, Any], payload),
                )
        except WebSocketDisconnect as error:
            logger.info(
                "[web_chat] websocket disconnect id=%s code=%s reason=%s",
                socket_id,
                error.code,
                error.reason,
            )
        finally:
            await self._remove_connection(websocket)
            logger.info("[web_chat] websocket closed id=%s", socket_id)

    def save_upload(self, data: bytes, filename: str) -> dict[str, Any]:
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
    ) -> dict[str, object]:
        """在分配正式附件前有界读取，并以 fsync + replace 原子发布。"""

        return await self._save_upload_stream(chunks, filename, max_bytes=max_bytes)

    async def _save_upload_stream(
        self,
        chunks: AsyncIterable[bytes],
        filename: str,
        *,
        max_bytes: int,
    ) -> dict[str, object]:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        suffix = Path(filename).suffix
        if not suffix:
            guessed = mimetypes.guess_extension(mimetypes.guess_type(filename)[0] or "")
            suffix = guessed or ".bin"
        if self._artifact_store is not None:
            data = bytearray()
            total = 0
            async for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError("上传 chunk 必须是 bytes")
                total += len(chunk)
                if total > max_bytes:
                    raise UploadTooLargeError(
                        f"上传内容超过 {max_bytes // (1024 * 1024)}MB 限制"
                    )
                data.extend(chunk)
            if total == 0:
                raise ValueError("上传内容不能为空")
            media_type = mimetypes.guess_type(filename)[0]
            kind = (
                V3AttachmentKind.IMAGE
                if media_type is not None and media_type.startswith("image/")
                else V3AttachmentKind.FILE
            )
            ref = await self._artifact_store.import_bytes(
                bytes(data),
                kind=kind,
                filename=filename,
                media_type=media_type,
            )
            return self._artifact_result(ref)

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
    def _upload_result(filename: str, path: Path) -> dict[str, Any]:
        return {
            "filename": filename,
            "upload_path": str(path),
            "upload_url": f"/api/chat/media?path={str(path)}",
        }

    @staticmethod
    def _artifact_result(ref: AttachmentRef) -> dict[str, Any]:
        """返回只含 opaque artifact identity 的 Web upload projection。"""

        return {
            "filename": ref.filename,
            "artifact_id": ref.artifact_id,
            "kind": ref.kind.value,
            "media_type": ref.media_type,
            "size_bytes": ref.size_bytes,
            "sha256": ref.sha256,
            "upload_url": f"/api/chat/artifacts/{ref.artifact_id}",
        }

    @staticmethod
    def artifact_descriptor(ref: AttachmentRef) -> dict[str, Any]:
        """把 Core-owned ref 投影为不含本地路径的 Web descriptor。"""

        return {
            "artifact_id": ref.artifact_id,
            "kind": ref.kind.value,
            "filename": ref.filename,
            "media_type": ref.media_type,
            "size_bytes": ref.size_bytes,
            "sha256": ref.sha256,
            "url": f"/api/chat/artifacts/{ref.artifact_id}",
        }

    async def read_artifact(self, artifact_id: str) -> tuple[bytes, str | None, str | None]:
        """通过 Core read lease 读取一个有界、不可变 artifact。"""

        store = self._artifact_store
        if store is None:
            raise RuntimeError("Web artifact store 尚未绑定")
        refs = store.resolve_refs((artifact_id,))
        if len(refs) != 1:
            raise RuntimeError("Web artifact resolve 返回数量无效")
        ref = refs[0]
        lease = await store.acquire(ref)
        try:
            data = await lease.read_bytes(max_bytes=MAX_UPLOAD_BYTES)
        finally:
            await lease.aclose()
        return data, ref.media_type, ref.filename

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
        if passive and (
            not message.execution_attempt_id or not message.control_turn_id
        ):
            raise RuntimeError("Web passive final 缺少 Turn/Attempt 身份")
        frame: dict[str, Any] = {
            "type": "message.final",
            "session_id": session_key,
            "turn_id": message.execution_attempt_id or "",
            "content": message.content,
            "thinking": message.thinking or "",
            "media": media,
            "metadata": metadata,
        }
        if passive:
            frame["control_turn_id"] = message.control_turn_id
            frame["execution_attempt_id"] = message.execution_attempt_id
        duration = metadata.get("turn_duration_ms")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            frame["duration_ms"] = duration
        delivered = await self._broadcast(session_key, frame)
        if delivered > 0:
            self._pending_terminal.pop(session_key, None)
            return DeliveryReceipt(
                DeliveryStatus.SUCCESS,
                canonical_media=tuple(media),
            )
        if message.control_turn_id:
            self._pending_terminal[session_key] = frame
            return DeliveryReceipt(
                DeliveryStatus.SUCCESS,
                canonical_media=tuple(media),
            )
        if delivered == 0:
            return DeliveryReceipt(
                DeliveryStatus.FAILED,
                detail="Web 会话没有可用连接",
            )
        raise RuntimeError("Web legacy delivery 状态无效")

    async def deliver_v3(
        self,
        request: ProviderDeliveryRequest,
        *,
        attachment_read: ChannelAttachmentReadPort | None,
    ) -> ProviderDeliveryReceipt:
        """Deliver one exact v3 request without exposing source paths to Web clients."""

        if not isinstance(request, ProviderDeliveryRequest):
            raise TypeError("Web native deliver 只接受 ProviderDeliveryRequest")
        session_key = self._session_key(request.recipient)

        leases: list[AttachmentReadLease] = []
        result: ProviderDeliveryReceipt | None = None
        delivered = 0
        try:
            if request.attachments:
                if attachment_read is None:
                    result = ProviderDeliveryReceipt(
                        request.delivery_id,
                        V3DeliveryStatus.REJECTED,
                        error="Web channel 缺少 attachment read port",
                    )
                else:
                    for ref in request.attachments:
                        lease = await attachment_read.acquire(ref)
                        leases.append(lease)

            if result is None:
                metadata = cast(dict[str, Any], _thaw_json(request.metadata))
                metadata.pop("_channel_commit_role", None)
                if request.commit_role.value != "passive":
                    metadata.setdefault("source", "message_push")
                independent_delivery = (
                    request.commit_role is not ChannelCommitRole.PASSIVE
                    or metadata.get("source") == "message_push"
                )
                if not independent_delivery and (
                    request.execution_attempt_id is None
                    or request.control_turn_id is None
                ):
                    raise RuntimeError("Web passive final 缺少 Turn/Attempt 身份")
                frame: dict[str, Any] = {
                    "type": "message.final",
                    "session_id": session_key,
                    "turn_id": (
                        f"delivery:{request.delivery_id}"
                        if independent_delivery
                        else request.execution_attempt_id
                    ),
                    "content": request.body,
                    "thinking": request.thinking or "",
                    "media": [self.artifact_descriptor(ref) for ref in request.attachments],
                    "metadata": metadata,
                }
                if request.reply_to is not None:
                    frame["reply_to"] = request.reply_to
                if request.session_message_id is not None:
                    frame["session_message_id"] = request.session_message_id
                if not independent_delivery and request.control_turn_id is not None:
                    frame["control_turn_id"] = request.control_turn_id
                if not independent_delivery and request.execution_attempt_id is not None:
                    frame["execution_attempt_id"] = request.execution_attempt_id
                if request.terminal_status is not None:
                    frame["terminal_status"] = request.terminal_status.value
                duration = metadata.get("turn_duration_ms")
                if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                    frame["duration_ms"] = duration
                delivered, failed = await self._broadcast_native(session_key, frame)
                if failed:
                    result = ProviderDeliveryReceipt(
                        request.delivery_id,
                        V3DeliveryStatus.UNKNOWN,
                        error="Web WebSocket frame 发送状态未知",
                    )
                elif delivered == 0:
                    if request.control_turn_id is not None:
                        self._pending_terminal[session_key] = frame
                        result = ProviderDeliveryReceipt(
                            request.delivery_id,
                            V3DeliveryStatus.DELIVERED,
                        )
                    else:
                        result = ProviderDeliveryReceipt(
                            request.delivery_id,
                            V3DeliveryStatus.REJECTED,
                            error="Web 会话没有可用连接",
                        )
                else:
                    self._pending_terminal.pop(session_key, None)
                    result = ProviderDeliveryReceipt(
                        request.delivery_id,
                        V3DeliveryStatus.DELIVERED,
                    )
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            result = ProviderDeliveryReceipt(
                request.delivery_id,
                V3DeliveryStatus.REJECTED,
                error=str(error),
            )
        finally:
            close_error: Exception | None = None
            cancellation: asyncio.CancelledError | None = None
            for lease in reversed(leases):
                try:
                    await lease.aclose()
                except asyncio.CancelledError as error:
                    cancellation = cancellation or error
                except Exception as error:
                    close_error = close_error or error
            if close_error is not None:
                logger.error("[web_chat] attachment read lease 关闭失败: %s", close_error)
                if delivered > 0:
                    result = ProviderDeliveryReceipt(
                        request.delivery_id,
                        V3DeliveryStatus.UNKNOWN,
                        error="Web attachment read lease 关闭状态未知",
                    )
            if cancellation is not None:
                raise cancellation
        if result is None:
            raise RuntimeError("Web native delivery 未产生 receipt")
        return result

    async def _broadcast_native(
        self,
        session_key: str,
        frame: dict[str, Any],
    ) -> tuple[int, int]:
        """Broadcast a native frame and preserve provider uncertainty after send errors."""

        async with self._connection_lock:
            sockets = list(self._connections.get(session_key, set()))
        if not sockets:
            return 0, 0
        stale: list[WebSocket] = []
        delivered = 0
        failed = 0
        for socket in sockets:
            if socket.application_state != WebSocketState.CONNECTED:
                stale.append(socket)
                continue
            try:
                await socket.send_json(frame)
                delivered += 1
            except Exception as error:
                logger.warning("[web_chat] native WebSocket frame 发送失败: %s", error)
                failed += 1
                stale.append(socket)
        if stale:
            async with self._connection_lock:
                current = self._connections.get(session_key)
                if current is not None:
                    for socket in stale:
                        current.discard(socket)
        return delivered, failed

    async def _handle_client_frame(
        self,
        websocket: WebSocket,
        payload: dict[str, Any],
    ) -> str:
        frame_type = str(payload.get("type") or "")
        try:
            request_id = _normalize_web_request_id(payload.get("request_id"))
        except ValueError as error:
            await self._send_error(websocket, "", str(error))
            return ""
        if frame_type == "session.create":
            return await self._create_session(websocket, request_id)
        if frame_type == "session.attach":
            return await self._attach_session(websocket, request_id, payload)
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

    async def _attach_session(
        self,
        websocket: WebSocket,
        request_id: str,
        payload: dict[str, Any],
    ) -> str:
        """把当前 socket 绑定到已知 Web Session，并补投积压终态。"""

        try:
            session_key = self._normalize_session_id(payload.get("session_id"))
        except ValueError as error:
            await self._send_error(websocket, request_id, str(error))
            return ""
        if not session_key:
            await self._send_error(websocket, request_id, "session_id 缺失或无效")
            return ""
        await self._add_connection(session_key, websocket)
        return session_key

    async def _send_user_message(
        self,
        websocket: WebSocket,
        request_id: str,
        payload: dict[str, Any],
    ) -> str:
        return await self._send_user_message_in_binding(
            websocket,
            request_id,
            payload,
        )

    async def _send_user_message_in_binding(
        self,
        websocket: WebSocket,
        request_id: str,
        payload: dict[str, Any],
    ) -> str:
        """验证并构造 Web exact inbound，再在首个 await 前捕获 binding。"""

        ctx = self._require_ctx()
        try:
            session_key = self._normalize_session_id(payload.get("session_id"))
        except ValueError as error:
            await self._send_error(websocket, request_id, str(error))
            return ""
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
        attachments: tuple[AttachmentRef, ...] = ()
        if media and self._artifact_store is not None:
            try:
                refs = self._artifact_store.resolve_refs(tuple(media))
            except ValueError as error:
                await self._send_error(websocket, request_id, "media 附件不存在")
                logger.info("[web_chat] rejected unknown inbound artifact: %s", error)
                return session_key
            if len(refs) != len(media):
                raise RuntimeError("Web artifact resolve 返回数量无效")
            attachments = tuple(refs)
            media = []
        elif media:
            await self._send_error(websocket, request_id, "Web artifact store 尚未绑定")
            return session_key
        if "model_runtime_id" in payload:
            model_runtime_id = payload["model_runtime_id"]
            if not isinstance(model_runtime_id, str):
                await self._send_error(
                    websocket,
                    request_id,
                    "model_runtime_id 必须是字符串",
                )
                return session_key
            metadata["model_runtime_id"] = model_runtime_id.strip()
            model_reasoning_effort = payload.get("model_reasoning_effort", "")
            if not isinstance(model_reasoning_effort, str):
                await self._send_error(
                    websocket,
                    request_id,
                    "model_reasoning_effort 必须是字符串",
                )
                return session_key
            metadata["model_reasoning_effort"] = model_reasoning_effort.strip()
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
        chat_id = self._chat_id(session_key)
        raw = RawInbound(
            message_id=request_id or uuid4().hex,
            provider_identity=chat_id,
            recipient=chat_id,
            message=ChannelInboundMessage(
                channel=self.name,
                sender="web",
                chat_id=chat_id,
                content=_normalize_v3_content(inbound_content),
                timestamp=datetime.now(timezone.utc),
                metadata=cast(Any, metadata),
                attachments=attachments,
            ),
        )
        try:
            adapter, runtime = self._begin_v3_inbound()
        except RuntimeError as error:
            logger.info("[web_chat] rejected inbound message: %s", error)
            await self._send_error(websocket, request_id, str(error))
            return session_key
        connection_added = False
        try:
            connection_added = await self._add_connection(session_key, websocket)
            await adapter.admit_captured(runtime, raw)
        except BaseException as error:
            if connection_added:
                cleanup_task = asyncio.create_task(
                    self._remove_connection_attempt(session_key, websocket),
                    name=f"web-ingress-connection-rollback:{session_key}",
                )
                try:
                    await _settle_cleanup_task(cleanup_task)
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "Web ingress connection 回滚失败",
                        [error, cleanup_error],
                    ) from error
            raise
        finally:
            adapter._finish_inbound()
        return session_key

    async def _stop_turn(
        self,
        websocket: WebSocket,
        request_id: str,
        payload: dict[str, Any],
    ) -> str:
        ctx = self._require_ctx()
        try:
            session_key = self._normalize_session_id(payload.get("session_id"))
        except ValueError as error:
            await self._send_error(websocket, request_id, str(error))
            return ""
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
        if not event.turn_id:
            raise RuntimeError("Web TurnStarted 缺少 Server 权威 turn_id")
        turn_id = event.turn_id
        await self._broadcast(event.session_key, {
            "type": "turn.started",
            "session_id": event.session_key,
            "turn_id": turn_id,
            "control_turn_id": event.control_turn_id or turn_id,
            "client_message_id": event.client_message_id,
            "content": event.content,
        })

    async def _on_stream_delta(self, event: StreamDeltaReady) -> None:
        if event.channel != self.name:
            return
        turn_id = self._event_turn_id(event.turn_id)
        if event.thinking_delta:
            await self._broadcast(event.session_key, {
                "type": "react.thinking.delta",
                "session_id": event.session_key,
                "turn_id": turn_id,
                "delta": event.thinking_delta,
            })
        if event.content_delta:
            await self._broadcast(event.session_key, {
                "type": "answer.delta",
                "session_id": event.session_key,
                "turn_id": turn_id,
                "delta": event.content_delta,
            })

    async def _on_tool_call_started(self, event: ToolCallStarted) -> None:
        if event.channel != self.name:
            return
        await self._broadcast(event.session_key, {
            "type": "react.tool.started",
            "session_id": event.session_key,
            "turn_id": self._event_turn_id(event.turn_id),
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
            "turn_id": self._event_turn_id(event.turn_id),
            "call_id": event.call_id,
            "tool_name": event.tool_name,
            "status": event.status,
            "result_preview": event.result_preview,
        })

    async def _on_output_completed(self, event: TurnOutputCompleted) -> None:
        if event.channel != self.name:
            return
        turn_id = self._event_turn_id(event.turn_id)
        await self._broadcast(event.session_key, {
            "type": "turn.output.completed",
            "session_id": event.session_key,
            "turn_id": turn_id,
            "client_message_id": event.client_message_id,
        })

    async def _add_connection(self, session_key: str, websocket: WebSocket) -> bool:
        """把一个 Web socket 投影到唯一的当前 Session。"""

        async with self._connection_lock:
            added = websocket not in self._connections.get(session_key, set())
            for bound_session_key, bound_sockets in tuple(self._connections.items()):
                if bound_session_key == session_key:
                    continue
                bound_sockets.discard(websocket)
                if not bound_sockets:
                    _ = self._connections.pop(bound_session_key, None)
            sockets = self._connections.setdefault(session_key, set())
            sockets.add(websocket)
        await self._refill_terminal(session_key, websocket)
        return added

    async def _refill_terminal(self, session_key: str, websocket: WebSocket) -> None:
        """新连接绑定后补投一个已完成但尚未送达的终态帧。"""

        frame = self._pending_terminal.get(session_key)
        if frame is None:
            return
        try:
            await websocket.send_json(frame)
        except Exception as error:
            logger.warning(
                "[web_chat] 终态帧补投失败，保留待下次连接 session=%s err=%r",
                session_key,
                error,
            )
            return
        self._pending_terminal.pop(session_key, None)

    async def _remove_connection_attempt(
        self,
        session_key: str,
        websocket: WebSocket,
    ) -> None:
        """只撤销本次失败 message.send 新增的 socket/session 关系。"""

        async with self._connection_lock:
            sockets = self._connections.get(session_key)
            if sockets is None:
                return
            sockets.discard(websocket)
            if not sockets:
                _ = self._connections.pop(session_key, None)

    async def _remove_connection(
        self,
        websocket: WebSocket,
    ) -> None:
        async with self._connection_lock:
            for session_key in tuple(self._connections):
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
        if value is None or value == "":
            return ""
        if not isinstance(value, str):
            raise ValueError("session_id 必须是字符串")
        text = value
        if (
            len(text) > MAX_WEB_INBOUND_ID_LENGTH
            or text != text.strip()
            or any(ord(char) < 32 for char in text)
        ):
            raise ValueError("session_id 格式无效")
        if not text.startswith(f"{self.name}:"):
            raise ValueError("session_id 必须属于当前 Web channel")
        chat_id = text[len(self.name) + 1:]
        if not chat_id:
            raise ValueError("session_id 缺少 chat id")
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

    @staticmethod
    def _event_turn_id(attempt_turn_id: str) -> str:
        if not attempt_turn_id:
            raise RuntimeError("Web lifecycle event 缺少 turn_id")
        return attempt_turn_id

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


def _normalize_v3_content(value: str) -> str:
    """把 Web 文本归一化为 Core inbound 允许的无控制字符正文。"""

    return "".join(
        "\u2028" if ord(char) in {10, 13} else " " if ord(char) < 32 else char
        for char in value
    )


def _thaw_json(value: object) -> object:
    """把 Core 冻结的 JSON 投影为 WebSocket 可序列化对象。"""

    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _normalize_web_request_id(value: object) -> str:
    """验证客户端 request_id，空值由 exact ingress 分配不可伪造的 id。"""

    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        raise ValueError("request_id 必须是字符串")
    if (
        len(value) > MAX_WEB_INBOUND_ID_LENGTH
        or value != value.strip()
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError("request_id 格式无效")
    return value


async def _settle_cleanup_task(task: asyncio.Task[Any]) -> Any:
    """完成 message.send 精确清理后再恢复原始失败。"""

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


def build_web_channel_definition(channel: WebChatChannel) -> Any:
    """Project the already-started Web owner into a Core native channel definition."""

    from agent.plugin_composition.channels import (
        ChannelCapability,
        CoreChannelDefinition,
    )

    if not isinstance(channel, WebChatChannel):
        raise TypeError("Web Core channel definition 只接受 WebChatChannel")

    return CoreChannelDefinition(
        name=channel.name,
        capabilities=frozenset({ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}),
        factory=channel.build_v3_adapter,
        inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
        source_revision="core-web-v3",
        config_revision="core-web-v3",
        generation_id="core-web-v3",
        config={"channel": channel.name},
        factory_export="infra.channels.web_chat_channel.WebChatChannel.build_v3_adapter",
    )


__all__ = [
    "MAX_UPLOAD_BYTES",
    "UploadTooLargeError",
    "WebChatChannel",
    "WebNativeChannelAdapter",
    "build_web_channel_definition",
]
