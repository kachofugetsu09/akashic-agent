from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, cast

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from starlette.middleware.gzip import GZipMiddleware

from agent.config_models import MobileRealtimeConfig
from infra.mobile_realtime.auth import (
    AuthenticationChallengeExpired,
    DeviceAuthenticationError,
    DeviceAuthenticator,
    DeviceProofPayload,
    DeviceRevokedError,
    UnknownAuthenticationChallenge,
)
from infra.mobile_realtime.attachments import (
    AttachmentChunk,
    decode_attachment_chunk,
    encode_attachment_chunk,
)
from infra.mobile_realtime.inbox import DurableInboxManager, InboxResetRequired
from infra.mobile_realtime.key_protection import (
    KeyProtectionError,
    KeysetManager,
    LoadedKeyset,
    MasterKeyStore,
    SecretServiceMasterKeyStore,
    create_server_ssl_context,
)
from infra.mobile_realtime.plugin_ui_http import (
    PluginUiHttpTicketError,
    PluginUiHttpTicketIssuer,
    format_plugin_ui_http_expiry,
)
from infra.mobile_realtime.message_content_http import (
    MessageContentTicketError,
    MessageContentTicketIssuer,
    format_message_content_expiry,
)
from infra.mobile_realtime.pairing import (
    PairClaimPayload,
    PairingSecretError,
    PairingService,
    PairingSignatureError,
    PendingPairingClaim,
)
from infra.mobile_realtime.protocol import (
    AckFrame,
    AttachmentBeginCommand,
    AttachmentDownloadCommand,
    AttachmentFinishCommand,
    AuthAcceptedControl,
    AuthAcceptedPayload,
    GenericCommand,
    GenericControl,
    MessageSendCommand,
    MobileFrame,
    ProtocolDecodeError,
    ResumeControl,
    frame_to_json,
    parse_frame,
)
from infra.mobile_realtime.storage import (
    AckOverflowError,
    AckRollbackError,
    CommandConflictError,
    DeviceRecord,
    DurableInboxEvent,
    MobileRealtimeStorage,
    PairingStateError,
    ServerIdentityReference,
    UnknownDeviceError,
    AttachmentRecord,
)

if TYPE_CHECKING:
    from infra.mobile_realtime.channel import MobileRealtimeChannel

_CLOSE_PROTOCOL = 4400
_CLOSE_UNAUTHENTICATED = 4401
_PLUGIN_UI_HTTP_PATH = "/mobile/plugin-ui/v1/query"
_MESSAGE_CONTENT_HTTP_PATH = "/mobile/message-content/v1"
_MAX_MESSAGE_CONTENT_RANGE_BYTES = 256 * 1024
_MAX_PLUGIN_UI_HTTP_REQUEST_BYTES = 72 * 1024
_CLOSE_REVOKED = 4403
_CLOSE_SLOW_CONSUMER = 4408
_CLOSE_COMMAND = 4410
_MAX_PENDING_CONNECTION_EVENTS = 64
_CLOSE_VERSION = 4406
_CONNECTION_CONTROL_SEND_TIMEOUT_SECONDS = 3.0
_CONNECTION_CONTROL_LOCK_TIMEOUT_SECONDS = 30.0
_CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

logger = logging.getLogger(__name__)


def _plugin_ui_task_set() -> set[asyncio.Task[None]]:
    return set()


class MobileGatewayError(RuntimeError):
    pass


class MobilePluginUiHttpError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class MobileMessageContentHttpError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class PairingApproval:
    pairing_id: str
    device_id: str


@dataclass(slots=True)
class ActiveMobileConnection:
    websocket: WebSocket
    connection_epoch: int
    send_lock: asyncio.Lock
    pending_events: deque[DurableInboxEvent]
    ready: bool
    delivery_task: asyncio.Task[None] | None
    plugin_ui_tasks: set[asyncio.Task[None]] = field(default_factory=_plugin_ui_task_set)
    sent_condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    reply_barrier: int | None = None


class PairingApprovalRegistry:
    """跨 WebChat 请求线程与 WebSocket 协程传递配对批准结果。"""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._waiters: dict[str, asyncio.Future[PairingApproval]] = {}
        self._early: dict[str, PairingApproval] = {}

    async def wait(self, pairing_id: str, timeout: float) -> PairingApproval:
        early = self._early.pop(pairing_id, None)
        if early is not None:
            return early
        future = self._loop.create_future()
        if pairing_id in self._waiters:
            raise PairingStateError("同一 pairing 已有等待中的 WebSocket")
        self._waiters[pairing_id] = future
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            _ = self._waiters.pop(pairing_id, None)

    def notify(self, approval: PairingApproval) -> None:
        _ = self._loop.call_soon_threadsafe(self._notify_on_loop, approval)

    def _notify_on_loop(self, approval: PairingApproval) -> None:
        future = self._waiters.get(approval.pairing_id)
        if future is None:
            self._early[approval.pairing_id] = approval
            return
        if not future.done():
            future.set_result(approval)


class MobilePairingAdmin:
    """向本机 WebChat 暴露配对创建、查看和批准操作。"""

    def __init__(
        self,
        pairing: PairingService,
        approvals: PairingApprovalRegistry,
    ) -> None:
        self._pairing = pairing
        self._approvals = approvals

    def create_offer(self) -> dict[str, object]:
        return self._pairing.create_offer().qr_payload()

    def pending_claim(self, pairing_id: str) -> dict[str, object] | None:
        claim = self._pairing.pending_claim(pairing_id)
        if claim is None:
            return None
        return _pending_claim_json(claim)

    def approve(self, pairing_id: str, confirmation_code: str) -> dict[str, object]:
        device = self._pairing.approve(pairing_id, confirmation_code)
        self._approvals.notify(
            PairingApproval(pairing_id=pairing_id, device_id=device.device_id)
        )
        return _device_json(device)


class MobileGatewayRuntime:
    """持有移动网关的身份、配对、认证、ACK 和持久化生命周期。"""

    def __init__(
        self,
        *,
        config: MobileRealtimeConfig,
        storage: MobileRealtimeStorage,
        pairing: PairingService,
        authenticator: DeviceAuthenticator,
        inbox: DurableInboxManager,
        approvals: PairingApprovalRegistry,
        keyset: LoadedKeyset,
    ) -> None:
        self.config = config
        self.storage = storage
        self.pairing = pairing
        self.authenticator = authenticator
        self.inbox = inbox
        self.approvals = approvals
        self.keyset = keyset
        self.admin = MobilePairingAdmin(pairing, approvals)
        self.plugin_ui_http_tickets = PluginUiHttpTicketIssuer(keyset, storage)
        self.message_content_tickets = MessageContentTicketIssuer(keyset, storage)
        self._channel: MobileRealtimeChannel | None = None
        self._connections: dict[str, ActiveMobileConnection] = {}
        self._delivery_lock = asyncio.Lock()

    @property
    def channel(self) -> MobileRealtimeChannel:
        if self._channel is None:
            raise RuntimeError("MobileRealtimeChannel 尚未绑定")
        return self._channel

    def bind_channel(self, channel: MobileRealtimeChannel) -> None:
        if self._channel is not None:
            raise RuntimeError("MobileRealtimeChannel 只能绑定一次")
        self._channel = channel

    async def handle_websocket(self, websocket: WebSocket) -> None:
        """执行 challenge、配对或设备认证，再进入已认证协议循环。"""

        await websocket.accept()
        connection_id = secrets.token_hex(16)
        challenge = self.authenticator.create_challenge(connection_id)
        await _send_control(websocket, "server.challenge", challenge.payload())
        try:
            first = parse_frame(await websocket.receive_text())
            if isinstance(first, GenericControl) and first.type == "pair.claim":
                await self._handle_pair_claim(websocket, first)
                return
            if not (isinstance(first, GenericControl) and first.type == "device.proof"):
                await _close_with_error(
                    websocket,
                    code=_CLOSE_UNAUTHENTICATED,
                    reason="认证前只允许 pair.claim 或 device.proof",
                )
                return
            try:
                proof = DeviceProofPayload.model_validate(first.payload, strict=True)
                device = self.authenticator.authenticate(connection_id, proof)
            except DeviceRevokedError:
                await _close_with_error(
                    websocket,
                    code=_CLOSE_REVOKED,
                    reason="设备已撤销",
                )
                return
            except (
                AuthenticationChallengeExpired,
                DeviceAuthenticationError,
                UnknownAuthenticationChallenge,
                UnknownDeviceError,
                ValidationError,
            ) as error:
                await _close_with_error(
                    websocket,
                    code=_CLOSE_UNAUTHENTICATED,
                    reason=str(error),
                )
                return

            accepted = AuthAcceptedControl(
                v=1,
                kind="control",
                type="auth.accepted",
                connection_epoch=device.connection_epoch,
                payload=AuthAcceptedPayload(
                    connection_epoch=device.connection_epoch,
                    device_id=device.device_id,
                ),
            )
            await websocket.send_text(frame_to_json(accepted))
            await self._authenticated_loop(
                websocket,
                device_id=device.device_id,
                connection_epoch=device.connection_epoch,
            )
        except WebSocketDisconnect:
            return
        except (ProtocolDecodeError, ValidationError) as error:
            if isinstance(error, ValidationError):
                logger.warning(
                    "mobile 协议帧校验失败: %s",
                    error.errors(
                        include_url=False,
                        include_context=False,
                        include_input=False,
                    ),
                )
            await _close_with_error(
                websocket,
                code=_protocol_close_code(error),
                reason=_protocol_error_reason(error),
            )

    async def _handle_pair_claim(
        self,
        websocket: WebSocket,
        frame: GenericControl,
    ) -> None:
        """完成手机 claim，并等待本机 WebChat 显式批准。"""

        try:
            payload = PairClaimPayload.model_validate(frame.payload, strict=True)
            claim = self.pairing.claim(payload)
        except (
            PairingSecretError,
            PairingSignatureError,
            PairingStateError,
            ValidationError,
        ) as error:
            await _close_with_error(
                websocket,
                code=_CLOSE_UNAUTHENTICATED,
                reason=str(error),
            )
            return
        session = self.storage.read_pairing_session(claim.pairing_id)
        if session is None:
            raise PairingStateError("已验签 pairing 会话在数据库中消失")
        await _send_control(
            websocket,
            "pair.pending",
            {
                "pairing_id": claim.pairing_id,
                "confirmation_code": claim.confirmation_code,
                "device_name": claim.device_name,
            },
        )
        timeout = max(0.0, (session.expires_at - _utc_now()).total_seconds())
        try:
            approval = await self.approvals.wait(claim.pairing_id, timeout)
        except TimeoutError:
            await _close_with_error(
                websocket,
                code=_CLOSE_UNAUTHENTICATED,
                reason="配对确认超时",
            )
            return
        await _send_control(
            websocket,
            "pair.accepted",
            {
                "pairing_id": approval.pairing_id,
                "device_id": approval.device_id,
            },
        )
        await websocket.close(code=1000, reason="配对完成，请使用设备密钥重新连接")

    async def _authenticated_loop(
        self,
        websocket: WebSocket,
        *,
        device_id: str,
        connection_epoch: int,
    ) -> None:
        """只处理 epoch 匹配的 resume、ACK 和基础 command。"""

        resumed = False
        try:
            while True:
                incoming = await _receive_authenticated_item(websocket)
                if isinstance(incoming, AttachmentChunk):
                    if not resumed:
                        await _close_with_error(
                            websocket,
                            code=_CLOSE_UNAUTHENTICATED,
                            reason="auth.accepted 后必须先 resume",
                        )
                        return
                    if not await self._is_active_connection(
                        device_id=device_id,
                        websocket=websocket,
                        connection_epoch=connection_epoch,
                    ):
                        return
                    try:
                        await self.channel.handle_attachment_chunk(
                            device_id=device_id,
                            chunk=incoming,
                        )
                    except ValueError as error:
                        await _close_with_error(
                            websocket,
                            code=_CLOSE_PROTOCOL,
                            reason=str(error),
                        )
                        return
                    continue
                frame = incoming
                if not isinstance(
                    frame,
                    (
                        ResumeControl,
                        AckFrame,
                        GenericCommand,
                        MessageSendCommand,
                        AttachmentBeginCommand,
                        AttachmentFinishCommand,
                        AttachmentDownloadCommand,
                    ),
                ):
                    await _close_with_error(
                        websocket,
                        code=_CLOSE_PROTOCOL,
                        reason="客户端发送了方向不允许的帧",
                    )
                    return
                frame_epoch = frame.connection_epoch
                if frame_epoch != connection_epoch:
                    await _close_with_error(
                        websocket,
                        code=_CLOSE_PROTOCOL,
                        reason="connection_epoch 不匹配",
                    )
                    return
                if isinstance(frame, ResumeControl):
                    if resumed:
                        await _close_with_error(
                            websocket,
                            code=_CLOSE_PROTOCOL,
                            reason="同一连接只能 resume 一次",
                        )
                        return
                    await self._resume_and_register(
                        websocket,
                        device_id=device_id,
                        connection_epoch=connection_epoch,
                        last_ack=frame.payload.last_ack,
                    )
                    resumed = True
                    continue
                if not resumed:
                    await _close_with_error(
                        websocket,
                        code=_CLOSE_UNAUTHENTICATED,
                        reason="auth.accepted 后必须先 resume",
                    )
                    return
                if isinstance(frame, AckFrame):
                    try:
                        acknowledged = await self._acknowledge_active_connection(
                            device_id=device_id,
                            websocket=websocket,
                            connection_epoch=connection_epoch,
                            through_event_seq=frame.payload.through_event_seq,
                        )
                    except (AckRollbackError, AckOverflowError) as error:
                        await _close_with_error(
                            websocket,
                            code=_CLOSE_PROTOCOL,
                            reason=str(error),
                        )
                        return
                    if not acknowledged:
                        return
                    continue
                if isinstance(
                    frame,
                    (
                        GenericCommand,
                        MessageSendCommand,
                        AttachmentBeginCommand,
                        AttachmentFinishCommand,
                        AttachmentDownloadCommand,
                    ),
                ):
                    if not await self._is_active_connection(
                        device_id=device_id,
                        websocket=websocket,
                        connection_epoch=connection_epoch,
                    ):
                        return
                    if (
                        isinstance(frame, GenericCommand)
                        and frame.type == "message.content.prepare"
                    ):
                        await self._handle_message_content_prepare(
                            websocket,
                            frame,
                            connection_epoch,
                            device_id,
                        )
                        continue
                    if isinstance(frame, GenericCommand) and _is_plugin_ui_command(frame):
                        if frame.type == "plugin.ui.query":
                            self._start_plugin_ui_command(
                                websocket,
                                frame,
                                connection_epoch,
                                device_id,
                            )
                        else:
                            await self._handle_plugin_ui_command(
                                websocket,
                                frame,
                                connection_epoch,
                                device_id,
                            )
                        continue
                    await self._handle_command(
                        websocket,
                        frame,
                        connection_epoch,
                        device_id,
                    )
                    continue
                raise AssertionError("已验证客户端帧没有对应处理分支")
        finally:
            if resumed:
                connection = self._connections.get(device_id)
                if connection is not None and connection.websocket is websocket:
                    await self._cancel_plugin_ui_connection(device_id, connection)
                await self._remove_connection(
                    device_id=device_id,
                    websocket=websocket,
                )

    async def _cancel_plugin_ui_connection(
        self,
        device_id: str,
        connection: ActiveMobileConnection,
    ) -> None:
        """取消单个连接代际的临时插件查询并释放设备调度状态。"""

        tasks = tuple(connection.plugin_ui_tasks)
        for task in tasks:
            _ = task.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)
        await self.channel.cancel_plugin_ui_device(device_id)

    async def _is_active_connection(
        self,
        *,
        device_id: str,
        websocket: WebSocket,
        connection_epoch: int,
    ) -> bool:
        """确认帧仍属于当前设备的活动连接代际。"""

        async with self._delivery_lock:
            connection = self._connections.get(device_id)
            return (
                connection is not None
                and connection.websocket is websocket
                and connection.connection_epoch == connection_epoch
            )

    async def _acknowledge_active_connection(
        self,
        *,
        device_id: str,
        websocket: WebSocket,
        connection_epoch: int,
        through_event_seq: int,
    ) -> bool:
        """仅允许当前连接代际在同一临界区推进 ACK cursor。"""

        async with self._delivery_lock:
            connection = self._connections.get(device_id)
            if (
                connection is None
                or connection.websocket is not websocket
                or connection.connection_epoch != connection_epoch
            ):
                return False
            _ = self.inbox.acknowledge(
                device_id,
                through_event_seq=through_event_seq,
            )
            return True

    async def _resume_and_register(
        self,
        websocket: WebSocket,
        *,
        device_id: str,
        connection_epoch: int,
        last_ack: int,
    ) -> None:
        """先占住设备投递槽，再在锁外重放并切换为实时投递。"""

        # 1. 在短临界区内冻结重放窗口，并让并发新事件进入新连接队列
        async with self._delivery_lock:
            replay_after, replay_through, terminal = self._prepare_resume(
                device_id=device_id,
                last_ack=last_ack,
            )
            previous = self._connections.get(device_id)
            connection = ActiveMobileConnection(
                websocket=websocket,
                connection_epoch=connection_epoch,
                send_lock=asyncio.Lock(),
                pending_events=deque(),
                ready=False,
                delivery_task=None,
            )
            self._connections[device_id] = connection

        # 2. 旧连接关闭和新连接重放都不占用全局投递锁
        if previous is not None and previous.websocket is not websocket:
            await self._cancel_plugin_ui_connection(device_id, previous)
            async with previous.sent_condition:
                previous.sent_condition.notify_all()
            _ = asyncio.create_task(
                self._close_connection(
                    device_id,
                    previous,
                    code=4001,
                    reason="设备已在新连接上线",
                )
            )
        resume_completed = False
        try:
            await self._send_resume_window(
                device_id=device_id,
                connection=connection,
                replay_after=replay_after,
                replay_through=replay_through,
                terminal=terminal,
            )
            resume_completed = True
        finally:
            if not resume_completed:
                async with self._delivery_lock:
                    if self._connections.get(device_id) is connection:
                        _ = self._connections.pop(device_id)

        # 3. 原子切换 ready；重放期间积累的事件由同一设备任务顺序发送
        async with self._delivery_lock:
            if self._connections.get(device_id) is not connection:
                return
            connection.ready = True
            self._schedule_delivery_locked(device_id, connection)

    def _prepare_resume(
        self,
        *,
        device_id: str,
        last_ack: int,
    ) -> tuple[int, int, DurableInboxEvent]:
        """冻结重放窗口，并返回需要最后发送的尾帧。"""

        # 1. cursor 已落后时只发送 reset，避免伪造可恢复历史
        cursor = self.storage.read_cursor(device_id)
        if last_ack < cursor.acknowledged_event_seq:
            terminal = self._enqueue_event(
                device_id=device_id,
                event_type="sync.reset_required",
                payload={"reason": "client_ack_behind_server_cursor"},
            )
            return last_ack, last_ack, terminal

        # 2. 先冻结当前已分配上限；服务端回退必须早于旧窗口保留期判断
        replay_through = cursor.next_event_seq - 1
        if last_ack > replay_through:
            event_id = _new_ulid()
            terminal = self.inbox.rebase_with_event(
                device_id=device_id,
                through_event_seq=last_ack,
                event_id=event_id,
                envelope_json=_encode_stored_event(
                    event_id=event_id,
                    event_type="sync.reset_required",
                    payload={"reason": "client_ack_ahead_of_server_cursor"},
                ),
            )
            return last_ack, last_ack, terminal

        # 3. 验证保留窗口；回退恢复之外不能重放已经过期的 durable 事件
        try:
            replay = self.inbox.replay(
                device_id,
                after_event_seq=last_ack,
                limit=1,
            )
        except InboxResetRequired:
            terminal = self._enqueue_event(
                device_id=device_id,
                event_type="sync.reset_required",
                payload={"reason": "inbox_retention_exceeded"},
            )
            return last_ack, last_ack, terminal

        # 已落盘 reset 是重建起点；重启后新写入的事件必须在同一冻结窗口续发
        if replay.events and _stored_event_type(replay.events[0]) == "sync.reset_required":
            reset = replay.events[0]
            if not self.storage.durable_event_range_is_contiguous(
                device_id,
                after_event_seq=reset.event_seq,
                through_event_seq=replay_through,
            ):
                terminal = self._enqueue_event(
                    device_id=device_id,
                    event_type="sync.reset_required",
                    payload={"reason": "inbox_sequence_gap_after_reset"},
                )
                return last_ack, last_ack, terminal
            if replay_through == reset.event_seq:
                return last_ack, last_ack, reset
            tail = self.storage.read_durable_events(
                device_id,
                after_event_seq=replay_through - 1,
                limit=1,
            )
            if not tail or tail[0].event_seq != replay_through:
                raise RuntimeError("mobile resume 冻结窗口末尾事件不存在")
            return reset.event_seq - 1, replay_through - 1, tail[0]

        # 4. 持久化窗口已有缺口时必须重建，不能把后续事件伪装成连续重放
        if not self.storage.durable_event_range_is_contiguous(
            device_id,
            after_event_seq=last_ack,
            through_event_seq=replay_through,
        ):
            terminal = self._enqueue_event(
                device_id=device_id,
                event_type="sync.reset_required",
                payload={"reason": "inbox_sequence_gap"},
            )
            return last_ack, last_ack, terminal

        # 5. 终止帧先占号，随后并发 publish 只能排在它之后
        terminal = self._enqueue_event(
            device_id=device_id,
            event_type="sync.completed",
            payload={
                "mode": "replay",
                "replayed_events": max(0, replay_through - last_ack),
            },
        )
        return last_ack, replay_through, terminal

    async def _send_resume_window(
        self,
        *,
        device_id: str,
        connection: ActiveMobileConnection,
        replay_after: int,
        replay_through: int,
        terminal: DurableInboxEvent,
    ) -> None:
        """在单连接写锁内连续发送重放窗口与尾帧。"""

        # 1. 同一连接的重放帧不可被 command 或二进制回复穿插
        async with connection.send_lock:
            after_event_seq = replay_after
            while after_event_seq < replay_through:
                page = self.storage.read_durable_events(
                    device_id,
                    after_event_seq=after_event_seq,
                    limit=512,
                )
                replay = tuple(
                    event for event in page if event.event_seq <= replay_through
                )
                if not replay:
                    raise RuntimeError("mobile resume 冻结窗口出现 event_seq 缺口")
                for event in replay:
                    await _send_stored_event(
                        connection.websocket,
                        event.envelope_json,
                        event.event_seq,
                        connection.connection_epoch,
                    )
                after_event_seq = replay[-1].event_seq
            await _send_stored_event(
                connection.websocket,
                terminal.envelope_json,
                terminal.event_seq,
                connection.connection_epoch,
            )

        # 2. 仅当前 epoch 可以推进 sent cursor
        async with self._delivery_lock:
            if self._connections.get(device_id) is not connection:
                return
            _ = self.inbox.mark_sent(
                device_id,
                through_event_seq=terminal.event_seq,
            )

    async def publish_event(
        self,
        *,
        event_type: str,
        payload: dict[str, object],
        session_id: str | None = None,
        turn_id: str | None = None,
        device_id: str | None = None,
        connection_epoch: int | None = None,
    ) -> None:
        """把 P0 事件写入每个设备 inbox，并向在线连接即时投递。"""

        # 1. 先在协议边界验证事件，拒绝把坏 envelope 写入 SQLite
        event_id = _new_ulid()
        stored = _encode_stored_event(
            event_id=event_id,
            event_type=event_type,
            payload=payload,
            session_id=session_id,
            turn_id=turn_id,
        )

        # 2. 临界区只负责持久化和排队，不等待任何设备网络 I/O
        async with self._delivery_lock:
            if device_id is not None:
                if connection_epoch is not None:
                    connection = self._connections.get(device_id)
                    if connection is None or connection.connection_epoch != connection_epoch:
                        return
                target_device_ids = (device_id,)
            else:
                if connection_epoch is not None:
                    raise ValueError("广播事件不能指定 connection_epoch")
                target_device_ids = tuple(
                    device.device_id for device in self.storage.list_active_devices()
                )
            events = self.inbox.enqueue_many(
                device_ids=target_device_ids,
                event_id=event_id,
                envelope_json=stored,
            )
            for event in events:
                target_device_id = event.device_id
                connection = self._connections.get(target_device_id)
                if connection is None:
                    continue
                if len(connection.pending_events) >= _MAX_PENDING_CONNECTION_EVENTS:
                    _ = self._connections.pop(target_device_id)
                    _ = asyncio.create_task(
                        self._close_connection(
                            target_device_id,
                            connection,
                            code=_CLOSE_SLOW_CONSUMER,
                            reason="设备实时投递队列已满，请重新连接恢复",
                        )
                    )
                    logger.warning(
                        "mobile 设备投递队列已满，转为下次 resume: device=%s",
                        target_device_id,
                    )
                    continue
                connection.pending_events.append(event)
                self._schedule_delivery_locked(target_device_id, connection)

    async def publish_event_with_outbound_attachments(
        self,
        *,
        candidates: tuple[AttachmentRecord, ...],
        payload_builder: Callable[[tuple[AttachmentRecord, ...]], dict[str, object]],
        session_id: str,
    ) -> tuple[AttachmentRecord, ...]:
        """原子提交附件和 proactive durable event，再排队在线投递。"""

        event_id = _new_ulid()
        # 1. delivery lock 内用一个 SQLite 事务提交附件与全部 inbox 行
        async with self._delivery_lock:
            target_device_ids = tuple(
                device.device_id for device in self.storage.list_active_devices()
            )
            resolved, events = self.storage.commit_outbound_event(
                candidates,
                device_ids=target_device_ids,
                event_id=event_id,
                envelope_builder=lambda records: _encode_stored_event(
                    event_id=event_id,
                    event_type="message.proactive",
                    payload=payload_builder(records),
                    session_id=session_id,
                    turn_id=None,
                ),
                created_at=datetime.now(timezone.utc),
            )

            # 2. 数据库提交已是送达事实；在线队列只优化即时可见性
            self._queue_committed_events_locked(events)
        return resolved

    def _queue_committed_events_locked(
        self,
        events: tuple[DurableInboxEvent, ...],
    ) -> None:
        """尽力排队已提交事件；失败时保留 durable resume 恢复路径。"""

        try:
            for event in events:
                connection = self._connections.get(event.device_id)
                if connection is None:
                    continue
                if len(connection.pending_events) >= _MAX_PENDING_CONNECTION_EVENTS:
                    _ = self._connections.pop(event.device_id)
                    _ = asyncio.create_task(
                        self._close_connection(
                            event.device_id,
                            connection,
                            code=_CLOSE_SLOW_CONSUMER,
                            reason="设备实时投递队列已满，请重新连接恢复",
                        )
                    )
                    logger.warning(
                        "mobile 设备投递队列已满，转为下次 resume: device=%s",
                        event.device_id,
                    )
                    continue
                connection.pending_events.append(event)
                self._schedule_delivery_locked(event.device_id, connection)
        except Exception:
            logger.exception(
                "mobile durable event 已提交，在线排队失败；等待设备 resume"
            )

    async def publish_connection_control(
        self,
        *,
        control_type: str,
        payload: dict[str, object],
        device_id: str,
        connection_epoch: int,
    ) -> None:
        """仅向指定的当前连接发送非持久化控制帧。"""

        # 1. 在协议边界构造并验证控制帧
        wire = parse_frame(
            json.dumps(
                {
                    "v": 1,
                    "kind": "control",
                    "type": control_type,
                    "connection_epoch": connection_epoch,
                    "payload": payload,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        )

        # 2. 只捕获订阅时的同一条在线连接，不进入 durable inbox
        async with self._delivery_lock:
            connection = self._connections.get(device_id)
            if (
                connection is None
                or connection.connection_epoch != connection_epoch
                or not connection.ready
            ):
                return
        try:
            async with asyncio.timeout(_CONNECTION_CONTROL_LOCK_TIMEOUT_SECONDS):
                _ = await connection.send_lock.acquire()
        except TimeoutError as error:
            await self._evict_failed_control_connection(
                device_id=device_id,
                connection=connection,
                error=error,
                force_close=True,
            )
            return
        try:
            try:
                async with self._delivery_lock:
                    if self._connections.get(device_id) is not connection:
                        return
                async with asyncio.timeout(_CONNECTION_CONTROL_SEND_TIMEOUT_SECONDS):
                    await connection.websocket.send_text(frame_to_json(wire))
            finally:
                connection.send_lock.release()
        except (WebSocketDisconnect, RuntimeError, OSError, TimeoutError) as error:
            await self._evict_failed_control_connection(
                device_id=device_id,
                connection=connection,
                error=error,
                force_close=False,
            )

    async def _evict_failed_control_connection(
        self,
        *,
        device_id: str,
        connection: ActiveMobileConnection,
        error: BaseException,
        force_close: bool,
    ) -> None:
        """移除控制帧投递失败的当前连接，并触发有界清退。"""

        logger.warning(
            "mobile 连接控制帧投递失败: device=%s error=%s",
            device_id,
            error,
        )
        async with self._delivery_lock:
            if self._connections.get(device_id) is not connection:
                return
            _ = self._connections.pop(device_id)
        _ = asyncio.create_task(
            self._close_connection(
                device_id,
                connection,
                code=_CLOSE_SLOW_CONSUMER,
                reason="连接控制帧投递失败，请重新连接恢复",
                force=force_close,
            )
        )

    def _schedule_delivery_locked(
        self,
        device_id: str,
        connection: ActiveMobileConnection,
    ) -> None:
        """为 ready 连接启动唯一的有序投递任务。"""

        if not connection.ready or not connection.pending_events:
            return
        if connection.delivery_task is not None and not connection.delivery_task.done():
            return
        task = asyncio.create_task(self._drain_connection(device_id, connection))
        connection.delivery_task = task
        task.add_done_callback(
            lambda completed: self._report_delivery_failure(device_id, completed)
        )

    async def _drain_connection(
        self,
        device_id: str,
        connection: ActiveMobileConnection,
    ) -> None:
        """按 event_seq 排空单设备队列，慢设备只阻塞自己的任务。"""

        try:
            while True:
                # 1. 在短临界区领取一个事件
                async with self._delivery_lock:
                    if self._connections.get(device_id) is not connection:
                        return
                    if not connection.pending_events:
                        connection.delivery_task = None
                        return
                    event = connection.pending_events.popleft()

                # 2. WebSocket 写入仅占用本连接写锁
                try:
                    async with connection.send_lock:
                        await _send_stored_event(
                            connection.websocket,
                            event.envelope_json,
                            event.event_seq,
                            connection.connection_epoch,
                        )
                except (WebSocketDisconnect, RuntimeError, OSError) as error:
                    logger.warning(
                        "mobile 在线投递失败，事件保留到下次 resume: device=%s error=%s",
                        device_id,
                        error,
                    )
                    async with self._delivery_lock:
                        if self._connections.get(device_id) is connection:
                            _ = self._connections.pop(device_id)
                    return
                # 3. 连接仍是当前 epoch 时才推进 sent cursor
                async with self._delivery_lock:
                    if self._connections.get(device_id) is connection:
                        _ = self.inbox.mark_sent(
                            device_id,
                            through_event_seq=event.event_seq,
                        )
                        if (
                            connection.reply_barrier is not None
                            and event.event_seq >= connection.reply_barrier
                        ):
                            connection.delivery_task = None
                            return
                async with connection.sent_condition:
                    connection.sent_condition.notify_all()
        finally:
            async with connection.sent_condition:
                connection.sent_condition.notify_all()

    async def _close_connection(
        self,
        device_id: str,
        connection: ActiveMobileConnection,
        *,
        code: int,
        reason: str,
        force: bool = False,
    ) -> None:
        """关闭已退出活动表的 WebSocket，锁失活时强制中断。"""

        try:
            if force:
                await connection.websocket.close(code=code, reason=reason)
                return
            async with connection.send_lock:
                await connection.websocket.close(
                    code=code,
                    reason=reason,
                )
        except (WebSocketDisconnect, RuntimeError, OSError):
            logger.info("mobile WebSocket 已先于服务端关闭: %s", device_id)

    @staticmethod
    def _report_delivery_failure(
        device_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "mobile 设备投递任务异常退出: %s",
                device_id,
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _remove_connection(
        self,
        *,
        device_id: str,
        websocket: WebSocket,
    ) -> None:
        async with self._delivery_lock:
            connection = self._connections.get(device_id)
            if connection is not None and connection.websocket is websocket:
                _ = self._connections.pop(device_id)

    def _enqueue_event(
        self,
        *,
        device_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> DurableInboxEvent:
        event_id = _new_ulid()
        stored = _encode_stored_event(
            event_id=event_id,
            event_type=event_type,
            payload=payload,
        )
        return self.inbox.enqueue(
            device_id=device_id,
            event_id=event_id,
            envelope_json=stored,
        )

    async def _handle_command(
        self,
        websocket: WebSocket,
        frame: (
            GenericCommand
            | MessageSendCommand
            | AttachmentBeginCommand
            | AttachmentFinishCommand
            | AttachmentDownloadCommand
        ),
        connection_epoch: int,
        device_id: str,
    ) -> None:
        connection = self._connections.get(device_id)
        if connection is None or connection.websocket is not websocket:
            raise RuntimeError("mobile command reply 缺少活动连接")
        if frame.type == "ping":
            async with connection.send_lock:
                await websocket.send_json(
                    {
                        "v": 1,
                        "kind": "reply",
                        "type": "ping.ok",
                        "id": frame.id,
                        "connection_epoch": connection_epoch,
                        "payload": {"server_time": _utc_now().isoformat()},
                    }
                )
            return
        from infra.mobile_realtime.channel import MobileCommandError

        try:
            reply = await self.channel.handle_command(
                device_id=device_id,
                frame=frame,
            )
        except CommandConflictError as error:
            async with connection.send_lock:
                await _send_reply(
                    websocket,
                    frame_id=frame.id,
                    connection_epoch=connection_epoch,
                    reply_type=f"{frame.type}.error",
                    payload={"code": "command_id_conflict", "message": str(error)},
                    session_id=frame.session_id,
                    turn_id=frame.turn_id,
                )
            return
        except MobileCommandError as error:
            async with connection.send_lock:
                await _send_reply(
                    websocket,
                    frame_id=frame.id,
                    connection_epoch=connection_epoch,
                    reply_type=f"{frame.type}.error",
                    payload={"code": error.code, "message": str(error)},
                    session_id=frame.session_id,
                    turn_id=frame.turn_id,
                )
            return
        async with self._delivery_lock:
            if self._connections.get(device_id) is not connection:
                raise RuntimeError("mobile command 连接已被新 epoch 替换")
            delivery_barrier = self.storage.read_cursor(device_id).next_event_seq - 1
            if (
                self.storage.read_cursor(device_id).sent_event_seq
                < delivery_barrier
            ):
                connection.reply_barrier = delivery_barrier
        await self._flush_connection_delivery(
            device_id,
            connection,
            through_event_seq=delivery_barrier,
        )
        try:
            async with connection.send_lock:
                if reply.binary is not None:
                    await websocket.send_bytes(encode_attachment_chunk(reply.binary))
                await _send_reply(
                    websocket,
                    frame_id=frame.id,
                    connection_epoch=connection_epoch,
                    reply_type=reply.type,
                    payload=reply.payload,
                    session_id=reply.session_id,
                    turn_id=reply.turn_id,
                )
        finally:
            async with self._delivery_lock:
                if connection.reply_barrier == delivery_barrier:
                    connection.reply_barrier = None
                    if self._connections.get(device_id) is connection:
                        self._schedule_delivery_locked(device_id, connection)

    def _start_plugin_ui_command(
        self,
        websocket: WebSocket,
        frame: GenericCommand,
        connection_epoch: int,
        device_id: str,
    ) -> None:
        """让只读插件查询离开 WebSocket 接收循环并跟随连接取消。"""

        connection = self._connections.get(device_id)
        if connection is None or connection.websocket is not websocket:
            raise RuntimeError("plugin UI query 缺少活动连接")
        task = asyncio.create_task(
            self._handle_plugin_ui_command(
                websocket,
                frame,
                connection_epoch,
                device_id,
            ),
            name=f"mobile_plugin_ui:{device_id}:{frame.id}",
        )
        connection.plugin_ui_tasks.add(task)

        def remove(completed: asyncio.Task[None]) -> None:
            connection.plugin_ui_tasks.discard(completed)
            if not completed.cancelled():
                error = completed.exception()
                if error is not None:
                    logger.error(
                        "plugin UI query task 失败 device=%s request=%s",
                        device_id,
                        frame.id,
                        exc_info=(type(error), error, error.__traceback__),
                    )

        task.add_done_callback(remove)

    async def _handle_plugin_ui_command(
        self,
        websocket: WebSocket,
        frame: GenericCommand,
        connection_epoch: int,
        device_id: str,
    ) -> None:
        """执行临时插件请求并直接回包，不经过 durable receipt 和事件屏障。"""

        from infra.mobile_realtime.channel import MobileCommandError

        connection = self._connections.get(device_id)
        if connection is None or connection.websocket is not websocket:
            return
        try:
            if frame.type == "plugin.ui.query.prepare":
                _ = self.channel.prepare_plugin_ui_query(
                    device_id=device_id,
                    frame=frame,
                )
                request_body = _plugin_ui_http_request_body(frame)
                grant = self.plugin_ui_http_tickets.issue(
                    device_id=device_id,
                    connection_epoch=connection_epoch,
                    request_body=request_body,
                )
                reply_type = "plugin.ui.query.ready"
                payload: dict[str, object] = {
                    "path": _PLUGIN_UI_HTTP_PATH,
                    "ticket": grant.ticket,
                    "expires_at": format_plugin_ui_http_expiry(grant.expires_at),
                }
            else:
                reply = await self.channel.handle_plugin_ui_command(
                    device_id=device_id,
                    frame=frame,
                )
                reply_type = reply.type
                payload = reply.payload
        except asyncio.CancelledError:
            return
        except MobileCommandError as error:
            reply_type = f"{frame.type}.error"
            payload = {
                "code": error.code,
                "message": str(error),
            }
        except Exception as error:
            logger.exception(
                "plugin UI query 执行失败 device=%s plugin_request=%s",
                device_id,
                frame.id,
            )
            reply_type = f"{frame.type}.error"
            payload = {"code": "plugin_error", "message": str(error)}
        if self._connections.get(device_id) is not connection:
            return
        async with connection.send_lock:
            await _send_reply(
                websocket,
                frame_id=frame.id,
                connection_epoch=connection_epoch,
                reply_type=reply_type,
                payload=payload,
                session_id=frame.session_id,
                turn_id=frame.turn_id,
            )

    async def _handle_message_content_prepare(
        self,
        websocket: WebSocket,
        frame: GenericCommand,
        connection_epoch: int,
        device_id: str,
    ) -> None:
        """校验 manifest 并直接签发不进入 durable receipt 的正文下载票据。"""

        from infra.mobile_realtime.channel import MobileCommandError

        connection = self._connections.get(device_id)
        if connection is None or connection.websocket is not websocket:
            return
        try:
            descriptor = self.channel.prepare_message_content(frame)
            session_id = frame.session_id
            if session_id is None:
                raise RuntimeError("message content prepare 缺少 session_id")
            grant = self.message_content_tickets.issue(
                device_id=device_id,
                connection_epoch=connection_epoch,
                session_id=session_id,
                message_id=cast(str, descriptor["message_id"]),
                byte_length=cast(int, descriptor["byte_length"]),
                sha256=cast(str, descriptor["sha256"]),
            )
            reply_type = "message.content.ready"
            payload: dict[str, object] = {
                **descriptor,
                "path": _MESSAGE_CONTENT_HTTP_PATH,
                "ticket": grant.ticket,
                "expires_at": format_message_content_expiry(grant.expires_at),
            }
        except MobileCommandError as error:
            reply_type = "message.content.prepare.error"
            payload = {"code": error.code, "message": str(error)}
        if self._connections.get(device_id) is not connection:
            return
        async with connection.send_lock:
            await _send_reply(
                websocket,
                frame_id=frame.id,
                connection_epoch=connection_epoch,
                reply_type=reply_type,
                payload=payload,
                session_id=frame.session_id,
                turn_id=None,
            )

    def read_message_content_http(
        self,
        *,
        ticket: str,
        range_header: str | None,
        if_range: str | None,
    ) -> tuple[bytes, int, int, int, str]:
        """校验短期授权并返回一个有界、连续的正文 byte range。"""

        from infra.mobile_realtime.channel import MobileCommandError

        verified = self.message_content_tickets.verify(ticket)
        connection = self._connections.get(verified.device_id)
        if (
            connection is None
            or connection.connection_epoch != verified.connection_epoch
        ):
            raise MessageContentTicketError("message content ticket 的连接已失效")
        expected_etag = f'"{verified.sha256}"'
        if if_range is not None and if_range != expected_etag:
            raise MobileMessageContentHttpError(
                "content_changed",
                "If-Range 与正文摘要不匹配",
                status_code=412,
            )
        try:
            content = self.channel.read_message_content(
                session_id=verified.session_id,
                message_id=verified.message_id,
                byte_length=verified.byte_length,
                sha256=verified.sha256,
            )
        except MobileCommandError as error:
            status = 404 if error.code == "message_not_found" else 412
            raise MobileMessageContentHttpError(
                error.code,
                str(error),
                status_code=status,
            ) from error
        start, end = _parse_message_content_range(range_header, len(content))
        return content[start : end + 1], start, end, len(content), verified.sha256

    async def handle_plugin_ui_http_query(
        self,
        *,
        ticket: str,
        request_body: dict[str, object],
    ) -> dict[str, object]:
        """验签并通过现有插件调度器执行一次 HTTPS 只读查询。"""

        from infra.mobile_realtime.channel import MobileCommandError

        verified = self.plugin_ui_http_tickets.verify(
            ticket,
            request_body=request_body,
        )
        connection = self._connections.get(verified.device_id)
        if (
            connection is None
            or connection.connection_epoch != verified.connection_epoch
        ):
            raise PluginUiHttpTicketError("plugin UI HTTP ticket 的连接已失效")
        frame = _plugin_ui_http_frame(
            request_body,
            connection_epoch=verified.connection_epoch,
        )
        try:
            reply = await self.channel.handle_plugin_ui_command(
                device_id=verified.device_id,
                frame=frame,
            )
        except MobileCommandError as error:
            raise MobilePluginUiHttpError(
                error.code,
                str(error),
                status_code=_plugin_ui_http_error_status(error.code),
            ) from error
        if reply.type != "plugin.ui.query.ok":
            raise RuntimeError(f"plugin UI HTTP query 返回了意外 reply: {reply.type}")
        result = reply.payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("plugin UI HTTP query 成功但缺少 object result")
        return cast(dict[str, object], result)

    async def _flush_connection_delivery(
        self,
        device_id: str,
        connection: ActiveMobileConnection,
        *,
        through_event_seq: int,
    ) -> None:
        """等待命令完成时已分配的本设备事件发送到指定屏障。"""

        while True:
            # 1. 在连接状态边界检查 epoch 与已发送序号
            async with self._delivery_lock:
                if self._connections.get(device_id) is not connection:
                    raise RuntimeError("mobile command 连接已被新 epoch 替换")
                cursor = self.storage.read_cursor(device_id)
                if cursor.sent_event_seq >= through_event_seq:
                    return
                if connection.delivery_task is None:
                    raise RuntimeError("mobile command 投递屏障缺少活动任务")

            # 2. 条件通知只表示 sent cursor 可能前进，醒来后重新校验
            async with connection.sent_condition:
                cursor = self.storage.read_cursor(device_id)
                if cursor.sent_event_seq >= through_event_seq:
                    return
                _ = await connection.sent_condition.wait()

    def close(self) -> None:
        self.storage.close()


def build_mobile_gateway_runtime(
    config: MobileRealtimeConfig,
    workspace: Path,
    *,
    master_keys: MasterKeyStore | None = None,
) -> tuple[MobileGatewayRuntime, LoadedKeyset]:
    """加载或初始化加密身份，并构造移动网关运行时。"""

    database_path = _resolve_workspace_path(workspace, config.database)
    current_path = _resolve_workspace_path(
        workspace,
        config.key_encryption.keyset_manifest,
    )
    keys = master_keys or SecretServiceMasterKeyStore(
        config.key_encryption.master_key_namespace
    )
    keysets = KeysetManager(current_path.parent, keys)
    storage = MobileRealtimeStorage(database_path)
    try:
        identity = storage.read_server_identity()
        if current_path.exists():
            keyset = keysets.load(
                expected_server_fingerprint=(
                    identity.public_key_fingerprint if identity is not None else None
                )
            )
        else:
            if identity is not None:
                raise KeyProtectionError("数据库存在 server identity，但 keyset 已丢失")
            keyset = keysets.initialize(lan_hostname=config.lan_hostname)
        storage.write_server_identity(
            ServerIdentityReference(
                server_id=keyset.manifest.server_id,
                keyset_manifest_path=str(config.key_encryption.keyset_manifest),
                public_key_fingerprint=keyset.server_fingerprint,
            )
        )
        loop = asyncio.get_running_loop()
        approvals = PairingApprovalRegistry(loop)
        lan_endpoints = (f"wss://{config.lan_hostname}:{config.port}/ws",)
        tunnel_endpoints = (config.public_url,) if config.public_url else ()
        pairing = PairingService(
            storage,
            keyset,
            lan_endpoints=lan_endpoints,
            tunnel_endpoints=tunnel_endpoints,
        )
        runtime = MobileGatewayRuntime(
            config=config,
            storage=storage,
            pairing=pairing,
            authenticator=DeviceAuthenticator(storage, keyset),
            inbox=DurableInboxManager(
                storage,
                retention=config.inbox_retention,
            ),
            approvals=approvals,
            keyset=keyset,
        )
        from infra.mobile_realtime.channel import MobileRealtimeChannel

        runtime.bind_channel(MobileRealtimeChannel(runtime))
        return runtime, keyset
    except BaseException:
        storage.close()
        raise


def create_mobile_gateway_app(runtime: MobileGatewayRuntime) -> FastAPI:
    app = FastAPI(title="Akasic Mobile Realtime Gateway", docs_url=None, redoc_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.websocket("/ws")
    async def mobile_websocket(websocket: WebSocket) -> None:
        await runtime.handle_websocket(websocket)

    @app.post(_PLUGIN_UI_HTTP_PATH)
    async def mobile_plugin_ui_query(request: Request) -> JSONResponse:
        """接收短期设备授权下的插件查询数据面请求。"""

        try:
            ticket = _plugin_ui_http_bearer(request)
            body = await _read_plugin_ui_http_body(request)
            result = await runtime.handle_plugin_ui_http_query(
                ticket=ticket,
                request_body=body,
            )
        except PluginUiHttpTicketError as error:
            return _plugin_ui_http_error_response(
                "invalid_ticket",
                str(error),
                status_code=401,
            )
        except MobilePluginUiHttpError as error:
            return _plugin_ui_http_error_response(
                error.code,
                str(error),
                status_code=error.status_code,
            )
        except (MobileGatewayError, ProtocolDecodeError, ValidationError) as error:
            return _plugin_ui_http_error_response(
                "invalid_request",
                str(error),
                status_code=400,
            )
        except Exception:
            logger.exception("plugin UI HTTP query 执行失败")
            return _plugin_ui_http_error_response(
                "plugin_error",
                "插件查询执行失败",
                status_code=500,
            )
        response = JSONResponse(result)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get(_MESSAGE_CONTENT_HTTP_PATH)
    async def mobile_message_content(request: Request) -> Response:
        """返回短期设备授权下的一个不可变正文 byte range。"""

        try:
            ticket = _message_content_http_bearer(request)
            content, start, end, total, sha256 = runtime.read_message_content_http(
                ticket=ticket,
                range_header=request.headers.get("range"),
                if_range=request.headers.get("if-range"),
            )
        except MessageContentTicketError as error:
            return _message_content_error_response(
                "invalid_ticket",
                str(error),
                status_code=401,
            )
        except MobileMessageContentHttpError as error:
            return _message_content_error_response(
                error.code,
                str(error),
                status_code=error.status_code,
            )
        content_digest = base64.b64encode(hashlib.sha256(content).digest()).decode("ascii")
        representation_digest = base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
        return Response(
            content=content,
            status_code=206,
            media_type="text/plain; charset=utf-8",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "private, no-store",
                "Content-Digest": f"sha-256=:{content_digest}:",
                "Content-Encoding": "identity",
                "Content-Range": f"bytes {start}-{end}/{total}",
                "ETag": f'"{sha256}"',
                "Repr-Digest": f"sha-256=:{representation_digest}:",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return app


def build_mobile_gateway_server(
    runtime: MobileGatewayRuntime,
    keyset: LoadedKeyset,
) -> uvicorn.Server:
    config = uvicorn.Config(
        create_mobile_gateway_app(runtime),
        host=runtime.config.host,
        port=runtime.config.port,
        log_level="warning",
        access_log=False,
        ws="websockets-sansio",
        ws_max_size=256 * 1024,
        ws_max_queue=32,
        ws_ping_interval=25.0,
        ws_ping_timeout=25.0,
        ws_per_message_deflate=False,
    )
    config.load()
    config.ssl_certfile = keyset.tls_certificate_path
    config.ssl = create_server_ssl_context(keyset)
    return uvicorn.Server(config)


async def _receive_authenticated_item(
    websocket: WebSocket,
) -> MobileFrame | AttachmentChunk:
    """接收认证后的 JSON 协议帧或附件二进制分片。"""

    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(code=int(message.get("code") or 1000))
    text = message.get("text")
    data = message.get("bytes")
    if isinstance(text, str) and data is None:
        return parse_frame(text)
    if isinstance(data, bytes) and text is None:
        try:
            return decode_attachment_chunk(data)
        except (ValueError, ValidationError) as error:
            raise ProtocolDecodeError(str(error)) from error
    raise ProtocolDecodeError("WebSocket frame 必须恰好包含 text 或 bytes")


async def _send_control(
    websocket: WebSocket,
    control_type: str,
    payload: dict[str, object],
) -> None:
    await websocket.send_json(
        {"v": 1, "kind": "control", "type": control_type, "payload": payload}
    )


async def _send_reply(
    websocket: WebSocket,
    *,
    frame_id: str,
    connection_epoch: int,
    reply_type: str,
    payload: dict[str, object],
    session_id: str | None,
    turn_id: str | None,
) -> None:
    frame: dict[str, object] = {
        "v": 1,
        "kind": "reply",
        "type": reply_type,
        "id": frame_id,
        "connection_epoch": connection_epoch,
        "payload": payload,
    }
    if session_id is not None:
        frame["session_id"] = session_id
    if turn_id is not None:
        frame["turn_id"] = turn_id
    wire = parse_frame(
        json.dumps(
            frame,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    )
    await websocket.send_text(frame_to_json(wire))


async def _close_with_error(
    websocket: WebSocket,
    *,
    code: int,
    reason: str,
) -> None:
    await _send_control(
        websocket,
        "protocol.error",
        {"code": code, "message": reason[:512]},
    )
    await websocket.close(code=code, reason=reason[:123])


async def _send_stored_event(
    websocket: WebSocket,
    envelope_json: str,
    event_seq: int,
    connection_epoch: int,
) -> None:
    wire = _stored_event_to_wire(
        envelope_json,
        event_seq=event_seq,
        connection_epoch=connection_epoch,
    )
    if wire.kind != "event":
        raise AssertionError("stored event 被解析成非 event 帧")
    await websocket.send_text(frame_to_json(wire))


def _encode_stored_event(
    *,
    event_id: str,
    event_type: str,
    payload: dict[str, object],
    session_id: str | None = None,
    turn_id: str | None = None,
) -> str:
    """在入箱前验证事件，并编码不含连接态字段的稳定 envelope。"""

    body: dict[str, object] = {
        "id": event_id,
        "type": event_type,
        "payload": payload,
    }
    if session_id is not None:
        body["session_id"] = session_id
    if turn_id is not None:
        body["turn_id"] = turn_id
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    _ = _stored_event_to_wire(encoded, event_seq=1, connection_epoch=1)
    return encoded


def _stored_event_type(event: DurableInboxEvent) -> str:
    """读取已经过入箱校验的 durable 事件类型。"""

    body = json.loads(event.envelope_json)
    event_type = body["type"]
    if not isinstance(event_type, str):
        raise TypeError("durable event type 必须为文本")
    return event_type


def _stored_event_to_wire(
    envelope_json: str,
    *,
    event_seq: int,
    connection_epoch: int,
) -> MobileFrame:
    raw = _decode_stored_envelope(envelope_json)
    allowed = {"id", "type", "payload", "session_id", "turn_id"}
    if not {"id", "type", "payload"}.issubset(raw) or not raw.keys() <= allowed:
        raise MobileGatewayError("durable inbox envelope 格式无效")
    body: dict[str, object] = {
        "v": 1,
        "kind": "event",
        "type": raw["type"],
        "id": raw["id"],
        "connection_epoch": connection_epoch,
        "event_seq": event_seq,
        "payload": raw["payload"],
    }
    if "session_id" in raw:
        body["session_id"] = raw["session_id"]
    if "turn_id" in raw:
        body["turn_id"] = raw["turn_id"]
    return parse_frame(
        json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    )


def _decode_stored_envelope(envelope_json: str) -> dict[str, object]:
    """在 SQLite 边界拒绝重复字段、非标准常量和非 object envelope。"""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise MobileGatewayError(f"durable inbox 包含重复字段: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise MobileGatewayError(f"durable inbox 包含非标准常量: {value}")

    try:
        raw = json.loads(
            envelope_json,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise MobileGatewayError("durable inbox envelope JSON 无效") from error
    if not isinstance(raw, dict):
        raise MobileGatewayError("durable inbox envelope 顶层必须是 object")
    return cast(dict[str, object], raw)


def _plugin_ui_http_request_body(frame: GenericCommand) -> dict[str, object]:
    """把已验证 WS prepare 帧冻结为 HTTPS 请求摘要输入。"""

    return {
        "request_id": frame.id,
        "owner_id": frame.payload["owner_id"],
        "plugin_id": frame.payload["plugin_id"],
        "plugin_revision": frame.payload["plugin_revision"],
        "method": frame.payload["method"],
        "payload": frame.payload["payload"],
        "slot": frame.payload["slot"],
        "session_id": frame.session_id,
        "turn_id": frame.turn_id,
    }


def _plugin_ui_http_frame(
    body: dict[str, object],
    *,
    connection_epoch: int,
) -> GenericCommand:
    """把验签后的 HTTPS body 还原为既有插件查询协议对象。"""

    expected = {
        "request_id",
        "owner_id",
        "plugin_id",
        "plugin_revision",
        "method",
        "payload",
        "slot",
        "session_id",
        "turn_id",
    }
    if set(body) != expected:
        raise MobileGatewayError("plugin UI HTTP 请求字段无效")
    return GenericCommand.model_validate(
        {
            "v": 1,
            "kind": "command",
            "type": "plugin.ui.query",
            "id": body["request_id"],
            "connection_epoch": connection_epoch,
            "session_id": body["session_id"],
            "turn_id": body["turn_id"],
            "payload": {
                "owner_id": body["owner_id"],
                "plugin_id": body["plugin_id"],
                "plugin_revision": body["plugin_revision"],
                "method": body["method"],
                "payload": body["payload"],
                "slot": body["slot"],
            },
        },
        strict=True,
    )


def _plugin_ui_http_bearer(request: Request) -> str:
    authorization = request.headers.get("authorization")
    if authorization is None or not authorization.startswith("Bearer "):
        raise PluginUiHttpTicketError("plugin UI HTTP 请求缺少 Bearer ticket")
    ticket = authorization.removeprefix("Bearer ")
    if not 1 <= len(ticket) <= 4096 or any(character.isspace() for character in ticket):
        raise PluginUiHttpTicketError("plugin UI HTTP Bearer ticket 无效")
    return ticket


def _message_content_http_bearer(request: Request) -> str:
    authorization = request.headers.get("authorization")
    if authorization is None or not authorization.startswith("Bearer "):
        raise MessageContentTicketError("message content 请求缺少 Bearer ticket")
    ticket = authorization.removeprefix("Bearer ")
    if not 1 <= len(ticket) <= 4096 or any(character.isspace() for character in ticket):
        raise MessageContentTicketError("message content Bearer ticket 无效")
    return ticket


def _parse_message_content_range(value: str | None, total: int) -> tuple[int, int]:
    """解析一个显式 bytes range，并强制单次响应保持有界。"""

    if total <= 0:
        raise MobileMessageContentHttpError(
            "range_not_satisfiable",
            "空正文不需要 range 下载",
            status_code=416,
        )
    if value is None or not value.startswith("bytes=") or "," in value:
        raise MobileMessageContentHttpError(
            "range_required",
            "正文下载必须携带单个 bytes Range",
            status_code=416,
        )
    bounds = value.removeprefix("bytes=").split("-", 1)
    if len(bounds) != 2 or not bounds[0].isdigit() or not bounds[1].isdigit():
        raise MobileMessageContentHttpError(
            "invalid_range",
            "正文 Range 格式无效",
            status_code=416,
        )
    start, requested_end = int(bounds[0]), int(bounds[1])
    if start >= total or requested_end < start:
        raise MobileMessageContentHttpError(
            "range_not_satisfiable",
            "正文 Range 超出内容范围",
            status_code=416,
        )
    end = min(requested_end, total - 1)
    if end - start + 1 > _MAX_MESSAGE_CONTENT_RANGE_BYTES:
        raise MobileMessageContentHttpError(
            "range_too_large",
            "正文 Range 超过单次下载预算",
            status_code=416,
        )
    return start, end


def _message_content_error_response(
    code: str,
    message: str,
    *,
    status_code: int,
) -> JSONResponse:
    response = JSONResponse(
        {"error": {"code": code, "message": message}},
        status_code=status_code,
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


async def _read_plugin_ui_http_body(request: Request) -> dict[str, object]:
    """有界读取并严格解析一个插件查询 JSON object。"""

    content_type = request.headers.get("content-type", "").partition(";")[0].strip()
    if content_type != "application/json":
        raise MobileGatewayError("plugin UI HTTP Content-Type 必须是 application/json")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as error:
            raise MobileGatewayError("plugin UI HTTP Content-Length 无效") from error
        if declared_length < 0 or declared_length > _MAX_PLUGIN_UI_HTTP_REQUEST_BYTES:
            raise MobileGatewayError("plugin UI HTTP 请求超过 72 KiB")

    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > _MAX_PLUGIN_UI_HTTP_REQUEST_BYTES:
            raise MobileGatewayError("plugin UI HTTP 请求超过 72 KiB")
    if not raw:
        raise MobileGatewayError("plugin UI HTTP 请求体不能为空")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise MobileGatewayError(f"plugin UI HTTP JSON 包含重复字段: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise MobileGatewayError(f"plugin UI HTTP JSON 包含非标准常量: {value}")

    try:
        body = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MobileGatewayError("plugin UI HTTP JSON 无效") from error
    if not isinstance(body, dict):
        raise MobileGatewayError("plugin UI HTTP 请求体必须是 object")
    return cast(dict[str, object], body)


def _plugin_ui_http_error_status(code: str) -> int:
    if code in {"plugin_timeout", "plugin_overloaded", "plugin_unavailable"}:
        return 503
    if code == "plugin_failed":
        return 500
    return 422


def _plugin_ui_http_error_response(
    code: str,
    message: str,
    *,
    status_code: int,
) -> JSONResponse:
    response = JSONResponse(
        {"code": code, "message": message},
        status_code=status_code,
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def _protocol_close_code(error: ProtocolDecodeError | ValidationError) -> int:
    if isinstance(error, ValidationError):
        for issue in error.errors(include_url=False):
            if issue["loc"] and issue["loc"][-1] == "v":
                return _CLOSE_VERSION
            if issue["loc"][:2] == ("command", "message.send"):
                return _CLOSE_COMMAND
    return _CLOSE_PROTOCOL


def _protocol_error_reason(error: ProtocolDecodeError | ValidationError) -> str:
    """把协议边界错误收敛为可展示且不泄露载荷的短原因。"""

    if isinstance(error, ProtocolDecodeError):
        return "协议帧无法解析"
    if _protocol_close_code(error) == _CLOSE_VERSION:
        return "协议版本不兼容"
    return "协议字段无效"


def _pending_claim_json(claim: PendingPairingClaim) -> dict[str, object]:
    return {
        "pairing_id": claim.pairing_id,
        "device_name": claim.device_name,
        "capabilities": list(claim.capabilities),
        "confirmation_code": claim.confirmation_code,
    }


def _device_json(device: DeviceRecord) -> dict[str, object]:
    return {
        "device_id": device.device_id,
        "display_name": device.display_name,
        "created_at": device.created_at.isoformat(),
        "capabilities": list(device.capabilities),
    }


def _resolve_workspace_path(workspace: Path, configured: Path) -> Path:
    if configured.is_absolute():
        return configured
    return workspace / configured


def _new_ulid() -> str:
    value = (int(time.time() * 1000) << 80) | secrets.randbits(80)
    chars = ["0"] * 26
    for index in range(25, -1, -1):
        chars[index] = _CROCKFORD32[value & 31]
        value >>= 5
    return "".join(chars)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_plugin_ui_command(frame: object) -> bool:
    return isinstance(frame, GenericCommand) and frame.type in {
        "plugin.ui.catalog",
        "plugin.ui.asset.get",
        "plugin.ui.query",
        "plugin.ui.query.prepare",
        "plugin.ui.cancel",
    }


__all__ = [
    "MobileGatewayRuntime",
    "MobilePairingAdmin",
    "PairingApprovalRegistry",
    "build_mobile_gateway_runtime",
    "build_mobile_gateway_server",
    "create_mobile_gateway_app",
]
