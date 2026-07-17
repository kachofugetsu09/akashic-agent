from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from agent.config_models import MobileRealtimeConfig
from infra.mobile_realtime.auth import (
    AuthenticationChallengeExpired,
    DeviceAuthenticationError,
    DeviceAuthenticator,
    DeviceProofPayload,
    DeviceRevokedError,
    UnknownAuthenticationChallenge,
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
from infra.mobile_realtime.pairing import (
    PairClaimPayload,
    PairingSecretError,
    PairingService,
    PairingSignatureError,
    PendingPairingClaim,
)
from infra.mobile_realtime.protocol import (
    AckFrame,
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
    MobileRealtimeStorage,
    PairingStateError,
    ServerIdentityReference,
    UnknownDeviceError,
)

if TYPE_CHECKING:
    from infra.mobile_realtime.channel import MobileRealtimeChannel

_CLOSE_PROTOCOL = 4400
_CLOSE_UNAUTHENTICATED = 4401
_CLOSE_REVOKED = 4403
_CLOSE_VERSION = 4406
_CLOSE_COMMAND = 4410
_CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

logger = logging.getLogger(__name__)


class MobileGatewayError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PairingApproval:
    pairing_id: str
    device_id: str


@dataclass(frozen=True, slots=True)
class ActiveMobileConnection:
    websocket: WebSocket
    connection_epoch: int


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
            first = await _receive_frame(websocket)
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
                frame = await _receive_frame(websocket)
                if not isinstance(
                    frame,
                    (ResumeControl, AckFrame, GenericCommand, MessageSendCommand),
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
                        _ = self.inbox.acknowledge(
                            device_id,
                            through_event_seq=frame.payload.through_event_seq,
                        )
                    except (AckRollbackError, AckOverflowError) as error:
                        await _close_with_error(
                            websocket,
                            code=_CLOSE_PROTOCOL,
                            reason=str(error),
                        )
                        return
                    continue
                if isinstance(frame, (GenericCommand, MessageSendCommand)):
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
                await self._remove_connection(
                    device_id=device_id,
                    websocket=websocket,
                )

    async def _resume_and_register(
        self,
        websocket: WebSocket,
        *,
        device_id: str,
        connection_epoch: int,
        last_ack: int,
    ) -> None:
        """在全局投递锁内完成重放并注册当前设备连接。"""

        async with self._delivery_lock:
            await self._resume(
                websocket,
                device_id=device_id,
                connection_epoch=connection_epoch,
                last_ack=last_ack,
            )
            previous = self._connections.get(device_id)
            if previous is not None and previous.websocket is not websocket:
                try:
                    await previous.websocket.close(
                        code=4001,
                        reason="设备已在新连接上线",
                    )
                except (RuntimeError, OSError):
                    logger.info("旧 mobile WebSocket 已先于新连接关闭: %s", device_id)
            self._connections[device_id] = ActiveMobileConnection(
                websocket=websocket,
                connection_epoch=connection_epoch,
            )

    async def _resume(
        self,
        websocket: WebSocket,
        *,
        device_id: str,
        connection_epoch: int,
        last_ack: int,
    ) -> None:
        """重放未确认 P0，并以新的 connection epoch 重建 wire envelope。"""

        cursor = self.storage.read_cursor(device_id)
        if last_ack < cursor.acknowledged_event_seq:
            await self._enqueue_and_send_event(
                websocket,
                device_id=device_id,
                connection_epoch=connection_epoch,
                event_type="sync.reset_required",
                payload={"reason": "client_ack_behind_server_cursor"},
            )
            return
        try:
            replay = self.inbox.replay(
                device_id,
                after_event_seq=last_ack,
                limit=512,
            )
        except InboxResetRequired:
            await self._enqueue_and_send_event(
                websocket,
                device_id=device_id,
                connection_epoch=connection_epoch,
                event_type="sync.reset_required",
                payload={"reason": "inbox_retention_exceeded"},
            )
            return
        replayed_events = 0
        highest_sent = cursor.sent_event_seq
        after_event_seq = last_ack
        while True:
            replay = self.inbox.replay(
                device_id,
                after_event_seq=after_event_seq,
                limit=512,
            )
            for event in replay.events:
                await _send_stored_event(
                    websocket,
                    event.envelope_json,
                    event.event_seq,
                    connection_epoch,
                )
                highest_sent = max(highest_sent, event.event_seq)
                after_event_seq = event.event_seq
                replayed_events += 1
            if len(replay.events) < 512:
                break
        if highest_sent > cursor.sent_event_seq:
            _ = self.inbox.mark_sent(
                device_id,
                through_event_seq=highest_sent,
            )
        await self._enqueue_and_send_event(
            websocket,
            device_id=device_id,
            connection_epoch=connection_epoch,
            event_type="sync.completed",
            payload={"mode": "replay", "replayed_events": replayed_events},
        )

    async def publish_event(
        self,
        *,
        event_type: str,
        payload: dict[str, object],
        session_id: str | None = None,
        turn_id: str | None = None,
        device_id: str | None = None,
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

        # 2. 串行化 resume 与 fanout，避免重放窗口漏掉并发事件
        async with self._delivery_lock:
            if device_id is not None:
                target_device_ids = (device_id,)
            elif session_id is not None:
                target_device_ids = (self.storage.session_owner(session_id),)
            else:
                target_device_ids = tuple(
                    device.device_id for device in self.storage.list_active_devices()
                )
            for target_device_id in target_device_ids:
                event = self.inbox.enqueue(
                    device_id=target_device_id,
                    event_id=event_id,
                    envelope_json=stored,
                )
                connection = self._connections.get(target_device_id)
                if connection is None:
                    continue
                try:
                    await _send_stored_event(
                        connection.websocket,
                        event.envelope_json,
                        event.event_seq,
                        connection.connection_epoch,
                    )
                except (WebSocketDisconnect, RuntimeError, OSError) as error:
                    logger.warning(
                        "mobile 在线投递失败，事件保留到下次 resume: device=%s seq=%s error=%s",
                        target_device_id,
                        event.event_seq,
                        error,
                    )
                    if self._connections.get(target_device_id) is connection:
                        _ = self._connections.pop(target_device_id)
                    continue
                _ = self.inbox.mark_sent(
                    target_device_id,
                    through_event_seq=event.event_seq,
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

    async def _enqueue_and_send_event(
        self,
        websocket: WebSocket,
        *,
        device_id: str,
        connection_epoch: int,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        event_id = _new_ulid()
        stored = _encode_stored_event(
            event_id=event_id,
            event_type=event_type,
            payload=payload,
        )
        event = self.inbox.enqueue(
            device_id=device_id,
            event_id=event_id,
            envelope_json=stored,
        )
        await _send_stored_event(
            websocket,
            event.envelope_json,
            event.event_seq,
            connection_epoch,
        )
        _ = self.inbox.mark_sent(
            device_id,
            through_event_seq=event.event_seq,
        )

    async def _handle_command(
        self,
        websocket: WebSocket,
        frame: GenericCommand | MessageSendCommand,
        connection_epoch: int,
        device_id: str,
    ) -> None:
        if frame.type == "ping":
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
        try:
            reply = await self.channel.handle_command(
                device_id=device_id,
                frame=frame,
            )
        except CommandConflictError as error:
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
        await _send_reply(
            websocket,
            frame_id=frame.id,
            connection_epoch=connection_epoch,
            reply_type=reply.type,
            payload=reply.payload,
            session_id=reply.session_id,
            turn_id=reply.turn_id,
        )

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

    @app.websocket("/ws")
    async def mobile_websocket(websocket: WebSocket) -> None:
        await runtime.handle_websocket(websocket)

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


async def _receive_frame(websocket: WebSocket) -> MobileFrame:
    return parse_frame(await websocket.receive_text())


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


def _protocol_close_code(error: ProtocolDecodeError | ValidationError) -> int:
    if isinstance(error, ValidationError):
        for issue in error.errors(include_url=False):
            if issue["loc"] and issue["loc"][-1] == "v":
                return _CLOSE_VERSION
            if issue["loc"][:2] == ("command", "message.send"):
                return _CLOSE_COMMAND
    return _CLOSE_PROTOCOL


def _protocol_error_reason(error: ProtocolDecodeError | ValidationError) -> str:
    """返回不包含用户载荷的稳定协议拒绝原因。"""

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


__all__ = [
    "MobileGatewayRuntime",
    "MobilePairingAdmin",
    "PairingApprovalRegistry",
    "build_mobile_gateway_runtime",
    "build_mobile_gateway_server",
    "create_mobile_gateway_app",
]
