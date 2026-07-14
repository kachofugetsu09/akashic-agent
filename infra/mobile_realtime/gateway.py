from __future__ import annotations

import asyncio
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import cast

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
    DeviceRecord,
    MobileRealtimeStorage,
    PairingStateError,
    ServerIdentityReference,
    UnknownDeviceError,
)

_CLOSE_PROTOCOL = 4400
_CLOSE_UNAUTHENTICATED = 4401
_CLOSE_REVOKED = 4403
_CLOSE_VERSION = 4406
_CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class MobileGatewayError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PairingApproval:
    pairing_id: str
    device_id: str


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
    ) -> None:
        self.config = config
        self.storage = storage
        self.pairing = pairing
        self.authenticator = authenticator
        self.inbox = inbox
        self.approvals = approvals
        self.admin = MobilePairingAdmin(pairing, approvals)

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
                reason=str(error),
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
                resumed = True
                await self._resume(
                    websocket,
                    device_id=device_id,
                    connection_epoch=connection_epoch,
                    last_ack=frame.payload.last_ack,
                )
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
                await self._handle_command(websocket, frame, connection_epoch)
                continue
            raise AssertionError("已验证客户端帧没有对应处理分支")

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
        stored = json.dumps(
            {"id": event_id, "type": event_type, "payload": payload},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
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
        await websocket.send_json(
            {
                "v": 1,
                "kind": "reply",
                "type": f"{frame.type}.error",
                "id": frame.id,
                "connection_epoch": connection_epoch,
                "payload": {
                    "code": "not_implemented",
                    "message": "该 command 尚未接入 mobile channel",
                },
            }
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
        )
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
    raw = _decode_stored_envelope(envelope_json)
    if not isinstance(raw, dict) or raw.keys() != {"id", "type", "payload"}:
        raise MobileGatewayError("durable inbox envelope 格式无效")
    wire = parse_frame(
        json.dumps(
            {
                "v": 1,
                "kind": "event",
                "type": raw["type"],
                "id": raw["id"],
                "connection_epoch": connection_epoch,
                "event_seq": event_seq,
                "payload": raw["payload"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    if wire.kind != "event":
        raise AssertionError("stored event 被解析成非 event 帧")
    await websocket.send_text(frame_to_json(wire))


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
    return _CLOSE_PROTOCOL


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
