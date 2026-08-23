import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeVar
from typing import Protocol, cast
from uuid import uuid4

from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentRef,
    ChannelInboundMessage,
    ChannelDeliveryReceipt,
    DeliveryStatus as ChannelDeliveryStatus,
    InboundEnvelope,
    InboundOwner,
    InboundState,
    OutboundEnvelope,
    RawInbound,
)
from bus.events import InboundItem, InboundMessage

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_DURABLE_INBOUND_RECOVERY_PAGE_SIZE = 256
_INBOUND_CLEANUP_RETRY_INITIAL_DELAY = 0.1
_INBOUND_CLEANUP_RETRY_MAX_DELAY = 5.0
_MOBILE_V3_HANDOFF = "mobile_v3_handoff"
_MOBILE_V3_HANDOFF_ID = "mobile_handoff_id"
_MOBILE_V3_ATTACHMENT_REFS = "mobile_v3_attachment_refs"

MobileInboundRecoverer = Callable[[RawInbound], Awaitable[bool]]


class DurableInboundStore(Protocol):
    """MessageBus 所需的最小移动 handoff 持久 owner 接口。"""

    def reserve_inbound_handoff(
        self,
        *,
        handoff_id: str,
        dedupe_key: str | None,
        channel: str,
        sender: str,
        chat_id: str,
        session_key: str,
        content: str,
        timestamp: str,
        media_json: str,
        metadata_json: str,
        created_at: str,
    ) -> tuple[str, bool]: ...

    def list_inbound_handoffs(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, str | None]]: ...

    def has_inbound_handoff(
        self,
        *,
        session_key: str,
        client_message_id: str,
    ) -> bool: ...

    def complete_inbound_handoff(self, handoff_id: str) -> None: ...


class MobileSessionAdmissionOwner(Protocol):
    """Mobile exact handoff 所需的最小 Session admission owner。"""

    def admit_existing(self, key: str) -> tuple[object, str]: ...

    def release_admission(self, admission_id: str) -> None: ...


@dataclass(slots=True)
class _MobileV3Admission:
    admission_id: str
    envelope: InboundEnvelope | None = None
    cleanup_pending: bool = False
    recoverable: bool = False


def _mobile_dedupe_key(message: InboundMessage) -> str | None:
    client_message_id = message.metadata.get("client_message_id")
    if not isinstance(client_message_id, str) or not client_message_id:
        return None
    return f"{message.session_key}:{client_message_id}"


def _serialize_handoff(message: InboundMessage) -> tuple[str, str]:
    media_json = json.dumps(
        message.media,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    metadata_json = json.dumps(
        message.metadata,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return media_json, metadata_json


def _inbound_from_handoff(row: dict[str, str | None]) -> InboundMessage:
    required = (
        "handoff_id",
        "channel",
        "sender",
        "chat_id",
        "content",
        "timestamp",
        "media_json",
        "metadata_json",
    )
    values = {key: row.get(key) for key in required}
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise ValueError(f"inbound handoff schema invalid: {row!r}")
    media = json.loads(cast(str, values["media_json"]))
    metadata = json.loads(cast(str, values["metadata_json"]))
    if not isinstance(media, list) or not all(isinstance(item, str) for item in media):
        raise ValueError(f"inbound handoff media invalid: {values['handoff_id']}")
    if not isinstance(metadata, dict):
        raise ValueError(f"inbound handoff metadata invalid: {values['handoff_id']}")
    timestamp = datetime.fromisoformat(cast(str, values["timestamp"]))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(
            f"inbound handoff timestamp missing timezone: {values['handoff_id']}"
        )
    return InboundMessage(
        channel=cast(str, values["channel"]),
        sender=cast(str, values["sender"]),
        chat_id=cast(str, values["chat_id"]),
        content=cast(str, values["content"]),
        timestamp=timestamp,
        media=cast(list[str], media),
        metadata=cast(dict[str, object], metadata),
        handoff_id=cast(str, values["handoff_id"]),
    )


def _raw_mobile_from_handoff(row: dict[str, str | None]) -> RawInbound | None:
    """Rebuild only a marked v3 Mobile handoff without reviving a legacy lease."""

    legacy = _inbound_from_handoff(row)
    metadata = legacy.metadata
    if metadata.get(_MOBILE_V3_HANDOFF) is not True:
        return None
    handoff_id = row.get("handoff_id")
    requested_handoff_id = metadata.get(_MOBILE_V3_HANDOFF_ID)
    client_message_id = metadata.get("client_message_id")
    refs_json = metadata.get(_MOBILE_V3_ATTACHMENT_REFS, [])
    if (
        not isinstance(handoff_id, str)
        or requested_handoff_id != handoff_id
        or not isinstance(client_message_id, str)
        or not isinstance(refs_json, list)
    ):
        raise ValueError("v3 Mobile inbound handoff identity invalid")
    refs: list[AttachmentRef] = []
    for item in refs_json:
        if not isinstance(item, dict):
            raise ValueError("v3 Mobile attachment handoff invalid")
        artifact_id = item.get("artifact_id")
        kind = item.get("kind")
        filename = item.get("filename")
        media_type = item.get("media_type")
        size_bytes = item.get("size_bytes")
        sha256 = item.get("sha256")
        if (
            not isinstance(artifact_id, str)
            or not isinstance(kind, str)
            or filename is not None and not isinstance(filename, str)
            or media_type is not None and not isinstance(media_type, str)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or not isinstance(sha256, str)
        ):
            raise ValueError("v3 Mobile attachment handoff invalid")
        try:
            refs.append(
                AttachmentRef(
                    artifact_id=artifact_id,
                    kind=AttachmentKind(kind),
                    filename=filename,
                    media_type=media_type,
                    size_bytes=size_bytes,
                    sha256=sha256,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("v3 Mobile attachment handoff invalid") from error
    return RawInbound(
        message_id=client_message_id,
        provider_identity=legacy.sender,
        recipient=legacy.chat_id,
        message=ChannelInboundMessage(
            channel=legacy.channel,
            sender=legacy.sender,
            chat_id=legacy.chat_id,
            content=legacy.content,
            timestamp=legacy.timestamp,
            metadata=metadata,
            attachments=tuple(refs),
        ),
    )


@dataclass
class _ChatLaneState:
    condition: asyncio.Condition
    active_users: int = 0
    passive_turns: int = 0
    passive_sends: int = 0
    next_non_passive_ticket: int = 0
    serving_non_passive_ticket: int = 0
    cancelled_non_passive_tickets: set[int] = field(default_factory=lambda: set[int]())
    sending: bool = False


@dataclass
class _InboundOwner:
    """在 durable cleanup 确认前保持 mobile handoff 的强引用。"""

    item: InboundItem
    cleanup_pending: bool = False


class _ChannelBindingOwner(Protocol):
    @property
    def snapshot_id(self) -> str: ...

    @property
    def generation_id(self) -> str: ...

    @property
    def channel_name(self) -> str: ...

    @property
    def binding_token(self) -> str: ...

    @property
    def active(self) -> bool: ...


@dataclass
class _AwaitedChannelOutbound:
    """Retain an exact binding owner until a tri-state provider receipt settles."""

    envelope: OutboundEnvelope
    binding: _ChannelBindingOwner
    receipt: "asyncio.Future[ChannelDeliveryReceipt]"
    passive: bool
    before_provider: Callable[[], None] | None = None
    provider_started: bool = False


class ChatLane:
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], _ChatLaneState] = {}

    def _acquire_state(
        self,
        channel: str,
        chat_id: str,
    ) -> tuple[tuple[str, str], _ChatLaneState]:
        key = (str(channel), str(chat_id))
        state = self._states.get(key)
        if state is None:
            state = _ChatLaneState(condition=asyncio.Condition())
            self._states[key] = state
        state.active_users += 1
        return key, state

    def _release_state(
        self,
        key: tuple[str, str],
        state: _ChatLaneState,
    ) -> None:
        state.active_users -= 1
        if (
            state.active_users
            or state.passive_turns
            or state.passive_sends
            or state.sending
            or state.next_non_passive_ticket != state.serving_non_passive_ticket
            or state.cancelled_non_passive_tickets
        ):
            return
        if self._states.get(key) is state:
            del self._states[key]

    def _skip_cancelled_non_passive(self, state: _ChatLaneState) -> None:
        while state.serving_non_passive_ticket in state.cancelled_non_passive_tickets:
            state.cancelled_non_passive_tickets.remove(state.serving_non_passive_ticket)
            state.serving_non_passive_ticket += 1

    async def mark_passive_pending(self, channel: str, chat_id: str) -> None:
        key, state = self._acquire_state(channel, chat_id)
        try:
            async with state.condition:
                state.passive_turns += 1
                state.condition.notify_all()
        finally:
            self._release_state(key, state)

    async def mark_passive_done(self, channel: str, chat_id: str) -> None:
        key, state = self._acquire_state(channel, chat_id)
        try:
            async with state.condition:
                if state.passive_turns > 0:
                    state.passive_turns -= 1
                state.condition.notify_all()
        finally:
            self._release_state(key, state)

    async def mark_passive_send_pending(self, channel: str, chat_id: str) -> None:
        key, state = self._acquire_state(channel, chat_id)
        try:
            async with state.condition:
                state.passive_sends += 1
                state.condition.notify_all()
        finally:
            self._release_state(key, state)

    async def mark_passive_send_done(self, channel: str, chat_id: str) -> None:
        """回滚尚未开始发送的出站 lane 计数。"""

        key, state = self._acquire_state(channel, chat_id)
        try:
            async with state.condition:
                if state.passive_sends > 0:
                    state.passive_sends -= 1
                state.condition.notify_all()
        finally:
            self._release_state(key, state)

    async def run_passive(
        self,
        channel: str,
        chat_id: str,
        send: Callable[[], Awaitable[_T]],
        *,
        pending_registered: bool = False,
    ) -> _T:
        """Serialize one passive send and preserve its exact pending ownership."""

        # 1. direct caller 自行登记；queued outbound 已在入队时登记。
        key, state = self._acquire_state(channel, chat_id)
        owns_pending = pending_registered
        sending = False
        try:
            try:
                async with state.condition:
                    if pending_registered:
                        if state.passive_sends <= 0:
                            raise RuntimeError("passive send pending 计数失衡")
                    else:
                        state.passive_sends += 1
                        owns_pending = True
                    while state.sending:
                        _ = await state.condition.wait()
                    state.sending = True
                    sending = True
                return await send()
            finally:
                # 2. 取消、发送失败与正常完成都只归还本次调用拥有的计数。
                if owns_pending:
                    async with state.condition:
                        if state.passive_sends <= 0:
                            raise RuntimeError("passive send pending 计数失衡")
                        state.passive_sends -= 1
                        if sending:
                            state.sending = False
                        state.condition.notify_all()
        finally:
            self._release_state(key, state)

    async def run_non_passive(
        self,
        channel: str,
        chat_id: str,
        send: Callable[[], Awaitable[_T]],
    ) -> _T:
        key, state = self._acquire_state(channel, chat_id)
        ticket = -1
        sending = False
        try:
            try:
                async with state.condition:
                    ticket = state.next_non_passive_ticket
                    state.next_non_passive_ticket += 1
                    self._skip_cancelled_non_passive(state)
                    while (
                        state.sending
                        or state.passive_turns > 0
                        or state.passive_sends > 0
                        or ticket != state.serving_non_passive_ticket
                    ):
                        _ = await state.condition.wait()
                        self._skip_cancelled_non_passive(state)
                    state.sending = True
                    sending = True
                return await send()
            finally:
                async with state.condition:
                    if ticket >= 0:
                        if sending:
                            state.serving_non_passive_ticket += 1
                            state.sending = False
                        else:
                            state.cancelled_non_passive_tickets.add(ticket)
                        self._skip_cancelled_non_passive(state)
                    state.condition.notify_all()
        finally:
            self._release_state(key, state)


class MessageBus:
    """在单用户 Companion 内传递消息，并持有 mobile handoff 的删除责任。"""

    def __init__(self, chat_lane: ChatLane | None = None) -> None:
        self._inbound: asyncio.Queue[InboundItem | InboundEnvelope] = asyncio.Queue()
        self._outbound: asyncio.Queue[_AwaitedChannelOutbound] = asyncio.Queue()
        self._inbound_accepted: dict[int, _InboundOwner] = {}
        self._inbound_cleanup_tasks: dict[int, asyncio.Task[None]] = {}
        self._inbound_cleanup_error: BaseException | None = None
        self._recovery_claimed: set[str] = set()
        self._mobile_v3_handoffs: dict[int, str] = {}
        self._mobile_v3_admissions: dict[str, _MobileV3Admission] = {}
        self._mobile_session_admission_owner: MobileSessionAdmissionOwner | None = None
        self._mobile_inbound_recoverer: MobileInboundRecoverer | None = None
        self._durable_handoff_lock = asyncio.Lock()
        self._chat_lane = chat_lane or ChatLane()
        self._running = False
        self._outbound_dispatch_stopped = False
        self._outbound_dispatch_task: asyncio.Task[None] | None = None
        self._outbound_closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._durable_inbound_store: DurableInboundStore | None = None
        self._pending_channel_receipts: set[
            asyncio.Future[ChannelDeliveryReceipt]
        ] = set()
        self._channel_outbound_dispatcher: (
            Callable[
                [OutboundEnvelope, _ChannelBindingOwner],
                Awaitable[ChannelDeliveryReceipt],
            ]
            | None
        ) = None

    def bind_durable_inbound_store(self, store: DurableInboundStore) -> None:
        """在 channel 启动前绑定一次由 session 持有的 handoff store。"""

        if self._durable_inbound_store is not None:
            raise RuntimeError("durable inbound store 已绑定")
        self._durable_inbound_store = store

    def bind_mobile_session_admission_owner(
        self,
        owner: MobileSessionAdmissionOwner,
    ) -> None:
        """Bind the Session owner retained by every durable Mobile handoff."""

        if self._mobile_session_admission_owner is not None:
            raise RuntimeError("mobile session admission owner 已绑定")
        self._mobile_session_admission_owner = owner

    def bind_mobile_channel_inbound_recoverer(
        self,
        recoverer: MobileInboundRecoverer,
    ) -> None:
        """Bind the current formal Mobile ingress used after a process restart."""

        if not callable(recoverer):
            raise TypeError("mobile v3 inbound recoverer 必须可调用")
        if self._mobile_inbound_recoverer is not None:
            raise RuntimeError("mobile v3 inbound recoverer 已绑定")
        self._mobile_inbound_recoverer = recoverer

    async def recover_durable_inbounds(self) -> None:
        """分页重放尚未完成的移动 handoff，不以 bus 容量拒绝消息。

        整页读取与 live publish 在同一 durable lock 内串行：live reserve 落库
        与 accepted owner 登记之间不存在可被恢复页观察到的窗口，同一 handoff
        不会被复制成第二个 owner。
        """

        self._raise_inbound_cleanup_error()
        store = self._durable_inbound_store
        if store is None:
            return
        exact_rows: list[tuple[str, RawInbound]] = []
        async with self._durable_handoff_lock:
            # 1. 只读取有限页，避免启动时把整个 durable backlog 搬入内存。
            rows = store.list_inbound_handoffs(
                limit=_DURABLE_INBOUND_RECOVERY_PAGE_SIZE
            )
            # 2. 仍在处理中的 live owner 不得被分页重放复制成第二个 owner。
            in_flight: set[str] = set()
            for owner in self._inbound_accepted.values():
                item = owner.item
                if isinstance(item, InboundMessage) and item.handoff_id is not None:
                    in_flight.add(item.handoff_id)
            in_flight.update(self._mobile_v3_handoffs.values())
            in_flight.update(
                handoff_id
                for handoff_id, admission in self._mobile_v3_admissions.items()
                if not admission.recoverable
            )
            for row in rows:
                handoff_id = row.get("handoff_id")
                if (
                    not isinstance(handoff_id, str)
                    or handoff_id in self._recovery_claimed
                    or handoff_id in in_flight
                ):
                    continue
                item = _inbound_from_handoff(row)
                self._recovery_claimed.add(handoff_id)
                raw = _raw_mobile_from_handoff(row)
                if raw is not None:
                    exact_rows.append((handoff_id, raw))
                    continue
                try:
                    await self._reserve_and_queue_mobile(
                        item, allow_existing_handoff=True
                    )
                except BaseException:
                    self._recovery_claimed.discard(handoff_id)
                    raise
        recoverer = self._mobile_inbound_recoverer
        for handoff_id, raw in exact_rows:
            if recoverer is None:
                raise RuntimeError("v3 Mobile inbound recovery port 未绑定")
            try:
                accepted = await recoverer(raw)
            except BaseException:
                self._recovery_claimed.discard(handoff_id)
                raise
            if not accepted:
                self._recovery_claimed.discard(handoff_id)
                raise RuntimeError("v3 Mobile inbound recovery 被 current binding 拒绝")

    async def reserve_mobile_channel_handoff(self, raw: RawInbound) -> bool:
        """Reserve the Mobile saga before attachment publication can become visible."""

        if raw.message.channel != "mobile":
            raise ValueError("mobile handoff reserve 只接受 Mobile RawInbound")
        async with self._durable_handoff_lock:
            if self._outbound_closed:
                raise RuntimeError("message bus 已关闭")
            handoff_id, session_key = self._mobile_v3_identity(
                raw.message_id, raw.message
            )
            _, acquired = self._ensure_mobile_v3_admission(
                handoff_id,
                session_key,
            )
            try:
                persisted_id, created = self._reserve_mobile_v3_handoff(
                    raw.message_id,
                    raw.message,
                )
            except BaseException:
                if acquired:
                    self._release_new_mobile_v3_admission(handoff_id)
                raise
            if not created and persisted_id != handoff_id:
                if acquired:
                    self._release_new_mobile_v3_admission(handoff_id)
                return False
            return True

    async def defer_mobile_channel_handoff(self, handoff_id: str) -> None:
        """Expose a failed provisional saga to same-process exact recovery."""

        task = asyncio.create_task(
            self._defer_mobile_channel_handoff(handoff_id),
            name=f"mobile-v3-defer:{handoff_id}",
        )
        await _await_cleanup_after_cancellation(task)

    async def _defer_mobile_channel_handoff(self, handoff_id: str) -> None:
        async with self._durable_handoff_lock:
            admission = self._mobile_v3_admissions.get(handoff_id)
            if admission is None:
                return
            if admission.envelope is not None:
                return
            admission.recoverable = True
            self._recovery_claimed.discard(handoff_id)

    def has_pending_mobile_handoff(
        self,
        *,
        session_key: str,
        client_message_id: str,
    ) -> bool:
        """检查移动命令是否仍由 durable queue owner 持有。"""

        store = self._durable_inbound_store
        return bool(
            store is not None
            and store.has_inbound_handoff(
                session_key=session_key,
                client_message_id=client_message_id,
            )
        )

    def bind_channel_outbound_dispatcher(
        self,
        callback: Callable[
            [OutboundEnvelope, _ChannelBindingOwner],
            Awaitable[ChannelDeliveryReceipt],
        ],
    ) -> None:
        """Bind the sole exact-binding v3 Channel delivery owner."""

        if not callable(callback):
            raise TypeError("v3 Channel outbound dispatcher 必须可调用")
        if self._channel_outbound_dispatcher is not None:
            raise RuntimeError("v3 Channel outbound dispatcher 已绑定")
        self._channel_outbound_dispatcher = callback

    async def publish_inbound(self, msg: InboundItem) -> None:
        """将渠道输入交给 Agent 消费。"""
        self._raise_inbound_cleanup_error()
        await self._publish_inbound(msg, allow_existing_handoff=False)

    async def publish_channel_inbound(self, envelope: InboundEnvelope) -> None:
        """Accept one exact Channel envelope into the Bus-owned queue."""

        self._raise_inbound_cleanup_error()
        if self._outbound_closed:
            await envelope.close(InboundOwner.INGRESS)
            raise RuntimeError("message bus 已关闭")
        if (
            envelope.owner is not InboundOwner.INGRESS
            or envelope.state is not InboundState.ADMITTED
        ):
            raise RuntimeError("v3 Channel inbound 必须由 INGRESS/ADMITTED 交给 Bus")
        if envelope.channel == "mobile":
            await self._reserve_and_queue_mobile_channel(envelope)
            return
        await self._chat_lane.mark_passive_pending(
            envelope.channel,
            envelope.chat_id,
        )
        if self._outbound_closed:
            await self._chat_lane.mark_passive_done(
                envelope.channel,
                envelope.chat_id,
            )
            await envelope.close(InboundOwner.INGRESS)
            raise RuntimeError("message bus 已关闭")
        envelope.handoff(InboundOwner.INGRESS, InboundOwner.BUS)
        try:
            self._inbound.put_nowait(envelope)
        except BaseException:
            await envelope.close(InboundOwner.BUS)
            await self._chat_lane.mark_passive_done(
                envelope.channel,
                envelope.chat_id,
            )
            raise

    async def _reserve_and_queue_mobile_channel(
        self,
        envelope: InboundEnvelope,
    ) -> None:
        """Durably reserve a v3 Mobile envelope before the Bus owns its lease."""

        metadata = dict(envelope.metadata)
        handoff_id = metadata.get(_MOBILE_V3_HANDOFF_ID)
        client_message_id = metadata.get("client_message_id")
        if (
            metadata.get(_MOBILE_V3_HANDOFF) is not True
            or not isinstance(handoff_id, str)
            or not handoff_id
            or not isinstance(client_message_id, str)
            or not client_message_id
        ):
            await envelope.close(InboundOwner.INGRESS)
            raise RuntimeError("v3 Mobile inbound 缺少 durable handoff identity")
        async with self._durable_handoff_lock:
            if self._outbound_closed:
                await envelope.close(InboundOwner.INGRESS)
                raise RuntimeError("message bus 已关闭")
            _, acquired = self._ensure_mobile_v3_admission(
                handoff_id,
                envelope.session_key,
            )
            try:
                persisted_id, created = self._reserve_mobile_v3_handoff(
                    envelope.message_id,
                    envelope.message,
                )
            except BaseException:
                if acquired:
                    self._release_new_mobile_v3_admission(handoff_id)
                await envelope.close(InboundOwner.INGRESS)
                raise
            is_recovery = handoff_id in self._recovery_claimed
            is_prepared = (
                persisted_id == handoff_id
                and handoff_id in self._mobile_v3_admissions
                and self._mobile_v3_admissions[handoff_id].envelope is None
            )
            if not created and not (
                (is_recovery or is_prepared) and persisted_id == handoff_id
            ):
                if acquired:
                    self._release_new_mobile_v3_admission(handoff_id)
                await envelope.close(InboundOwner.INGRESS)
                return
            if persisted_id != handoff_id:
                if acquired:
                    self._release_new_mobile_v3_admission(handoff_id)
                await envelope.close(InboundOwner.INGRESS)
                raise RuntimeError("v3 Mobile inbound durable handoff identity 漂移")
            admission = self._mobile_v3_admissions[handoff_id]
            lane_pending = False
            try:
                await self._chat_lane.mark_passive_pending(
                    envelope.channel,
                    envelope.chat_id,
                )
                lane_pending = True
                if self._outbound_closed:
                    raise RuntimeError("message bus 已关闭")
                envelope.handoff(InboundOwner.INGRESS, InboundOwner.BUS)
                self._inbound.put_nowait(envelope)
            except BaseException:
                admission.recoverable = True
                if lane_pending:
                    await self._chat_lane.mark_passive_done(
                        envelope.channel,
                        envelope.chat_id,
                    )
                if envelope.owner in {InboundOwner.INGRESS, InboundOwner.BUS}:
                    await envelope.close(envelope.owner)
                raise
            admission.envelope = envelope
            admission.recoverable = False
            self._mobile_v3_handoffs[id(envelope)] = handoff_id

    def _reserve_mobile_v3_handoff(
        self,
        message_id: str,
        message: ChannelInboundMessage,
    ) -> tuple[str, bool]:
        """Persist one exact Mobile identity while the durable lock is held."""

        metadata = dict(message.metadata)
        handoff_id = metadata.get(_MOBILE_V3_HANDOFF_ID)
        client_message_id = metadata.get("client_message_id")
        session_key = metadata.get("session_key_override")
        if (
            metadata.get(_MOBILE_V3_HANDOFF) is not True
            or not isinstance(handoff_id, str)
            or not handoff_id
            or client_message_id != message_id
            or not isinstance(session_key, str)
            or not session_key.strip()
        ):
            raise RuntimeError("v3 Mobile inbound 缺少 durable handoff identity")
        store = self._durable_inbound_store
        if store is None:
            raise RuntimeError("mobile inbound durable handoff store 未绑定")
        persisted_metadata: dict[str, object] = dict(metadata)
        persisted_metadata[_MOBILE_V3_ATTACHMENT_REFS] = [
            {
                "artifact_id": ref.artifact_id,
                "kind": ref.kind.value,
                "filename": ref.filename,
                "media_type": ref.media_type,
                "size_bytes": ref.size_bytes,
                "sha256": ref.sha256,
            }
            for ref in message.attachments
        ]
        return store.reserve_inbound_handoff(
            handoff_id=handoff_id,
            dedupe_key=f"{session_key.strip()}:{client_message_id}",
            channel=message.channel,
            sender=message.sender,
            chat_id=message.chat_id,
            session_key=session_key.strip(),
            content=message.content,
            timestamp=message.timestamp.astimezone(timezone.utc).isoformat(),
            media_json="[]",
            metadata_json=json.dumps(
                persisted_metadata,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _mobile_v3_identity(
        message_id: str,
        message: ChannelInboundMessage,
    ) -> tuple[str, str]:
        metadata = message.metadata
        handoff_id = metadata.get(_MOBILE_V3_HANDOFF_ID)
        session_key = metadata.get("session_key_override")
        if (
            metadata.get(_MOBILE_V3_HANDOFF) is not True
            or metadata.get("client_message_id") != message_id
            or not isinstance(handoff_id, str)
            or not handoff_id
            or not isinstance(session_key, str)
            or not session_key.strip()
        ):
            raise RuntimeError("v3 Mobile inbound 缺少 durable handoff identity")
        return handoff_id, session_key.strip()

    def _ensure_mobile_v3_admission(
        self,
        handoff_id: str,
        session_key: str,
    ) -> tuple[_MobileV3Admission, bool]:
        """Acquire or reuse the Session admission owned by a durable handoff."""

        existing = self._mobile_v3_admissions.get(handoff_id)
        if existing is not None:
            return existing, False
        owner = self._mobile_session_admission_owner
        if owner is None:
            raise RuntimeError("mobile session admission owner 未绑定")
        _, admission_id = owner.admit_existing(session_key)
        admission = _MobileV3Admission(admission_id=admission_id)
        self._mobile_v3_admissions[handoff_id] = admission
        return admission, True

    def _release_new_mobile_v3_admission(self, handoff_id: str) -> None:
        """Release an admission whose durable reserve never became an owner."""

        admission = self._mobile_v3_admissions.pop(handoff_id)
        if admission.envelope is not None:
            raise RuntimeError("new Mobile admission 已绑定 envelope")
        owner = self._mobile_session_admission_owner
        if owner is None:
            raise RuntimeError("mobile session admission owner 未绑定")
        owner.release_admission(admission.admission_id)

    def mobile_session_admission_id(self, envelope: InboundEnvelope) -> str:
        """Return the Bus-owned admission already retained for this envelope."""

        handoff_id = self._mobile_v3_handoffs.get(id(envelope))
        if handoff_id is None:
            raise RuntimeError("Mobile exact envelope 缺少 Bus handoff owner")
        admission = self._mobile_v3_admissions.get(handoff_id)
        if admission is None or admission.envelope is not envelope:
            raise RuntimeError("Mobile exact envelope 缺少 Session admission")
        return admission.admission_id

    def mobile_inbound_cleanup_pending(self, envelope: InboundEnvelope) -> bool:
        """Report the cleanup-only owner that must not be converted to recovery."""

        handoff_id = self._mobile_v3_handoffs.get(id(envelope))
        if handoff_id is None:
            return False
        admission = self._mobile_v3_admissions.get(handoff_id)
        return bool(
            admission is not None
            and admission.envelope is envelope
            and admission.cleanup_pending
        )

    async def _publish_inbound(
        self,
        msg: InboundItem,
        *,
        allow_existing_handoff: bool,
    ) -> None:
        """将消息入队；mobile 先持久化并由本类负责删除确认。"""

        if not isinstance(msg, InboundMessage) or msg.channel != "mobile":
            await self._chat_lane.mark_passive_pending(msg.channel, msg.chat_id)
            try:
                self._inbound.put_nowait(msg)
            except BaseException:
                await self._chat_lane.mark_passive_done(msg.channel, msg.chat_id)
                raise
            return

        async with self._durable_handoff_lock:
            await self._reserve_and_queue_mobile(
                msg, allow_existing_handoff=allow_existing_handoff
            )

    async def _reserve_and_queue_mobile(
        self,
        msg: InboundMessage,
        *,
        allow_existing_handoff: bool,
    ) -> None:
        """在 durable lock 内完成 mobile reserve + queue + owner 的唯一登记。

        调用方必须已持有 _durable_handoff_lock：live publish 与整页恢复共享
        这一个登记点，保证同一 handoff 至多产生一个 queue item 和一个
        accepted owner。
        """

        if id(msg) in self._inbound_accepted:
            raise RuntimeError("同一 mobile inbound 对象被重复接受")
        store = self._durable_inbound_store
        if store is None:
            raise RuntimeError("mobile inbound durable handoff store 未绑定")
        requested_handoff_id = msg.handoff_id
        media_json, metadata_json = _serialize_handoff(msg)
        handoff_id, created = store.reserve_inbound_handoff(
            handoff_id=msg.handoff_id or uuid4().hex,
            dedupe_key=_mobile_dedupe_key(msg),
            channel=msg.channel,
            sender=msg.sender,
            chat_id=msg.chat_id,
            session_key=msg.session_key,
            content=msg.content,
            timestamp=msg.timestamp.astimezone(timezone.utc).isoformat(),
            media_json=media_json,
            metadata_json=metadata_json,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        msg.handoff_id = handoff_id
        if not created and not (
            allow_existing_handoff and requested_handoff_id == handoff_id
        ):
            return
        await self._chat_lane.mark_passive_pending(msg.channel, msg.chat_id)
        try:
            self._inbound.put_nowait(msg)
        except BaseException:
            await self._chat_lane.mark_passive_done(msg.channel, msg.chat_id)
            raise
        self._inbound_accepted[id(msg)] = _InboundOwner(item=msg)

    async def consume_inbound(self) -> InboundItem | InboundEnvelope:
        """Transfer one queued Channel envelope to the lane owner."""

        item = await self._inbound.get()
        if isinstance(item, InboundEnvelope):
            item.handoff(InboundOwner.BUS, InboundOwner.LANE)
        return item

    async def complete_inbound(self, msg: InboundItem | InboundEnvelope) -> None:
        self._raise_inbound_cleanup_error()
        if isinstance(msg, InboundEnvelope):
            handoff_id = self._mobile_v3_handoffs.get(id(msg))
            if handoff_id is not None:
                task = asyncio.create_task(
                    self._complete_mobile_v3_inbound(msg, handoff_id),
                    name=f"mobile-v3-complete:{handoff_id}",
                )
                await _await_cleanup_after_cancellation(task)
                return
            await self.release_channel_inbound(msg, InboundOwner.LOOP)
            return
        owner = self._inbound_accepted.get(id(msg))
        if owner is None:
            await self._chat_lane.mark_passive_done(msg.channel, msg.chat_id)
            return
        if owner.item is not msg:
            raise RuntimeError("mobile inbound ownership changed")
        if owner.cleanup_pending:
            raise RuntimeError("inbound cleanup 已在重试中")
        if isinstance(msg, InboundMessage) and msg.handoff_id is not None:
            store = self._durable_inbound_store
            if store is None:
                raise RuntimeError("mobile inbound durable handoff store 未绑定")
            try:
                store.complete_inbound_handoff(msg.handoff_id)
            except OSError as error:
                logger.error(
                    "message_bus cleanup_degraded: retained inbound owner "
                    "handoff=%s error=%s",
                    msg.handoff_id,
                    error,
                )
                owner.cleanup_pending = True
                self._schedule_inbound_cleanup_retry(id(msg))
                raise
            except Exception as error:
                self._record_inbound_cleanup_fatal(error, id(msg))
                raise
        await self._finalize_inbound_owner(id(msg), owner)

    async def _complete_mobile_v3_inbound(
        self,
        envelope: InboundEnvelope,
        handoff_id: str,
    ) -> None:
        """Delete the durable row before releasing its exact binding and Session owner."""

        async with self._durable_handoff_lock:
            admission = self._mobile_v3_admissions.get(handoff_id)
            if admission is None or admission.envelope is not envelope:
                raise RuntimeError("Mobile exact completion owner 丢失")
            if admission.cleanup_pending:
                raise RuntimeError("Mobile exact cleanup 已在重试中")
            store = self._durable_inbound_store
            if store is None:
                raise RuntimeError("mobile inbound durable handoff store 未绑定")
            try:
                store.complete_inbound_handoff(handoff_id)
            except OSError as error:
                logger.error(
                    "message_bus cleanup_degraded: retained exact Mobile owner "
                    "handoff=%s error=%s",
                    handoff_id,
                    error,
                )
                admission.cleanup_pending = True
                self._schedule_inbound_cleanup_retry(id(envelope))
                raise
            except Exception as error:
                self._record_inbound_cleanup_fatal(error, id(envelope))
                raise
            await self._finalize_mobile_v3_owner_locked(
                envelope,
                handoff_id,
                admission,
            )

    async def retain_mobile_channel_inbound(
        self,
        envelope: InboundEnvelope,
        expected_owner: InboundOwner,
    ) -> None:
        """Release a failed exact binding while retaining durable Session recovery owner."""

        task = asyncio.create_task(
            self._retain_mobile_channel_inbound(envelope, expected_owner),
            name=f"mobile-v3-retain:{envelope.message_id}",
        )
        await _await_cleanup_after_cancellation(task)

    async def _retain_mobile_channel_inbound(
        self,
        envelope: InboundEnvelope,
        expected_owner: InboundOwner,
    ) -> None:
        async with self._durable_handoff_lock:
            handoff_id = self._mobile_v3_handoffs.get(id(envelope))
            if handoff_id is None:
                raise RuntimeError("Mobile exact recovery owner 丢失")
            admission = self._mobile_v3_admissions.get(handoff_id)
            if admission is None or admission.envelope is not envelope:
                raise RuntimeError("Mobile exact Session recovery owner 丢失")
            await self._release_channel_inbound(envelope, expected_owner)
            admission.envelope = None
            admission.cleanup_pending = False
            admission.recoverable = True
            self._mobile_v3_handoffs.pop(id(envelope), None)
            self._recovery_claimed.discard(handoff_id)

    async def release_channel_inbound(
        self,
        envelope: InboundEnvelope,
        expected_owner: InboundOwner,
    ) -> None:
        """Close one exact inbound lease and release its lane admission."""

        task = asyncio.create_task(
            self._release_channel_inbound(envelope, expected_owner),
            name=f"channel-inbound-release:{envelope.message_id}",
        )
        await _await_cleanup_after_cancellation(task)

    async def _release_channel_inbound(
        self,
        envelope: InboundEnvelope,
        expected_owner: InboundOwner,
    ) -> None:
        await envelope.close(expected_owner)
        await self._chat_lane.mark_passive_done(
            envelope.channel,
            envelope.chat_id,
        )

    def _raise_inbound_cleanup_error(self) -> None:
        error = self._inbound_cleanup_error
        if error is not None:
            raise RuntimeError("message bus inbound cleanup owner failed") from error

    def _record_inbound_cleanup_fatal(
        self,
        error: BaseException,
        owner_key: int,
    ) -> None:
        if self._inbound_cleanup_error is None:
            self._inbound_cleanup_error = error
        logger.exception(
            "message_bus event=runtime_fatal owner=message_bus.inbound_cleanup "
            "owner_key=%s error=%s",
            owner_key,
            error,
        )

    def _schedule_inbound_cleanup_retry(self, owner_key: int) -> None:
        """为 cleanup-only owner 启动唯一的退避重试 task。"""

        existing = self._inbound_cleanup_tasks.get(owner_key)
        if existing is not None and not existing.done():
            raise RuntimeError(f"inbound cleanup retry 已存在: {owner_key}")
        self._inbound_cleanup_tasks[owner_key] = asyncio.create_task(
            self._retry_inbound_cleanup(owner_key),
            name=f"message-bus-cleanup:{owner_key}",
        )

    async def _retry_inbound_cleanup(self, owner_key: int) -> None:
        """只重试 durable handoff 删除，成功后释放原 accepted owner。"""

        delay = _INBOUND_CLEANUP_RETRY_INITIAL_DELAY
        attempt = 0
        try:
            while True:
                await asyncio.sleep(delay)
                async with self._durable_handoff_lock:
                    exact_handoff_id = self._mobile_v3_handoffs.get(owner_key)
                    if exact_handoff_id is not None:
                        exact = self._mobile_v3_admissions.get(exact_handoff_id)
                        if exact is None or exact.envelope is None:
                            raise RuntimeError(
                                f"Mobile exact cleanup owner 丢失: {owner_key}"
                            )
                        if not exact.cleanup_pending:
                            raise RuntimeError(
                                f"Mobile exact cleanup owner 状态非法: {owner_key}"
                            )
                        store = self._durable_inbound_store
                        if store is None:
                            raise RuntimeError(
                                "mobile inbound durable handoff store 未绑定"
                            )
                        try:
                            store.complete_inbound_handoff(exact_handoff_id)
                        except OSError as error:
                            attempt += 1
                            delay = min(
                                _INBOUND_CLEANUP_RETRY_MAX_DELAY,
                                delay * 2,
                            )
                            logger.error(
                                "message_bus cleanup_degraded: retry failed "
                                "handoff=%s attempt=%s next_delay=%.3f error=%s",
                                exact_handoff_id,
                                attempt,
                                delay,
                                error,
                            )
                            continue
                        await self._finalize_mobile_v3_owner_locked(
                            exact.envelope,
                            exact_handoff_id,
                            exact,
                        )
                        return
                owner = self._inbound_accepted.get(owner_key)
                if owner is None:
                    raise RuntimeError(f"inbound cleanup owner 丢失: {owner_key}")
                if not owner.cleanup_pending:
                    raise RuntimeError(f"inbound cleanup owner 状态非法: {owner_key}")
                item = owner.item
                if not isinstance(item, InboundMessage) or item.handoff_id is None:
                    raise RuntimeError(
                        f"cleanup owner 缺少 mobile handoff: {owner_key}"
                    )
                store = self._durable_inbound_store
                if store is None:
                    raise RuntimeError("mobile inbound durable handoff store 未绑定")
                try:
                    store.complete_inbound_handoff(item.handoff_id)
                except OSError as error:
                    attempt += 1
                    delay = min(_INBOUND_CLEANUP_RETRY_MAX_DELAY, delay * 2)
                    logger.error(
                        "message_bus cleanup_degraded: retry failed "
                        "handoff=%s attempt=%s next_delay=%.3f error=%s",
                        item.handoff_id,
                        attempt,
                        delay,
                        error,
                    )
                    continue
                try:
                    await self._finalize_inbound_owner(owner_key, owner)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._record_inbound_cleanup_fatal(error, owner_key)
                return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._record_inbound_cleanup_fatal(error, owner_key)
        finally:
            current = self._inbound_cleanup_tasks.get(owner_key)
            if current is asyncio.current_task():
                self._inbound_cleanup_tasks.pop(owner_key, None)

    async def _finalize_mobile_v3_owner_locked(
        self,
        envelope: InboundEnvelope,
        handoff_id: str,
        admission: _MobileV3Admission,
    ) -> None:
        """Release the exact owner after DELETE while the durable lock is held."""

        if self._mobile_v3_admissions.get(handoff_id) is not admission:
            raise RuntimeError("Mobile exact admission 在完成期间变更")
        if self._mobile_v3_handoffs.get(id(envelope)) != handoff_id:
            raise RuntimeError("Mobile exact handoff 在完成期间变更")
        await self._release_channel_inbound(envelope, InboundOwner.LOOP)
        owner = self._mobile_session_admission_owner
        if owner is None:
            raise RuntimeError("mobile session admission owner 未绑定")
        owner.release_admission(admission.admission_id)
        self._mobile_v3_handoffs.pop(id(envelope))
        self._mobile_v3_admissions.pop(handoff_id)
        self._recovery_claimed.discard(handoff_id)

    async def _finalize_inbound_owner(
        self,
        owner_key: int,
        owner: _InboundOwner,
    ) -> None:
        """确认 lane 完成后释放 mobile handoff owner，并继续分页 pump。"""

        item = owner.item
        await self._chat_lane.mark_passive_done(item.channel, item.chat_id)
        async with self._durable_handoff_lock:
            current = self._inbound_accepted.pop(owner_key, None)
            if current is not owner:
                raise RuntimeError("inbound ownership changed during completion")
        if isinstance(item, InboundMessage) and item.handoff_id is not None:
            self._recovery_claimed.discard(item.handoff_id)
        await self.recover_durable_inbounds()

    async def aclose(self) -> None:
        """停止出站循环、排空未 dispatch 出站项并收束全部 cleanup-only retry task。

        阶段1：只排空尚未 dispatch 的队列项（收束 receipt、回滚 lane pending）；
        阶段2：dispatch 正在处理中的 in-flight 项由 run_passive 自己 finally 释放，
        此处绝不双减；
        阶段3：取消 cleanup-only retry task，并暴露已发生的 cleanup fatal。
        """

        self._outbound_closed = True
        self.stop()
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_all(),
                name="message-bus-close",
            )
        await _await_cleanup_after_cancellation(self._close_task)

    async def _close_all(self) -> None:
        """Complete all terminal Bus cleanup after admission is closed."""

        await self._stop_outbound_dispatcher()
        await self._drain_channel_inbound_queue()
        await self._drain_outbound_queue()
        tasks = tuple(self._inbound_cleanup_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._inbound_cleanup_tasks.clear()
        await self._release_mobile_v3_admissions_for_shutdown()
        self._raise_inbound_cleanup_error()

    async def _stop_outbound_dispatcher(self) -> None:
        """Cancel and drain the sole dispatcher before returning from close."""

        task = self._outbound_dispatch_task
        if task is None:
            return
        if task is asyncio.current_task():
            raise RuntimeError("message bus 不能从 outbound dispatcher 内关闭自身")
        if not task.done():
            task.cancel()
        result = await asyncio.gather(task, return_exceptions=True)
        error = result[0]
        if isinstance(error, BaseException) and not isinstance(
            error,
            asyncio.CancelledError,
        ):
            raise error

    async def _drain_channel_inbound_queue(self) -> None:
        """Close only Bus-owned v3 envelopes without rewriting legacy recovery."""

        retained: list[InboundItem] = []
        while True:
            try:
                item = self._inbound.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, InboundEnvelope):
                if id(item) in self._mobile_v3_handoffs:
                    await self.retain_mobile_channel_inbound(
                        item,
                        InboundOwner.BUS,
                    )
                else:
                    await self.release_channel_inbound(item, InboundOwner.BUS)
            else:
                retained.append(item)
        for item in retained:
            self._inbound.put_nowait(item)

    async def _release_mobile_v3_admissions_for_shutdown(self) -> None:
        """Drop process-local owners while leaving durable rows for the next boot."""

        async with self._durable_handoff_lock:
            owner = self._mobile_session_admission_owner
            if self._mobile_v3_admissions and owner is None:
                raise RuntimeError("mobile session admission owner 未绑定")
            for handoff_id, admission in tuple(self._mobile_v3_admissions.items()):
                envelope = admission.envelope
                if envelope is not None:
                    if envelope.owner is not InboundOwner.CLOSED:
                        await self._release_channel_inbound(envelope, envelope.owner)
                    else:
                        await self._chat_lane.mark_passive_done(
                            envelope.channel,
                            envelope.chat_id,
                        )
                    self._mobile_v3_handoffs.pop(id(envelope), None)
                assert owner is not None
                owner.release_admission(admission.admission_id)
                self._mobile_v3_admissions.pop(handoff_id)
                self._recovery_claimed.discard(handoff_id)

    async def _drain_outbound_queue(self) -> None:
        """排空尚未 dispatch 的出站项：收束其 receipt 并回滚 lane pending 计数。"""

        while True:
            try:
                item = self._outbound.get_nowait()
            except asyncio.QueueEmpty:
                return
            if not isinstance(item, _AwaitedChannelOutbound):
                raise RuntimeError("MessageBus outbound queue 含有未授权 legacy item")
            if not item.receipt.done():
                item.receipt.set_result(
                    _channel_delivery_receipt(
                        item.envelope,
                        ChannelDeliveryStatus.REJECTED,
                        "message bus 已关闭，delivery 尚未执行",
                    )
                )
            self._pending_channel_receipts.discard(item.receipt)
            if item.passive:
                await self._chat_lane.mark_passive_send_done(
                    item.envelope.channel,
                    item.envelope.recipient,
                )

    async def publish_outbound(self, msg: object) -> None:
        """Reject the removed OutboundMessage queue and require an exact envelope."""

        raise RuntimeError(
            "MessageBus legacy publish_outbound 已删除；请使用 committed Channel OutboundEnvelope"
        )

    async def publish_outbound_awaited(self, msg: object) -> bool:
        """Reject the removed bool receipt path instead of silently degrading delivery."""

        raise RuntimeError(
            "MessageBus legacy publish_outbound_awaited 已删除；请使用 exact Channel receipt"
        )

    async def publish_channel_outbound_awaited(
        self,
        envelope: OutboundEnvelope,
        binding: _ChannelBindingOwner,
        *,
        passive: bool = True,
        before_provider: Callable[[], None] | None = None,
    ) -> ChannelDeliveryReceipt:
        """Queue one exact v3 delivery and wait for its non-retryable receipt."""

        _validate_channel_binding_owner(envelope, binding)
        if self._outbound_closed or self._outbound_dispatch_stopped:
            return _channel_delivery_receipt(
                envelope,
                ChannelDeliveryStatus.REJECTED,
                "message bus outbound admission 已关闭",
            )
        future: asyncio.Future[ChannelDeliveryReceipt] = (
            asyncio.get_running_loop().create_future()
        )
        if passive:
            await self._chat_lane.mark_passive_send_pending(
                envelope.channel,
                envelope.recipient,
            )
            if self._outbound_closed or self._outbound_dispatch_stopped:
                await self._chat_lane.mark_passive_send_done(
                    envelope.channel,
                    envelope.recipient,
                )
                return _channel_delivery_receipt(
                    envelope,
                    ChannelDeliveryStatus.REJECTED,
                    "message bus outbound admission 已关闭",
                )
        try:
            self._outbound.put_nowait(
                _AwaitedChannelOutbound(
                    envelope, binding, future, passive, before_provider
                )
            )
        except BaseException:
            if passive:
                await self._chat_lane.mark_passive_send_done(
                    envelope.channel,
                    envelope.recipient,
                )
            raise
        self._pending_channel_receipts.add(future)
        future.add_done_callback(self._pending_channel_receipts.discard)
        return await _await_channel_receipt_after_cancellation(future)

    async def dispatch_outbound(self) -> None:
        """后台任务：把 exact v3 envelope 交给唯一 Channel dispatcher。"""
        if self._outbound_closed:
            return
        current_task = asyncio.current_task()
        active_dispatcher = self._outbound_dispatch_task
        if (
            active_dispatcher is not None
            and active_dispatcher is not current_task
            and not active_dispatcher.done()
        ):
            raise RuntimeError("message bus outbound dispatcher 已在运行")
        self._outbound_dispatch_task = current_task
        self._running = True
        self._outbound_dispatch_stopped = False
        in_flight_channel: _AwaitedChannelOutbound | None = None
        try:
            while self._running:
                try:
                    item = await asyncio.wait_for(self._outbound.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if self._outbound_closed:
                    await self._reject_outbound_after_close(item)
                    break
                if isinstance(item, _AwaitedChannelOutbound):
                    channel_item = item
                    in_flight_channel = channel_item
                    if channel_item.passive:
                        channel_receipt = await self._chat_lane.run_passive(
                            channel_item.envelope.channel,
                            channel_item.envelope.recipient,
                            lambda: self._send_channel_outbound(channel_item),
                            pending_registered=True,
                        )
                    else:
                        channel_receipt = await self._chat_lane.run_non_passive(
                            channel_item.envelope.channel,
                            channel_item.envelope.recipient,
                            lambda: self._send_channel_outbound(channel_item),
                        )
                    if not channel_item.receipt.done():
                        channel_item.receipt.set_result(channel_receipt)
                    in_flight_channel = None
                    continue
                raise RuntimeError("MessageBus outbound queue 含有未授权 legacy item")
        finally:
            self._running = False
            self._outbound_dispatch_stopped = True
            if in_flight_channel is not None and not in_flight_channel.receipt.done():
                in_flight_channel.receipt.set_result(
                    _channel_delivery_receipt(
                        in_flight_channel.envelope,
                        (
                            ChannelDeliveryStatus.UNKNOWN
                            if in_flight_channel.provider_started
                            else ChannelDeliveryStatus.REJECTED
                        ),
                        (
                            "message bus dispatch 在 provider receipt 前停止"
                            if in_flight_channel.provider_started
                            else "message bus 已关闭，delivery 尚未执行"
                        ),
                    )
                )
            if self._outbound_dispatch_task is current_task:
                self._outbound_dispatch_task = None

    async def _send_channel_outbound(
        self,
        item: _AwaitedChannelOutbound,
    ) -> ChannelDeliveryReceipt:
        """Invoke a v3 provider exactly once and preserve after-effect uncertainty."""

        dispatcher = self._channel_outbound_dispatcher
        if self._outbound_closed:
            return _channel_delivery_receipt(
                item.envelope,
                ChannelDeliveryStatus.REJECTED,
                "message bus outbound admission 已关闭",
            )
        if dispatcher is None:
            raise RuntimeError("v3 Channel outbound dispatcher 未绑定")
        if item.before_provider is not None:
            try:
                item.before_provider()
            except BaseException as error:
                logger.error(
                    "v3 channel pre-provider commit rejected channel=%s "
                    "delivery_id=%s error=%s",
                    item.envelope.channel,
                    item.envelope.delivery_id,
                    error,
                )
                return _channel_delivery_receipt(
                    item.envelope,
                    ChannelDeliveryStatus.REJECTED,
                    str(error) or type(error).__name__,
                )
        item.provider_started = True
        try:
            receipt = await dispatcher(item.envelope, item.binding)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            logger.error(
                "v3 channel delivery unknown channel=%s delivery_id=%s error=%s",
                item.envelope.channel,
                item.envelope.delivery_id,
                error,
            )
            return _channel_delivery_receipt(
                item.envelope,
                ChannelDeliveryStatus.UNKNOWN,
                str(error) or type(error).__name__,
            )
        if not isinstance(receipt, ChannelDeliveryReceipt):
            raise TypeError("v3 Channel dispatcher 必须返回 ChannelDeliveryReceipt")
        if receipt.delivery_id != item.envelope.delivery_id:
            raise RuntimeError("v3 Channel receipt delivery_id 不匹配")
        return receipt

    async def _reject_outbound_after_close(
        self,
        item: _AwaitedChannelOutbound,
    ) -> None:
        """Settle one item dequeued concurrently with terminal Bus close."""

        if not isinstance(item, _AwaitedChannelOutbound):
            raise RuntimeError("MessageBus outbound queue 含有未授权 legacy item")
        if not item.receipt.done():
            item.receipt.set_result(
                _channel_delivery_receipt(
                    item.envelope,
                    ChannelDeliveryStatus.REJECTED,
                    "message bus outbound admission 已关闭",
                )
            )
        if item.passive:
            await self._chat_lane.mark_passive_send_done(
                item.envelope.channel,
                item.envelope.recipient,
            )

    def stop(self) -> None:
        self._running = False

    @property
    def chat_lane(self) -> ChatLane:
        return self._chat_lane

    @property
    def inbound_size(self) -> int:
        return self._inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self._outbound.qsize()


def _validate_channel_binding_owner(
    envelope: OutboundEnvelope,
    binding: _ChannelBindingOwner,
) -> None:
    """Fence one outbound attempt to its exact live snapshot binding."""

    if not binding.active:
        raise RuntimeError("v3 Channel binding lease 已关闭")
    if (
        binding.snapshot_id != envelope.snapshot_id
        or binding.generation_id != envelope.generation_id
        or binding.channel_name != envelope.channel
        or binding.binding_token != envelope.binding_token
    ):
        raise RuntimeError("OutboundEnvelope 与 exact Channel binding 不一致")


def _channel_delivery_receipt(
    envelope: OutboundEnvelope,
    status: ChannelDeliveryStatus,
    error: str,
) -> ChannelDeliveryReceipt:
    return ChannelDeliveryReceipt(
        delivery_id=envelope.delivery_id,
        status=status,
        error=error,
    )


async def _await_channel_receipt_after_cancellation(
    future: asyncio.Future[ChannelDeliveryReceipt],
) -> ChannelDeliveryReceipt:
    """Wait for provider settlement before restoring caller cancellation."""

    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled = True
    receipt = future.result()
    if cancelled:
        raise asyncio.CancelledError
    return receipt


async def _await_cleanup_after_cancellation(task: asyncio.Task[None]) -> None:
    """Finish terminal cleanup before restoring caller cancellation."""

    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError
