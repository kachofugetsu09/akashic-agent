import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, TypeVar, cast
from uuid import uuid4

from agent.plugin_contracts import json_value
from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentRef,
    ChannelInboundMessage,
    DURABLE_ATTACHMENT_REFS,
    DURABLE_HANDOFF_ID,
    DURABLE_INBOUND_MARKER,
    DURABLE_PROVIDER_MESSAGE_ID,
    InboundEnvelope,
    InboundOwner,
    InboundState,
    JsonValue,
    RawInbound,
)
from bus.events import InboundItem, InboundMessage
from session.inbound_store import (
    add_handoff_provider_identity,
    read_handoff_provider_identity,
    strip_handoff_provider_identity,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_DURABLE_INBOUND_RECOVERY_PAGE_SIZE = 256
_INBOUND_CLEANUP_RETRY_INITIAL_DELAY = 0.1
_INBOUND_CLEANUP_RETRY_MAX_DELAY = 5.0
DurableInboundRecoverer = Callable[[RawInbound], Awaitable[bool]]


def _has_durable_handoff(value: object) -> bool:
    """Identify a durable handoff by its neutral marker, not a channel name."""

    metadata = getattr(value, "metadata", None)
    return bool(
        isinstance(metadata, Mapping)
        and metadata.get(DURABLE_INBOUND_MARKER) is True
    ) or (
        isinstance(value, InboundMessage)
        and value.handoff_id is not None
    )


class DurableInboundStore(Protocol):
    """MessageBus 所需的最小 durable handoff 持久 owner 接口。"""

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
        after: tuple[str, str] | None = None,
    ) -> list[dict[str, str | None]]: ...

    def has_inbound_handoff(
        self,
        *,
        channel: str,
        session_key: str,
        provider_message_id: str,
    ) -> bool: ...

    def read_inbound_handoff(
        self,
        *,
        channel: str,
        session_key: str,
        provider_message_id: str,
    ) -> dict[str, str | None] | None: ...

    def complete_inbound_handoff(self, handoff_id: str) -> None: ...


class SessionAdmissionOwner(Protocol):
    """durable handoff 所需的最小 Session admission owner。"""

    def acquire(self, key: str, *, require_existing: bool = True) -> str: ...

    def release_admission(self, admission_id: str) -> None: ...


@dataclass(slots=True)
class _DurableAdmission:
    admission_id: str
    channel: str | None = None
    envelope: InboundEnvelope | None = None
    cleanup_pending: bool = False
    recoverable: bool = False


def _durable_dedupe_key(message: InboundMessage) -> str | None:
    provider_message_id = _provider_message_id(message.metadata)
    if provider_message_id is None:
        return None
    return f"{message.channel}:{message.session_key}:{provider_message_id}"


def _provider_message_id(metadata: Mapping[str, object]) -> str | None:
    """Read the provider identity at the durable boundary."""

    value = metadata.get(DURABLE_PROVIDER_MESSAGE_ID)
    if not isinstance(value, str) or not value:
        return None
    return value


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


def _raw_durable_from_handoff(row: dict[str, str | None]) -> RawInbound | None:
    """Rebuild one marked durable handoff without reviving a legacy lease."""

    legacy = _inbound_from_handoff(row)
    metadata = legacy.metadata
    if metadata.get(DURABLE_INBOUND_MARKER) is not True:
        return None
    handoff_id = row.get("handoff_id")
    requested_handoff_id = metadata.get(DURABLE_HANDOFF_ID)
    provider_message_id = _provider_message_id(metadata)
    refs_json = metadata.get(DURABLE_ATTACHMENT_REFS, [])
    if (
        not isinstance(handoff_id, str)
        or requested_handoff_id != handoff_id
        or provider_message_id is None
        or not isinstance(refs_json, list)
    ):
        raise ValueError("durable inbound handoff identity invalid")
    refs: list[AttachmentRef] = []
    for item in refs_json:
        if not isinstance(item, dict):
            raise ValueError("durable attachment handoff invalid")
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
            raise ValueError("durable attachment handoff invalid")
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
            raise ValueError("durable attachment handoff invalid") from error
    provider_identity = read_handoff_provider_identity(metadata)
    message_metadata = strip_handoff_provider_identity(metadata)
    if provider_identity is None:
        provider_identity_value, recipient_value = legacy.sender, legacy.chat_id
    else:
        provider_identity_value, recipient_value = provider_identity
    return RawInbound(
        message_id=provider_message_id,
        provider_identity=provider_identity_value,
        recipient=recipient_value,
        message=ChannelInboundMessage(
            channel=legacy.channel,
            sender=legacy.sender,
            chat_id=legacy.chat_id,
            content=legacy.content,
            timestamp=legacy.timestamp,
            metadata=cast(Mapping[str, JsonValue], message_metadata),
            attachments=tuple(refs),
        ),
    )


@dataclass
class _ChatLaneState:
    condition: asyncio.Condition
    active_users: int = 0
    passive_turns: int = 0


@dataclass
class _InboundOwner:
    """在 durable cleanup 确认前保持 durable handoff 的强引用。"""

    item: InboundItem
    cleanup_pending: bool = False


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
        ):
            return
        if self._states.get(key) is state:
            del self._states[key]

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


class MessageBus:
    """在单用户 Companion 内传递消息，并持有 durable handoff 的删除责任。"""

    def __init__(self, chat_lane: ChatLane | None = None) -> None:
        self._inbound: asyncio.Queue[InboundItem | InboundEnvelope] = asyncio.Queue()
        self._inbound_accepted: dict[int, _InboundOwner] = {}
        self._inbound_cleanup_tasks: dict[int, asyncio.Task[None]] = {}
        self._inbound_cleanup_error: BaseException | None = None
        self._recovery_claimed: set[str] = set()
        self._durable_handoffs: dict[int, str] = {}
        self._durable_admissions: dict[str, _DurableAdmission] = {}
        self._session_admission_owner: SessionAdmissionOwner | None = None
        self._durable_inbound_recoverer: DurableInboundRecoverer | None = None
        self._durable_handoff_lock = asyncio.Lock()
        self._chat_lane = chat_lane or ChatLane()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._durable_inbound_store: DurableInboundStore | None = None

    def bind_durable_inbound_store(self, store: DurableInboundStore) -> None:
        """在 channel 启动前绑定一次由 session 持有的 handoff store。"""

        if self._durable_inbound_store is not None:
            raise RuntimeError("durable inbound store 已绑定")
        self._durable_inbound_store = store

    def bind_session_admission_owner(
        self,
        owner: SessionAdmissionOwner,
    ) -> None:
        """Bind the Session owner retained by every durable handoff."""

        if self._session_admission_owner is not None:
            raise RuntimeError("durable session admission owner 已绑定")
        self._session_admission_owner = owner

    def bind_durable_inbound_recoverer(
        self,
        recoverer: DurableInboundRecoverer,
    ) -> None:
        """Bind the current formal durable ingress used after a process restart."""

        if not callable(recoverer):
            raise TypeError("durable inbound recoverer 必须可调用")
        if self._durable_inbound_recoverer is not None:
            raise RuntimeError("durable inbound recoverer 已绑定")
        self._durable_inbound_recoverer = recoverer

    async def recover_durable_inbounds(self) -> None:
        """分页重放尚未完成的 durable handoff，不以 bus 容量拒绝消息。

        整页读取与 live publish 在同一 durable lock 内串行：live reserve 落库
        与 accepted owner 登记之间不存在可被恢复页观察到的窗口，同一 handoff
        不会被复制成第二个 owner。
        """

        if self._closed:
            raise RuntimeError("message bus 已关闭")
        self._raise_inbound_cleanup_error()
        store = self._durable_inbound_store
        if store is None:
            return
        after: tuple[str, str] | None = None
        while not self._closed:
            exact_rows: list[tuple[str, RawInbound]] = []
            legacy_page = bool(self._inbound_accepted)
            try:
                async with self._durable_handoff_lock:
                    # 1. 只读取有限页，避免启动时把整个 durable backlog 搬入内存。
                    rows = store.list_inbound_handoffs(
                        limit=_DURABLE_INBOUND_RECOVERY_PAGE_SIZE, after=after,
                    )
                    # 2. 仍在处理中的 live owner 不得被分页重放复制成第二个 owner。
                    in_flight: set[str] = set()
                    for owner in self._inbound_accepted.values():
                        item = owner.item
                        if isinstance(item, InboundMessage) and item.handoff_id is not None:
                            in_flight.add(item.handoff_id)
                    in_flight.update(self._durable_handoffs.values())
                    in_flight.update(
                        handoff_id
                        for handoff_id, admission in self._durable_admissions.items()
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
                        raw = _raw_durable_from_handoff(row)
                        self._recovery_claimed.add(handoff_id)
                        if raw is not None:
                            exact_rows.append((handoff_id, raw))
                            continue
                        legacy_page = True
                        try:
                            await self._reserve_and_queue_durable(
                                item, allow_existing_handoff=True
                            )
                        except BaseException:
                            self._recovery_claimed.discard(handoff_id)
                            raise
                if not rows:
                    return
                last = rows[-1]
                created_at, last_id = last["created_at"], last["handoff_id"]
                if created_at is None or last_id is None:
                    raise RuntimeError("durable handoff 缺少分页身份")
                after = created_at, last_id
                recoverer = self._durable_inbound_recoverer
                for handoff_id, raw in exact_rows:
                    if recoverer is None:
                        raise RuntimeError("durable inbound recovery port 未绑定")
                    if not await recoverer(raw):
                        # No exact current channel owner is a normal partial
                        # catalog state.  Release only this page claim and
                        # leave the durable row for the generation that owns
                        # its persisted channel; other channels can proceed.
                        self._recovery_claimed.discard(handoff_id)
                if legacy_page:
                    # 旧 worker 删除前维持原本逐次完成再 pump 一页的容量边界。
                    return
            finally:
                # 失败后的未执行行必须允许重试；已接纳行由 exact admission 去重。
                self._recovery_claimed.difference_update(row_id for row_id, _ in exact_rows)

    async def reserve_durable_inbound(self, raw: RawInbound) -> bool:
        """Reserve the durable saga before attachment publication can become visible."""

        if not _has_durable_handoff(raw.message):
            raise ValueError("durable handoff reserve 缺少 durable marker")
        async with self._durable_handoff_lock:
            if self._closed:
                raise RuntimeError("message bus 已关闭")
            handoff_id, session_key = self._durable_identity(
                raw.message_id, raw.message
            )
            admission, acquired = self._ensure_durable_admission(
                handoff_id,
                session_key,
                channel=raw.message.channel,
                require_existing=cast(bool, raw.message.metadata.get("require_existing_session", True)),
            )
            if admission.envelope is not None:
                # 一个已被当前 binding 接管的 handoff 不能被第二次 reserve
                # 伪装成新的 owner；调用方保留自己的 duplicate 语义。
                return False
            try:
                persisted_id, created = self._reserve_durable_handoff(raw)
            except BaseException:
                if acquired:
                    self._release_new_durable_admission(handoff_id)
                raise
            if not created and persisted_id != handoff_id:
                if acquired:
                    self._release_new_durable_admission(handoff_id)
                return False
            admission.recoverable = False
            self._recovery_claimed.discard(handoff_id)
            return True

    async def defer_durable_inbound(self, handoff_id: str) -> bool:
        """Expose a failed provisional saga to same-process exact recovery."""

        task = asyncio.create_task(
            self._defer_durable_inbound(handoff_id),
            name=f"durable-inbound-defer:{handoff_id}",
        )
        return await _await_cleanup_after_cancellation(task)

    async def _defer_durable_inbound(self, handoff_id: str) -> bool:
        async with self._durable_handoff_lock:
            if self._closed:
                raise RuntimeError("message bus 已关闭")
            admission = self._durable_admissions.get(handoff_id)
            if admission is None:
                return True
            if admission.envelope is not None:
                return False
            admission.recoverable = True
            self._recovery_claimed.discard(handoff_id)
            return True

    async def settle_rejected_inbound(
        self,
        *,
        channel: str,
        session_key: str,
        provider_message_id: str,
    ) -> None:
        """明确拒绝的命令收据落库后，释放未接纳的交接及 Session 租约。"""
        async with self._durable_handoff_lock:
            if self._closed:
                raise RuntimeError("message bus 已关闭")
            store = self._durable_inbound_store
            if store is None:
                raise RuntimeError("durable inbound durable handoff store 未绑定")
            row = store.read_inbound_handoff(
                channel=channel,
                session_key=session_key,
                provider_message_id=provider_message_id,
            )
            if row is None:
                return
            handoff_id = row["handoff_id"]
            if not isinstance(handoff_id, str):
                raise RuntimeError("durable handoff 缺少身份")
            admission = self._durable_admissions.get(handoff_id)
            if admission is not None and admission.envelope is not None:
                raise RuntimeError("不得结算仍有执行 owner 的 durable 输入")
            # 删除失败保留行和租约，由同一失败收据在恢复时重新结算。
            store.complete_inbound_handoff(handoff_id)
            if admission is not None:
                self._release_new_durable_admission(handoff_id)
            self._recovery_claimed.discard(handoff_id)

    def has_pending_durable_inbound(
        self,
        *,
        channel: str,
        session_key: str,
        provider_message_id: str,
    ) -> bool:
        """检查 provider message 是否仍由 durable queue owner 持有。"""

        store = self._durable_inbound_store
        return bool(
            store is not None
            and store.has_inbound_handoff(
                channel=channel,
                session_key=session_key,
                provider_message_id=provider_message_id,
            )
        )

    def pending_durable_attachment_refs(
        self,
        *,
        channel: str,
        session_key: str,
        provider_message_id: str,
    ) -> tuple[AttachmentRef, ...] | None:
        """读取 durable handoff 冻结的 exact attachment refs。"""

        store = self._durable_inbound_store
        if store is None:
            return None
        row = store.read_inbound_handoff(
            channel=channel,
            session_key=session_key,
            provider_message_id=provider_message_id,
        )
        if row is None:
            return None
        raw = _raw_durable_from_handoff(row)
        if raw is None:
            raise RuntimeError("pending handoff 不是 durable exact handoff")
        return raw.message.attachments

    async def publish_inbound(self, msg: InboundItem) -> None:
        """将渠道输入交给 Agent 消费。"""
        self._raise_inbound_cleanup_error()
        await self._publish_inbound(msg, allow_existing_handoff=False)

    async def prepare_channel_input(self, envelope: InboundEnvelope) -> None:
        """接管耐久 durable handoff；普通输入不排队，也不占用回复 lane。"""
        self._raise_inbound_cleanup_error()
        if self._closed:
            raise RuntimeError("message bus 已关闭")
        if envelope.owner is not InboundOwner.INGRESS or envelope.state is not InboundState.ADMITTED:
            raise RuntimeError("Channel input 必须仍由 INGRESS 持有")
        if _has_durable_handoff(envelope):
            await self._reserve_durable_input(envelope)

    async def complete_channel_input(self, envelope: InboundEnvelope) -> None:
        """Input 已持久提交后结算传输；不等待任何回复或学习消费者。"""
        handoff_id = self._durable_handoffs.get(id(envelope))
        if handoff_id is None:
            await envelope.close(InboundOwner.INGRESS)
        else:
            await self._complete_durable_inbound(envelope, handoff_id)

    async def retain_channel_input(self, envelope: InboundEnvelope) -> None:
        """提交前失败只释放 exact lease，保留已有耐久 handoff 供恢复。"""
        if id(envelope) in self._durable_handoffs:
            await self._retain_durable_inbound(envelope, InboundOwner.INGRESS)
        else:
            await envelope.close(InboundOwner.INGRESS)

    async def _reserve_durable_input(
        self,
        envelope: InboundEnvelope,
    ) -> None:
        """接管一个已经由 durable ingress reserve 的 exact envelope。

        prepare 只消费既有 reservation；它不能因为调用者漏掉 reserve
        就悄悄创建新的持久 handoff。
        """

        metadata = dict(envelope.metadata)
        handoff_id = metadata.get(DURABLE_HANDOFF_ID)
        provider_message_id = _provider_message_id(metadata)
        if (
            metadata.get(DURABLE_INBOUND_MARKER) is not True
            or not isinstance(handoff_id, str)
            or not handoff_id
            or provider_message_id is None
            or metadata.get("session_key_override") != envelope.session_key
        ):
            await envelope.close(InboundOwner.INGRESS)
            raise RuntimeError("durable inbound 缺少 durable handoff identity")
        async with self._durable_handoff_lock:
            if self._closed:
                await envelope.close(InboundOwner.INGRESS)
                raise RuntimeError("message bus 已关闭")
            store = self._durable_inbound_store
            if store is None:
                await envelope.close(InboundOwner.INGRESS)
                raise RuntimeError("durable inbound durable handoff store 未绑定")
            admission = self._durable_admissions.get(handoff_id)
            if admission is None:
                if handoff_id not in self._recovery_claimed:
                    await envelope.close(InboundOwner.INGRESS)
                    raise RuntimeError("durable inbound reservation 未绑定")
                admission, _ = self._ensure_durable_admission(
                    handoff_id,
                    envelope.session_key,
                    channel=envelope.message.channel,
                    require_existing=cast(
                        bool, metadata.get("require_existing_session", True)
                    ),
                )
            elif admission.channel != envelope.message.channel:
                await envelope.close(InboundOwner.INGRESS)
                raise RuntimeError("durable inbound channel ownership 不一致")
            if admission.envelope is not None:
                await envelope.close(InboundOwner.INGRESS)
                raise RuntimeError("durable inbound reservation 已有执行 owner")
            row = store.read_inbound_handoff(
                channel=envelope.message.channel,
                session_key=envelope.session_key,
                provider_message_id=envelope.message_id,
            )
            if row is None or row.get("handoff_id") != handoff_id:
                await envelope.close(InboundOwner.INGRESS)
                raise RuntimeError("durable inbound reservation identity 不匹配")
            admission.envelope = envelope
            admission.recoverable = False
            self._durable_handoffs[id(envelope)] = handoff_id

    def _reserve_durable_handoff(
        self,
        raw: RawInbound,
    ) -> tuple[str, bool]:
        """Persist one exact durable identity while the durable lock is held."""

        message = raw.message
        message_id = raw.message_id
        metadata = dict(message.metadata)
        handoff_id = metadata.get(DURABLE_HANDOFF_ID)
        provider_message_id = _provider_message_id(metadata)
        session_key = metadata.get("session_key_override")
        if (
            metadata.get(DURABLE_INBOUND_MARKER) is not True
            or not isinstance(handoff_id, str)
            or not handoff_id
            or provider_message_id != message_id
            or not isinstance(session_key, str)
            or not session_key.strip()
        ):
            raise RuntimeError("durable inbound 缺少 durable handoff identity")
        store = self._durable_inbound_store
        if store is None:
            raise RuntimeError("durable inbound durable handoff store 未绑定")
        persisted_metadata = add_handoff_provider_identity(
            metadata,
            raw.provider_identity,
            raw.recipient,
        )
        persisted_metadata[DURABLE_PROVIDER_MESSAGE_ID] = message_id
        persisted_metadata[DURABLE_ATTACHMENT_REFS] = [
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
            dedupe_key=f"{message.channel}:{session_key.strip()}:{provider_message_id}",
            channel=message.channel,
            sender=message.sender,
            chat_id=message.chat_id,
            session_key=session_key.strip(),
            content=message.content,
            timestamp=message.timestamp.astimezone(timezone.utc).isoformat(),
            media_json="[]",
            metadata_json=json.dumps(
                json_value(persisted_metadata),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _durable_identity(
        message_id: str,
        message: ChannelInboundMessage,
    ) -> tuple[str, str]:
        metadata = message.metadata
        handoff_id = metadata.get(DURABLE_HANDOFF_ID)
        session_key = metadata.get("session_key_override")
        if (
            metadata.get(DURABLE_INBOUND_MARKER) is not True
            or _provider_message_id(metadata) != message_id
            or not isinstance(handoff_id, str)
            or not handoff_id
            or not isinstance(session_key, str)
            or not session_key.strip()
        ):
            raise RuntimeError("durable inbound 缺少 durable handoff identity")
        return handoff_id, session_key.strip()

    def _ensure_durable_admission(
        self,
        handoff_id: str,
        session_key: str,
        *,
        channel: str,
        require_existing: bool,
    ) -> tuple[_DurableAdmission, bool]:
        """Acquire or reuse the Session admission owned by a durable handoff."""

        existing = self._durable_admissions.get(handoff_id)
        if existing is not None:
            if existing.channel != channel:
                raise RuntimeError("durable inbound channel ownership 不一致")
            return existing, False
        owner = self._session_admission_owner
        if owner is None:
            raise RuntimeError("durable session admission owner 未绑定")
        admission_id = owner.acquire(session_key, require_existing=require_existing)
        admission = _DurableAdmission(admission_id=admission_id, channel=channel)
        self._durable_admissions[handoff_id] = admission
        return admission, True

    def _release_new_durable_admission(self, handoff_id: str) -> None:
        """Release an admission whose durable reserve never became an owner."""

        admission = self._durable_admissions.pop(handoff_id)
        if admission.envelope is not None:
            raise RuntimeError("new durable admission 已绑定 envelope")
        owner = self._session_admission_owner
        if owner is None:
            raise RuntimeError("durable session admission owner 未绑定")
        owner.release_admission(admission.admission_id)

    def durable_session_admission_id(self, envelope: InboundEnvelope) -> str:
        """Return the Bus-owned admission already retained for this envelope."""

        handoff_id = self._durable_handoffs.get(id(envelope))
        if handoff_id is None:
            raise RuntimeError("durable exact envelope 缺少 Bus handoff owner")
        admission = self._durable_admissions.get(handoff_id)
        if admission is None or admission.envelope is not envelope:
            raise RuntimeError("durable exact envelope 缺少 Session admission")
        return admission.admission_id

    def durable_inbound_cleanup_pending(self, envelope: InboundEnvelope) -> bool:
        """Report the cleanup-only owner that must not be converted to recovery."""

        handoff_id = self._durable_handoffs.get(id(envelope))
        if handoff_id is None:
            return False
        admission = self._durable_admissions.get(handoff_id)
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
        """将消息入队；durable 先持久化并由本类负责删除确认。"""

        if not _has_durable_handoff(msg):
            await self._chat_lane.mark_passive_pending(msg.channel, msg.chat_id)
            try:
                self._inbound.put_nowait(msg)
            except BaseException:
                await self._chat_lane.mark_passive_done(msg.channel, msg.chat_id)
                raise
            return

        async with self._durable_handoff_lock:
            await self._reserve_and_queue_durable(
                msg, allow_existing_handoff=allow_existing_handoff
            )

    async def _reserve_and_queue_durable(
        self,
        msg: InboundMessage,
        *,
        allow_existing_handoff: bool,
    ) -> None:
        """在 durable lock 内完成 reserve + queue + owner 的唯一登记。

        调用方必须已持有 _durable_handoff_lock：live publish 与整页恢复共享
        这一个登记点，保证同一 handoff 至多产生一个 queue item 和一个
        accepted owner。
        """

        if id(msg) in self._inbound_accepted:
            raise RuntimeError("同一 durable inbound 对象被重复接受")
        store = self._durable_inbound_store
        if store is None:
            raise RuntimeError("durable inbound store 未绑定")
        requested_handoff_id = msg.handoff_id
        media_json, metadata_json = _serialize_handoff(msg)
        handoff_id, created = store.reserve_inbound_handoff(
            handoff_id=msg.handoff_id or uuid4().hex,
            dedupe_key=_durable_dedupe_key(msg),
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
            handoff_id = self._durable_handoffs.get(id(msg))
            if handoff_id is not None:
                task = asyncio.create_task(
                    self._complete_durable_inbound(msg, handoff_id),
                    name=f"durable-inbound-complete:{handoff_id}",
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
            raise RuntimeError("durable inbound ownership changed")
        if owner.cleanup_pending:
            raise RuntimeError("inbound cleanup 已在重试中")
        if isinstance(msg, InboundMessage) and msg.handoff_id is not None:
            store = self._durable_inbound_store
            if store is None:
                raise RuntimeError("durable inbound durable handoff store 未绑定")
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

    async def _complete_durable_inbound(
        self,
        envelope: InboundEnvelope,
        handoff_id: str,
    ) -> None:
        """Delete the durable row before releasing its exact binding and Session owner."""

        async with self._durable_handoff_lock:
            admission = self._durable_admissions.get(handoff_id)
            if admission is None or admission.envelope is not envelope:
                raise RuntimeError("durable exact completion owner 丢失")
            if admission.cleanup_pending:
                raise RuntimeError("durable exact cleanup 已在重试中")
            store = self._durable_inbound_store
            if store is None:
                raise RuntimeError("durable inbound durable handoff store 未绑定")
            try:
                store.complete_inbound_handoff(handoff_id)
            except OSError as error:
                logger.error(
                    "message_bus cleanup_degraded: retained exact durable owner "
                    "handoff=%s error=%s",
                    handoff_id,
                    error,
                )
                admission.cleanup_pending = True
                self._schedule_inbound_cleanup_retry(id(envelope))
                return
            except Exception as error:
                self._record_inbound_cleanup_fatal(error, id(envelope))
                raise
            await self._finalize_durable_owner_locked(
                envelope,
                handoff_id,
                admission,
            )

    async def retain_durable_inbound(
        self,
        envelope: InboundEnvelope,
        expected_owner: InboundOwner,
    ) -> None:
        """Release a failed exact binding while retaining durable Session recovery owner."""

        task = asyncio.create_task(
            self._retain_durable_inbound(envelope, expected_owner),
            name=f"durable-inbound-retain:{envelope.message_id}",
        )
        await _await_cleanup_after_cancellation(task)

    async def _retain_durable_inbound(
        self,
        envelope: InboundEnvelope,
        expected_owner: InboundOwner,
    ) -> None:
        async with self._durable_handoff_lock:
            handoff_id = self._durable_handoffs.get(id(envelope))
            if handoff_id is None:
                raise RuntimeError("durable exact recovery owner 丢失")
            admission = self._durable_admissions.get(handoff_id)
            if admission is None or admission.envelope is not envelope:
                raise RuntimeError("durable exact Session recovery owner 丢失")
            await envelope.close(expected_owner)
            admission.envelope = None
            admission.cleanup_pending = False
            admission.recoverable = True
            self._durable_handoffs.pop(id(envelope), None)
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
                    exact_handoff_id = self._durable_handoffs.get(owner_key)
                    if exact_handoff_id is not None:
                        exact = self._durable_admissions.get(exact_handoff_id)
                        if exact is None or exact.envelope is None:
                            raise RuntimeError(
                                f"durable exact cleanup owner 丢失: {owner_key}"
                            )
                        if not exact.cleanup_pending:
                            raise RuntimeError(
                                f"durable exact cleanup owner 状态非法: {owner_key}"
                            )
                        store = self._durable_inbound_store
                        if store is None:
                            raise RuntimeError(
                                "durable inbound durable handoff store 未绑定"
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
                        await self._finalize_durable_owner_locked(
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
                        f"cleanup owner 缺少 durable handoff: {owner_key}"
                    )
                store = self._durable_inbound_store
                if store is None:
                    raise RuntimeError("durable inbound durable handoff store 未绑定")
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

    async def _finalize_durable_owner_locked(
        self,
        envelope: InboundEnvelope,
        handoff_id: str,
        admission: _DurableAdmission,
    ) -> None:
        """Release the exact owner after DELETE while the durable lock is held."""

        if self._durable_admissions.get(handoff_id) is not admission:
            raise RuntimeError("durable exact admission 在完成期间变更")
        if self._durable_handoffs.get(id(envelope)) != handoff_id:
            raise RuntimeError("durable exact handoff 在完成期间变更")
        await envelope.close(InboundOwner.INGRESS)
        owner = self._session_admission_owner
        if owner is None:
            raise RuntimeError("durable session admission owner 未绑定")
        owner.release_admission(admission.admission_id)
        self._durable_handoffs.pop(id(envelope))
        self._durable_admissions.pop(handoff_id)
        self._recovery_claimed.discard(handoff_id)

    async def _finalize_inbound_owner(
        self,
        owner_key: int,
        owner: _InboundOwner,
    ) -> None:
        """确认 lane 完成后释放 durable handoff owner，并继续分页 pump。"""

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
        """关闭 Bus 接纳，排空入站 owner 并收束 cleanup-only retry task。"""

        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_all(),
                name="message-bus-close",
            )
        await _await_cleanup_after_cancellation(self._close_task)

    async def _close_all(self) -> None:
        """Complete all terminal Bus cleanup after admission is closed."""

        await self._drain_channel_inbound_queue()
        tasks = tuple(self._inbound_cleanup_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._inbound_cleanup_tasks.clear()
        await self._release_durable_admissions_for_shutdown()
        self._raise_inbound_cleanup_error()

    async def _drain_channel_inbound_queue(self) -> None:
        """Close only Bus-owned v3 envelopes without rewriting legacy recovery."""

        retained: list[InboundItem] = []
        while True:
            try:
                item = self._inbound.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, InboundEnvelope):
                if id(item) in self._durable_handoffs:
                    await self.retain_durable_inbound(
                        item,
                        InboundOwner.BUS,
                    )
                else:
                    await self.release_channel_inbound(item, InboundOwner.BUS)
            else:
                retained.append(item)
        for item in retained:
            self._inbound.put_nowait(item)

    async def _release_durable_admissions_for_shutdown(self) -> None:
        """Drop process-local owners while leaving durable rows for the next boot."""

        async with self._durable_handoff_lock:
            owner = self._session_admission_owner
            if self._durable_admissions and owner is None:
                raise RuntimeError("durable session admission owner 未绑定")
            for handoff_id, admission in tuple(self._durable_admissions.items()):
                envelope = admission.envelope
                if envelope is not None:
                    if envelope.owner is not InboundOwner.CLOSED:
                        await envelope.close(envelope.owner)
                    self._durable_handoffs.pop(id(envelope), None)
                assert owner is not None
                owner.release_admission(admission.admission_id)
                self._durable_admissions.pop(handoff_id)
                self._recovery_claimed.discard(handoff_id)

    @property
    def inbound_size(self) -> int:
        return self._inbound.qsize()


async def _await_cleanup_after_cancellation(task: asyncio.Task[_T]) -> _T:
    """Finish terminal cleanup before restoring caller cancellation."""

    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result
