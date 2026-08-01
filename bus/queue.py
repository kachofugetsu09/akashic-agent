import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeVar
from typing import Protocol, cast
from uuid import uuid4

from bus.events import InboundItem, InboundMessage, OutboundMessage, SpawnCompletionItem

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

DEFAULT_MESSAGE_BUS_CAPACITY = 256
DEFAULT_MESSAGE_BUS_BYTES = 4 * 1024 * 1024
_INBOUND_CLEANUP_RETRY_INITIAL_DELAY = 0.1
_INBOUND_CLEANUP_RETRY_MAX_DELAY = 5.0


class MessageBusCapacityError(RuntimeError):
    """表示消息在进入有界队列前被拒绝。"""

    error_type = "resource-exhausted"
    failure_type = "operation_rejected"
    code = "resource-exhausted"
    retryable = True

    def __init__(self, direction: str, *, item_bytes: int, reason: str) -> None:
        self.direction = direction
        self.item_bytes = item_bytes
        self.reason = reason
        super().__init__(
            f"resource-exhausted: {direction} message bus admission {reason} "
            f"(item_bytes={item_bytes})"
        )


# Keep a descriptive alias for callers that use the transport terminology.
MessageBusBusyError = MessageBusCapacityError


def _wire_size(item: object) -> int:
    """计算消息进入队列后占用的 UTF-8 JSON 字节数。"""

    if isinstance(item, InboundMessage):
        payload = {
            "channel": item.channel,
            "sender": item.sender,
            "chat_id": item.chat_id,
            "content": item.content,
            "timestamp": item.timestamp.isoformat(),
            "media": list(item.media),
            "metadata": dict(item.metadata),
        }
    elif isinstance(item, SpawnCompletionItem):
        payload = {
            "channel": item.channel,
            "chat_id": item.chat_id,
            "event": repr(item.event),
            "decision": repr(item.decision),
            "timestamp": item.timestamp.isoformat(),
        }
    elif isinstance(item, OutboundMessage):
        payload = {
            "channel": item.channel,
            "chat_id": item.chat_id,
            "content": item.content,
            "thinking": item.thinking,
            "reply_to": item.reply_to,
            "media": list(item.media),
            "metadata": dict(item.metadata),
        }
    else:
        raise TypeError(f"unsupported MessageBus item: {type(item).__name__}")
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode())


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
        raise ValueError(f"inbound handoff timestamp missing timezone: {values['handoff_id']}")
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


@dataclass
class _ChatLaneState:
    condition: asyncio.Condition
    active_users: int = 0
    passive_turns: int = 0
    passive_sends: int = 0
    next_non_passive_ticket: int = 0
    serving_non_passive_ticket: int = 0
    cancelled_non_passive_tickets: set[int] = field(
        default_factory=lambda: set[int]()
    )
    sending: bool = False


@dataclass
class _InboundOwner:
    """在 durable cleanup 确认前保持 accepted inbound item 的强引用。"""

    item: InboundItem
    item_bytes: int
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
            state.cancelled_non_passive_tickets.remove(
                state.serving_non_passive_ticket
            )
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
    ) -> _T:
        key, state = self._acquire_state(channel, chat_id)
        try:
            async with state.condition:
                while state.sending:
                    _ = await state.condition.wait()
                state.sending = True
            try:
                return await send()
            finally:
                async with state.condition:
                    if state.passive_sends > 0:
                        state.passive_sends -= 1
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


class OutboundSubscription:
    def __init__(
        self,
        bus: "MessageBus",
        channel: str,
        callback: Callable[[OutboundMessage], Awaitable[None]],
    ) -> None:
        self._bus = bus
        self._channel = channel
        self._callback = callback
        self._active = True

    def close(self) -> None:
        if not self._active:
            return
        self._active = False
        self._bus.unsubscribe_outbound(self._channel, self._callback)


class MessageBus:
    """agent 与各 channel 之间的异步消息总线"""

    def __init__(
        self,
        chat_lane: ChatLane | None = None,
        *,
        inbound_capacity: int = DEFAULT_MESSAGE_BUS_CAPACITY,
        outbound_capacity: int = DEFAULT_MESSAGE_BUS_CAPACITY,
        inbound_bytes: int = DEFAULT_MESSAGE_BUS_BYTES,
        outbound_bytes: int = DEFAULT_MESSAGE_BUS_BYTES,
    ) -> None:
        for name, value in (
            ("inbound_capacity", inbound_capacity),
            ("outbound_capacity", outbound_capacity),
            ("inbound_bytes", inbound_bytes),
            ("outbound_bytes", outbound_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        self._inbound_capacity = inbound_capacity
        self._outbound_capacity = outbound_capacity
        self._inbound_bytes_capacity = inbound_bytes
        self._outbound_bytes_capacity = outbound_bytes
        self._inbound: asyncio.Queue[InboundItem] = asyncio.Queue(maxsize=inbound_capacity)
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=outbound_capacity)
        self._inbound_bytes = 0
        self._inbound_accepted: dict[int, _InboundOwner] = {}
        self._inbound_cleanup_tasks: dict[int, asyncio.Task[None]] = {}
        self._inbound_cleanup_error: BaseException | None = None
        self._recovery_claimed: set[str] = set()
        self._outbound_bytes = 0
        self._inbound_reserved_items = 0
        self._outbound_reserved_items = 0
        self._inbound_reserved_bytes = 0
        self._outbound_reserved_bytes = 0
        self._admission_lock = asyncio.Lock()
        self._subscribers: dict[
            str, list[Callable[[OutboundMessage], Awaitable[None]]]
        ] = {}
        self._chat_lane = chat_lane or ChatLane()
        self._running = False
        self._delivery_observer: (
            Callable[[OutboundMessage, bool], Awaitable[None]] | None
        ) = None
        self._durable_inbound_store: DurableInboundStore | None = None

    def bind_durable_inbound_store(self, store: DurableInboundStore) -> None:
        """在 channel 启动前绑定一次由 session 持有的 handoff store。"""

        if self._durable_inbound_store is not None:
            raise RuntimeError("durable inbound store 已绑定")
        self._durable_inbound_store = store

    async def recover_durable_inbounds(self) -> None:
        """在有界 bus slot 内重放尚未完成的移动 handoff。"""

        self._raise_inbound_cleanup_error()
        store = self._durable_inbound_store
        if store is None:
            return
        # 1. 只读取有限页，避免启动时把整个 durable backlog 搬入内存。
        available = self._inbound_capacity - len(self._inbound_accepted)
        if available <= 0:
            return
        rows = store.list_inbound_handoffs(
            limit=min(self._inbound_capacity, available + len(self._recovery_claimed))
        )
        for row in rows:
            handoff_id = row.get("handoff_id")
            if not isinstance(handoff_id, str) or handoff_id in self._recovery_claimed:
                continue
            if len(self._inbound_accepted) >= self._inbound_capacity:
                break
            item = _inbound_from_handoff(row)
            self._recovery_claimed.add(handoff_id)
            try:
                await self._publish_inbound(item, allow_existing_handoff=True)
            except MessageBusCapacityError as error:
                self._recovery_claimed.discard(handoff_id)
                if error.item_bytes > self._inbound_bytes_capacity:
                    raise
                logger.info(
                    "durable inbound recovery waiting for bus capacity: handoff=%s reason=%s",
                    handoff_id,
                    error.reason,
                )
                break
            except BaseException:
                self._recovery_claimed.discard(handoff_id)
                raise

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

    def bind_outbound_delivery_observer(
        self,
        callback: Callable[[OutboundMessage, bool], Awaitable[None]],
    ) -> None:
        """绑定唯一出站送达观察者。"""

        if self._delivery_observer is not None:
            raise RuntimeError("outbound delivery observer 已绑定")
        self._delivery_observer = callback

    async def publish_inbound(self, msg: InboundItem) -> None:
        """将渠道输入交给 Agent 消费。"""
        self._raise_inbound_cleanup_error()
        await self._publish_inbound(msg, allow_existing_handoff=False)

    async def _publish_inbound(
        self,
        msg: InboundItem,
        *,
        allow_existing_handoff: bool,
    ) -> None:
        """在容量、lane 和 durable handoff 三者一致后入队一条消息。"""

        item_bytes = _wire_size(msg)
        async with self._admission_lock:
            if id(msg) in self._inbound_accepted:
                raise RuntimeError("同一 inbound 对象被重复接受")
            if len(self._inbound_accepted) + self._inbound_reserved_items >= self._inbound_capacity:
                raise MessageBusCapacityError("inbound", item_bytes=item_bytes, reason="queue full")
            if (
                self._inbound_bytes
                + self._inbound_reserved_bytes
                + item_bytes
                > self._inbound_bytes_capacity
            ):
                raise MessageBusCapacityError(
                    "inbound",
                    item_bytes=item_bytes,
                    reason="byte budget full",
                )
            self._inbound_reserved_items += 1
            self._inbound_reserved_bytes += item_bytes
            marked = False
            try:
                if isinstance(msg, InboundMessage) and msg.channel == "mobile":
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
                        allow_existing_handoff
                        and requested_handoff_id == handoff_id
                    ):
                        return
                await self._chat_lane.mark_passive_pending(msg.channel, msg.chat_id)
                marked = True
                self._inbound.put_nowait(msg)
            except BaseException:
                if marked:
                    await self._chat_lane.mark_passive_done(msg.channel, msg.chat_id)
                raise
            finally:
                self._inbound_reserved_items -= 1
                self._inbound_reserved_bytes -= item_bytes
            self._inbound_accepted[id(msg)] = _InboundOwner(
                item=msg,
                item_bytes=item_bytes,
            )
            self._inbound_bytes += item_bytes

    async def consume_inbound(self) -> InboundItem:
        """阻塞直到有消息可消费"""
        return await self._inbound.get()

    async def complete_inbound(self, msg: InboundItem) -> None:
        self._raise_inbound_cleanup_error()
        item_bytes = _wire_size(msg)
        async with self._admission_lock:
            owner = self._inbound_accepted.get(id(msg))
            if owner is None or owner.item is not msg:
                raise RuntimeError("inbound 未被接受或已完成")
            if owner.item_bytes != item_bytes:
                raise RuntimeError("inbound ownership bytes 不一致")
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
                owner = self._inbound_accepted.get(owner_key)
                if owner is None:
                    raise RuntimeError(f"inbound cleanup owner 丢失: {owner_key}")
                if not owner.cleanup_pending:
                    raise RuntimeError(f"inbound cleanup owner 状态非法: {owner_key}")
                item = owner.item
                if not isinstance(item, InboundMessage) or item.handoff_id is None:
                    raise RuntimeError(f"cleanup owner 缺少 mobile handoff: {owner_key}")
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

    async def _finalize_inbound_owner(
        self,
        owner_key: int,
        owner: _InboundOwner,
    ) -> None:
        """确认 lane 完成后释放 accepted owner，并继续有限 durable pump。"""

        item = owner.item
        await self._chat_lane.mark_passive_done(item.channel, item.chat_id)
        async with self._admission_lock:
            current = self._inbound_accepted.pop(owner_key, None)
            if current is not owner:
                raise RuntimeError("inbound ownership changed during completion")
            self._inbound_bytes -= owner.item_bytes
        if isinstance(item, InboundMessage) and item.handoff_id is not None:
            self._recovery_claimed.discard(item.handoff_id)
        await self.recover_durable_inbounds()

    async def aclose(self) -> None:
        """停止出站循环并收束所有 cleanup-only retry task。"""

        self.stop()
        tasks = tuple(self._inbound_cleanup_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._inbound_cleanup_tasks.clear()
        self._raise_inbound_cleanup_error()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """将 Agent 输出交给对应渠道发送。"""
        item_bytes = _wire_size(msg)
        async with self._admission_lock:
            if self._outbound.qsize() + self._outbound_reserved_items >= self._outbound_capacity:
                raise MessageBusCapacityError("outbound", item_bytes=item_bytes, reason="queue full")
            if self._outbound_bytes + self._outbound_reserved_bytes + item_bytes > self._outbound_bytes_capacity:
                raise MessageBusCapacityError("outbound", item_bytes=item_bytes, reason="byte budget full")
            self._outbound_reserved_items += 1
            self._outbound_reserved_bytes += item_bytes
            marked = False
            try:
                await self._chat_lane.mark_passive_send_pending(msg.channel, msg.chat_id)
                marked = True
                self._outbound.put_nowait(msg)
            except BaseException:
                if marked:
                    await self._chat_lane.mark_passive_send_done(msg.channel, msg.chat_id)
                raise
            finally:
                self._outbound_reserved_items -= 1
                self._outbound_reserved_bytes -= item_bytes
            self._outbound_bytes += item_bytes

    def subscribe_outbound(
        self,
        channel: str,
        callback: Callable[[OutboundMessage], Awaitable[None]],
    ) -> OutboundSubscription:
        """订阅某 channel 的出站消息"""
        self._subscribers.setdefault(channel, []).append(callback)
        return OutboundSubscription(self, channel, callback)

    def unsubscribe_outbound(
        self,
        channel: str,
        callback: Callable[[OutboundMessage], Awaitable[None]],
    ) -> None:
        callbacks = self._subscribers.get(channel)
        if callbacks is None:
            return
        try:
            callbacks.remove(callback)
        except ValueError:
            return
        if not callbacks:
            del self._subscribers[channel]

    async def dispatch_outbound(self) -> None:
        """后台任务：将出站消息分发给对应 channel 的订阅者。

        发送失败时退避 2s 重试一次；仍失败则向用户发送降级错误通知，不静默丢弃。
        """
        self._running = True
        while self._running:
            try:
                msg = await asyncio.wait_for(self._outbound.get(), timeout=1.0)
                async with self._admission_lock:
                    self._outbound_bytes -= _wire_size(msg)
                delivered = await self._chat_lane.run_passive(
                    msg.channel,
                    msg.chat_id,
                    lambda: self._send_outbound(msg),
                )
                if self._delivery_observer is not None:
                    await self._delivery_observer(msg, delivered)
            except asyncio.TimeoutError:
                continue

    async def _send_outbound(self, msg: OutboundMessage) -> bool:
        """发送原始消息，并区分原消息与降级文案的结果。"""

        callbacks = tuple(self._subscribers.get(msg.channel, []))
        delivered = bool(callbacks)
        for cb in callbacks:
            try:
                await cb(msg)
            except Exception as first_err:
                logger.warning(
                    f"分发消息到 {msg.channel} 首次失败，2s 后重试: {first_err}"
                )
                await asyncio.sleep(2)
                try:
                    await cb(msg)
                except Exception as second_err:
                    delivered = False
                    logger.error(
                        f"分发消息到 {msg.channel} 重试仍失败，发送降级通知: {second_err}"
                    )
                    fallback = OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="（消息发送失败，请稍后重试）",
                    )
                    try:
                        await cb(fallback)
                    except Exception:
                        logger.error(
                            f"降级通知也失败，消息彻底丢失  channel={msg.channel} "
                            f"chat_id={msg.chat_id}"
                        )
        return delivered

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

    @property
    def inbound_bytes(self) -> int:
        return self._inbound_bytes

    @property
    def outbound_bytes(self) -> int:
        return self._outbound_bytes

    @property
    def inbound_capacity(self) -> int:
        return self._inbound_capacity

    @property
    def outbound_capacity(self) -> int:
        return self._outbound_capacity

    @property
    def inbound_bytes_capacity(self) -> int:
        return self._inbound_bytes_capacity

    @property
    def outbound_bytes_capacity(self) -> int:
        return self._outbound_bytes_capacity

    @property
    def inbound_reserved_bytes(self) -> int:
        return self._inbound_reserved_bytes

    @property
    def outbound_reserved_bytes(self) -> int:
        return self._outbound_reserved_bytes
