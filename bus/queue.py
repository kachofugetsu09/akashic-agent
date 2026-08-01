import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeVar
from typing import Protocol, cast
from uuid import uuid4

from bus.events import InboundItem, InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_DURABLE_INBOUND_RECOVERY_PAGE_SIZE = 256
_INBOUND_CLEANUP_RETRY_INITIAL_DELAY = 0.1
_INBOUND_CLEANUP_RETRY_MAX_DELAY = 5.0


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
    """在 durable cleanup 确认前保持 mobile handoff 的强引用。"""

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
    """在单用户 Companion 内传递消息，并持有 mobile handoff 的删除责任。"""

    def __init__(self, chat_lane: ChatLane | None = None) -> None:
        self._inbound: asyncio.Queue[InboundItem] = asyncio.Queue()
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._inbound_accepted: dict[int, _InboundOwner] = {}
        self._inbound_cleanup_tasks: dict[int, asyncio.Task[None]] = {}
        self._inbound_cleanup_error: BaseException | None = None
        self._recovery_claimed: set[str] = set()
        self._durable_handoff_lock = asyncio.Lock()
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
        """分页重放尚未完成的移动 handoff，不以 bus 容量拒绝消息。"""

        self._raise_inbound_cleanup_error()
        store = self._durable_inbound_store
        if store is None:
            return
        # 1. 只读取有限页，避免启动时把整个 durable backlog 搬入内存。
        rows = store.list_inbound_handoffs(limit=_DURABLE_INBOUND_RECOVERY_PAGE_SIZE)
        for row in rows:
            handoff_id = row.get("handoff_id")
            if not isinstance(handoff_id, str) or handoff_id in self._recovery_claimed:
                continue
            item = _inbound_from_handoff(row)
            self._recovery_claimed.add(handoff_id)
            try:
                await self._publish_inbound(item, allow_existing_handoff=True)
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

    async def consume_inbound(self) -> InboundItem:
        """阻塞直到有消息可消费"""
        return await self._inbound.get()

    async def complete_inbound(self, msg: InboundItem) -> None:
        self._raise_inbound_cleanup_error()
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
        await self._chat_lane.mark_passive_send_pending(msg.channel, msg.chat_id)
        try:
            self._outbound.put_nowait(msg)
        except BaseException:
            await self._chat_lane.mark_passive_send_done(msg.channel, msg.chat_id)
            raise

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
