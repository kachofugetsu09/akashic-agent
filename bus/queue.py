import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from bus.events import InboundItem, InboundMessage, OutboundMessage, SpawnCompletionItem

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

DEFAULT_MESSAGE_BUS_CAPACITY = 256
DEFAULT_MESSAGE_BUS_BYTES = 4 * 1024 * 1024


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
        item_bytes = _wire_size(msg)
        async with self._admission_lock:
            if self._inbound.qsize() + self._inbound_reserved_items >= self._inbound_capacity:
                raise MessageBusCapacityError("inbound", item_bytes=item_bytes, reason="queue full")
            if self._inbound_bytes + self._inbound_reserved_bytes + item_bytes > self._inbound_bytes_capacity:
                raise MessageBusCapacityError("inbound", item_bytes=item_bytes, reason="byte budget full")
            self._inbound_reserved_items += 1
            self._inbound_reserved_bytes += item_bytes
            marked = False
            try:
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
            self._inbound_bytes += item_bytes

    async def consume_inbound(self) -> InboundItem:
        """阻塞直到有消息可消费"""
        msg = await self._inbound.get()
        async with self._admission_lock:
            self._inbound_bytes -= _wire_size(msg)
        return msg

    async def complete_inbound(self, msg: InboundItem) -> None:
        await self._chat_lane.mark_passive_done(msg.channel, msg.chat_id)

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
