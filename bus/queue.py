import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from bus.events import InboundItem, OutboundMessage

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


@dataclass
class _ChatLaneState:
    condition: asyncio.Condition
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

    def _state(self, channel: str, chat_id: str) -> _ChatLaneState:
        key = (str(channel), str(chat_id))
        state = self._states.get(key)
        if state is None:
            state = _ChatLaneState(condition=asyncio.Condition())
            self._states[key] = state
        return state

    def _skip_cancelled_non_passive(self, state: _ChatLaneState) -> None:
        while state.serving_non_passive_ticket in state.cancelled_non_passive_tickets:
            state.cancelled_non_passive_tickets.remove(
                state.serving_non_passive_ticket
            )
            state.serving_non_passive_ticket += 1

    async def mark_passive_pending(self, channel: str, chat_id: str) -> None:
        state = self._state(channel, chat_id)
        async with state.condition:
            state.passive_turns += 1
            state.condition.notify_all()

    async def mark_passive_done(self, channel: str, chat_id: str) -> None:
        state = self._state(channel, chat_id)
        async with state.condition:
            if state.passive_turns > 0:
                state.passive_turns -= 1
            state.condition.notify_all()

    async def mark_passive_send_pending(self, channel: str, chat_id: str) -> None:
        state = self._state(channel, chat_id)
        async with state.condition:
            state.passive_sends += 1
            state.condition.notify_all()

    async def run_passive(
        self,
        channel: str,
        chat_id: str,
        send: Callable[[], Awaitable[_T]],
    ) -> _T:
        state = self._state(channel, chat_id)
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

    async def run_non_passive(
        self,
        channel: str,
        chat_id: str,
        send: Callable[[], Awaitable[_T]],
    ) -> _T:
        state = self._state(channel, chat_id)
        ticket = -1
        sending = False
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


class MessageBus:
    """agent 与各 channel 之间的异步消息总线"""

    def __init__(self, chat_lane: ChatLane | None = None) -> None:
        self._inbound: asyncio.Queue[InboundItem] = asyncio.Queue()
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._subscribers: dict[
            str, list[Callable[[OutboundMessage], Awaitable[None]]]
        ] = {}
        self._chat_lane = chat_lane or ChatLane()
        self._running = False

    async def publish_inbound(self, msg: InboundItem) -> None:
        """channel → agent"""
        await self._chat_lane.mark_passive_pending(msg.channel, msg.chat_id)
        await self._inbound.put(msg)

    async def consume_inbound(self) -> InboundItem:
        """阻塞直到有消息可消费"""
        return await self._inbound.get()

    async def complete_inbound(self, msg: InboundItem) -> None:
        await self._chat_lane.mark_passive_done(msg.channel, msg.chat_id)

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """agent → channel"""
        await self._chat_lane.mark_passive_send_pending(msg.channel, msg.chat_id)
        await self._outbound.put(msg)

    def subscribe_outbound(
        self,
        channel: str,
        callback: Callable[[OutboundMessage], Awaitable[None]],
    ) -> None:
        """订阅某 channel 的出站消息"""
        self._subscribers.setdefault(channel, []).append(callback)

    async def dispatch_outbound(self) -> None:
        """后台任务：将出站消息分发给对应 channel 的订阅者。

        发送失败时退避 2s 重试一次；仍失败则向用户发送降级错误通知，不静默丢弃。
        """
        self._running = True
        while self._running:
            try:
                msg = await asyncio.wait_for(self._outbound.get(), timeout=1.0)
                await self._chat_lane.run_passive(
                    msg.channel,
                    msg.chat_id,
                    lambda: self._send_outbound(msg),
                )
            except asyncio.TimeoutError:
                continue

    async def _send_outbound(self, msg: OutboundMessage) -> None:
        for cb in self._subscribers.get(msg.channel, []):
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
