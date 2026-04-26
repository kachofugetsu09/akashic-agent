import asyncio
import logging
from collections.abc import Awaitable, Callable

from bus.events import InboundItem, OutboundMessage

logger = logging.getLogger(__name__)


class MessageBus:
    """agent 与各 channel 之间的异步消息总线"""

    def __init__(self) -> None:
        self._inbound: asyncio.Queue[InboundItem] = asyncio.Queue()
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._subscribers: dict[
            str, list[Callable[[OutboundMessage], Awaitable[None]]]
        ] = {}
        self._running = False

    async def publish_inbound(self, msg: InboundItem) -> None:
        """channel → agent"""
        await self._inbound.put(msg)

    async def consume_inbound(self) -> InboundItem:
        """阻塞直到有消息可消费"""
        return await self._inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """agent → channel"""
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
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        self._running = False

    @property
    def inbound_size(self) -> int:
        return self._inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self._outbound.qsize()
